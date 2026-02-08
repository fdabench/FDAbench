# -*- coding: utf-8 -*-
"""Graph node functions for tree-structured exploration (Algorithm 1).

Nodes are grouped by phase:
  Phase 0: Query loading and base context
  Phase 1: Tree-structured exploration
  Phase 2: Report generation and expert review
  Phase 3: Validation and finalization
"""

import json
import sys
import logging
from typing import Any, Dict

from langgraph.types import interrupt

from .state import TreeExplorationState
from ..models.tree_models import (
    ExplorationTree, ExplorationNode, TerminalPath, ToolAction, BranchDecision,
)
from ..tools.tool_executor import ToolExecutor
from ..exploration.candidate_spawner import CandidateSpawner
from ..exploration.branch_reflector import BranchReflector
from ..generators.report_generator import ReportGenerator
from ..validation.single_source_validator import SingleSourceValidator
from ..validation.dag_annotator import DAGAnnotator
from ..utils.io import load_queries as load_queries_from_file, append_to_output

logger = logging.getLogger(__name__)

_tool_executor: ToolExecutor = None
_candidate_spawner: CandidateSpawner = None
_branch_reflector: BranchReflector = None
_report_generator: ReportGenerator = None
_single_source_validator: SingleSourceValidator = None
_dag_annotator: DAGAnnotator = None


def _get_tool_executor(vector_index_path: str = None, sql_dir: str = None,
                       gold_result_dir: str = None) -> ToolExecutor:
    global _tool_executor
    if _tool_executor is None:
        _tool_executor = ToolExecutor(
            vector_index_path=vector_index_path,
            sql_dir=sql_dir,
            gold_result_dir=gold_result_dir,
        )
    return _tool_executor


def _get_candidate_spawner() -> CandidateSpawner:
    global _candidate_spawner
    if _candidate_spawner is None:
        _candidate_spawner = CandidateSpawner()
    return _candidate_spawner


def _get_branch_reflector() -> BranchReflector:
    global _branch_reflector
    if _branch_reflector is None:
        _branch_reflector = BranchReflector()
    return _branch_reflector


def _get_report_generator() -> ReportGenerator:
    global _report_generator
    if _report_generator is None:
        _report_generator = ReportGenerator()
    return _report_generator


def _get_single_source_validator() -> SingleSourceValidator:
    global _single_source_validator
    if _single_source_validator is None:
        _single_source_validator = SingleSourceValidator()
    return _single_source_validator


def _get_dag_annotator() -> DAGAnnotator:
    global _dag_annotator
    if _dag_annotator is None:
        _dag_annotator = DAGAnnotator()
    return _dag_annotator


def load_queries(state: TreeExplorationState) -> Dict[str, Any]:
    """Load queries from JSONL input file."""
    input_path = state.get("input_path", "")
    query_limit = state.get("_query_limit")
    logger.info(f"Loading queries from: {input_path}")
    try:
        queries = load_queries_from_file(input_path)
        if query_limit:
            queries = queries[:query_limit]
            logger.info(f"Applied limit: {query_limit}")
        total = len(queries)
        logger.info(f"Loaded {total} queries")
        return {
            "queries": queries,
            "total_queries": total,
            "current_query_idx": 0,
            "should_continue": total > 0,
            "error_message": None if total > 0 else "No queries found",
        }
    except Exception as e:
        logger.error(f"Failed to load queries: {e}")
        return {
            "queries": [],
            "total_queries": 0,
            "should_continue": False,
            "error_message": str(e),
        }


def load_next_query(state: TreeExplorationState) -> Dict[str, Any]:
    """Extract current query fields and reset per-query state."""
    queries = state.get("queries", [])
    idx = state.get("current_query_idx", 0)

    if idx >= len(queries):
        return {"should_continue": False, "error_message": "No more queries"}

    item = queries[idx]
    instance_id = item.get("instance_id") or item.get("id") or str(idx)
    db = item.get("db_id") or item.get("db") or ""

    # Infer database_type from instance_id prefix if not provided
    database_type = item.get("database_type", "")
    if not database_type:
        iid = instance_id.lower()
        if iid.startswith("bird"):
            database_type = "bird"
        elif iid.startswith("spider"):
            database_type = "Spider2-lite"
        elif iid.startswith("bq") or iid.startswith("sf_bq"):
            database_type = "Spider2-lite"
        elif iid.startswith("local"):
            database_type = "local"
        else:
            database_type = "bird"

    task_id = item.get("task_id", f"FDA{idx + 1:04d}")
    original_query = (
        item.get("instruction")
        or item.get("question")
        or item.get("query")
        or ""
    )

    logger.info(f"Processing query {idx + 1}/{len(queries)}: {instance_id}")

    return {
        "instance_id": instance_id,
        "db": db,
        "database_type": database_type,
        "task_id": task_id,
        "original_query": original_query,
        # Reset per-query state
        "exploration_tree": {},
        "current_iteration": 0,
        "candidate_actions": [],
        "current_candidates_processed": False,
        "terminal_paths": [],
        "current_path_idx": 0,
        "enhanced_query": "",
        "report_draft": "",
        "reflection": "",
        "expert_status": "",
        "expert_feedback": "",
        "reinit_feedback": "",
        "revision_count": 0,
        "is_multi_source": False,
        "annotated_dag": {},
        "annotated_rubric": {},
        "annotated_subtasks": [],
        "error_message": None,
    }


def build_base_context(state: TreeExplorationState) -> Dict[str, Any]:
    """Phase 0: Build σ₀ = ⟨SchemaExtract, ExecuteSQL, EnterpriseKB⟩."""
    instance_id = state.get("instance_id", "")
    gold_result_dir = state.get("gold_result_dir", "")
    sql_dir = state.get("sql_dir", "")
    vector_index_path = state.get("vector_index_path", "")

    logger.info(f"Building base context for: {instance_id}")

    executor = _get_tool_executor(vector_index_path, sql_dir, gold_result_dir)
    sql_result, sql_statement = executor.get_base_context(
        instance_id, gold_result_dir, sql_dir
    )

    logger.info(f"Base context: SQL result={len(sql_result)} chars, SQL stmt={len(sql_statement)} chars")

    return {
        "sql_result": sql_result,
        "sql_statement": sql_statement,
    }


def init_exploration(state: TreeExplorationState) -> Dict[str, Any]:
    """Initialize ExplorationTree with root node containing base context."""
    max_iterations = state.get("max_iterations", 3)

    tree = ExplorationTree(max_iterations=max_iterations)
    tree.create_root(base_actions=[])  # Root has no search actions yet

    logger.info(f"Initialized exploration tree (max_iterations={max_iterations})")

    return {
        "exploration_tree": tree.to_dict(),
        "current_iteration": 0,
    }


def spawn_candidates(state: TreeExplorationState) -> Dict[str, Any]:
    """SpawnCandidates: For each frontier node, propose 1-4 candidate branches."""
    tree_dict = state.get("exploration_tree", {})
    tree = ExplorationTree.from_dict(tree_dict)
    original_query = state.get("original_query", "")
    sql_result = state.get("sql_result", "")
    reinit_feedback = state.get("reinit_feedback", "")

    spawner = _get_candidate_spawner()
    all_candidates = []

    frontier_nodes = tree.get_frontier_nodes()
    logger.info(f"Spawning candidates for {len(frontier_nodes)} frontier nodes (iteration {tree.iteration})")

    for node in frontier_nodes:
        candidates = spawner.spawn_candidates(
            node, original_query, sql_result,
            revision_feedback=reinit_feedback,
        )
        for candidate in candidates:
            all_candidates.append({
                "parent_id": node.node_id,
                "action": candidate.to_dict(),
            })

    logger.info(f"Generated {len(all_candidates)} total candidates")

    # Clear frontier - nodes will be re-added based on reflection decisions
    tree.frontier = []

    return {
        "exploration_tree": tree.to_dict(),
        "candidate_actions": all_candidates,
        "current_candidates_processed": False,
        "reinit_feedback": "",  # Clear after use
    }


def execute_and_reflect(state: TreeExplorationState) -> Dict[str, Any]:
    """Execute each candidate and reflect to decide PRUNE/CONTINUE/SUFFICIENT.

    Processes all candidates sequentially (LLM is the bottleneck).
    """
    tree_dict = state.get("exploration_tree", {})
    tree = ExplorationTree.from_dict(tree_dict)
    candidates = state.get("candidate_actions", [])
    original_query = state.get("original_query", "")
    sql_result = state.get("sql_result", "")
    vector_index_path = state.get("vector_index_path", "")

    executor = _get_tool_executor(vector_index_path)
    reflector = _get_branch_reflector()

    for candidate_info in candidates:
        parent_id = candidate_info["parent_id"]
        action = ToolAction.from_dict(candidate_info["action"])

        # Execute the tool
        executor.execute(action)

        # Add child node to tree
        child = tree.add_node(parent_id, action)

        # Self-reflect on the child node
        decision, rationale = reflector.reflect(child, original_query, sql_result)

        if decision == BranchDecision.CONTINUE:
            tree.mark_continue(child.node_id, rationale)
            logger.info(f"  Node {child.node_id}: CONTINUE - {rationale[:80]}")
        elif decision == BranchDecision.SUFFICIENT:
            tree.mark_sufficient(child.node_id, rationale)
            logger.info(f"  Node {child.node_id}: SUFFICIENT - {rationale[:80]}")
        elif decision == BranchDecision.PRUNE:
            tree.mark_pruned(child.node_id, rationale)
            logger.info(f"  Node {child.node_id}: PRUNE - {rationale[:80]}")

    # Advance iteration
    tree.advance_iteration()

    logger.info(
        f"After iteration {tree.iteration}: "
        f"frontier={len(tree.frontier)}, "
        f"terminal_paths={len(tree.terminal_paths)}, "
        f"pruned={len(tree.pruned)}"
    )

    # If no terminal paths and no frontier, create a fallback terminal path from root
    if tree.is_complete() and not tree.terminal_paths:
        logger.warning("No terminal paths found, creating fallback from deepest node")
        _create_fallback_terminal_path(tree)

    return {
        "exploration_tree": tree.to_dict(),
        "terminal_paths": [p.to_dict() for p in tree.terminal_paths],
        "candidate_actions": [],
        "current_candidates_processed": True,
    }


def _create_fallback_terminal_path(tree: ExplorationTree):
    """Create a fallback terminal path from the deepest non-pruned node."""
    best_node = None
    best_depth = -1
    for node in tree.nodes.values():
        if node.decision != BranchDecision.PRUNE and node.depth > best_depth:
            best_depth = node.depth
            best_node = node
    if best_node and best_node.cumulative_actions:
        tree.mark_sufficient(best_node.node_id, "Fallback: no SUFFICIENT path found")


def select_next_path(state: TreeExplorationState) -> Dict[str, Any]:
    """Pick the next TerminalPath for report generation."""
    terminal_paths = state.get("terminal_paths", [])
    current_path_idx = state.get("current_path_idx", 0)

    if current_path_idx >= len(terminal_paths):
        logger.info("No more terminal paths to process")
        return {"current_path_idx": current_path_idx}

    path_dict = terminal_paths[current_path_idx]
    path = TerminalPath.from_dict(path_dict)
    logger.info(
        f"Selected path {current_path_idx + 1}/{len(terminal_paths)}: "
        f"{path.path_id} ({len(path.actions)} actions)"
    )

    # Reset report state for this path
    generator = _get_report_generator()
    generator.reset_conversation()

    return {
        "revision_count": 0,
        "expert_status": "",
        "expert_feedback": "",
        "enhanced_query": "",
        "report_draft": "",
        "reflection": "",
    }


def generate_report_draft(state: TreeExplorationState) -> Dict[str, Any]:
    """Generate enhanced_query + ground_truth_report from terminal path."""
    terminal_paths = state.get("terminal_paths", [])
    current_path_idx = state.get("current_path_idx", 0)
    original_query = state.get("original_query", "")
    sql_result = state.get("sql_result", "")

    path = TerminalPath.from_dict(terminal_paths[current_path_idx])

    generator = _get_report_generator()

    # Generate enhanced query
    enhanced_query = generator.generate_enhanced_query(original_query, path)
    logger.info(f"Enhanced query: {enhanced_query[:100]}...")

    # Generate report
    report = generator.generate_report(enhanced_query, sql_result, path)
    logger.info(f"Generated report: {len(report)} chars")

    return {
        "enhanced_query": enhanced_query,
        "report_draft": report,
    }


def auto_reflect(state: TreeExplorationState) -> Dict[str, Any]:
    """Automated quality self-reflection on the report draft."""
    report = state.get("report_draft", "")
    original_query = state.get("original_query", "")
    sql_result = state.get("sql_result", "")
    terminal_paths = state.get("terminal_paths", [])
    current_path_idx = state.get("current_path_idx", 0)

    path = TerminalPath.from_dict(terminal_paths[current_path_idx])

    generator = _get_report_generator()
    reflection = generator.reflect_on_report(report, original_query, sql_result, path)

    logger.info(f"Auto-reflection: {reflection[:100]}...")

    return {"reflection": reflection}


def display_draft(state: TreeExplorationState) -> Dict[str, Any]:
    """Display the report draft and reflection to the user."""
    enhanced_query = state.get("enhanced_query", "")
    report = state.get("report_draft", "")
    reflection = state.get("reflection", "")
    instance_id = state.get("instance_id", "")
    current_path_idx = state.get("current_path_idx", 0)
    total_paths = len(state.get("terminal_paths", []))

    print(f"\n{'='*60}")
    print(f"REPORT DRAFT - {instance_id} (Path {current_path_idx + 1}/{total_paths})")
    print(f"{'='*60}")
    print(f"\nEnhanced Query:\n{enhanced_query}")
    print(f"\nReport:\n{report}")
    print(f"\nAuto-Reflection:\n{reflection}")
    print(f"{'='*60}")

    return {}


def expert_review(state: TreeExplorationState) -> Dict[str, Any]:
    """Interrupt for expert review: ACCEPT / REVISE / REJECT."""
    interactive_mode = state.get("interactive_mode", True)

    if not interactive_mode or not sys.stdin.isatty():
        logger.info("Non-interactive mode - auto-accepting report")
        return {"expert_status": "ACCEPT", "expert_feedback": ""}

    feedback = interrupt({
        "type": "expert_review",
        "enhanced_query": state.get("enhanced_query", ""),
        "report_draft": state.get("report_draft", ""),
        "reflection": state.get("reflection", ""),
        "options": ["(a) Accept", "(r) Revise", "(d) Reject"],
        "revision_count": state.get("revision_count", 0),
        "max_revisions": state.get("max_revisions", 3),
        "prompt": "Your choice (a/r/d): ",
    })

    choice = feedback.get("choice", "a")
    status_map = {"a": "ACCEPT", "r": "REVISE", "d": "REJECT"}
    expert_status = status_map.get(choice, "ACCEPT")

    return {
        "expert_status": expert_status,
        "expert_feedback": feedback.get("feedback", ""),
    }


def reinit_exploration(state: TreeExplorationState) -> Dict[str, Any]:
    """Re-enter Phase 1 after expert REVISE: reopen the leaf node for more exploration.

    - Re-opens the current terminal path's leaf node (SUFFICIENT → CONTINUE)
    - Removes the current terminal path from the list
    - Stores expert feedback for candidate spawner to use
    - Increments revision_count
    """
    tree_dict = state.get("exploration_tree", {})
    tree = ExplorationTree.from_dict(tree_dict)
    terminal_paths = state.get("terminal_paths", [])
    current_path_idx = state.get("current_path_idx", 0)
    expert_feedback = state.get("expert_feedback", "")
    revision_count = state.get("revision_count", 0)

    if current_path_idx < len(terminal_paths):
        path = TerminalPath.from_dict(terminal_paths[current_path_idx])

        # Find the leaf node of this path and re-open it
        for node in tree.nodes.values():
            if (node.decision == BranchDecision.SUFFICIENT
                    and node.cumulative_actions
                    and len(node.cumulative_actions) == len(path.actions)):
                # Re-open: change SUFFICIENT → CONTINUE, add back to frontier
                node.decision = BranchDecision.CONTINUE
                node.reflection_rationale = f"Re-opened for revision: {expert_feedback[:100]}"
                if node.node_id not in tree.frontier:
                    tree.frontier.append(node.node_id)
                logger.info(f"Re-opened node {node.node_id} for revision")
                break

        # Remove the terminal path that was re-opened
        tree.terminal_paths = [
            tp for i, tp in enumerate(tree.terminal_paths)
            if tp.path_id != path.path_id
        ]

        # Update the serialized terminal paths list too
        updated_paths = [p for i, p in enumerate(terminal_paths) if i != current_path_idx]
    else:
        updated_paths = terminal_paths

    logger.info(
        f"Reinit exploration: feedback='{expert_feedback[:80]}', "
        f"revision_count={revision_count + 1}, frontier={len(tree.frontier)}"
    )

    return {
        "exploration_tree": tree.to_dict(),
        "terminal_paths": [p if isinstance(p, dict) else p.to_dict()
                           for p in tree.terminal_paths],
        "reinit_feedback": expert_feedback,
        "revision_count": revision_count + 1,
        # Reset path index since we removed a path
        "current_path_idx": 0,
    }


def validate_single_source(state: TreeExplorationState) -> Dict[str, Any]:
    """Check if the task truly requires multiple sources."""
    original_query = state.get("original_query", "")
    sql_result = state.get("sql_result", "")
    report = state.get("report_draft", "")
    terminal_paths = state.get("terminal_paths", [])
    current_path_idx = state.get("current_path_idx", 0)

    path = TerminalPath.from_dict(terminal_paths[current_path_idx])

    validator = _get_single_source_validator()
    is_multi = validator.is_multi_source(original_query, sql_result, report, path)

    logger.info(f"Single-source validation: is_multi_source={is_multi}")

    if not is_multi:
        return {
            "is_multi_source": False,
            "skipped_count": state.get("skipped_count", 0) + 1,
        }

    return {"is_multi_source": True}


def annotate_dag(state: TreeExplorationState) -> Dict[str, Any]:
    """Build DAG, rubric, gold_subtasks from exploration trace."""
    terminal_paths = state.get("terminal_paths", [])
    current_path_idx = state.get("current_path_idx", 0)
    sql_result = state.get("sql_result", "")
    sql_statement = state.get("sql_statement", "")
    enhanced_query = state.get("enhanced_query", "")
    report = state.get("report_draft", "")
    queries = state.get("queries", [])
    current_query_idx = state.get("current_query_idx", 0)
    interactive_mode = state.get("interactive_mode", True)

    path = TerminalPath.from_dict(terminal_paths[current_path_idx])
    record = dict(queries[current_query_idx]) if current_query_idx < len(queries) else {}
    # Ensure task_id from state is available to annotator
    record["task_id"] = state.get("task_id", record.get("task_id", "unknown"))

    annotator = _get_dag_annotator()
    annotations = annotator.annotate(
        record=record,
        terminal_path=path,
        sql_result=sql_result,
        sql_statement=sql_statement,
        enhanced_query=enhanced_query,
        report=report,
    )

    level = annotations["level"]

    # Interactive override for difficulty
    if interactive_mode and sys.stdin.isatty():
        print(f"\nDifficulty recommendation: {level}")
        override = input("Override? (e)asy/(m)edium/(h)ard/[Enter=accept]: ").strip().lower()
        if override in ("e", "easy"):
            level = "easy"
        elif override in ("m", "medium"):
            level = "medium"
        elif override in ("h", "hard"):
            level = "hard"
        annotations["level"] = level

    logger.info(
        f"Annotated: level={level}, "
        f"subtasks={len(annotations['gold_subtasks'])}, "
        f"dag_nodes={len(annotations['dag']['nodes'])}"
    )

    return {
        "annotated_dag": annotations["dag"],
        "annotated_rubric": annotations["rubric"],
        "annotated_subtasks": annotations["gold_subtasks"],
        "difficulty_level": level,
        "frozen_web_search": annotations["frozen_web_search"],
        "frozen_vector_search": annotations["frozen_vector_search"],
        "db_type": annotations["db_type"],
        "tools_available": annotations["tools_available"],
    }


def save_entry(state: TreeExplorationState) -> Dict[str, Any]:
    """Serialize and save the complete task entry to output JSONL."""
    output_path = state.get("output_path", "")
    instance_id = state.get("instance_id", "")
    task_id = state.get("task_id", "")
    db = state.get("db", "")
    database_type = state.get("database_type", "")
    original_query = state.get("original_query", "")

    entry = {
        "task_id": task_id,
        "instance_id": instance_id,
        "db": db,
        "db_type": state.get("db_type", "sqlite"),
        "level": state.get("difficulty_level", "hard"),
        "database_type": database_type,
        "question_type": "report",
        "tools_available": state.get("tools_available", []),
        "gold_subtasks": state.get("annotated_subtasks", []),
        "query": state.get("enhanced_query", original_query),
        "ground_truth_report": state.get("report_draft", ""),
        "dag": state.get("annotated_dag", {}),
        "rubric": state.get("annotated_rubric", {}),
        "frozen_web_search": state.get("frozen_web_search", {"searches": []}),
        "frozen_vector_search": state.get("frozen_vector_search", {"searches": []}),
        "sql_result": state.get("sql_result", ""),
    }

    try:
        append_to_output(entry, output_path)
        logger.info(f"Entry saved: {task_id} ({instance_id})")
        print(f"\nEntry saved: {task_id} ({instance_id})\n")
        return {
            "output_entries": [entry],
            "accepted_count": state.get("accepted_count", 0) + 1,
            "processed_count": state.get("processed_count", 0) + 1,
        }
    except Exception as e:
        logger.error(f"Failed to save entry: {e}")
        return {
            "error_count": state.get("error_count", 0) + 1,
            "error_message": str(e),
        }


def route_next_path_node(state: TreeExplorationState) -> Dict[str, Any]:
    """Advance to next terminal path."""
    current_path_idx = state.get("current_path_idx", 0)
    return {"current_path_idx": current_path_idx + 1}


def check_more_queries(state: TreeExplorationState) -> Dict[str, Any]:
    """Increment query index and check if more queries remain."""
    current_idx = state.get("current_query_idx", 0)
    total_queries = state.get("total_queries", 0)
    next_idx = current_idx + 1
    has_more = next_idx < total_queries

    logger.info(f"Progress: {next_idx}/{total_queries}, has_more={has_more}")

    return {
        "current_query_idx": next_idx,
        "should_continue": has_more,
        "processed_count": state.get("processed_count", 0),
    }
