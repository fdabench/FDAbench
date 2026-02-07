# -*- coding: utf-8 -*-
"""
DAG Evaluator - Evaluation system for DAG-based task execution.

This module provides comprehensive evaluation of agent performance on DAG-structured
tasks, including:
- Graph coverage metrics (required, critical path, optional nodes)
- Tool use quality (recall, precision, parameter accuracy)
- Dependency violation detection
- End-to-end scoring integration
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Set, Tuple

# Import from PUDDING models
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from PUDDING.models.dag_models import (
    TaskDAG, DAGNode, DAGEdge, AlternativeGroup,
    NodeType, EdgeType, NodeStatus, DAGExecutionState,
    get_node_type_for_tool,
)

logger = logging.getLogger(__name__)


@dataclass
class DAGEvaluationResult:
    """
    Result of DAG-based evaluation.

    Contains metrics across three layers:
    1. Graph Coverage - How well did the agent cover the expected workflow
    2. Tool Use Quality - How well did the agent use tools
    3. End-to-End - Final answer quality (integrated from rubric evaluator)
    """
    # Graph Coverage Metrics
    required_node_coverage: float = 0.0      # Required nodes completed / total required
    critical_path_coverage: float = 0.0      # Critical path nodes completed / total critical
    optional_node_coverage: float = 0.0      # Optional nodes completed / total optional
    alt_group_satisfaction: Dict[str, bool] = field(default_factory=dict)  # group_id -> satisfied

    # Tool Use Quality Metrics
    tool_recall: float = 0.0                 # Tools used that should have been used
    tool_precision: float = 0.0              # Tools used that were correct
    tool_f1: float = 0.0                     # Harmonic mean of recall and precision
    param_accuracy: float = 0.0              # Accuracy of tool parameters
    sequence_sanity: float = 1.0             # 1.0 if no hard_dep violations, 0 otherwise

    # Detail Information
    completed_nodes: List[str] = field(default_factory=list)
    missed_required_nodes: List[str] = field(default_factory=list)
    extra_nodes: List[str] = field(default_factory=list)
    dep_violations: List[Tuple[str, str]] = field(default_factory=list)  # (node, missing_dep)
    execution_order: List[str] = field(default_factory=list)

    # End-to-End Score (from rubric evaluator)
    end_to_end_score: float = 0.0

    # Composite Score
    composite_score: float = 0.0

    def compute_composite_score(
        self,
        graph_weight: float = 0.5,
        tool_weight: float = 0.5,
    ) -> float:
        """
        Compute weighted composite score (TOS).

        TOS measures tool orchestration quality only, independent of
        rubric-based report scoring (RS).

        Args:
            graph_weight: Weight for graph coverage
            tool_weight: Weight for tool use quality

        Returns:
            Weighted composite score
        """
        # Graph coverage score (average of required and critical path)
        graph_score = (self.required_node_coverage + self.critical_path_coverage) / 2

        # Tool use score (F1 weighted by sequence sanity)
        tool_score = self.tool_f1 * self.sequence_sanity

        # Composite (pure tool metrics, no RS)
        self.composite_score = (
            graph_weight * graph_score +
            tool_weight * tool_score
        )
        return self.composite_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            # Graph coverage
            "required_node_coverage": self.required_node_coverage,
            "critical_path_coverage": self.critical_path_coverage,
            "optional_node_coverage": self.optional_node_coverage,
            "alt_group_satisfaction": self.alt_group_satisfaction,

            # Tool use quality
            "tool_recall": self.tool_recall,
            "tool_precision": self.tool_precision,
            "tool_f1": self.tool_f1,
            "param_accuracy": self.param_accuracy,
            "sequence_sanity": self.sequence_sanity,

            # Details
            "completed_nodes": self.completed_nodes,
            "missed_required_nodes": self.missed_required_nodes,
            "extra_nodes": self.extra_nodes,
            "dep_violations": self.dep_violations,
            "execution_order": self.execution_order,

            # End-to-end and composite
            "end_to_end_score": self.end_to_end_score,
            "composite_score": self.composite_score,
        }

    def to_database_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for DuckDB storage."""
        return {
            # Graph coverage
            "dag_required_node_coverage": self.required_node_coverage,
            "dag_critical_path_coverage": self.critical_path_coverage,
            "dag_optional_node_coverage": self.optional_node_coverage,
            "dag_alt_group_satisfaction": json.dumps(self.alt_group_satisfaction),

            # Tool use quality
            "dag_tool_recall": self.tool_recall,
            "dag_tool_precision": self.tool_precision,
            "dag_tool_f1": self.tool_f1,
            "dag_param_accuracy": self.param_accuracy,
            "dag_sequence_sanity": self.sequence_sanity,

            # Details
            "dag_completed_nodes": json.dumps(self.completed_nodes),
            "dag_missed_required_nodes": json.dumps(self.missed_required_nodes),
            "dag_extra_nodes": json.dumps(self.extra_nodes),
            "dag_dep_violations": json.dumps(self.dep_violations),
            "dag_execution_order": json.dumps(self.execution_order),

            # Composite
            "dag_composite_score": self.composite_score,
        }


class DAGEvaluator:
    """
    Evaluator for DAG-based task execution.

    Evaluates agent performance against a gold-standard TaskDAG.
    """

    def __init__(self):
        """Initialize the DAG evaluator."""
        # Tool name aliases for matching
        self.tool_aliases: Dict[str, Set[str]] = {
            "get_schema_info": {"schema_understanding", "get_schema", "schema"},
            "generated_sql": {"sql_generate", "generate_sql", "sql_gen"},
            "execute_sql": {"sql_execute", "run_sql", "exec_sql"},
            "perplexity_search": {"web_search", "web_context_search", "search"},
            "vectorDB_search": {"vector_search", "vec_search", "embedding_search"},
            "file_system": {"file_search", "file_system_search", "fs_search"},
            "sql_optimize": {"optimize_sql", "sql_optimization"},
            "sql_debug": {"debug_sql", "sql_debugging"},
        }

    def evaluate(
        self,
        gold_dag: TaskDAG,
        agent_execution: Dict[str, Any],
        test_case: Optional[Dict[str, Any]] = None,
        end_to_end_score: float = 0.0,
    ) -> DAGEvaluationResult:
        """
        Evaluate agent execution against a gold-standard DAG.

        Args:
            gold_dag: The expected TaskDAG structure
            agent_execution: Agent's execution data containing:
                - tools_executed: List[str] - Tools that were executed
                - tool_results: Dict[str, Any] - Results from each tool
                - execution_order: List[str] - Order of tool execution
                - subtask_results: Dict[str, Any] - Subtask results (optional)
            test_case: Original test case data (optional)
            end_to_end_score: End-to-end score from rubric evaluator

        Returns:
            DAGEvaluationResult with all metrics
        """
        result = DAGEvaluationResult()
        result.end_to_end_score = end_to_end_score

        # Extract execution data
        tools_executed = self._extract_tools_executed(agent_execution)
        tool_results = agent_execution.get("tool_results", {})
        execution_order = agent_execution.get("execution_order", tools_executed)

        # Store execution order
        result.execution_order = execution_order

        # Compute graph coverage
        self._compute_graph_coverage(gold_dag, tools_executed, result)

        # Compute tool metrics
        self._compute_tool_metrics(gold_dag, tools_executed, result)

        # Check dependency violations
        self._check_dependency_violations(gold_dag, execution_order, result)

        # Compute parameter accuracy if we have tool results
        if tool_results:
            self._compute_param_accuracy(gold_dag, tool_results, result)

        # Compute composite score
        result.compute_composite_score()

        return result

    def _extract_tools_executed(self, agent_execution: Dict[str, Any]) -> List[str]:
        """Extract list of tools executed from agent execution data."""
        # Try different locations for tools executed
        if "tools_executed" in agent_execution:
            return agent_execution["tools_executed"]

        if "metrics" in agent_execution:
            metrics = agent_execution["metrics"]
            if "tools_executed" in metrics:
                return metrics["tools_executed"]
            if "completed_tools" in metrics:
                return metrics["completed_tools"]

        # Try to extract from subtask_results
        if "subtask_results" in agent_execution:
            tools = []
            for subtask_id, result in agent_execution["subtask_results"].items():
                if isinstance(result, dict) and result.get("status") == "success":
                    tool = result.get("tool", subtask_id)
                    tools.append(tool)
            return tools

        return []

    def _normalize_tool_name(self, tool: str) -> str:
        """Normalize tool name to canonical form."""
        tool_lower = tool.lower().strip()

        # Check if it's an alias
        for canonical, aliases in self.tool_aliases.items():
            if tool_lower == canonical.lower() or tool_lower in {a.lower() for a in aliases}:
                return canonical

        return tool

    def _tools_match(self, tool1: str, tool2: str) -> bool:
        """Check if two tool names refer to the same tool."""
        return self._normalize_tool_name(tool1) == self._normalize_tool_name(tool2)

    def _match_execution_to_nodes(
        self,
        gold_dag: TaskDAG,
        tools_executed: List[str],
    ) -> Dict[str, str]:
        """
        Match executed tools to DAG nodes in order, considering dependencies.

        Returns a mapping of node_id -> executed_tool for matched nodes.
        This handles the case where the same tool is used by multiple nodes.
        """
        # Track which nodes have been matched
        matched_nodes: Dict[str, str] = {}
        completed_node_ids: Set[str] = set()

        # Build tool -> nodes mapping
        tool_to_nodes: Dict[str, List[str]] = {}
        for node_id, node in gold_dag.nodes.items():
            normalized_tool = self._normalize_tool_name(node.tool)
            if normalized_tool not in tool_to_nodes:
                tool_to_nodes[normalized_tool] = []
            tool_to_nodes[normalized_tool].append(node_id)

        # Process executed tools in order
        for executed_tool in tools_executed:
            normalized = self._normalize_tool_name(executed_tool)
            if normalized not in tool_to_nodes:
                continue

            # Find the first unmatched node that uses this tool and has deps satisfied
            candidate_nodes = tool_to_nodes[normalized]

            for node_id in candidate_nodes:
                if node_id in matched_nodes:
                    continue

                # Check if hard dependencies are satisfied
                hard_deps = gold_dag.get_hard_dependencies(node_id)
                deps_satisfied = all(dep_id in completed_node_ids for dep_id in hard_deps)

                if deps_satisfied:
                    matched_nodes[node_id] = executed_tool
                    completed_node_ids.add(node_id)
                    break
            else:
                # No suitable node found with deps satisfied
                # Match to first unmatched node anyway (for coverage counting)
                for node_id in candidate_nodes:
                    if node_id not in matched_nodes:
                        matched_nodes[node_id] = executed_tool
                        completed_node_ids.add(node_id)
                        break

        return matched_nodes

    def _compute_graph_coverage(
        self,
        gold_dag: TaskDAG,
        tools_executed: List[str],
        result: DAGEvaluationResult,
    ) -> None:
        """Compute graph coverage metrics."""
        # Get expected nodes
        required_nodes = gold_dag.get_required_nodes()
        critical_path = gold_dag.critical_path
        optional_nodes = gold_dag.get_optional_nodes()

        # Match executed tools to nodes in order
        matched_nodes = self._match_execution_to_nodes(gold_dag, tools_executed)
        matched_node_ids = set(matched_nodes.keys())

        # Calculate required node coverage
        required_completed = 0
        missed_required = []
        for node_id in required_nodes:
            if node_id in matched_node_ids:
                required_completed += 1
                result.completed_nodes.append(node_id)
            else:
                missed_required.append(node_id)

        result.required_node_coverage = (
            required_completed / len(required_nodes) if required_nodes else 1.0
        )
        result.missed_required_nodes = missed_required

        # Calculate critical path coverage
        critical_completed = 0
        for node_id in critical_path:
            if node_id in matched_node_ids:
                critical_completed += 1

        result.critical_path_coverage = (
            critical_completed / len(critical_path) if critical_path else 1.0
        )

        # Calculate optional node coverage
        optional_completed = 0
        for node_id in optional_nodes:
            if node_id in matched_node_ids:
                optional_completed += 1
                if node_id not in result.completed_nodes:
                    result.completed_nodes.append(node_id)

        result.optional_node_coverage = (
            optional_completed / len(optional_nodes) if optional_nodes else 1.0
        )

        # Calculate alternative group satisfaction
        for group_id, group in gold_dag.alt_groups.items():
            satisfied = False
            completed_in_group = 0
            for node_id in group.node_ids:
                if node_id in matched_node_ids:
                    completed_in_group += 1
            satisfied = completed_in_group >= group.min_required
            result.alt_group_satisfaction[group_id] = satisfied

        # Find extra tools (executed but not matched to any node)
        matched_tools = set(matched_nodes.values())
        expected_tools = {
            self._normalize_tool_name(node.tool)
            for node in gold_dag.nodes.values()
        }
        extra = []
        for tool in tools_executed:
            normalized = self._normalize_tool_name(tool)
            if normalized not in expected_tools:
                extra.append(tool)
        result.extra_nodes = extra

    def _compute_tool_metrics(
        self,
        gold_dag: TaskDAG,
        tools_executed: List[str],
        result: DAGEvaluationResult,
    ) -> None:
        """Compute tool recall, precision, and F1."""
        # Get expected tools from gold DAG
        expected_tools = {
            self._normalize_tool_name(node.tool)
            for node in gold_dag.nodes.values()
        }

        # Normalize executed tools
        executed_tools = {self._normalize_tool_name(t) for t in tools_executed}

        # True positives: tools that should be called and were called
        true_positives = expected_tools & executed_tools

        # False negatives: tools that should be called but weren't
        false_negatives = expected_tools - executed_tools

        # False positives: tools that were called but shouldn't be
        false_positives = executed_tools - expected_tools

        # Calculate metrics
        tp = len(true_positives)
        fn = len(false_negatives)
        fp = len(false_positives)

        result.tool_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        result.tool_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if result.tool_recall + result.tool_precision > 0:
            result.tool_f1 = (
                2 * result.tool_recall * result.tool_precision /
                (result.tool_recall + result.tool_precision)
            )
        else:
            result.tool_f1 = 0.0

    def _check_dependency_violations(
        self,
        gold_dag: TaskDAG,
        execution_order: List[str],
        result: DAGEvaluationResult,
    ) -> None:
        """Check for hard dependency violations in execution order."""
        violations = []

        # Match execution to nodes first
        matched_nodes = self._match_execution_to_nodes(gold_dag, execution_order)

        # Build a map of node_id to position in execution order
        node_positions: Dict[str, int] = {}
        position = 0
        for tool in execution_order:
            normalized = self._normalize_tool_name(tool)
            # Find which node was matched to this tool execution
            for node_id, matched_tool in matched_nodes.items():
                if self._normalize_tool_name(matched_tool) == normalized:
                    if node_id not in node_positions:
                        node_positions[node_id] = position
                        break
            position += 1

        # Check each hard dependency
        for edge in gold_dag.edges:
            if edge.edge_type != EdgeType.HARD_DEP:
                continue

            source_id = edge.source_id
            target_id = edge.target_id

            source_pos = node_positions.get(source_id, -1)
            target_pos = node_positions.get(target_id, -1)

            # Only check if target node was actually matched/executed
            if target_pos >= 0:  # Target was executed
                if source_pos < 0:  # Source wasn't executed
                    violations.append((target_id, source_id))
                elif source_pos > target_pos:  # Source came after target
                    violations.append((target_id, source_id))

        result.dep_violations = violations
        result.sequence_sanity = 1.0 if not violations else 0.0

    def _compute_param_accuracy(
        self,
        gold_dag: TaskDAG,
        tool_results: Dict[str, Any],
        result: DAGEvaluationResult,
    ) -> None:
        """Compute parameter accuracy based on expected inputs."""
        total_params = 0
        correct_params = 0

        for node_id, node in gold_dag.nodes.items():
            expected_input = node.input
            if not expected_input:
                continue

            # Find the matching tool result
            normalized_tool = self._normalize_tool_name(node.tool)
            actual_result = None

            for tool_name, tool_result in tool_results.items():
                if self._normalize_tool_name(tool_name) == normalized_tool:
                    actual_result = tool_result
                    break

            if actual_result is None:
                continue

            # Check each expected parameter
            for param_name, expected_value in expected_input.items():
                total_params += 1

                # Try to find the actual parameter value
                actual_value = None
                if isinstance(actual_result, dict):
                    actual_value = actual_result.get(param_name)

                if actual_value == expected_value:
                    correct_params += 1
                elif self._values_match(expected_value, actual_value):
                    correct_params += 1

        result.param_accuracy = correct_params / total_params if total_params > 0 else 1.0

    def _values_match(self, expected: Any, actual: Any) -> bool:
        """Check if two parameter values match (with some flexibility)."""
        if expected == actual:
            return True

        # String comparison (case-insensitive, stripped)
        if isinstance(expected, str) and isinstance(actual, str):
            return expected.lower().strip() == actual.lower().strip()

        # List comparison (order-independent)
        if isinstance(expected, list) and isinstance(actual, list):
            return set(expected) == set(actual)

        return False

    def evaluate_from_test_case(
        self,
        test_case: Dict[str, Any],
        agent_result: Dict[str, Any],
        end_to_end_score: float = 0.0,
    ) -> DAGEvaluationResult:
        """
        Evaluate agent result against a test case that may have a DAG or linear subtasks.

        Args:
            test_case: Test case with 'gold_dag' or 'gold_subtasks'
            agent_result: Agent's execution result
            end_to_end_score: End-to-end score from rubric evaluator

        Returns:
            DAGEvaluationResult
        """
        # Get or create gold DAG
        # Check both "gold_dag" and "dag" keys for compatibility
        dag_key = "gold_dag" if "gold_dag" in test_case else "dag" if "dag" in test_case else None
        if dag_key:
            dag_data = test_case[dag_key]
            if isinstance(dag_data, TaskDAG):
                gold_dag = dag_data
            elif isinstance(dag_data, dict):
                gold_dag = TaskDAG.from_dict(dag_data)
            else:
                gold_dag = TaskDAG.from_json(dag_data)
        elif "gold_subtasks" in test_case:
            # Convert linear subtasks to DAG
            from PUDDING.tools.dag_builder import convert_subtasks_to_dag
            gold_dag = convert_subtasks_to_dag(
                instance_id=test_case.get("instance_id", "unknown"),
                subtasks=test_case["gold_subtasks"],
                db_name=test_case.get("db"),
                query=test_case.get("query"),
            )
        else:
            # No gold standard, return empty result
            logger.warning("No gold_dag or gold_subtasks in test case")
            return DAGEvaluationResult(end_to_end_score=end_to_end_score)

        return self.evaluate(
            gold_dag=gold_dag,
            agent_execution=agent_result,
            test_case=test_case,
            end_to_end_score=end_to_end_score,
        )


def create_dag_evaluation_metrics(
    result: DAGEvaluationResult,
) -> Dict[str, Any]:
    """
    Create a dictionary of DAG evaluation metrics suitable for database storage.

    Args:
        result: DAGEvaluationResult

    Returns:
        Dictionary with dag_ prefixed column names
    """
    return result.to_database_dict()
