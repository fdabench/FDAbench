# -*- coding: utf-8 -*-
"""Graph nodes for LangGraph workflow."""

import sys
import logging
from typing import Dict, Any

from langgraph.types import interrupt

from .state import DatasetBuildState
from ..tools.subtask_executor import GoldSubtaskManager
from ..generators.question_generator import SingleChoiceGenerator
from ..models.data_models import DatasetEntry, SubtaskResult
from ..utils.display import display_results, display_review_options, display_difficulty_options
from ..utils.io import load_queries as load_queries_from_file, append_to_output

logger = logging.getLogger(__name__)

_subtask_manager: GoldSubtaskManager = None
_question_generator: SingleChoiceGenerator = None


def _get_subtask_manager(file_system_path: str = None) -> GoldSubtaskManager:
    global _subtask_manager
    if _subtask_manager is None:
        _subtask_manager = GoldSubtaskManager(file_system_path)
    return _subtask_manager


def _get_question_generator() -> SingleChoiceGenerator:
    global _question_generator
    if _question_generator is None:
        _question_generator = SingleChoiceGenerator()
    return _question_generator


def load_queries(state: DatasetBuildState) -> Dict[str, Any]:
    bird_path = state.get("bird_path", "")
    logger.info(f"Loading queries from: {bird_path}")
    try:
        queries = load_queries_from_file(bird_path)
        total = len(queries)
        logger.info(f"Loaded {total} queries")
        return {
            "queries": queries,
            "total_queries": total,
            "current_query_idx": 0,
            "should_continue": total > 0,
            "error_message": None if total > 0 else "No queries found"
        }
    except Exception as e:
        logger.error(f"Failed to load queries: {e}")
        return {"queries": [], "total_queries": 0, "should_continue": False, "error_message": str(e)}


def load_next_query(state: DatasetBuildState) -> Dict[str, Any]:
    queries = state.get("queries", [])
    idx = state.get("current_query_idx", 0)
    if idx >= len(queries):
        return {"should_continue": False, "error_message": "No more queries"}

    item = queries[idx]
    instance_id = item.get("instance_id") or item.get("id") or str(idx)
    db = item.get("db_id") or item.get("db") or "SQLite"
    original_query = item.get("instruction") or item.get("question") or item.get("query") or ""
    logger.info(f"Processing query {idx + 1}/{len(queries)}: {instance_id}")

    return {
        "current_item": item, "instance_id": instance_id, "db": db,
        "original_query": original_query, "revision_count": 0,
        "user_choice": None, "user_feedback": "", "difficulty_vote": None, "error_message": None
    }


def fetch_subtasks(state: DatasetBuildState) -> Dict[str, Any]:
    instance_id = state.get("instance_id", "")
    original_query = state.get("original_query", "")
    gold_result_dir = state.get("gold_result_dir", "")
    sql_path = state.get("sql_path", "")
    file_system_path = state.get("file_system_path", "")
    logger.info(f"Fetching subtasks for: {instance_id}")

    try:
        manager = _get_subtask_manager(file_system_path)
        results = manager.execute_subtasks_concurrent(instance_id, original_query, gold_result_dir, sql_path)
        gold_result = results.get('gold_result', SubtaskResult('gold_result', 'N/A')).result
        sql_statement = results.get('sql_statement', SubtaskResult('sql_statement', 'N/A')).result
        external_search = results.get('external_search', SubtaskResult('external_search', 'N/A'))
        external_result = external_search.result
        selected_tool_type = getattr(external_search, 'selected_tool_type', None)

        return {
            "subtask_results": {k: v.to_dict() if hasattr(v, 'to_dict') else {"result": str(v)} for k, v in results.items()},
            "gold_result": gold_result, "sql_statement": sql_statement,
            "external_result": external_result, "selected_tool_type": selected_tool_type, "error_message": None
        }
    except Exception as e:
        logger.error(f"Failed to fetch subtasks: {e}")
        return {"subtask_results": {}, "error_message": str(e)}


def generate_content(state: DatasetBuildState) -> Dict[str, Any]:
    original_query = state.get("original_query", "")
    subtask_results = state.get("subtask_results", {})
    logger.info("Generating content...")

    try:
        generator = _get_question_generator()
        generator.reset_conversation()
        results_objects = {}
        for key, value in subtask_results.items():
            if isinstance(value, dict):
                results_objects[key] = SubtaskResult.from_dict(value)
            else:
                results_objects[key] = value

        content_result = generator.generate_single_choice_content(original_query, results_objects)
        return {
            "content_result": content_result,
            "question_data": content_result.get("question_data", {}),
            "conversation_history": content_result.get("conversation_history", []),
            "generation_success": content_result.get("success", False),
            "error_message": None
        }
    except Exception as e:
        logger.error(f"Failed to generate content: {e}")
        return {"content_result": {}, "generation_success": False, "error_message": str(e)}


def reflect(state: DatasetBuildState) -> Dict[str, Any]:
    question_data = state.get("question_data", {})
    original_query = state.get("original_query", "")
    gold_result = state.get("gold_result", "")
    external_result = state.get("external_result", "")
    generation_success = state.get("generation_success", False)

    if not generation_success or not question_data:
        return {"reflection": "Content generation failed or used fallback."}

    try:
        generator = _get_question_generator()
        reflection = generator.reflect_on_content(question_data, original_query, gold_result, external_result)
        return {"reflection": reflection}
    except Exception as e:
        logger.error(f"Failed to generate reflection: {e}")
        return {"reflection": f"Reflection failed: {e}"}


def display(state: DatasetBuildState) -> Dict[str, Any]:
    original_query = state.get("original_query", "")
    content_result = state.get("content_result", {})
    subtask_results = state.get("subtask_results", {})
    reflection = state.get("reflection", "")

    results_objects = {}
    for key, value in subtask_results.items():
        if isinstance(value, dict):
            results_objects[key] = SubtaskResult.from_dict(value)
        else:
            results_objects[key] = value

    display_results(original_query, content_result, results_objects, reflection)
    return {}


def human_feedback(state: DatasetBuildState) -> Dict[str, Any]:
    interactive_mode = state.get("interactive_mode", True)
    if not interactive_mode or not sys.stdin.isatty():
        logger.info("Non-interactive mode - auto-accepting content")
        return {"user_choice": "a", "user_feedback": ""}

    display_review_options()
    feedback = interrupt({
        "type": "feedback",
        "question_data": state.get("question_data", {}),
        "reflection": state.get("reflection", ""),
        "options": ["(a) Accept", "(d) Dispose", "(r) Revise"],
        "revision_count": state.get("revision_count", 0),
        "max_revisions": state.get("max_revisions", 3),
        "prompt": "Your choice (a/d/r): "
    })
    return {"user_choice": feedback.get("choice", "a"), "user_feedback": feedback.get("feedback", "")}


def difficulty_vote(state: DatasetBuildState) -> Dict[str, Any]:
    interactive_mode = state.get("interactive_mode", True)
    if not interactive_mode or not sys.stdin.isatty():
        logger.info("Non-interactive mode - defaulting to medium difficulty")
        return {"difficulty_vote": "medium"}

    display_difficulty_options()
    vote = interrupt({
        "type": "difficulty_vote",
        "options": ["(e) Easy", "(m) Medium", "(h) Hard"],
        "prompt": "Difficulty vote (e/m/h): "
    })
    return {"difficulty_vote": vote.get("difficulty", "medium")}


def revise(state: DatasetBuildState) -> Dict[str, Any]:
    feedback = state.get("user_feedback", "")
    conversation_history = state.get("conversation_history", [])
    question_data = state.get("question_data", {})
    original_query = state.get("original_query", "")
    gold_result = state.get("gold_result", "")
    external_result = state.get("external_result", "")
    revision_count = state.get("revision_count", 0)
    logger.info(f"Revising content (attempt {revision_count + 1})")

    try:
        generator = _get_question_generator()
        revised_result = generator.revise_content(
            feedback, conversation_history, question_data, original_query, gold_result, external_result
        )
        if revised_result.get("success", False):
            return {
                "content_result": revised_result,
                "question_data": revised_result.get("question_data", question_data),
                "conversation_history": revised_result.get("conversation_history", conversation_history),
                "revision_count": revision_count + 1,
                "revised_count": state.get("revised_count", 0) + 1,
                "generation_success": True
            }
        else:
            logger.warning("Revision failed, keeping original content")
            return {"revision_count": revision_count + 1}
    except Exception as e:
        logger.error(f"Failed to revise content: {e}")
        return {"revision_count": revision_count + 1, "error_message": str(e)}


def save_entry(state: DatasetBuildState) -> Dict[str, Any]:
    instance_id = state.get("instance_id", "")
    db = state.get("db", "")
    original_query = state.get("original_query", "")
    question_data = state.get("question_data", {})
    difficulty = state.get("difficulty_vote")
    output_path = state.get("output_path", "")
    sql_statement = state.get("sql_statement", "")
    gold_result = state.get("gold_result", "")
    selected_tool_type = state.get("selected_tool_type")
    file_system_path = state.get("file_system_path", "")
    logger.info(f"Saving entry: {instance_id}")

    try:
        entry = DatasetEntry(instance_id, db, level=difficulty)
        manager = _get_subtask_manager(file_system_path)
        gold_subtasks = manager.build_gold_subtasks(
            db, original_query, sql_statement, gold_result,
            question_data.get("question", ""), selected_tool_type
        )
        entry_dict = entry.to_dict(question_data, gold_subtasks)
        append_to_output(entry_dict, output_path)
        logger.info(f"Entry saved successfully")
        print(f"\nEntry saved successfully!\n")
        return {
            "accepted_count": state.get("accepted_count", 0) + 1,
            "processed_count": state.get("processed_count", 0) + 1
        }
    except Exception as e:
        logger.error(f"Failed to save entry: {e}")
        return {"error_count": state.get("error_count", 0) + 1, "error_message": str(e)}


def dispose(state: DatasetBuildState) -> Dict[str, Any]:
    instance_id = state.get("instance_id", "")
    current_idx = state.get("current_query_idx", 0)
    logger.info(f"Query {current_idx + 1} disposed: {instance_id}")
    print(f"\nQuery {current_idx + 1} disposed and skipped.\n")
    return {
        "disposed_count": state.get("disposed_count", 0) + 1,
        "processed_count": state.get("processed_count", 0) + 1
    }


def check_more(state: DatasetBuildState) -> Dict[str, Any]:
    current_idx = state.get("current_query_idx", 0)
    total_queries = state.get("total_queries", 0)
    next_idx = current_idx + 1
    has_more = next_idx < total_queries
    logger.info(f"Progress: {next_idx}/{total_queries}, has_more={has_more}")
    return {"current_query_idx": next_idx, "should_continue": has_more}
