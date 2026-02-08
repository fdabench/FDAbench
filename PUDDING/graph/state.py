# -*- coding: utf-8 -*-
"""TreeExplorationState - LangGraph state for tree-structured exploration."""

import operator
from typing import Any, Dict, List, Optional
from typing_extensions import Annotated, TypedDict


def add_messages(left: list, right: list) -> list:
    return left + right


class TreeExplorationState(TypedDict, total=False):
    queries: List[Dict[str, Any]]
    current_query_idx: int
    total_queries: int
    instance_id: str
    db: str
    database_type: str
    task_id: str
    original_query: str

    schema_info: str
    sql_result: str
    sql_statement: str

    exploration_tree: dict
    current_iteration: int
    max_iterations: int
    candidate_actions: list
    current_candidates_processed: bool

    terminal_paths: list
    current_path_idx: int
    enhanced_query: str
    report_draft: str
    conversation_history: Annotated[list, add_messages]
    reflection: str

    expert_status: str
    expert_feedback: str
    reinit_feedback: str
    revision_count: int
    max_revisions: int

    is_multi_source: bool
    annotated_dag: dict
    annotated_rubric: dict
    annotated_subtasks: list
    difficulty_level: str
    frozen_web_search: dict
    frozen_vector_search: dict
    db_type: str
    tools_available: list

    output_entries: Annotated[list, operator.add]

    input_path: str
    output_path: str
    gold_result_dir: str
    sql_dir: str
    vector_index_path: str
    interactive_mode: bool

    processed_count: int
    accepted_count: int
    skipped_count: int
    error_count: int
    should_continue: bool
    error_message: Optional[str]


def create_initial_state(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "queries": [],
        "current_query_idx": 0,
        "total_queries": 0,
        "instance_id": "",
        "db": "",
        "database_type": "",
        "task_id": "",
        "original_query": "",

        "schema_info": "",
        "sql_result": "",
        "sql_statement": "",

        "exploration_tree": {},
        "current_iteration": 0,
        "max_iterations": config.get("max_tree_depth", 3),
        "candidate_actions": [],
        "current_candidates_processed": False,

        "terminal_paths": [],
        "current_path_idx": 0,
        "enhanced_query": "",
        "report_draft": "",
        "conversation_history": [],
        "reflection": "",

        "expert_status": "",
        "expert_feedback": "",
        "reinit_feedback": "",
        "revision_count": 0,
        "max_revisions": config.get("max_revisions", 3),

        "is_multi_source": False,
        "annotated_dag": {},
        "annotated_rubric": {},
        "annotated_subtasks": [],
        "difficulty_level": "",
        "frozen_web_search": {},
        "frozen_vector_search": {},
        "db_type": "",
        "tools_available": [],

        "output_entries": [],

        "input_path": config.get("input_path", ""),
        "output_path": config.get("output_path", ""),
        "gold_result_dir": config.get("gold_result_dir", ""),
        "sql_dir": config.get("sql_dir", ""),
        "vector_index_path": config.get("vector_index_path", ""),
        "interactive_mode": config.get("interactive_mode", True),

        "processed_count": 0,
        "accepted_count": 0,
        "skipped_count": 0,
        "error_count": 0,
        "should_continue": True,
        "error_message": None,
    }
