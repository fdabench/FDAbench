# -*- coding: utf-8 -*-
"""LangGraph state definition for dataset building workflow."""

from typing import TypedDict, List, Dict, Any, Optional, Literal
from typing_extensions import Annotated


def add_messages(left: List[Dict], right: List[Dict]) -> List[Dict]:
    return left + right


class DatasetBuildState(TypedDict, total=False):
    queries: List[Dict[str, Any]]
    current_query_idx: int
    total_queries: int
    current_item: Dict[str, Any]
    instance_id: str
    db: str
    original_query: str

    subtask_results: Dict[str, Any]
    gold_result: str
    sql_statement: str
    external_result: str
    selected_tool_type: Optional[str]

    question_data: Dict[str, Any]
    conversation_history: Annotated[List[Dict], add_messages]
    reflection: str
    content_result: Dict[str, Any]

    user_choice: Optional[Literal['a', 'd', 'r']]
    user_feedback: str
    difficulty_vote: Optional[Literal['easy', 'medium', 'hard']]
    revision_count: int
    max_revisions: int

    processed_count: int
    accepted_count: int
    revised_count: int
    disposed_count: int
    error_count: int

    generation_success: bool
    should_continue: bool
    error_message: Optional[str]
    interactive_mode: bool

    bird_path: str
    gold_result_dir: str
    sql_path: str
    output_path: str
    file_system_path: str


def create_initial_state(config: Dict[str, Any]) -> DatasetBuildState:
    return DatasetBuildState(
        queries=[], current_query_idx=0, total_queries=0, current_item={},
        instance_id="", db="", original_query="",
        subtask_results={}, gold_result="", sql_statement="", external_result="", selected_tool_type=None,
        question_data={}, conversation_history=[], reflection="", content_result={},
        user_choice=None, user_feedback="", difficulty_vote=None,
        revision_count=0, max_revisions=config.get("max_revisions", 3),
        processed_count=0, accepted_count=0, revised_count=0, disposed_count=0, error_count=0,
        generation_success=False, should_continue=True, error_message=None,
        interactive_mode=config.get("interactive", True),
        bird_path=config.get("bird_path", ""), gold_result_dir=config.get("gold_result_dir", ""),
        sql_path=config.get("sql_path", ""), output_path=config.get("output_path", ""),
        file_system_path=config.get("file_system_path", ""),
    )
