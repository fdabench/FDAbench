# -*- coding: utf-8 -*-
"""LangGraph StateGraph builder for dataset building workflow."""

import sqlite3
from typing import Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import DatasetBuildState
from .nodes import (
    load_queries, load_next_query, fetch_subtasks, generate_content,
    reflect, display, human_feedback, difficulty_vote, revise,
    save_entry, dispose, check_more,
)
from .edges import (
    route_after_load_queries, route_after_human_feedback,
    route_after_revise, route_after_check_more, route_after_fetch_subtasks,
)


def build_dataset_graph(db_path: Optional[str] = None, checkpointer: Optional[SqliteSaver] = None) -> StateGraph:
    builder = StateGraph(DatasetBuildState)

    builder.add_node("load_queries", load_queries)
    builder.add_node("load_next_query", load_next_query)
    builder.add_node("fetch_subtasks", fetch_subtasks)
    builder.add_node("generate_content", generate_content)
    builder.add_node("reflect", reflect)
    builder.add_node("display", display)
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("difficulty_vote", difficulty_vote)
    builder.add_node("revise", revise)
    builder.add_node("save_entry", save_entry)
    builder.add_node("dispose", dispose)
    builder.add_node("check_more", check_more)

    builder.add_edge(START, "load_queries")
    builder.add_conditional_edges("load_queries", route_after_load_queries,
        {"load_next_query": "load_next_query", "__end__": END})
    builder.add_edge("load_next_query", "fetch_subtasks")
    builder.add_conditional_edges("fetch_subtasks", route_after_fetch_subtasks,
        {"generate_content": "generate_content", "check_more": "check_more"})
    builder.add_edge("generate_content", "reflect")
    builder.add_edge("reflect", "display")
    builder.add_edge("display", "human_feedback")
    builder.add_conditional_edges("human_feedback", route_after_human_feedback,
        {"difficulty_vote": "difficulty_vote", "dispose": "dispose", "revise": "revise", "check_more": "check_more"})
    builder.add_edge("difficulty_vote", "save_entry")
    builder.add_edge("save_entry", "check_more")
    builder.add_edge("dispose", "check_more")
    builder.add_conditional_edges("revise", route_after_revise, {"reflect": "reflect"})
    builder.add_conditional_edges("check_more", route_after_check_more,
        {"load_next_query": "load_next_query", "__end__": END})

    if checkpointer is None:
        db_path = db_path or "checkpoints.db"
        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        checkpointer.setup()

    return builder.compile(checkpointer=checkpointer)


def build_dataset_graph_without_checkpointing() -> StateGraph:
    builder = StateGraph(DatasetBuildState)

    builder.add_node("load_queries", load_queries)
    builder.add_node("load_next_query", load_next_query)
    builder.add_node("fetch_subtasks", fetch_subtasks)
    builder.add_node("generate_content", generate_content)
    builder.add_node("reflect", reflect)
    builder.add_node("display", display)
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("difficulty_vote", difficulty_vote)
    builder.add_node("revise", revise)
    builder.add_node("save_entry", save_entry)
    builder.add_node("dispose", dispose)
    builder.add_node("check_more", check_more)

    builder.add_edge(START, "load_queries")
    builder.add_conditional_edges("load_queries", route_after_load_queries,
        {"load_next_query": "load_next_query", "__end__": END})
    builder.add_edge("load_next_query", "fetch_subtasks")
    builder.add_conditional_edges("fetch_subtasks", route_after_fetch_subtasks,
        {"generate_content": "generate_content", "check_more": "check_more"})
    builder.add_edge("generate_content", "reflect")
    builder.add_edge("reflect", "display")
    builder.add_edge("display", "human_feedback")
    builder.add_conditional_edges("human_feedback", route_after_human_feedback,
        {"difficulty_vote": "difficulty_vote", "dispose": "dispose", "revise": "revise", "check_more": "check_more"})
    builder.add_edge("difficulty_vote", "save_entry")
    builder.add_edge("save_entry", "check_more")
    builder.add_edge("dispose", "check_more")
    builder.add_conditional_edges("revise", route_after_revise, {"reflect": "reflect"})
    builder.add_conditional_edges("check_more", route_after_check_more,
        {"load_next_query": "load_next_query", "__end__": END})

    return builder.compile()
