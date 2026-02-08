# -*- coding: utf-8 -*-
"""LangGraph StateGraph builder for 3-phase tree-structured exploration."""

import sqlite3
from typing import Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import TreeExplorationState
from .nodes import (
    load_queries,
    load_next_query,
    build_base_context,
    init_exploration,
    spawn_candidates,
    execute_and_reflect,
    select_next_path,
    generate_report_draft,
    auto_reflect,
    display_draft,
    expert_review,
    reinit_exploration,
    validate_single_source,
    annotate_dag,
    save_entry,
    route_next_path_node,
    check_more_queries,
)
from .edges import (
    route_after_load_queries,
    route_exploration,
    route_after_expert_review,
    route_after_validation,
    route_next_path,
    route_after_check_more,
)


def _build_graph(builder: StateGraph) -> StateGraph:
    """Wire up all nodes and edges for the 3-phase graph.

    Graph structure:
        START → load_queries → load_next_query → build_base_context → init_exploration

        [Phase 1 - Tree Loop]:
          → route_exploration
            ├─ spawn_candidates → execute_and_reflect → route_exploration (loop)
            └─ select_next_path (Phase 2)

        [Phase 2 - Per Terminal Path]:
          select_next_path → generate_report_draft → auto_reflect → display_draft
            → expert_review
              ├─ ACCEPT → validate_single_source (Phase 3)
              ├─ REVISE → reinit_exploration → route_exploration (re-enter Phase 1)
              └─ REJECT → route_next_path_node

        [Phase 3 - Finalization]:
          validate_single_source
            ├─ multi-source → annotate_dag → save_entry → route_next_path_node
            └─ single-source → route_next_path_node (skip)

          route_next_path_node → route_next_path
            ├─ more paths → select_next_path
            └─ no more → check_more_queries → load_next_query | END
    """

    builder.add_node("load_queries", load_queries)
    builder.add_node("load_next_query", load_next_query)
    builder.add_node("build_base_context", build_base_context)
    builder.add_node("init_exploration", init_exploration)
    builder.add_node("spawn_candidates", spawn_candidates)
    builder.add_node("execute_and_reflect", execute_and_reflect)
    builder.add_node("select_next_path", select_next_path)
    builder.add_node("generate_report_draft", generate_report_draft)
    builder.add_node("auto_reflect", auto_reflect)
    builder.add_node("display_draft", display_draft)
    builder.add_node("expert_review", expert_review)
    builder.add_node("reinit_exploration", reinit_exploration)
    builder.add_node("validate_single_source", validate_single_source)
    builder.add_node("annotate_dag", annotate_dag)
    builder.add_node("save_entry", save_entry)
    builder.add_node("route_next_path_node", route_next_path_node)
    builder.add_node("check_more_queries", check_more_queries)

    builder.add_edge(START, "load_queries")
    builder.add_conditional_edges(
        "load_queries",
        route_after_load_queries,
        {"load_next_query": "load_next_query", "__end__": END},
    )
    builder.add_edge("load_next_query", "build_base_context")
    builder.add_edge("build_base_context", "init_exploration")


    builder.add_conditional_edges(
        "init_exploration",
        route_exploration,
        {"spawn_candidates": "spawn_candidates", "select_next_path": "select_next_path"},
    )
    builder.add_edge("spawn_candidates", "execute_and_reflect")
    builder.add_conditional_edges(
        "execute_and_reflect",
        route_exploration,
        {"spawn_candidates": "spawn_candidates", "select_next_path": "select_next_path"},
    )


    builder.add_edge("select_next_path", "generate_report_draft")
    builder.add_edge("generate_report_draft", "auto_reflect")
    builder.add_edge("auto_reflect", "display_draft")
    builder.add_edge("display_draft", "expert_review")
    builder.add_conditional_edges(
        "expert_review",
        route_after_expert_review,
        {
            "validate_single_source": "validate_single_source",
            "reinit_exploration": "reinit_exploration",
            "route_next_path": "route_next_path_node",
        },
    )
    builder.add_conditional_edges(
        "reinit_exploration",
        route_exploration,
        {"spawn_candidates": "spawn_candidates", "select_next_path": "select_next_path"},
    )


    builder.add_conditional_edges(
        "validate_single_source",
        route_after_validation,
        {"annotate_dag": "annotate_dag", "route_next_path": "route_next_path_node"},
    )
    builder.add_edge("annotate_dag", "save_entry")
    builder.add_edge("save_entry", "route_next_path_node")


    builder.add_conditional_edges(
        "route_next_path_node",
        route_next_path,
        {"select_next_path": "select_next_path", "check_more_queries": "check_more_queries"},
    )
    builder.add_conditional_edges(
        "check_more_queries",
        route_after_check_more,
        {"load_next_query": "load_next_query", "__end__": END},
    )

    return builder


def build_dataset_graph(
    db_path: Optional[str] = None,
    checkpointer: Optional[SqliteSaver] = None,
) -> StateGraph:
    """Build and compile the 3-phase tree exploration graph with checkpointing."""
    builder = StateGraph(TreeExplorationState)
    _build_graph(builder)

    if checkpointer is None:
        db_path = db_path or "checkpoints.db"
        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        checkpointer.setup()

    return builder.compile(checkpointer=checkpointer)


def build_dataset_graph_without_checkpointing() -> StateGraph:
    """Build and compile the graph without checkpointing (for testing)."""
    builder = StateGraph(TreeExplorationState)
    _build_graph(builder)
    return builder.compile()
