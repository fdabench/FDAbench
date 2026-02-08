# -*- coding: utf-8 -*-
"""Tree-aware routing functions for the 3-phase LangGraph workflow."""

from typing import Literal

from .state import TreeExplorationState


def route_after_load_queries(
    state: TreeExplorationState,
) -> Literal["load_next_query", "__end__"]:
    """Route after loading queries: continue if queries exist."""
    if state.get("should_continue", False) and state.get("total_queries", 0) > 0:
        return "load_next_query"
    return "__end__"


def route_exploration(
    state: TreeExplorationState,
) -> Literal["spawn_candidates", "select_next_path"]:
    """Route Phase 1: continue exploring or move to Phase 2.

    Transitions to Phase 2 when:
    - Frontier is empty (all branches resolved), OR
    - Max iterations reached
    """
    tree_dict = state.get("exploration_tree", {})
    frontier = tree_dict.get("frontier", [])
    iteration = tree_dict.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    if len(frontier) == 0 or iteration >= max_iterations:
        return "select_next_path"
    return "spawn_candidates"


def route_after_expert_review(
    state: TreeExplorationState,
) -> Literal["validate_single_source", "reinit_exploration", "route_next_path"]:
    """Route after expert review: accept, revise (re-explore), or reject."""
    expert_status = state.get("expert_status", "")
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 3)

    if expert_status == "ACCEPT":
        return "validate_single_source"
    elif expert_status == "REVISE":
        if revision_count >= max_revisions:
            # Max revisions hit, accept as-is
            return "validate_single_source"
        return "reinit_exploration"
    else:  # REJECT
        return "route_next_path"


def route_after_validation(
    state: TreeExplorationState,
) -> Literal["annotate_dag", "route_next_path"]:
    """Route after single-source validation: annotate if multi-source, skip otherwise."""
    if state.get("is_multi_source", False):
        return "annotate_dag"
    return "route_next_path"


def route_next_path(
    state: TreeExplorationState,
) -> Literal["select_next_path", "check_more_queries"]:
    """Route to next terminal path or next query."""
    terminal_paths = state.get("terminal_paths", [])
    current_path_idx = state.get("current_path_idx", 0)

    if current_path_idx < len(terminal_paths):
        return "select_next_path"
    return "check_more_queries"


def route_after_check_more(
    state: TreeExplorationState,
) -> Literal["load_next_query", "__end__"]:
    """Route after checking for more queries."""
    if state.get("should_continue", False):
        return "load_next_query"
    return "__end__"
