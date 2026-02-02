# -*- coding: utf-8 -*-
"""Routing functions for LangGraph workflow conditional edges."""

from typing import Literal
from .state import DatasetBuildState


def route_after_load_queries(state: DatasetBuildState) -> Literal["load_next_query", "__end__"]:
    if state.get("should_continue", False) and state.get("total_queries", 0) > 0:
        return "load_next_query"
    return "__end__"


def route_after_human_feedback(state: DatasetBuildState) -> Literal["difficulty_vote", "dispose", "revise", "check_more"]:
    user_choice = state.get("user_choice", "a")
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 3)

    if user_choice == "a":
        return "difficulty_vote"
    elif user_choice == "d":
        return "dispose"
    elif user_choice == "r":
        if revision_count >= max_revisions:
            print(f"\nMaximum revisions ({max_revisions}) reached. Proceeding to save.\n")
            return "difficulty_vote"
        return "revise"
    return "difficulty_vote"


def route_after_revise(state: DatasetBuildState) -> Literal["reflect"]:
    return "reflect"


def route_after_check_more(state: DatasetBuildState) -> Literal["load_next_query", "__end__"]:
    if state.get("should_continue", False):
        return "load_next_query"
    return "__end__"


def route_after_fetch_subtasks(state: DatasetBuildState) -> Literal["generate_content", "check_more"]:
    if state.get("error_message"):
        return "check_more"
    return "generate_content"
