# -*- coding: utf-8 -*-
"""
LangGraph-based state machine for tree-structured exploration workflow.
"""

from .state import TreeExplorationState, create_initial_state
from .builder import build_dataset_graph
from .runner import DatasetBuildRunner

# Backwards compatibility alias
DatasetBuildState = TreeExplorationState

__all__ = [
    "TreeExplorationState",
    "DatasetBuildState",
    "create_initial_state",
    "build_dataset_graph",
    "DatasetBuildRunner",
]
