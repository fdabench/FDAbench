# -*- coding: utf-8 -*-
"""
Dataset Build Module - LangGraph-based Stateful Agent for Dataset Construction

This module provides a human-in-the-loop workflow for building datasets using:
- State machine driven processing with LangGraph
- Human feedback via interrupt() mechanism
- Session persistence with SqliteSaver checkpointing

Usage:
    # Interactive mode
    python -m dataset_build.main

    # Auto mode
    python -m dataset_build.main --auto

    # Resume session
    python -m dataset_build.main --resume <thread_id>
"""

from .graph import DatasetBuildState, build_dataset_graph, DatasetBuildRunner
from .models import DatasetEntry, SubtaskResult
from .generators import SingleChoiceGenerator
from .tools import GoldSubtaskManager, ExternalToolManager

__version__ = "2.0.0"

__all__ = [
    "DatasetBuildState",
    "build_dataset_graph",
    "DatasetBuildRunner",
    "DatasetEntry",
    "SubtaskResult",
    "SingleChoiceGenerator",
    "GoldSubtaskManager",
    "ExternalToolManager",
]
