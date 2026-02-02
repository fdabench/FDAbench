# -*- coding: utf-8 -*-
"""
PUDDING - Agent-Expert Collaboration Framework for Dataset Construction

An agentic dataset construction framework that leverages LLM generation with
iterative expert validation for reliable and scalable benchmark construction.

Three-Phase Workflow:
    Phase 1 (Initialization): Gather structured/unstructured data context
    Phase 2 (Expert Verification): Iterative agent-expert collaboration
    Phase 3 (Finalization): Quality validation and difficulty classification

Usage:
    python -m PUDDING.main          # Interactive mode (with expert review)
    python -m PUDDING.main --auto   # Automatic mode
    python -m PUDDING.main --resume <thread_id>  # Resume session
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
