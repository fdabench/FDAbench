# -*- coding: utf-8 -*-
"""
PUDDING - Tree-Structured Exploration with Self-Reflection

An agentic dataset construction framework implementing Algorithm 1:
tree-structured context grounding with per-branch self-reflection,
agent-expert collaboration, and multi-source validation.

Three-Phase Workflow:
    Phase 1 (Tree Exploration): Tree-structured context grounding with
        SpawnCandidates and SelfReflect (PRUNE/CONTINUE/SUFFICIENT)
    Phase 2 (Expert Verification): Report generation and iterative review
    Phase 3 (Finalization): Single-source validation, DAG annotation, rubric

Usage:
    python -m PUDDING.main                    # Interactive mode
    python -m PUDDING.main --auto             # Automatic mode (no human review)
    python -m PUDDING.main --limit 10         # Process limited queries
    python -m PUDDING.main --resume <thread>  # Resume session
"""

from .graph import (
    TreeExplorationState,
    DatasetBuildState,  # Backwards compat alias
    create_initial_state,
    build_dataset_graph,
    DatasetBuildRunner,
)
from .models import DatasetEntry, SubtaskResult
from .models.tree_models import (
    BranchDecision,
    ToolAction,
    ExplorationNode,
    TerminalPath,
    ExplorationTree,
)
from .generators import SingleChoiceGenerator
from .generators.report_generator import ReportGenerator

__version__ = "3.0.0"

__all__ = [
    # Graph
    "TreeExplorationState",
    "DatasetBuildState",
    "create_initial_state",
    "build_dataset_graph",
    "DatasetBuildRunner",
    # Models
    "DatasetEntry",
    "SubtaskResult",
    "BranchDecision",
    "ToolAction",
    "ExplorationNode",
    "TerminalPath",
    "ExplorationTree",
    # Generators
    "SingleChoiceGenerator",
    "ReportGenerator",
]
