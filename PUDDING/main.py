#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PUDDING - Tree-Structured Exploration with Self-Reflection

Implements Algorithm 1: tree-structured context grounding with per-branch
self-reflection (PRUNE/CONTINUE/SUFFICIENT), agent-expert collaboration
for report generation, and multi-source validation.

Usage:
    # Interactive mode (with expert review)
    python -m PUDDING.main --input path/to/queries.jsonl

    # Auto mode (no human review)
    python -m PUDDING.main --input path/to/queries.jsonl --auto

    # Process limited queries with custom depth
    python -m PUDDING.main --input path/to/queries.jsonl --limit 5 --max-tree-depth 2

    # Resume previous session
    python -m PUDDING.main --resume tree-explore-20260208-abc123
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv

from .graph.runner import DatasetBuildRunner


def setup_logging(log_path: str) -> logging.Logger:
    """Setup logging configuration."""
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PUDDING: Tree-Structured Exploration with Self-Reflection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test (1 case, auto mode)
    python -m PUDDING.main --input PUDDING/input_data/original_data/bird.jsonl --limit 1 --auto

    # Full run with custom depth
    python -m PUDDING.main --input data.jsonl --max-tree-depth 3 --auto

    # Interactive mode with expert review
    python -m PUDDING.main --input data.jsonl

    # Resume session
    python -m PUDDING.main --resume tree-explore-20260208-abc123
        """,
    )

    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=3,
        help="Maximum tree exploration iterations (default: 3)",
    )

    parser.add_argument(
        "--max-revisions",
        type=int,
        default=3,
        help="Maximum report revision attempts (default: 3)",
    )

    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto mode: skip expert review, auto-accept all reports",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to process",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="THREAD_ID",
        help="Resume a previous session with the given thread ID",
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input JSONL file",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSONL file (default: PUDDING/output_data/report_output.jsonl)",
    )

    parser.add_argument(
        "--vector-index",
        type=str,
        default=None,
        help="Path to FAISS vector index directory (default: storage_faiss)",
    )

    parser.add_argument(
        "--checkpoint-db",
        type=str,
        default=None,
        help="Path to checkpoint database",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    """Build configuration dictionary from arguments."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    config = {
        # Input paths
        "input_path": args.input or os.path.join(
            current_dir, "input_data", "original_data", "bird.jsonl"
        ),
        "gold_result_dir": os.path.join(current_dir, "input_data", "gold_sql_result"),
        "sql_dir": os.path.join(current_dir, "input_data", "gold_sql_query"),
        "vector_index_path": args.vector_index or os.path.join(
            project_root, "storage_faiss"
        ),

        # Output paths
        "output_path": args.output or os.path.join(
            current_dir, "output_data", "report_output.jsonl"
        ),

        # Tree exploration settings
        "max_tree_depth": args.max_tree_depth,
        "max_revisions": args.max_revisions,
        "interactive_mode": not args.auto,
    }

    return config


def main():
    """Main entry point."""
    load_dotenv()
    args = parse_args()
    config = build_config(args)

    # Setup logging
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "log", "tree_exploration.log"
    )
    logger = setup_logging(log_path)

    mode_text = "Automatic" if args.auto else "Interactive"
    logger.info(f"PUDDING Tree Exploration ({mode_text} Mode)")
    logger.info(f"  Input:          {config['input_path']}")
    logger.info(f"  Output:         {config['output_path']}")
    logger.info(f"  Vector Index:   {config['vector_index_path']}")
    logger.info(f"  Max Tree Depth: {config['max_tree_depth']}")
    logger.info(f"  Max Revisions:  {config['max_revisions']}")

    if args.resume:
        logger.info(f"  Resuming:       {args.resume}")
    if args.limit:
        logger.info(f"  Limit:          {args.limit} queries")

    try:
        runner = DatasetBuildRunner(
            config=config,
            db_path=args.checkpoint_db,
        )

        runner.run(
            thread_id=args.resume,
            limit=args.limit,
        )

        logger.info("Tree exploration completed successfully")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        print("\n\nInterrupted. Use --resume to continue later.")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
