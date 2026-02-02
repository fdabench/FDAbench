#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bird Dataset Build Script - LangGraph-based Stateful Agent

This script builds a dataset using a LangGraph-based workflow that supports:
- State machine driven processing
- Human-in-the-loop with interrupt() for feedback
- Checkpointing with SqliteSaver for session persistence and resume

Usage:
    # Interactive mode (default)
    python -m PUDDING.main

    # Automatic mode (no human review)
    python -m PUDDING.main --auto

    # Process limited queries
    python -m PUDDING.main --limit 10

    # Resume a previous session
    python -m PUDDING.main --resume dataset-build-20260131-abc123

    # Custom paths
    python -m PUDDING.main --input path/to/queries.jsonl --output path/to/output.json
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

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Bird Dataset Build Script with LangGraph (Human-in-the-Loop)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive mode
    python -m PUDDING.main

    # Auto mode (no human review)
    python -m PUDDING.main --auto

    # Process only 5 queries
    python -m PUDDING.main --limit 5

    # Resume previous session
    python -m PUDDING.main --resume dataset-build-20260131-abc123
        """
    )

    parser.add_argument(
        '--auto',
        action='store_true',
        help='Run in automatic mode without human review (default is interactive)'
    )

    parser.add_argument(
        '--max-revisions',
        type=int,
        default=3,
        help='Maximum number of revisions per item (default: 3)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of queries to process'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        metavar='THREAD_ID',
        help='Resume a previous session with the given thread ID'
    )

    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Path to input JSONL file (overrides default)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output JSON file (overrides default)'
    )

    parser.add_argument(
        '--checkpoint-db',
        type=str,
        default=None,
        help='Path to checkpoint database (default: output_data/checkpoints.db)'
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    """Build configuration dictionary from arguments."""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    config = {
        # Input paths
        "bird_path": args.input or os.path.join(
            current_dir, "input_data", "original_data", "bird.jsonl"
        ),
        "gold_result_dir": os.path.join(current_dir, "input_data", "gold_sql_result"),
        "sql_path": os.path.join(current_dir, "input_data", "gold_sql_query"),
        "file_system_path": os.path.join(current_dir, "input_data", "file_system"),

        # Output paths
        "output_path": args.output or os.path.join(
            current_dir, "output_data", "medium_singlechoice.json"
        ),
        "log_path": os.path.join(current_dir, "log", "bird_singlechoice.log"),

        # Settings
        "database_type": "bird",
        "interactive": not args.auto,
        "max_revisions": args.max_revisions,
    }

    return config


def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    # Parse arguments
    args = parse_args()

    # Build configuration
    config = build_config(args)

    # Setup logging
    logger = setup_logging(config["log_path"])

    # Log startup info
    mode_text = "Automatic" if args.auto else "Interactive Human Review"
    logger.info(f"Starting Bird Dataset Build ({mode_text} Mode)")
    logger.info(f"Using LangGraph-based stateful agent")
    logger.info(f"Input: {config['bird_path']}")
    logger.info(f"Output: {config['output_path']}")

    if args.resume:
        logger.info(f"Resuming session: {args.resume}")

    if args.limit:
        logger.info(f"Processing limit: {args.limit} queries")

    try:
        # Create and run the workflow
        runner = DatasetBuildRunner(
            config=config,
            db_path=args.checkpoint_db
        )

        runner.run(
            thread_id=args.resume,
            limit=args.limit
        )

        logger.info("Dataset build completed successfully")

    except KeyboardInterrupt:
        logger.warning("Build interrupted by user")
        print("\n\nBuild interrupted. Use --resume to continue later.")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Build failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
