# -*- coding: utf-8 -*-
"""
Graph Runner - CLI interaction and session management for the LangGraph workflow.

This module handles:
- Running the graph with interrupt handling
- CLI interaction for human feedback
- Session persistence and resume
"""

import os
import sys
import uuid
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from langgraph.types import Command
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import DatasetBuildState, create_initial_state
from .builder import build_dataset_graph
from ..utils.display import display_welcome_message, display_statistics


logger = logging.getLogger(__name__)


class DatasetBuildRunner:
    """
    Runner for the dataset building workflow.

    Handles graph execution, interrupt handling, and session management.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        db_path: Optional[str] = None
    ):
        """
        Initialize the runner.

        Args:
            config: Configuration dictionary with paths and settings
            db_path: Path to SQLite database for checkpointing
        """
        self.config = config
        self.db_path = db_path or os.path.join(
            os.path.dirname(config.get("output_path", "")),
            "checkpoints.db"
        )

        # Ensure checkpoint directory exists
        checkpoint_dir = os.path.dirname(self.db_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Build the graph
        self.graph = build_dataset_graph(db_path=self.db_path)

        # Session tracking
        self.thread_id: Optional[str] = None
        self.start_time: float = 0

    def generate_thread_id(self) -> str:
        """Generate a unique thread ID for the session."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"dataset-build-{timestamp}-{unique_id}"

    def run(
        self,
        thread_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run the dataset building workflow.

        Args:
            thread_id: Optional thread ID to resume from. If None, starts a new session.
            limit: Optional limit on number of queries to process.

        Returns:
            Final state after completion or interruption.
        """
        self.start_time = time.time()
        interactive_mode = self.config.get("interactive", True)

        # Generate or use provided thread ID
        if thread_id:
            self.thread_id = thread_id
            logger.info(f"Resuming session: {self.thread_id}")
            print(f"\nResuming session: {self.thread_id}\n")
        else:
            self.thread_id = self.generate_thread_id()
            logger.info(f"Starting new session: {self.thread_id}")
            print(f"\nSession started with thread_id: {self.thread_id}\n")

        # Create run config
        run_config = {"configurable": {"thread_id": self.thread_id}}

        # Display welcome message for interactive mode
        if interactive_mode:
            display_welcome_message(self.config.get("max_revisions", 3))

        # Check if we're resuming
        try:
            state = self.graph.get_state(run_config)
            if state.values and thread_id:
                # Resuming from existing state
                logger.info("Resuming from checkpoint...")
                result = self._resume_execution(run_config)
            else:
                # New execution
                initial_state = create_initial_state(self.config)

                # Apply limit if specified
                if limit:
                    initial_state["max_queries"] = limit

                result = self._run_with_interrupt_handling(initial_state, run_config)
        except Exception as e:
            logger.error(f"Error during execution: {e}")
            raise

        # Display final statistics
        self._display_final_stats(result)

        return result

    def _run_with_interrupt_handling(
        self,
        initial_state: DatasetBuildState,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run the graph with interrupt handling for human-in-the-loop.

        Args:
            initial_state: Initial state to start from
            config: Run configuration

        Returns:
            Final state
        """
        result = self.graph.invoke(initial_state, config)

        # Handle interrupts in a loop
        while "__interrupt__" in result or self._is_interrupted(result):
            result = self._handle_interrupt(result, config)

        return result

    def _resume_execution(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resume execution from a checkpoint.

        Args:
            config: Run configuration

        Returns:
            Final state
        """
        # Get current state
        state = self.graph.get_state(config)

        if not state.values:
            raise ValueError(f"No checkpoint found for thread_id: {self.thread_id}")

        # Check if there's a pending interrupt
        if state.next:
            # There's a pending node, likely waiting for input
            logger.info(f"Resuming at node: {state.next}")

            # Get the interrupt info
            tasks = state.tasks
            if tasks:
                for task in tasks:
                    if hasattr(task, 'interrupts') and task.interrupts:
                        interrupt_info = task.interrupts[0]
                        return self._handle_interrupt(
                            {"__interrupt__": [interrupt_info]},
                            config
                        )

        # No pending interrupt, continue normal execution
        result = self.graph.invoke(None, config)

        while "__interrupt__" in result or self._is_interrupted(result):
            result = self._handle_interrupt(result, config)

        return result

    def _is_interrupted(self, result: Dict[str, Any]) -> bool:
        """Check if the result indicates an interrupt."""
        if isinstance(result, dict):
            return "__interrupt__" in result
        return False

    def _handle_interrupt(
        self,
        result: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle an interrupt by getting user input and resuming.

        Args:
            result: Current result with interrupt info
            config: Run configuration

        Returns:
            New result after resuming
        """
        interrupt_info = result.get("__interrupt__", [])

        if not interrupt_info:
            # No interrupt, just return
            return result

        interrupt_data = interrupt_info[0]

        # Get the value from the interrupt
        if hasattr(interrupt_data, 'value'):
            interrupt_value = interrupt_data.value
        else:
            interrupt_value = interrupt_data

        interrupt_type = interrupt_value.get("type", "unknown")

        # Handle different interrupt types
        if interrupt_type == "feedback":
            user_response = self._get_user_feedback(interrupt_value)
        elif interrupt_type == "difficulty_vote":
            user_response = self._get_difficulty_vote(interrupt_value)
        else:
            logger.warning(f"Unknown interrupt type: {interrupt_type}")
            user_response = {}

        # Resume with user response
        result = self.graph.invoke(Command(resume=user_response), config)

        # Check for more interrupts
        while "__interrupt__" in result or self._is_interrupted(result):
            result = self._handle_interrupt(result, config)

        return result

    def _get_user_feedback(self, interrupt_value: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get user feedback from CLI.

        Args:
            interrupt_value: Interrupt data with context

        Returns:
            User response dictionary
        """
        # Check for non-interactive environment
        if not sys.stdin.isatty():
            logger.warning("Non-interactive environment - auto-accepting")
            print("\nNon-interactive environment - auto-accepting content")
            return {"choice": "a", "feedback": ""}

        revision_count = interrupt_value.get("revision_count", 0)
        max_revisions = interrupt_value.get("max_revisions", 3)

        print(f"\nRevisions so far: {revision_count}/{max_revisions}")

        while True:
            try:
                user_input = input("\nYour choice (a/d/r): ").strip().lower()

                if user_input in ['a', 'd', 'r']:
                    feedback = ""
                    if user_input == 'r':
                        if revision_count >= max_revisions:
                            print(f"\nMaximum revisions ({max_revisions}) reached.")
                            print("Content will be accepted as-is.")
                            return {"choice": "a", "feedback": ""}

                        print("\nPlease provide specific feedback for improvement:")
                        feedback = input("Your feedback: ").strip()

                        if not feedback:
                            print("Feedback cannot be empty. Please try again.")
                            continue

                    return {"choice": user_input, "feedback": feedback}
                else:
                    print("Invalid input. Please enter 'a', 'd', or 'r'.")

            except (EOFError, KeyboardInterrupt):
                logger.warning("Input interrupted - auto-accepting")
                print("\nInput interrupted - auto-accepting content")
                return {"choice": "a", "feedback": ""}

    def _get_difficulty_vote(self, interrupt_value: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get difficulty vote from CLI.

        Args:
            interrupt_value: Interrupt data with context

        Returns:
            User response dictionary
        """
        # Check for non-interactive environment
        if not sys.stdin.isatty():
            logger.warning("Non-interactive environment - defaulting to medium")
            return {"difficulty": "medium"}

        while True:
            try:
                user_input = input("\nDifficulty vote (e/m/h): ").strip().lower()

                if user_input == 'e':
                    return {"difficulty": "easy"}
                elif user_input == 'm':
                    return {"difficulty": "medium"}
                elif user_input == 'h':
                    return {"difficulty": "hard"}
                else:
                    print("Invalid input. Please enter 'e', 'm', or 'h'.")

            except (EOFError, KeyboardInterrupt):
                logger.warning("Input interrupted - defaulting to medium")
                return {"difficulty": "medium"}

    def _display_final_stats(self, result: Dict[str, Any]) -> None:
        """Display final statistics after run completion."""
        total_time = time.time() - self.start_time

        # Extract stats from result
        total_queries = result.get("total_queries", 0)
        processed_count = result.get("processed_count", 0)
        error_count = result.get("error_count", 0)
        accepted_count = result.get("accepted_count", 0)
        revised_count = result.get("revised_count", 0)
        disposed_count = result.get("disposed_count", 0)
        interactive_mode = result.get("interactive_mode", True)

        display_statistics(
            total_queries=total_queries,
            processed_count=processed_count,
            error_count=error_count,
            accepted_count=accepted_count,
            revised_count=revised_count,
            disposed_count=disposed_count,
            total_time=total_time,
            interactive_mode=interactive_mode
        )

        print(f"\nSession ID: {self.thread_id}")
        print(f"To resume this session, use: --resume {self.thread_id}")
        print(f"Output saved to: {self.config.get('output_path', 'N/A')}")


def run_PUDDING(
    config: Dict[str, Any],
    thread_id: Optional[str] = None,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convenience function to run the dataset building workflow.

    Args:
        config: Configuration dictionary
        thread_id: Optional thread ID to resume from
        limit: Optional limit on queries to process

    Returns:
        Final state
    """
    runner = DatasetBuildRunner(config)
    return runner.run(thread_id=thread_id, limit=limit)
