# -*- coding: utf-8 -*-
"""Tree-aware runner with interrupt handling and session management."""

import os
import sys
import uuid
import time
import logging
from typing import Any, Dict, Optional
from datetime import datetime

from langgraph.types import Command

from .state import TreeExplorationState, create_initial_state
from .builder import build_dataset_graph

logger = logging.getLogger(__name__)


class DatasetBuildRunner:
    """Runner for the tree-structured dataset building workflow.

    Handles:
    - Graph execution with interrupt handling
    - Expert review interaction (ACCEPT/REVISE/REJECT)
    - Tree exploration progress display
    - Session persistence and resume
    """

    def __init__(self, config: Dict[str, Any], db_path: Optional[str] = None):
        self.config = config
        self.db_path = db_path or os.path.join(
            os.path.dirname(config.get("output_path", ".")),
            "checkpoints.db",
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
        """Generate a unique thread ID."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"tree-explore-{timestamp}-{unique_id}"

    def run(
        self,
        thread_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run the tree-structured exploration workflow.

        Args:
            thread_id: Optional ID to resume from. If None, starts new session.
            limit: Optional limit on number of queries to process.

        Returns:
            Final state after completion.
        """
        self.start_time = time.time()

        if thread_id:
            self.thread_id = thread_id
            logger.info(f"Resuming session: {self.thread_id}")
            print(f"\nResuming session: {self.thread_id}\n")
        else:
            self.thread_id = self.generate_thread_id()
            logger.info(f"Starting new session: {self.thread_id}")
            print(f"\nSession started: {self.thread_id}\n")

        run_config = {"configurable": {"thread_id": self.thread_id}}

        try:
            state = self.graph.get_state(run_config)
            if state.values and thread_id:
                result = self._resume_execution(run_config)
            else:
                initial_state = create_initial_state(self.config)

                # Apply limit by truncating queries after load
                if limit:
                    initial_state["_query_limit"] = limit

                result = self._run_with_interrupt_handling(initial_state, run_config)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Session saved.")
            print(f"Resume with: --resume {self.thread_id}")
            return {}
        except Exception as e:
            logger.error(f"Error during execution: {e}")
            raise

        self._display_final_stats(result)
        return result

    def _run_with_interrupt_handling(
        self,
        initial_state: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run graph with interrupt handling for expert review."""
        result = self.graph.invoke(initial_state, config)

        while self._is_interrupted(result):
            result = self._handle_interrupt(result, config)

        return result

    def _resume_execution(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resume from a checkpoint."""
        state = self.graph.get_state(config)

        if not state.values:
            raise ValueError(f"No checkpoint found for: {self.thread_id}")

        if state.next:
            logger.info(f"Resuming at node: {state.next}")
            tasks = state.tasks
            if tasks:
                for task in tasks:
                    if hasattr(task, "interrupts") and task.interrupts:
                        interrupt_info = task.interrupts[0]
                        return self._handle_interrupt(
                            {"__interrupt__": [interrupt_info]}, config
                        )

        result = self.graph.invoke(None, config)

        while self._is_interrupted(result):
            result = self._handle_interrupt(result, config)

        return result

    def _is_interrupted(self, result: Dict[str, Any]) -> bool:
        """Check if the result indicates an interrupt."""
        return isinstance(result, dict) and "__interrupt__" in result

    def _handle_interrupt(
        self,
        result: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle an interrupt by getting user input and resuming."""
        interrupt_info = result.get("__interrupt__", [])
        if not interrupt_info:
            return result

        interrupt_data = interrupt_info[0]
        interrupt_value = (
            interrupt_data.value
            if hasattr(interrupt_data, "value")
            else interrupt_data
        )

        interrupt_type = interrupt_value.get("type", "unknown")

        if interrupt_type == "expert_review":
            user_response = self._get_expert_review(interrupt_value)
        else:
            logger.warning(f"Unknown interrupt type: {interrupt_type}")
            user_response = {"choice": "a", "feedback": ""}

        result = self.graph.invoke(Command(resume=user_response), config)

        while self._is_interrupted(result):
            result = self._handle_interrupt(result, config)

        return result

    def _get_expert_review(self, interrupt_value: Dict[str, Any]) -> Dict[str, Any]:
        """Get expert review from CLI."""
        if not sys.stdin.isatty():
            logger.warning("Non-interactive - auto-accepting")
            return {"choice": "a", "feedback": ""}

        revision_count = interrupt_value.get("revision_count", 0)
        max_revisions = interrupt_value.get("max_revisions", 3)

        print(f"\nRevisions: {revision_count}/{max_revisions}")
        print("Options: (a) Accept  (r) Revise  (d) Reject")

        while True:
            try:
                choice = input("\nYour choice (a/r/d): ").strip().lower()

                if choice in ("a", "d", "r"):
                    feedback = ""
                    if choice == "r":
                        if revision_count >= max_revisions:
                            print(f"\nMax revisions ({max_revisions}) reached. Accepting.")
                            return {"choice": "a", "feedback": ""}

                        feedback = input("Feedback: ").strip()
                        if not feedback:
                            print("Feedback cannot be empty.")
                            continue

                    return {"choice": choice, "feedback": feedback}
                else:
                    print("Invalid input. Enter 'a', 'r', or 'd'.")

            except (EOFError, KeyboardInterrupt):
                print("\nInput interrupted - auto-accepting")
                return {"choice": "a", "feedback": ""}

    def _display_final_stats(self, result: Dict[str, Any]) -> None:
        """Display final statistics."""
        if not result:
            return

        total_time = time.time() - self.start_time
        total_queries = result.get("total_queries", 0)
        processed = result.get("processed_count", 0)
        accepted = result.get("accepted_count", 0)
        skipped = result.get("skipped_count", 0)
        errors = result.get("error_count", 0)

        print(f"\n{'='*50}")
        print(f"PUDDING Tree Exploration Complete")
        print(f"{'='*50}")
        print(f"  Total queries:    {total_queries}")
        print(f"  Processed:        {processed}")
        print(f"  Accepted:         {accepted}")
        print(f"  Skipped (single): {skipped}")
        print(f"  Errors:           {errors}")
        print(f"  Time:             {total_time:.1f}s")
        print(f"\n  Session: {self.thread_id}")
        print(f"  Resume:  --resume {self.thread_id}")
        print(f"  Output:  {self.config.get('output_path', 'N/A')}")
        print(f"{'='*50}\n")
