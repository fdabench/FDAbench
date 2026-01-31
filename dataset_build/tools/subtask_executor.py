# -*- coding: utf-8 -*-
"""
Subtask Executor - Manages gold subtasks including SQL tools and external search tools.
Extracted from main.py.
"""

import os
import glob
import time
import logging
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..models.data_models import SubtaskResult
from .external_tools import ExternalToolManager


class GoldSubtaskManager:
    """Manages gold subtasks including SQL tools and external search tools."""

    def __init__(self, file_system_path: Optional[str] = None):
        self.external_tool_manager = ExternalToolManager(file_system_path=file_system_path)
        self.logger = logging.getLogger(__name__)

    def get_gold_result_for_instance(self, instance_id: str, gold_result_dir: str) -> str:
        """Get gold result for a specific instance."""
        try:
            # Try exact match first
            gold_file_path = os.path.join(gold_result_dir, f"{instance_id}.csv")
            if os.path.exists(gold_file_path):
                with open(gold_file_path, "r", encoding="utf-8") as f:
                    return f.read()

            # Try with suffix patterns
            pattern = os.path.join(gold_result_dir, f"{instance_id}_*.csv")
            matching_files = glob.glob(pattern)

            if matching_files:
                matching_files.sort()
                with open(matching_files[0], "r", encoding="utf-8") as f:
                    return f.read()

            self.logger.warning(f"No gold result found for instance: {instance_id}")
            return "N/A"

        except Exception as e:
            self.logger.error(f"Error loading gold result for {instance_id}: {e}")
            return f"Error: {e}"

    def get_sql_for_instance(self, instance_id: str, sql_dir: str) -> str:
        """Get SQL statement for a specific instance."""
        try:
            # Try exact match first
            sql_file_path = os.path.join(sql_dir, f"{instance_id}.sql")
            if os.path.exists(sql_file_path):
                with open(sql_file_path, "r", encoding="utf-8") as f:
                    return f.read().strip()

            # Try with suffix patterns
            pattern = os.path.join(sql_dir, f"{instance_id}_*.sql")
            matching_files = glob.glob(pattern)

            if matching_files:
                matching_files.sort()
                with open(matching_files[0], "r", encoding="utf-8") as f:
                    return f.read().strip()

            self.logger.warning(f"No SQL found for instance: {instance_id}")
            return "N/A"

        except Exception as e:
            self.logger.error(f"Error loading SQL for {instance_id}: {e}")
            return f"Error: {e}"

    def execute_subtasks_concurrent(
        self,
        instance_id: str,
        original_query: str,
        gold_result_dir: str,
        sql_dir: str
    ) -> Dict[str, SubtaskResult]:
        """Execute all subtasks concurrently and return results."""
        self.logger.info("Gathering data from subtasks concurrently...")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_gold = executor.submit(self.get_gold_result_for_instance, instance_id, gold_result_dir)
            future_sql = executor.submit(self.get_sql_for_instance, instance_id, sql_dir)
            future_external = executor.submit(self.external_tool_manager.search, original_query, original_query)

            futures = {
                'get_schema_info': None,
                'gold_result': future_gold,
                'sql_statement': future_sql,
                'external_search': future_external
            }

            results: Dict[str, SubtaskResult] = {}
            for future in as_completed([f for f in futures.values() if f is not None]):
                for name, fut in futures.items():
                    if fut == future:
                        try:
                            if name == 'external_search':
                                result = future.result()
                                results[name] = result
                            else:
                                result = future.result()
                                results[name] = SubtaskResult(name, result, success=True)

                            elapsed = time.time() - start_time
                            self.logger.info(f"{name} completed at {elapsed:.2f}s")
                        except Exception as e:
                            self.logger.error(f"{name} failed: {e}")
                            results[name] = SubtaskResult(name, f"Error: {e}", success=False, error=str(e))
                        break

        # Add schema info placeholder
        results['get_schema_info'] = SubtaskResult('get_schema_info', 'Schema info placeholder', success=True)

        end_time = time.time()
        self.logger.info(f"All subtasks completed: {end_time - start_time:.2f}s")

        return results

    def build_gold_subtasks(
        self,
        db: str,
        original_query: str,
        sql_statement: str,
        gold_result: str,
        query: str,
        selected_tool_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Build gold subtasks following original script structure with dynamic external tool selection."""
        subtasks: List[Dict[str, Any]] = [
            {
                "subtask_id": "get_schema_info",
                "tool": "get_schema_info",
                "input": {"database_name": db},
                "description": f"Provide schema information about the database"
            },
            {
                "subtask_id": "generated_sql",
                "tool": "generated_sql",
                "input": {
                    "natural_language_query": original_query,
                    "database_name": db
                },
                "expected_SQL": sql_statement,
                "description": f"Provide SQL to answer: {original_query}"
            },
            {
                "subtask_id": "execute_sql",
                "tool": "execute_sql",
                "input": {
                    "database_name": db
                },
                "expected_result": gold_result,
                "description": f"Execute SQL to answer: {original_query}"
            }
        ]

        # Add dynamic fourth subtask based on selected tool type
        if selected_tool_type == "VECTOR_SEARCH":
            subtasks.append({
                "subtask_id": "vectorDB_search",
                "tool": "vectorDB_search",
                "description": f"Retrieve relevant context for: {query}"
            })
        elif selected_tool_type == "FILE_SYSTEM":
            subtasks.append({
                "subtask_id": "file_system",
                "tool": "file_system",
                "input": {
                    "natural_language_query": original_query
                },
                "description": f"Provide file information to answer: {query}"
            })
        else:  # Default to perplexity_search
            subtasks.append({
                "subtask_id": "web_context_search",
                "tool": "perplexity_search",
                "description": f"Retrieve relevant external context for: {query}"
            })

        return subtasks
