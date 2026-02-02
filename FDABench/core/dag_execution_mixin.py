# -*- coding: utf-8 -*-
"""
DAG Execution Mixin - Mixin class for DAG-aware agent execution.

This module provides a mixin class that agents can inherit to gain
DAG-based execution capabilities, including:
- Loading and managing TaskDAG structures
- Determining next executable nodes
- Tracking execution state
- Parallel execution detection
"""

import time
import logging
from typing import Dict, List, Any, Optional, Set, Tuple

# Import DAG models
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from PUDDING.models.dag_models import (
    TaskDAG, DAGNode, DAGEdge, AlternativeGroup,
    NodeType, EdgeType, NodeStatus, DAGExecutionState,
)
from PUDDING.tools.dag_builder import DAGBuilder, convert_subtasks_to_dag

logger = logging.getLogger(__name__)


class DAGExecutionMixin:
    """
    Mixin class providing DAG-based execution capabilities for agents.

    This mixin should be used alongside BaseAgent to add DAG execution support.
    It provides methods for:
    - Loading DAG from various sources
    - Getting next executable nodes
    - Managing execution state
    - Recording execution results
    """

    def init_dag_execution(self) -> None:
        """Initialize DAG execution state. Call in agent __init__."""
        self._current_dag: Optional[TaskDAG] = None
        self._execution_state: Optional[DAGExecutionState] = None
        self._dag_enabled: bool = False

    def load_dag(
        self,
        dag_source: Any,
        force_linear: bool = False,
    ) -> Optional[TaskDAG]:
        """
        Load a TaskDAG from various sources.

        Args:
            dag_source: Can be:
                - TaskDAG instance
                - Dict with 'gold_dag' key containing DAG data
                - Dict with 'gold_subtasks' key containing linear subtasks
                - List of subtask dicts (linear conversion)
                - JSON string of DAG
            force_linear: If True, always convert to linear chain

        Returns:
            Loaded TaskDAG or None if loading failed
        """
        try:
            # Already a TaskDAG
            if isinstance(dag_source, TaskDAG):
                self._current_dag = dag_source
                self._dag_enabled = True
                logger.info(f"Loaded DAG with {len(dag_source.nodes)} nodes")
                return self._current_dag

            # Dict with gold_dag
            if isinstance(dag_source, dict) and "gold_dag" in dag_source:
                dag_data = dag_source["gold_dag"]
                if isinstance(dag_data, TaskDAG):
                    self._current_dag = dag_data
                elif isinstance(dag_data, dict):
                    self._current_dag = TaskDAG.from_dict(dag_data)
                elif isinstance(dag_data, str):
                    self._current_dag = TaskDAG.from_json(dag_data)
                self._dag_enabled = True
                logger.info(f"Loaded DAG from gold_dag with {len(self._current_dag.nodes)} nodes")
                return self._current_dag

            # Dict with gold_subtasks (convert to DAG)
            if isinstance(dag_source, dict) and "gold_subtasks" in dag_source:
                subtasks = dag_source["gold_subtasks"]
                instance_id = dag_source.get("instance_id", "unknown")
                db_name = dag_source.get("db")
                query = dag_source.get("query")

                self._current_dag = convert_subtasks_to_dag(
                    instance_id=instance_id,
                    subtasks=subtasks,
                    db_name=db_name,
                    query=query,
                )
                self._dag_enabled = True
                logger.info(f"Converted {len(subtasks)} subtasks to DAG")
                return self._current_dag

            # List of subtasks (convert to DAG)
            if isinstance(dag_source, list):
                self._current_dag = DAGBuilder.create_from_linear_subtasks(
                    dag_id="converted",
                    subtasks=dag_source,
                )
                self._dag_enabled = True
                logger.info(f"Converted {len(dag_source)} subtasks to DAG")
                return self._current_dag

            # JSON string
            if isinstance(dag_source, str):
                self._current_dag = TaskDAG.from_json(dag_source)
                self._dag_enabled = True
                logger.info(f"Loaded DAG from JSON with {len(self._current_dag.nodes)} nodes")
                return self._current_dag

            logger.warning(f"Unable to load DAG from source type: {type(dag_source)}")
            return None

        except Exception as e:
            logger.error(f"Failed to load DAG: {e}")
            self._dag_enabled = False
            return None

    def create_dag_for_query(
        self,
        query_data: Dict[str, Any],
        template: str = "sql_with_search",
    ) -> TaskDAG:
        """
        Create a DAG for a query using a template.

        Args:
            query_data: Query data dictionary
            template: Template name to use

        Returns:
            Created TaskDAG
        """
        params = {
            "db_name": query_data.get("db", ""),
            "query": query_data.get("query", query_data.get("advanced_query", "")),
            "expected_sql": None,
            "expected_result": None,
        }

        # Extract expected values from gold_subtasks if available
        gold_subtasks = query_data.get("gold_subtasks", [])
        for subtask in gold_subtasks:
            if subtask.get("tool") in ["generated_sql", "sql_generate"]:
                params["expected_sql"] = subtask.get("expected_SQL")
            if subtask.get("tool") in ["execute_sql", "sql_execute"]:
                params["expected_result"] = subtask.get("expected_result")

        self._current_dag = DAGBuilder.create_dag_from_template(
            dag_id=query_data.get("instance_id", "unknown"),
            template_name=template,
            params=params,
        )
        self._dag_enabled = True

        logger.info(f"Created DAG from template '{template}' with {len(self._current_dag.nodes)} nodes")
        return self._current_dag

    def start_dag_execution(self) -> None:
        """Start DAG execution and initialize execution state."""
        if not self._current_dag:
            logger.warning("No DAG loaded, cannot start execution")
            return

        self._execution_state = DAGExecutionState(dag=self._current_dag)
        logger.info(f"Started DAG execution for {self._current_dag.dag_id}")

    def get_next_executable_nodes(self) -> List[str]:
        """
        Get list of node IDs that can be executed now.

        Returns:
            List of executable node IDs (may be empty)
        """
        if not self._current_dag:
            return []

        return self._current_dag.get_executable_nodes()

    def get_next_executable_tools(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Get list of executable tools with their node IDs and inputs.

        Returns:
            List of (node_id, tool_name, input_params) tuples
        """
        if not self._current_dag:
            return []

        executable_nodes = self.get_next_executable_nodes()
        tools = []

        for node_id in executable_nodes:
            node = self._current_dag.nodes.get(node_id)
            if node:
                tools.append((node_id, node.tool, node.input.copy()))

        return tools

    def can_execute_parallel(self, node_ids: List[str] = None) -> bool:
        """
        Check if nodes can be executed in parallel.

        Args:
            node_ids: Specific nodes to check, or None for all executable

        Returns:
            True if parallel execution is safe
        """
        if not self._current_dag:
            return False

        if node_ids is None:
            node_ids = self.get_next_executable_nodes()

        return self._current_dag.can_execute_parallel(node_ids)

    def mark_node_complete(
        self,
        node_id: str,
        result: Any = None,
        execution_time: float = 0.0,
    ) -> None:
        """
        Mark a node as completed.

        Args:
            node_id: ID of the completed node
            result: Execution result
            execution_time: Time taken in seconds
        """
        if not self._current_dag:
            return

        self._current_dag.mark_node_complete(node_id, result, execution_time)

        if self._execution_state:
            self._execution_state.record_execution(
                node_id=node_id,
                result=result,
                execution_time=execution_time,
                success=True,
            )

        logger.info(f"Completed DAG node: {node_id}")

    def mark_node_failed(
        self,
        node_id: str,
        error: str,
        execution_time: float = 0.0,
    ) -> None:
        """
        Mark a node as failed.

        Args:
            node_id: ID of the failed node
            error: Error message
            execution_time: Time taken in seconds
        """
        if not self._current_dag:
            return

        self._current_dag.mark_node_failed(node_id, error, execution_time)

        if self._execution_state:
            self._execution_state.record_execution(
                node_id=node_id,
                result=None,
                execution_time=execution_time,
                success=False,
                error=error,
            )

        logger.warning(f"Failed DAG node: {node_id} - {error}")

    def mark_node_skipped(self, node_id: str) -> None:
        """
        Mark a node as skipped.

        Args:
            node_id: ID of the skipped node
        """
        if not self._current_dag:
            return

        self._current_dag.mark_node_skipped(node_id)
        logger.info(f"Skipped DAG node: {node_id}")

    def is_dag_complete(self) -> bool:
        """
        Check if DAG execution is complete.

        Returns:
            True if all required nodes are completed
        """
        if not self._current_dag:
            return True

        required_nodes = self._current_dag.get_required_nodes()
        completed_nodes = self._current_dag.get_completed_nodes()

        return all(node_id in completed_nodes for node_id in required_nodes)

    def is_critical_path_complete(self) -> bool:
        """
        Check if critical path is complete.

        Returns:
            True if all critical path nodes are completed
        """
        if not self._current_dag:
            return True

        critical_path = self._current_dag.critical_path
        completed_nodes = self._current_dag.get_completed_nodes()

        return all(node_id in completed_nodes for node_id in critical_path)

    def get_dag_progress(self) -> Dict[str, Any]:
        """
        Get current DAG execution progress.

        Returns:
            Dictionary with progress metrics
        """
        if not self._current_dag:
            return {"enabled": False}

        total_nodes = len(self._current_dag.nodes)
        completed_nodes = len(self._current_dag.get_completed_nodes())
        failed_nodes = len(self._current_dag.get_failed_nodes())
        required_nodes = len(self._current_dag.get_required_nodes())
        required_completed = len([
            n for n in self._current_dag.get_required_nodes()
            if n in self._current_dag.get_completed_nodes()
        ])

        return {
            "enabled": True,
            "dag_id": self._current_dag.dag_id,
            "total_nodes": total_nodes,
            "completed_nodes": completed_nodes,
            "failed_nodes": failed_nodes,
            "required_nodes": required_nodes,
            "required_completed": required_completed,
            "progress_percent": (completed_nodes / total_nodes * 100) if total_nodes > 0 else 0,
            "required_progress_percent": (required_completed / required_nodes * 100) if required_nodes > 0 else 0,
            "is_complete": self.is_dag_complete(),
            "critical_path_complete": self.is_critical_path_complete(),
        }

    def get_dag_execution_data(self) -> Dict[str, Any]:
        """
        Get execution data suitable for evaluation.

        Returns:
            Dictionary with execution data for DAGEvaluator
        """
        if not self._current_dag:
            return {}

        completed_nodes = self._current_dag.get_completed_nodes()
        tools_executed = []
        tool_results = {}
        execution_order = []

        for node_id in completed_nodes:
            node = self._current_dag.nodes.get(node_id)
            if node:
                tools_executed.append(node.tool)
                if node.actual_result is not None:
                    tool_results[node.tool] = node.actual_result

        if self._execution_state:
            execution_order = self._execution_state.execution_order
            tool_results.update(self._execution_state.tool_results)

        return {
            "tools_executed": tools_executed,
            "tool_results": tool_results,
            "execution_order": execution_order,
            "dag_progress": self.get_dag_progress(),
        }

    def finish_dag_execution(self) -> Dict[str, Any]:
        """
        Finish DAG execution and return final state.

        Returns:
            Dictionary with final execution state
        """
        if self._execution_state:
            self._execution_state.finish()

        execution_data = self.get_dag_execution_data()

        # Add timing information
        if self._execution_state:
            execution_data["total_execution_time"] = self._execution_state.get_total_execution_time()
            execution_data["dep_violations"] = self._execution_state.dep_violations

        logger.info(f"Finished DAG execution: {execution_data.get('dag_progress', {})}")
        return execution_data

    def get_node_by_tool(self, tool_name: str) -> Optional[DAGNode]:
        """
        Get a node by its tool name.

        Args:
            tool_name: Tool name to search for

        Returns:
            DAGNode or None
        """
        if not self._current_dag:
            return None

        for node in self._current_dag.nodes.values():
            if node.tool == tool_name:
                return node

        return None

    def get_pending_required_nodes(self) -> List[str]:
        """
        Get list of required nodes that are still pending.

        Returns:
            List of pending required node IDs
        """
        if not self._current_dag:
            return []

        required = self._current_dag.get_required_nodes()
        return [
            node_id for node_id in required
            if self._current_dag.nodes[node_id].status == NodeStatus.PENDING
        ]

    def should_skip_optional_nodes(self) -> bool:
        """
        Determine if optional nodes should be skipped.

        Returns:
            True if optional nodes can be skipped (e.g., time constraint)
        """
        # Can be overridden by subclasses for custom logic
        return False

    def get_alt_group_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all alternative groups.

        Returns:
            Dictionary of group_id -> status info
        """
        if not self._current_dag:
            return {}

        status = {}
        for group_id, group in self._current_dag.alt_groups.items():
            completed_count = sum(
                1 for node_id in group.node_ids
                if node_id in self._current_dag.nodes and
                self._current_dag.nodes[node_id].status == NodeStatus.COMPLETED
            )
            status[group_id] = {
                "nodes": group.node_ids,
                "min_required": group.min_required,
                "completed": completed_count,
                "satisfied": completed_count >= group.min_required,
            }

        return status
