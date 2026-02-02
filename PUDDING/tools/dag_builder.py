# -*- coding: utf-8 -*-
"""
DAG Builder - Utilities for constructing Task DAGs.

This module provides functions to build TaskDAG objects from:
- Linear subtask lists (backwards compatible)
- Template-based configurations
- Custom specifications
"""

import logging
from typing import Dict, List, Any, Optional

from ..models.dag_models import (
    TaskDAG, DAGNode, DAGEdge, AlternativeGroup,
    NodeType, EdgeType, NodeStatus,
    get_node_type_for_tool, TOOL_TO_NODE_TYPE
)

logger = logging.getLogger(__name__)


class DAGBuilder:
    """
    Builder class for constructing TaskDAG objects.

    Provides static methods and a fluent interface for building DAGs.
    """

    def __init__(self, dag_id: str):
        """
        Initialize a new DAG builder.

        Args:
            dag_id: Unique identifier for the DAG
        """
        self.dag = TaskDAG(dag_id=dag_id)
        self._node_counter = 0

    def add_node(
        self,
        tool: str,
        input_params: Dict[str, Any] = None,
        description: str = "",
        is_required: bool = True,
        is_critical_path: bool = False,
        alt_group_id: Optional[str] = None,
        expected_result: Optional[str] = None,
        expected_sql: Optional[str] = None,
        node_id: Optional[str] = None,
    ) -> str:
        """
        Add a node to the DAG.

        Args:
            tool: Tool name to execute
            input_params: Input parameters for the tool
            description: Human-readable description
            is_required: Whether this node must be completed
            is_critical_path: Whether on critical path
            alt_group_id: Alternative group ID if part of OR branch
            expected_result: Expected result for validation
            expected_sql: Expected SQL for SQL nodes
            node_id: Optional custom node ID

        Returns:
            The node ID
        """
        if node_id is None:
            self._node_counter += 1
            node_id = f"node_{self._node_counter}"

        node = DAGNode(
            node_id=node_id,
            node_type=get_node_type_for_tool(tool),
            tool=tool,
            input=input_params or {},
            description=description,
            is_required=is_required,
            is_critical_path=is_critical_path,
            alt_group_id=alt_group_id,
            expected_result=expected_result,
            expected_sql=expected_sql,
        )
        self.dag.add_node(node)
        return node_id

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType = EdgeType.HARD_DEP,
        weight: float = 1.0,
    ) -> "DAGBuilder":
        """
        Add an edge between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of dependency
            weight: Weight for scheduling

        Returns:
            Self for chaining
        """
        edge = DAGEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
        )
        self.dag.add_edge(edge)
        return self

    def add_alt_group(
        self,
        group_id: str,
        node_ids: List[str],
        min_required: int = 1,
    ) -> "DAGBuilder":
        """
        Add an alternative group.

        Args:
            group_id: Unique group identifier
            node_ids: List of node IDs in the group
            min_required: Minimum number of nodes to complete

        Returns:
            Self for chaining
        """
        group = AlternativeGroup(
            group_id=group_id,
            node_ids=node_ids,
            min_required=min_required,
        )
        self.dag.add_alt_group(group)
        return self

    def set_metadata(self, metadata: Dict[str, Any]) -> "DAGBuilder":
        """
        Set DAG metadata.

        Args:
            metadata: Metadata dictionary

        Returns:
            Self for chaining
        """
        self.dag.metadata = metadata
        return self

    def build(self) -> TaskDAG:
        """
        Build and return the TaskDAG.

        Also computes the critical path.

        Returns:
            The constructed TaskDAG
        """
        self.dag._update_entry_exit_nodes()
        self.dag.compute_critical_path()
        return self.dag

    @staticmethod
    def create_from_linear_subtasks(
        dag_id: str,
        subtasks: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskDAG:
        """
        Create a TaskDAG from a list of subtasks.

        If subtasks contain 'depends_on' fields, those dependencies are used.
        Otherwise, creates a linear chain for backwards compatibility.

        Args:
            dag_id: Unique identifier for the DAG
            subtasks: List of subtask dictionaries with format:
                {
                    "subtask_id": str,
                    "tool": str,
                    "input": Dict,
                    "description": str,
                    "expected_result": str (optional),
                    "expected_SQL": str (optional),
                    "depends_on": List[str] (optional) - explicit dependencies
                }
            metadata: Optional metadata

        Returns:
            A TaskDAG with proper dependencies
        """
        builder = DAGBuilder(dag_id)

        if metadata:
            builder.set_metadata(metadata)

        # Check if any subtask has explicit depends_on
        has_explicit_deps = any("depends_on" in st for st in subtasks)

        node_ids = []
        node_deps = {}  # Store depends_on for each node

        for subtask in subtasks:
            node_id = builder.add_node(
                tool=subtask["tool"],
                input_params=subtask.get("input", {}),
                description=subtask.get("description", ""),
                is_required=True,
                expected_result=subtask.get("expected_result"),
                expected_sql=subtask.get("expected_SQL") or subtask.get("expected_output"),
                node_id=subtask.get("subtask_id"),
            )
            node_ids.append(node_id)

            # Store depends_on if present
            if "depends_on" in subtask:
                node_deps[node_id] = subtask["depends_on"]

        # Create edges based on dependencies
        if has_explicit_deps:
            # Use explicit depends_on from subtasks
            for node_id, deps in node_deps.items():
                for dep in deps:
                    builder.add_edge(dep, node_id, EdgeType.HARD_DEP)

            # For nodes without explicit deps (typically first few nodes),
            # create sequential deps if they appear before any node with deps
            nodes_with_deps = set(node_deps.keys())
            prev_node = None
            for node_id in node_ids:
                if node_id in nodes_with_deps:
                    break  # Stop when we hit a node with explicit deps
                if prev_node is not None:
                    builder.add_edge(prev_node, node_id, EdgeType.HARD_DEP)
                prev_node = node_id
        else:
            # Fallback: create sequential hard dependencies (linear chain)
            for i in range(len(node_ids) - 1):
                builder.add_edge(node_ids[i], node_ids[i + 1], EdgeType.HARD_DEP)

        return builder.build()

    @staticmethod
    def create_sql_workflow_dag(
        dag_id: str,
        db_name: str,
        query: str,
        include_search: bool = True,
        search_type: str = "web",  # "web", "vector", or "both"
        expected_sql: Optional[str] = None,
        expected_result: Optional[str] = None,
    ) -> TaskDAG:
        """
        Create a standard SQL analysis workflow DAG.

        The workflow follows:
        1. Schema understanding (required, critical)
        2. SQL generation (required, critical) - depends on 1
        3. SQL execution (required, critical) - depends on 2
        4. Optional: Web/Vector search (soft dependency from 1)
        5. Report synthesis (required, critical) - depends on 3, soft from 4

        Args:
            dag_id: Unique identifier
            db_name: Database name
            query: The query to answer
            include_search: Whether to include search nodes
            search_type: Type of search ("web", "vector", "both")
            expected_sql: Expected SQL for validation
            expected_result: Expected result for validation

        Returns:
            TaskDAG for SQL workflow
        """
        builder = DAGBuilder(dag_id)
        builder.set_metadata({
            "db_name": db_name,
            "query": query,
            "workflow_type": "sql_analysis",
        })

        # 1. Schema understanding (entry, critical)
        schema_id = builder.add_node(
            tool="get_schema_info",
            input_params={"database_name": db_name},
            description=f"Get schema information for database {db_name}",
            is_required=True,
            is_critical_path=True,
            node_id="schema",
        )

        # 2. SQL generation (critical)
        sql_gen_id = builder.add_node(
            tool="generated_sql",
            input_params={"natural_language_query": query, "database_name": db_name},
            description=f"Generate SQL to answer: {query}",
            is_required=True,
            is_critical_path=True,
            expected_sql=expected_sql,
            node_id="sql_gen",
        )
        builder.add_edge(schema_id, sql_gen_id, EdgeType.HARD_DEP)

        # 3. SQL execution (critical)
        sql_exec_id = builder.add_node(
            tool="execute_sql",
            input_params={"database_name": db_name},
            description=f"Execute SQL query",
            is_required=True,
            is_critical_path=True,
            expected_result=expected_result,
            node_id="sql_exec",
        )
        builder.add_edge(sql_gen_id, sql_exec_id, EdgeType.HARD_DEP)

        # 4. Optional search nodes
        search_node_ids = []
        if include_search:
            if search_type in ["web", "both"]:
                web_search_id = builder.add_node(
                    tool="perplexity_search",
                    input_params={"query": query},
                    description=f"Web search for context: {query}",
                    is_required=False,
                    is_critical_path=False,
                    node_id="web_search",
                )
                builder.add_edge(schema_id, web_search_id, EdgeType.SOFT_DEP)
                search_node_ids.append(web_search_id)

            if search_type in ["vector", "both"]:
                vector_search_id = builder.add_node(
                    tool="vectorDB_search",
                    input_params={"query": query},
                    description=f"Vector search for context: {query}",
                    is_required=False,
                    is_critical_path=False,
                    node_id="vector_search",
                )
                builder.add_edge(schema_id, vector_search_id, EdgeType.SOFT_DEP)
                search_node_ids.append(vector_search_id)

            # Create alternative group if both search types
            if len(search_node_ids) > 1:
                builder.add_alt_group("search_group", search_node_ids, min_required=1)

        # 5. Report synthesis (exit, critical)
        synthesis_id = builder.add_node(
            tool="synthesize_report",
            input_params={"query": query},
            description=f"Synthesize report for: {query}",
            is_required=True,
            is_critical_path=True,
            node_id="synthesis",
        )
        builder.add_edge(sql_exec_id, synthesis_id, EdgeType.HARD_DEP)

        # Add soft dependencies from search to synthesis
        for search_id in search_node_ids:
            builder.add_edge(search_id, synthesis_id, EdgeType.SOFT_DEP)

        return builder.build()

    @staticmethod
    def create_dag_from_template(
        dag_id: str,
        template_name: str,
        params: Dict[str, Any],
    ) -> TaskDAG:
        """
        Create a DAG from a predefined template.

        Available templates:
        - "sql_basic": Schema -> SQL Gen -> SQL Exec -> Synthesis
        - "sql_with_search": Adds web/vector search branches
        - "sql_with_validation": Adds validation step
        - "multi_source": Multiple data sources with aggregation

        Args:
            dag_id: Unique identifier
            template_name: Name of the template
            params: Parameters for the template

        Returns:
            TaskDAG based on template
        """
        templates = {
            "sql_basic": DAGBuilder._template_sql_basic,
            "sql_with_search": DAGBuilder._template_sql_with_search,
            "sql_with_validation": DAGBuilder._template_sql_with_validation,
            "multi_source": DAGBuilder._template_multi_source,
        }

        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}. Available: {list(templates.keys())}")

        return templates[template_name](dag_id, params)

    @staticmethod
    def _template_sql_basic(dag_id: str, params: Dict[str, Any]) -> TaskDAG:
        """Basic SQL workflow template."""
        return DAGBuilder.create_sql_workflow_dag(
            dag_id=dag_id,
            db_name=params.get("db_name", ""),
            query=params.get("query", ""),
            include_search=False,
            expected_sql=params.get("expected_sql"),
            expected_result=params.get("expected_result"),
        )

    @staticmethod
    def _template_sql_with_search(dag_id: str, params: Dict[str, Any]) -> TaskDAG:
        """SQL workflow with search branches template."""
        return DAGBuilder.create_sql_workflow_dag(
            dag_id=dag_id,
            db_name=params.get("db_name", ""),
            query=params.get("query", ""),
            include_search=True,
            search_type=params.get("search_type", "both"),
            expected_sql=params.get("expected_sql"),
            expected_result=params.get("expected_result"),
        )

    @staticmethod
    def _template_sql_with_validation(dag_id: str, params: Dict[str, Any]) -> TaskDAG:
        """SQL workflow with validation step template."""
        builder = DAGBuilder(dag_id)
        db_name = params.get("db_name", "")
        query = params.get("query", "")

        builder.set_metadata({
            "db_name": db_name,
            "query": query,
            "workflow_type": "sql_with_validation",
        })

        # Schema
        schema_id = builder.add_node(
            tool="get_schema_info",
            input_params={"database_name": db_name},
            description="Get schema information",
            is_required=True,
            is_critical_path=True,
            node_id="schema",
        )

        # SQL generation
        sql_gen_id = builder.add_node(
            tool="generated_sql",
            input_params={"natural_language_query": query, "database_name": db_name},
            description="Generate SQL",
            is_required=True,
            is_critical_path=True,
            expected_sql=params.get("expected_sql"),
            node_id="sql_gen",
        )
        builder.add_edge(schema_id, sql_gen_id, EdgeType.HARD_DEP)

        # SQL execution
        sql_exec_id = builder.add_node(
            tool="execute_sql",
            input_params={"database_name": db_name},
            description="Execute SQL",
            is_required=True,
            is_critical_path=True,
            expected_result=params.get("expected_result"),
            node_id="sql_exec",
        )
        builder.add_edge(sql_gen_id, sql_exec_id, EdgeType.HARD_DEP)

        # Validation with external knowledge
        validate_id = builder.add_node(
            tool="validate",
            input_params={"query": query},
            description="Validate results with external knowledge",
            is_required=False,
            is_critical_path=False,
            node_id="validate",
        )
        builder.add_edge(sql_exec_id, validate_id, EdgeType.SOFT_DEP)

        # Synthesis
        synthesis_id = builder.add_node(
            tool="synthesize_report",
            input_params={"query": query},
            description="Synthesize report",
            is_required=True,
            is_critical_path=True,
            node_id="synthesis",
        )
        builder.add_edge(sql_exec_id, synthesis_id, EdgeType.HARD_DEP)
        builder.add_edge(validate_id, synthesis_id, EdgeType.SOFT_DEP)

        return builder.build()

    @staticmethod
    def _template_multi_source(dag_id: str, params: Dict[str, Any]) -> TaskDAG:
        """Multi-source data aggregation template."""
        builder = DAGBuilder(dag_id)
        db_name = params.get("db_name", "")
        query = params.get("query", "")

        builder.set_metadata({
            "db_name": db_name,
            "query": query,
            "workflow_type": "multi_source",
        })

        # Schema (entry)
        schema_id = builder.add_node(
            tool="get_schema_info",
            input_params={"database_name": db_name},
            description="Get schema information",
            is_required=True,
            is_critical_path=True,
            node_id="schema",
        )

        # SQL path
        sql_gen_id = builder.add_node(
            tool="generated_sql",
            input_params={"natural_language_query": query, "database_name": db_name},
            description="Generate SQL",
            is_required=True,
            is_critical_path=True,
            node_id="sql_gen",
        )
        builder.add_edge(schema_id, sql_gen_id, EdgeType.HARD_DEP)

        sql_exec_id = builder.add_node(
            tool="execute_sql",
            input_params={"database_name": db_name},
            description="Execute SQL",
            is_required=True,
            is_critical_path=True,
            node_id="sql_exec",
        )
        builder.add_edge(sql_gen_id, sql_exec_id, EdgeType.HARD_DEP)

        # Web search path
        web_search_id = builder.add_node(
            tool="perplexity_search",
            input_params={"query": query},
            description="Web search for context",
            is_required=False,
            node_id="web_search",
        )
        builder.add_edge(schema_id, web_search_id, EdgeType.SOFT_DEP)

        # Vector search path
        vector_search_id = builder.add_node(
            tool="vectorDB_search",
            input_params={"query": query},
            description="Vector search for context",
            is_required=False,
            node_id="vector_search",
        )
        builder.add_edge(schema_id, vector_search_id, EdgeType.SOFT_DEP)

        # File system path
        file_search_id = builder.add_node(
            tool="file_system",
            input_params={"query": query},
            description="File system search",
            is_required=False,
            node_id="file_search",
        )
        builder.add_edge(schema_id, file_search_id, EdgeType.SOFT_DEP)

        # Alt group for search options
        builder.add_alt_group(
            "search_group",
            [web_search_id, vector_search_id, file_search_id],
            min_required=1,
        )

        # Evidence extraction from all sources
        extract_id = builder.add_node(
            tool="extract_evidence",
            input_params={"query": query},
            description="Extract evidence from all sources",
            is_required=False,
            node_id="extract",
        )
        builder.add_edge(sql_exec_id, extract_id, EdgeType.SOFT_DEP)
        builder.add_edge(web_search_id, extract_id, EdgeType.SOFT_DEP)
        builder.add_edge(vector_search_id, extract_id, EdgeType.SOFT_DEP)
        builder.add_edge(file_search_id, extract_id, EdgeType.SOFT_DEP)

        # Synthesis (exit)
        synthesis_id = builder.add_node(
            tool="synthesize_report",
            input_params={"query": query},
            description="Synthesize report from all sources",
            is_required=True,
            is_critical_path=True,
            node_id="synthesis",
        )
        builder.add_edge(sql_exec_id, synthesis_id, EdgeType.HARD_DEP)
        builder.add_edge(extract_id, synthesis_id, EdgeType.SOFT_DEP)

        return builder.build()


def convert_subtasks_to_dag(
    instance_id: str,
    subtasks: List[Dict[str, Any]],
    db_name: Optional[str] = None,
    query: Optional[str] = None,
) -> TaskDAG:
    """
    Convenience function to convert linear subtasks to a DAG.

    Args:
        instance_id: Instance ID to use as DAG ID
        subtasks: List of subtask dictionaries
        db_name: Optional database name for metadata
        query: Optional query for metadata

    Returns:
        TaskDAG representation of the subtasks
    """
    metadata = {}
    if db_name:
        metadata["db_name"] = db_name
    if query:
        metadata["query"] = query

    return DAGBuilder.create_from_linear_subtasks(
        dag_id=instance_id,
        subtasks=subtasks,
        metadata=metadata,
    )
