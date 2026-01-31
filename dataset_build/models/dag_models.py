# -*- coding: utf-8 -*-
"""
DAG Models - Data structures for Task Graph (DAG) based benchmark.

This module defines the core data models for representing task execution
as a Directed Acyclic Graph (DAG) with support for:
- Multiple node types (SQL_QUERY, RETRIEVE_DOC, COMPUTE, etc.)
- Different edge types (HARD_DEP, SOFT_DEP, ALT_GROUP)
- Alternative execution paths (OR branches)
- Critical path identification
"""

import json
import time
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict


class NodeType(Enum):
    """Types of nodes in the task DAG."""
    SQL_QUERY = "SQL_QUERY"           # Access DB, produce structured results
    RETRIEVE_DOC = "RETRIEVE_DOC"     # Retrieve unstructured content
    EXTRACT_EVIDENCE = "EXTRACT_EVIDENCE"  # Extract key facts from docs
    COMPUTE = "COMPUTE"               # Compute on SQL results
    VALIDATE = "VALIDATE"             # Sanity check with external knowledge
    SYNTHESIZE_REPORT = "SYNTHESIZE_REPORT"  # Generate final output


class EdgeType(Enum):
    """Types of edges in the task DAG."""
    HARD_DEP = "HARD_DEP"    # Must complete A before B
    SOFT_DEP = "SOFT_DEP"    # Recommended but not required
    ALT_GROUP = "ALT_GROUP"  # Alternative branches (OR)


class NodeStatus(Enum):
    """Execution status of a DAG node."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# Tool to NodeType mapping
TOOL_TO_NODE_TYPE: Dict[str, NodeType] = {
    # SQL_QUERY tools
    "get_schema_info": NodeType.SQL_QUERY,
    "schema_understanding": NodeType.SQL_QUERY,
    "generated_sql": NodeType.SQL_QUERY,
    "sql_generate": NodeType.SQL_QUERY,
    "generate_sql": NodeType.SQL_QUERY,
    "execute_sql": NodeType.SQL_QUERY,
    "sql_execute": NodeType.SQL_QUERY,

    # RETRIEVE_DOC tools
    "perplexity_search": NodeType.RETRIEVE_DOC,
    "web_search": NodeType.RETRIEVE_DOC,
    "web_context_search": NodeType.RETRIEVE_DOC,
    "vectorDB_search": NodeType.RETRIEVE_DOC,
    "vector_search": NodeType.RETRIEVE_DOC,
    "file_system": NodeType.RETRIEVE_DOC,
    "file_system_search": NodeType.RETRIEVE_DOC,

    # EXTRACT_EVIDENCE tools
    "extract_evidence": NodeType.EXTRACT_EVIDENCE,

    # COMPUTE tools
    "sql_optimize": NodeType.COMPUTE,
    "sql_debug": NodeType.COMPUTE,

    # VALIDATE tools
    "validate": NodeType.VALIDATE,

    # SYNTHESIZE_REPORT tools
    "synthesize_report": NodeType.SYNTHESIZE_REPORT,
    "generate_report": NodeType.SYNTHESIZE_REPORT,
}


def get_node_type_for_tool(tool: str) -> NodeType:
    """Get the NodeType for a given tool name."""
    return TOOL_TO_NODE_TYPE.get(tool, NodeType.SQL_QUERY)


@dataclass
class DAGNode:
    """
    A node in the task DAG representing a single subtask.

    Attributes:
        node_id: Unique identifier for the node
        node_type: Type of the node (SQL_QUERY, RETRIEVE_DOC, etc.)
        tool: Tool name to execute
        input: Input parameters for the tool
        description: Human-readable description of the subtask
        is_required: Whether this node must be completed
        is_critical_path: Whether this node is on the critical path
        alt_group_id: ID of the alternative group (if part of OR branch)
        expected_result: Expected result for validation
        expected_sql: Expected SQL query for SQL nodes
        status: Current execution status
        actual_result: Actual execution result
        execution_time: Time taken to execute in seconds
        error: Error message if execution failed
    """
    node_id: str
    node_type: NodeType
    tool: str
    input: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    is_required: bool = True
    is_critical_path: bool = False
    alt_group_id: Optional[str] = None
    expected_result: Optional[str] = None
    expected_sql: Optional[str] = None
    status: NodeStatus = NodeStatus.PENDING
    actual_result: Optional[Any] = None
    execution_time: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "tool": self.tool,
            "input": self.input,
            "description": self.description,
            "is_required": self.is_required,
            "is_critical_path": self.is_critical_path,
            "alt_group_id": self.alt_group_id,
            "expected_result": self.expected_result,
            "expected_sql": self.expected_sql,
            "status": self.status.value,
            "actual_result": self.actual_result,
            "execution_time": self.execution_time,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DAGNode":
        """Create DAGNode from dictionary."""
        return cls(
            node_id=data["node_id"],
            node_type=NodeType(data["node_type"]) if isinstance(data["node_type"], str) else data["node_type"],
            tool=data["tool"],
            input=data.get("input", {}),
            description=data.get("description", ""),
            is_required=data.get("is_required", True),
            is_critical_path=data.get("is_critical_path", False),
            alt_group_id=data.get("alt_group_id"),
            expected_result=data.get("expected_result"),
            expected_sql=data.get("expected_sql"),
            status=NodeStatus(data.get("status", "pending")),
            actual_result=data.get("actual_result"),
            execution_time=data.get("execution_time", 0.0),
            error=data.get("error"),
        )


@dataclass
class DAGEdge:
    """
    An edge in the task DAG representing a dependency.

    Attributes:
        source_id: ID of the source node
        target_id: ID of the target node
        edge_type: Type of dependency (HARD_DEP, SOFT_DEP, ALT_GROUP)
        weight: Weight for scheduling priority (default 1.0)
    """
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DAGEdge":
        """Create DAGEdge from dictionary."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=EdgeType(data["edge_type"]) if isinstance(data["edge_type"], str) else data["edge_type"],
            weight=data.get("weight", 1.0),
        )


@dataclass
class AlternativeGroup:
    """
    A group of alternative nodes where at least min_required must be completed.

    Attributes:
        group_id: Unique identifier for the group
        node_ids: List of node IDs in this alternative group
        min_required: Minimum number of nodes that must be completed
    """
    group_id: str
    node_ids: List[str]
    min_required: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "group_id": self.group_id,
            "node_ids": self.node_ids,
            "min_required": self.min_required,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlternativeGroup":
        """Create AlternativeGroup from dictionary."""
        return cls(
            group_id=data["group_id"],
            node_ids=data["node_ids"],
            min_required=data.get("min_required", 1),
        )


@dataclass
class TaskDAG:
    """
    A Directed Acyclic Graph representing a task workflow.

    Attributes:
        dag_id: Unique identifier for the DAG
        nodes: Dictionary of node_id to DAGNode
        edges: List of DAGEdge
        alt_groups: Dictionary of group_id to AlternativeGroup
        entry_nodes: List of entry node IDs (no incoming edges)
        exit_nodes: List of exit node IDs (no outgoing edges)
        critical_path: Ordered list of node IDs on the critical path
        metadata: Additional metadata for the DAG
    """
    dag_id: str
    nodes: Dict[str, DAGNode] = field(default_factory=dict)
    edges: List[DAGEdge] = field(default_factory=list)
    alt_groups: Dict[str, AlternativeGroup] = field(default_factory=dict)
    entry_nodes: List[str] = field(default_factory=list)
    exit_nodes: List[str] = field(default_factory=list)
    critical_path: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: DAGNode) -> None:
        """Add a node to the DAG."""
        self.nodes[node.node_id] = node
        self._update_entry_exit_nodes()

    def add_edge(self, edge: DAGEdge) -> None:
        """Add an edge to the DAG."""
        self.edges.append(edge)
        self._update_entry_exit_nodes()

    def add_alt_group(self, group: AlternativeGroup) -> None:
        """Add an alternative group to the DAG."""
        self.alt_groups[group.group_id] = group
        # Update nodes with alt_group_id
        for node_id in group.node_ids:
            if node_id in self.nodes:
                self.nodes[node_id].alt_group_id = group.group_id

    def _update_entry_exit_nodes(self) -> None:
        """Update entry and exit node lists based on edges."""
        if not self.nodes:
            return

        # Find nodes with incoming edges
        nodes_with_incoming = set()
        # Find nodes with outgoing edges
        nodes_with_outgoing = set()

        for edge in self.edges:
            nodes_with_incoming.add(edge.target_id)
            nodes_with_outgoing.add(edge.source_id)

        all_node_ids = set(self.nodes.keys())

        # Entry nodes have no incoming edges
        self.entry_nodes = list(all_node_ids - nodes_with_incoming)
        # Exit nodes have no outgoing edges
        self.exit_nodes = list(all_node_ids - nodes_with_outgoing)

    def get_node_dependencies(self, node_id: str) -> List[str]:
        """Get list of node IDs that the given node depends on (incoming edges)."""
        deps = []
        for edge in self.edges:
            if edge.target_id == node_id:
                deps.append(edge.source_id)
        return deps

    def get_node_dependents(self, node_id: str) -> List[str]:
        """Get list of node IDs that depend on the given node (outgoing edges)."""
        dependents = []
        for edge in self.edges:
            if edge.source_id == node_id:
                dependents.append(edge.target_id)
        return dependents

    def get_hard_dependencies(self, node_id: str) -> List[str]:
        """Get list of node IDs with HARD_DEP to the given node."""
        deps = []
        for edge in self.edges:
            if edge.target_id == node_id and edge.edge_type == EdgeType.HARD_DEP:
                deps.append(edge.source_id)
        return deps

    def get_executable_nodes(self) -> List[str]:
        """
        Get list of node IDs that can be executed now.
        A node is executable if:
        - Status is PENDING
        - All HARD_DEP dependencies are COMPLETED
        """
        executable = []
        for node_id, node in self.nodes.items():
            if node.status != NodeStatus.PENDING:
                continue

            # Check all hard dependencies are completed
            hard_deps = self.get_hard_dependencies(node_id)
            all_deps_completed = all(
                self.nodes[dep_id].status == NodeStatus.COMPLETED
                for dep_id in hard_deps
                if dep_id in self.nodes
            )

            if all_deps_completed:
                executable.append(node_id)

        return executable

    def can_execute_parallel(self, node_ids: List[str]) -> bool:
        """Check if a list of nodes can be executed in parallel."""
        # Nodes can be parallel if there are no HARD_DEP between them
        for i, node_a in enumerate(node_ids):
            for node_b in node_ids[i+1:]:
                # Check if there's a hard dependency between them
                for edge in self.edges:
                    if edge.edge_type == EdgeType.HARD_DEP:
                        if (edge.source_id == node_a and edge.target_id == node_b) or \
                           (edge.source_id == node_b and edge.target_id == node_a):
                            return False
        return True

    def mark_node_complete(self, node_id: str, result: Any = None, execution_time: float = 0.0) -> None:
        """Mark a node as completed with its result."""
        if node_id in self.nodes:
            self.nodes[node_id].status = NodeStatus.COMPLETED
            self.nodes[node_id].actual_result = result
            self.nodes[node_id].execution_time = execution_time

    def mark_node_failed(self, node_id: str, error: str, execution_time: float = 0.0) -> None:
        """Mark a node as failed with error message."""
        if node_id in self.nodes:
            self.nodes[node_id].status = NodeStatus.FAILED
            self.nodes[node_id].error = error
            self.nodes[node_id].execution_time = execution_time

    def mark_node_skipped(self, node_id: str) -> None:
        """Mark a node as skipped."""
        if node_id in self.nodes:
            self.nodes[node_id].status = NodeStatus.SKIPPED

    def compute_critical_path(self) -> List[str]:
        """
        Compute the critical path through the DAG.
        The critical path is the longest path considering only HARD_DEP edges
        from entry to exit nodes.
        """
        if not self.entry_nodes or not self.exit_nodes:
            self._update_entry_exit_nodes()

        if not self.entry_nodes:
            return []

        # Build adjacency list for hard dependencies only
        adj: Dict[str, List[str]] = {node_id: [] for node_id in self.nodes}
        for edge in self.edges:
            if edge.edge_type == EdgeType.HARD_DEP:
                adj[edge.source_id].append(edge.target_id)

        # Find longest path using DFS with memoization
        memo: Dict[str, tuple] = {}  # node_id -> (length, path)

        def dfs(node_id: str) -> tuple:
            if node_id in memo:
                return memo[node_id]

            if node_id in self.exit_nodes or not adj[node_id]:
                memo[node_id] = (1, [node_id])
                return memo[node_id]

            max_length = 0
            best_path = [node_id]

            for next_node in adj[node_id]:
                if next_node in self.nodes:
                    length, path = dfs(next_node)
                    if length + 1 > max_length:
                        max_length = length + 1
                        best_path = [node_id] + path

            memo[node_id] = (max_length, best_path)
            return memo[node_id]

        # Find the longest path starting from any entry node
        longest_path = []
        for entry in self.entry_nodes:
            _, path = dfs(entry)
            if len(path) > len(longest_path):
                longest_path = path

        self.critical_path = longest_path

        # Mark nodes on critical path
        for node_id in longest_path:
            if node_id in self.nodes:
                self.nodes[node_id].is_critical_path = True

        return longest_path

    def get_required_nodes(self) -> List[str]:
        """Get list of required node IDs."""
        return [node_id for node_id, node in self.nodes.items() if node.is_required]

    def get_optional_nodes(self) -> List[str]:
        """Get list of optional node IDs."""
        return [node_id for node_id, node in self.nodes.items() if not node.is_required]

    def get_completed_nodes(self) -> List[str]:
        """Get list of completed node IDs."""
        return [node_id for node_id, node in self.nodes.items() if node.status == NodeStatus.COMPLETED]

    def get_failed_nodes(self) -> List[str]:
        """Get list of failed node IDs."""
        return [node_id for node_id, node in self.nodes.items() if node.status == NodeStatus.FAILED]

    def is_alt_group_satisfied(self, group_id: str) -> bool:
        """Check if an alternative group has been satisfied."""
        if group_id not in self.alt_groups:
            return True

        group = self.alt_groups[group_id]
        completed_count = sum(
            1 for node_id in group.node_ids
            if node_id in self.nodes and self.nodes[node_id].status == NodeStatus.COMPLETED
        )
        return completed_count >= group.min_required

    def get_execution_order(self) -> List[str]:
        """Get the topological order of node execution based on completion times."""
        completed_nodes = [
            (node_id, node.execution_time)
            for node_id, node in self.nodes.items()
            if node.status == NodeStatus.COMPLETED
        ]
        # Sort by execution completion (using actual order if times are same)
        return [node_id for node_id, _ in completed_nodes]

    def validate(self) -> List[str]:
        """
        Validate the DAG structure.
        Returns a list of validation errors (empty if valid).
        """
        errors = []

        # Check for cycles (would make it not a DAG)
        visited = set()
        rec_stack = set()

        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            for edge in self.edges:
                if edge.source_id == node_id:
                    next_node = edge.target_id
                    if next_node not in visited:
                        if has_cycle(next_node):
                            return True
                    elif next_node in rec_stack:
                        return True

            rec_stack.remove(node_id)
            return False

        for node_id in self.nodes:
            if node_id not in visited:
                if has_cycle(node_id):
                    errors.append(f"Cycle detected involving node {node_id}")
                    break

        # Check that all edge references exist
        for edge in self.edges:
            if edge.source_id not in self.nodes:
                errors.append(f"Edge source node {edge.source_id} not found")
            if edge.target_id not in self.nodes:
                errors.append(f"Edge target node {edge.target_id} not found")

        # Check alt groups
        for group_id, group in self.alt_groups.items():
            for node_id in group.node_ids:
                if node_id not in self.nodes:
                    errors.append(f"Alt group {group_id} references non-existent node {node_id}")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "dag_id": self.dag_id,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges],
            "alt_groups": {group_id: group.to_dict() for group_id, group in self.alt_groups.items()},
            "entry_nodes": self.entry_nodes,
            "exit_nodes": self.exit_nodes,
            "critical_path": self.critical_path,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskDAG":
        """Create TaskDAG from dictionary."""
        dag = cls(
            dag_id=data["dag_id"],
            entry_nodes=data.get("entry_nodes", []),
            exit_nodes=data.get("exit_nodes", []),
            critical_path=data.get("critical_path", []),
            metadata=data.get("metadata", {}),
        )

        # Load nodes
        for node_id, node_data in data.get("nodes", {}).items():
            dag.nodes[node_id] = DAGNode.from_dict(node_data)

        # Load edges
        for edge_data in data.get("edges", []):
            dag.edges.append(DAGEdge.from_dict(edge_data))

        # Load alt groups
        for group_id, group_data in data.get("alt_groups", {}).items():
            dag.alt_groups[group_id] = AlternativeGroup.from_dict(group_data)

        return dag

    @classmethod
    def from_json(cls, json_str: str) -> "TaskDAG":
        """Create TaskDAG from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class DAGExecutionState:
    """
    Tracks the execution state of a TaskDAG.

    Attributes:
        dag: The TaskDAG being executed
        start_time: Execution start time
        end_time: Execution end time
        execution_order: Ordered list of executed node IDs
        tool_results: Results from each tool execution
        dep_violations: List of dependency violations
    """
    dag: TaskDAG
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    execution_order: List[str] = field(default_factory=list)
    tool_results: Dict[str, Any] = field(default_factory=dict)
    dep_violations: List[tuple] = field(default_factory=list)

    def record_execution(self, node_id: str, result: Any, execution_time: float, success: bool, error: Optional[str] = None) -> None:
        """Record the execution of a node."""
        self.execution_order.append(node_id)
        self.tool_results[node_id] = result

        if success:
            self.dag.mark_node_complete(node_id, result, execution_time)
        else:
            self.dag.mark_node_failed(node_id, error or "Unknown error", execution_time)

        # Check for dependency violations
        self._check_dep_violations(node_id)

    def _check_dep_violations(self, node_id: str) -> None:
        """Check if executing this node violates any hard dependencies."""
        hard_deps = self.dag.get_hard_dependencies(node_id)
        for dep_id in hard_deps:
            if dep_id not in self.execution_order:
                self.dep_violations.append((node_id, dep_id))

    def finish(self) -> None:
        """Mark execution as finished."""
        self.end_time = time.time()

    def get_total_execution_time(self) -> float:
        """Get total execution time in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "dag": self.dag.to_dict(),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "execution_order": self.execution_order,
            "tool_results": self.tool_results,
            "dep_violations": self.dep_violations,
            "total_execution_time": self.get_total_execution_time(),
        }
