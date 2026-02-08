"""Tree data structures for Algorithm 1: Tree-Structured Exploration."""

from __future__ import annotations
import uuid
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


class BranchDecision(Enum):
    CONTINUE = "CONTINUE"
    SUFFICIENT = "SUFFICIENT"
    PRUNE = "PRUNE"


@dataclass
class ToolAction:
    """A single tool execution in the exploration tree."""
    tool_name: str        # web_search, vector_search, file_search
    input_params: dict    # tool-specific query/params
    rationale: str = ""   # why this branch was explored
    output: str = ""      # tool result text
    provenance: dict = field(default_factory=dict)  # URLs, chunk IDs, timestamps, etc.

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ToolAction":
        return ToolAction(**d)


@dataclass
class ExplorationNode:
    """A node in the exploration tree. Each node represents a state after executing an action."""
    node_id: str
    parent_id: Optional[str] = None
    depth: int = 0
    action: Optional[ToolAction] = None          # None for root node
    cumulative_actions: List[ToolAction] = field(default_factory=list)  # all actions root->this
    decision: Optional[BranchDecision] = None
    reflection_rationale: str = ""

    def to_dict(self) -> dict:
        d = {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "action": self.action.to_dict() if self.action else None,
            "cumulative_actions": [a.to_dict() for a in self.cumulative_actions],
            "decision": self.decision.value if self.decision else None,
            "reflection_rationale": self.reflection_rationale,
        }
        return d

    @staticmethod
    def from_dict(d: dict) -> "ExplorationNode":
        return ExplorationNode(
            node_id=d["node_id"],
            parent_id=d.get("parent_id"),
            depth=d.get("depth", 0),
            action=ToolAction.from_dict(d["action"]) if d.get("action") else None,
            cumulative_actions=[ToolAction.from_dict(a) for a in d.get("cumulative_actions", [])],
            decision=BranchDecision(d["decision"]) if d.get("decision") else None,
            reflection_rationale=d.get("reflection_rationale", ""),
        )


@dataclass
class TerminalPath:
    """A complete exploration path that was marked SUFFICIENT."""
    path_id: str
    actions: List[ToolAction] = field(default_factory=list)  # ordered root->leaf

    def to_dict(self) -> dict:
        return {
            "path_id": self.path_id,
            "actions": [a.to_dict() for a in self.actions],
        }

    @staticmethod
    def from_dict(d: dict) -> "TerminalPath":
        return TerminalPath(
            path_id=d["path_id"],
            actions=[ToolAction.from_dict(a) for a in d.get("actions", [])],
        )


class ExplorationTree:
    """The full exploration tree tracking frontier, terminal paths, and pruned nodes."""

    def __init__(self, max_iterations: int = 3):
        self.nodes: Dict[str, ExplorationNode] = {}
        self.root_id: Optional[str] = None
        self.frontier: List[str] = []           # node_ids to expand next
        self.terminal_paths: List[TerminalPath] = []  # SUFFICIENT paths
        self.pruned: List[str] = []             # PRUNE'd node_ids
        self.iteration: int = 0
        self.max_iterations: int = max_iterations

    def create_root(self, base_actions: List[ToolAction] = None) -> ExplorationNode:
        """Create root node with initial base context actions."""
        root_id = f"root_{uuid.uuid4().hex[:8]}"
        root = ExplorationNode(
            node_id=root_id,
            parent_id=None,
            depth=0,
            action=None,
            cumulative_actions=base_actions or [],
        )
        self.nodes[root_id] = root
        self.root_id = root_id
        self.frontier = [root_id]
        return root

    def add_node(self, parent_id: str, action: ToolAction) -> ExplorationNode:
        """Add a child node after executing an action from a parent."""
        parent = self.nodes[parent_id]
        node_id = f"node_{uuid.uuid4().hex[:8]}"
        node = ExplorationNode(
            node_id=node_id,
            parent_id=parent_id,
            depth=parent.depth + 1,
            action=action,
            cumulative_actions=parent.cumulative_actions + [action],
        )
        self.nodes[node_id] = node
        return node

    def mark_continue(self, node_id: str, rationale: str = ""):
        """Mark node as CONTINUE - add to frontier for further expansion."""
        node = self.nodes[node_id]
        node.decision = BranchDecision.CONTINUE
        node.reflection_rationale = rationale
        if node_id not in self.frontier:
            self.frontier.append(node_id)

    def mark_sufficient(self, node_id: str, rationale: str = ""):
        """Mark node as SUFFICIENT - create terminal path."""
        node = self.nodes[node_id]
        node.decision = BranchDecision.SUFFICIENT
        node.reflection_rationale = rationale
        path = TerminalPath(
            path_id=f"path_{uuid.uuid4().hex[:8]}",
            actions=list(node.cumulative_actions),
        )
        self.terminal_paths.append(path)
        # Remove from frontier if present
        if node_id in self.frontier:
            self.frontier.remove(node_id)

    def mark_pruned(self, node_id: str, rationale: str = ""):
        """Mark node as PRUNE - discard this branch."""
        node = self.nodes[node_id]
        node.decision = BranchDecision.PRUNE
        node.reflection_rationale = rationale
        self.pruned.append(node_id)
        if node_id in self.frontier:
            self.frontier.remove(node_id)

    def get_frontier_nodes(self) -> List[ExplorationNode]:
        """Get all nodes currently in the frontier."""
        return [self.nodes[nid] for nid in self.frontier if nid in self.nodes]

    def advance_iteration(self):
        """Advance iteration counter."""
        self.iteration += 1

    def is_complete(self) -> bool:
        """Check if exploration should stop."""
        return len(self.frontier) == 0 or self.iteration >= self.max_iterations

    def to_dict(self) -> dict:
        return {
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "root_id": self.root_id,
            "frontier": list(self.frontier),
            "terminal_paths": [p.to_dict() for p in self.terminal_paths],
            "pruned": list(self.pruned),
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
        }

    @staticmethod
    def from_dict(d: dict) -> "ExplorationTree":
        tree = ExplorationTree(max_iterations=d.get("max_iterations", 3))
        tree.nodes = {nid: ExplorationNode.from_dict(nd) for nid, nd in d.get("nodes", {}).items()}
        tree.root_id = d.get("root_id")
        tree.frontier = d.get("frontier", [])
        tree.terminal_paths = [TerminalPath.from_dict(p) for p in d.get("terminal_paths", [])]
        tree.pruned = d.get("pruned", [])
        tree.iteration = d.get("iteration", 0)
        return tree
