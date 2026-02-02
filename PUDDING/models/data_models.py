# -*- coding: utf-8 -*-
"""
Data Models - Core data structures for dataset building.
Extracted from main.py and external_tools.py.
"""

import time
from typing import Dict, List, Optional, Any


class SubtaskResult:
    """Container for subtask execution results."""

    def __init__(
        self,
        tool_name: str,
        result: str,
        success: bool = True,
        error: Optional[str] = None,
        selected_tool_type: Optional[str] = None
    ):
        self.tool_name = tool_name
        self.result = result
        self.success = success
        self.error = error
        self.selected_tool_type = selected_tool_type
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "tool_name": self.tool_name,
            "result": self.result,
            "success": self.success,
            "error": self.error,
            "selected_tool_type": self.selected_tool_type,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubtaskResult":
        """Create SubtaskResult from dictionary."""
        instance = cls(
            tool_name=data.get("tool_name", ""),
            result=data.get("result", ""),
            success=data.get("success", True),
            error=data.get("error"),
            selected_tool_type=data.get("selected_tool_type"),
        )
        if "timestamp" in data:
            instance.timestamp = data["timestamp"]
        return instance


class DatasetEntry:
    """Represents a single dataset entry."""

    def __init__(
        self,
        instance_id: str,
        db: str,
        level: Optional[str] = None,
        database_type: str = "bird"
    ):
        self.instance_id = instance_id
        self.db = db
        self.level = level
        self.database_type = database_type
        self.timestamp = time.time()

    def to_dict(self, question_data: Dict, gold_subtasks: List[Dict]) -> Dict:
        """Convert to dictionary format following original script structure."""
        choice_question = question_data.get("question", "")

        return {
            "instance_id": self.instance_id,
            "db": self.db,
            "level": self.level,
            "database_type": self.database_type,
            "question_type": "single_choice",
            "tools_available": [
                "get_schema_info",
                "generated_sql",
                "execute_sql",
                "perplexity_search",
                "vectorDB_search",
                "sql_optimize",
                "file_system",
                "context_history",
                "sql_debug"
            ],
            "gold_subtasks": gold_subtasks,
            "query": choice_question,
            "options": question_data.get("options", {}),
            "correct_answer": question_data.get("correct_answer", []),
            "explanation": question_data.get("explanation", "")
        }
