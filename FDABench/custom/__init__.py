"""
Custom interfaces for extending DB Agent Bench.

This module provides base classes and interfaces for creating custom:
- Agent patterns
- Tools and capabilities
- Evaluation methods
- Data processing pipelines

Users can inherit from these base classes to implement their own
agent behaviors and tool integrations.
"""

from .custom_agent import CustomAgent, CustomAgentBase
from .custom_tool import CustomTool, CustomToolBase

__all__ = [
    "CustomAgent",
    "CustomAgentBase", 
    "CustomTool",
    "CustomToolBase",
]