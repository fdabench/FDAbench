"""
FDABench - A modular framework for database agents with LLM capabilities.

This package provides a flexible framework for building database agents that can:
- Execute SQL queries across different database types
- Perform web and vector searches
- Search and manage file systems
- Generate reports and analyze data
- Support multiple agent patterns (planning, tool-use, multi-agent, reflection)
- Allow custom agent patterns and tools through extensible interfaces

Example usage:
    from FDABench import BaseAgent, PlanningAgent, ToolUseAgent
    from FDABench.tools import SQLGenerationTool, WebSearchTool, FileSystemSearchTool
    from FDABench.custom import CustomAgent, CustomTool
    from FDABench.prompts import get_prompt
    
    # Create an agent (uses OpenRouter by default)
    agent = PlanningAgent(
        model="moonshotai/kimi-k2",
        api_key="your-openrouter-key"
    )
    
    # Register tools
    agent.register_tool("sql_generate", SQLGenerationTool(llm_client=agent))
    agent.register_tool("web_search", WebSearchTool()) 
    agent.register_tool("file_search", FileSystemSearchTool())
    
    # Use prompts
    prompt = get_prompt("SQL_GENERATION_SIMPLE", 
                       natural_language_query="find all users",
                       schema_summary="users(id, name, email)",
                       db_type_info="SQLite",
                       syntax_notes="Standard SQLite")
    
    # Process queries
    result = agent.process_query_from_json(query_data)
"""

__version__ = "0.1.0"
__author__ = "FDABench Team"
__email__ = "contact@fdabench.com"

# Core imports
from .core.base_agent import BaseAgent
from .core.tool_registry import ToolRegistry, register_tool
from .core.token_tracker import TokenTracker, TokenTrackingEntry

# Agent patterns  
from .agents.planning_agent import PlanningAgent
from .agents.tool_use_agent import ToolUseAgent
from .agents.multi_agent import MultiAgent
from .agents.reflection_agent import ReflectionAgent

# Prompts
from .prompts import get_prompt, PROMPTS

# Tools
from .tools.sql_tools import SQLGenerationTool, SQLExecutionTool, SQLOptimizationTool, SQLDebugTool
from .tools.search_tools import WebSearchTool, VectorSearchTool
from .tools.file_tools import FileSystemTool
from .tools.schema_tools import SchemaInfoTool
from .tools.context_tools import ContextHistoryTool
from .tools.optimization_tools import QueryOptimizationTool, PerformanceTool

# Utilities and evaluation
from .evaluation.evaluation_tools import ReportEvaluator
from .utils.test_utils import (
    load_test_data,
    generate_task_name,
    create_query_row,
    store_in_duckdb,
    evaluate_agent_result,
    validate_agent_basic,
    print_summary,
    _calculate_tool_success_rate
)

__all__ = [
    # Core
    "BaseAgent",
    "ToolRegistry", 
    "register_tool",
    "TokenTracker",
    "TokenTrackingEntry",
    
    # Agent patterns
    "PlanningAgent",
    "ToolUseAgent", 
    "MultiAgent",
    "ReflectionAgent",
    
    # Custom interfaces
    "CustomAgent",
    "CustomAgentBase",
    "CustomTool",
    "CustomToolBase",
    
    # Prompts
    "get_prompt",
    "PROMPTS",
    
    # Tools
    "SQLGenerationTool",
    "SQLExecutionTool", 
    "SQLOptimizationTool",
    "SQLDebugTool",
    "SchemaInfoTool",
    "WebSearchTool",
    "VectorSearchTool",
    "FileSystemTool",
    "ContextHistoryTool",
    "QueryOptimizationTool",
    "PerformanceTool",
    
    # Utilities and evaluation
    "ReportEvaluator",
    "load_test_data",
    "generate_task_name",
    "create_query_row",
    "store_in_duckdb",
    "evaluate_agent_result",
    "validate_agent_basic",
    "print_summary",
    "_calculate_tool_success_rate",
]