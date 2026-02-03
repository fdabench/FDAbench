"""
Base agent implementation for FDABench.

This module provides the core BaseAgent class that all agent patterns inherit from.
It includes LLM integration, tool management, and token tracking.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from openai import OpenAI
from contextlib import contextmanager

from .token_tracker import TokenTracker
from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class Subtask:
    subtask_id: str
    tool: str
    input: Dict[str, Any]
    description: str
    expected_result: Optional[str] = None
    expected_SQL: Optional[str] = None


@dataclass 
class Query:
    instance_id: str
    db: str
    database_type: str
    tools_available: List[str]
    gold_subtasks: List[Subtask]
    query: str = ""
    advanced_query: str = ""
    original_query: str = ""
    ground_truth_report: str = ""
    multiple_choice_questions: List[Dict[str, Any]] = None
    level: str = ""
    question_type: str = ""
    options: Dict[str, str] = None
    correct_answer: List[str] = None
    explanation: str = ""
    
    def __post_init__(self):
        if self.multiple_choice_questions is None:
            self.multiple_choice_questions = []
        
        if self.options is None:
            self.options = {}
        
        if self.correct_answer is None:
            self.correct_answer = []
        
        if not self.advanced_query and self.query:
            self.advanced_query = self.query
        
        if not self.original_query and self.query:
            self.original_query = self.query


class BaseAgent:
    """
    Base agent class providing core functionality for all agent patterns.
    
    This class provides:
    - LLM integration with OpenRouter by default
    - Token usage tracking
    - Tool registry management
    - Common utilities for database operations
    """
    
    def __init__(self, 
                 model: str = "deepseek/deepseek-chat-v3-0324",
                 api_base: str = "https://openrouter.ai/api/v1",
                 api_key: Optional[str] = None):
        """
        Initialize the base agent.
        
        Args:
            model: LLM model to use (default: DeepSeek via OpenRouter)
            api_base: API base URL (default: OpenRouter)
            api_key: API key (will fallback to environment variables)
        """
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        
        self.token_tracker = TokenTracker()
        self.tool_registry = ToolRegistry()
        
        try:
            from ..utils.database_connection_manager import DatabaseConnectionManager
            self.db_manager = DatabaseConnectionManager()
            logger.info("Initialized DatabaseConnectionManager")
        except ImportError as e:
            logger.warning(f"Could not initialize DatabaseConnectionManager: {e}")
            self.db_manager = None
        
        self.tool_results = {}
        self.completed_subtasks = set()
        self.context_history = []
        self.max_history_length = 10
        
        self.phase_operation_counts = {
            'decision': 0, 'execute': 0, 'retry': 0, 'generate': 0
        }
        
        self._llm_cache = {}
        self._sql_cache = {}
        self._vector_cache = {}
        self._schema_cache = {}
        
        self._setup_api_client()
        
        self._register_basic_tools()
        
        logger.info(f"Initialized BaseAgent with model: {model}")
    
    def _setup_api_client(self):
        """Set up API client based on configuration."""
        if not self.api_key:
            if "openrouter.ai" in self.api_base:
                self.api_key = os.environ.get("OPENROUTER_API_KEY")
            elif "api.openai.com" in self.api_base:
                self.api_key = os.environ.get("OPENAI_API_KEY")
            else:
                self.api_key = os.environ.get("CUSTOM_API_KEY")
        
        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key
        )
    
    def _register_basic_tools(self):
        """Register basic tools that all agents can use."""
        try:
            from ..tools.sql_tools import SQLGenerationTool, SQLExecutionTool, SQLOptimizationTool, SQLDebugTool
            from ..tools.schema_tools import SchemaInspectionTool
            from ..tools.search_tools import WebSearchTool, VectorSearchTool
            from ..tools.file_tools import FileSystemSearchTool, FileReaderTool, FileWriterTool
            from ..tools.context_tools import ContextHistoryTool
            from ..tools.optimization_tools import QueryOptimizationTool, PerformanceTool
            
            self.register_tool("generated_sql", SQLGenerationTool(llm_client=self, db_manager=self.db_manager), category="sql", description="Generate SQL queries from natural language")
            self.register_tool("generate_sql", SQLGenerationTool(llm_client=self, db_manager=self.db_manager), category="sql", description="Generate SQL queries")
            self.register_tool("execute_sql", SQLExecutionTool(db_manager=self.db_manager), category="sql", description="Execute SQL queries against databases")
            self.register_tool("sql_optimize", SQLOptimizationTool(llm_client=self, db_manager=self.db_manager), category="sql", description="Optimize SQL queries for better performance")
            self.register_tool("sql_debug", SQLDebugTool(llm_client=self, db_manager=self.db_manager), category="sql", description="Debug and fix SQL query issues")
            
            self.register_tool("get_schema_info", SchemaInspectionTool(db_manager=self.db_manager), category="schema", description="Get database schema information")
            self.register_tool("schema_understanding", SchemaInspectionTool(db_manager=self.db_manager), category="schema", description="Understand database structure")
            
            self.register_tool("web_context_search", WebSearchTool(api_key=self.api_key), category="search", description="Search web for context information")
            self.register_tool("web_search", WebSearchTool(api_key=self.api_key), category="search", description="Search web for information")
            self.register_tool("vectorDB_search", VectorSearchTool(), category="search", description="Search vector database for similar content")
            self.register_tool("vector_search", VectorSearchTool(), category="search", description="Search vector database")
            
            self.register_tool("file_system_search", FileSystemSearchTool(), category="file", description="Search file system for relevant files")
            self.register_tool("context_history", ContextHistoryTool(), category="context", description="Manage conversation context and history")
            
            self.register_tool("query_optimize", QueryOptimizationTool(), category="optimization", description="Optimize database queries")
            self.register_tool("performance_analyze", PerformanceTool(), category="optimization", description="Analyze query performance")
            
            logger.info(f"Registered {len(self.list_tools())} basic tools")
            
        except ImportError as e:
            logger.warning(f"Could not import some tools: {e}")
        except Exception as e:
            logger.error(f"Error registering tools: {e}")
    
    
    def call_llm(self, messages: List[Dict[str, str]], model: Optional[str] = None, 
                 category: str = "general", phase: Optional[str] = None) -> str:
        """
        Call LLM with messages and track token usage.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use (defaults to self.model)
            category: Category for token tracking
            
        Returns:
            LLM response text
        """
        model = model or self.model
        
        input_text = " ".join([msg.get("content", "") for msg in messages])
        cache_key = f"{model}:{category}:{hash(input_text)}"
        
        if cache_key in self._llm_cache:
            cached_result = self._llm_cache[cache_key]
            self.token_tracker.track_call(
                category, 
                cached_result['input_tokens'], 
                cached_result['output_tokens'], 
                model,
                cached_result.get('cost'),
                phase
            )
            return cached_result['response']
        
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=20,
                extra_headers={
                    "HTTP-Referer": "https://github.com/wa../FDABench",
                    "X-Title": "Offical First Round FDABenchmark Agent Call"
                }
            )
            
            response = completion.choices[0].message.content
            
            usage = getattr(completion, 'usage', None)
            if usage:
                actual_input_tokens = getattr(usage, 'prompt_tokens', 0)
                actual_output_tokens = getattr(usage, 'completion_tokens', 0)
            else:
                logger.error(f"No usage data in API response for model {model}")
                raise ValueError(f"API response does not contain token usage data for model {model}")
            
            cost = getattr(completion, 'cost', None)
            
            self.token_tracker.track_call(category, actual_input_tokens, actual_output_tokens, model, cost, phase)
            
            self._llm_cache[cache_key] = {
                'response': response,
                'input_tokens': actual_input_tokens,
                'output_tokens': actual_output_tokens,
                'cost': cost
            }
            
            if len(self._llm_cache) > 1000:
                keys_to_remove = list(self._llm_cache.keys())[:100]
                for key in keys_to_remove:
                    del self._llm_cache[key]
            
            return response
            
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            self.token_tracker.track_call(category, 0, 0, model, 0.0, phase)
            raise e
    
    def register_tool(self, name: str, tool: Any, **metadata):
        """Register a tool with this agent."""
        self.tool_registry.register(name, tool, **metadata)
    
    def execute_tool(self, name: str, *args, **kwargs) -> Dict[str, Any]:
        """Execute a registered tool."""
        return self.tool_registry.execute_tool(name, *args, **kwargs)
    
    def list_tools(self) -> List[str]:
        """List all registered tools."""
        return self.tool_registry.list_tools()
    
    def process_query_from_json(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a query from JSON data. To be implemented by subclasses.
        
        Args:
            query_data: Query data dictionary
            
        Returns:
            Processing results
        """
        raise NotImplementedError("Subclasses must implement process_query_from_json()")
    
    def generate_report(self, query: Query) -> str:
        """
        Generate report based on tool results. To be implemented by subclasses.
        
        Args:
            query: Query object
            
        Returns:
            Generated report
        """
        raise NotImplementedError("Subclasses must implement generate_report()")
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this agent."""
        return {
            "agent_type": self.__class__.__name__,
            "model": self.model,
            "api_base": self.api_base,
            "registered_tools": self.list_tools(),
            "token_summary": self.token_tracker.get_token_summary(),
            "phase_summary": self.token_tracker.get_phase_summary()
        }
    
    @contextmanager
    def phase_timing(self, phase: str, operation_name: str = ""):
        """Context manager for tracking phase timing"""
        self.token_tracker.start_phase(phase)
        self.phase_operation_counts[phase] += 1
        start_time = time.time()
        
        logger.debug(f"Starting {phase} phase: {operation_name}")
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.token_tracker.track_phase_operation(phase, operation_name or "operation", duration)
            logger.debug(f"Completed {phase} phase: {operation_name} in {duration:.3f}s")
    
    def call_llm_with_phase(self, messages: List[Dict[str, str]], 
                           phase: str, category: str = "general", 
                           model: Optional[str] = None) -> str:
        """Call LLM with explicit phase tracking"""
        return self.call_llm(messages, model, category, phase)
    
    def get_phase_results(self) -> Dict[str, Any]:
        """Get comprehensive phase statistics"""
        phase_summary = self.token_tracker.get_phase_summary()
        phase_columns = self.token_tracker.get_phase_database_columns()
        
        total_latency = sum(stats['latency_seconds'] for stats in phase_summary.values())
        total_tokens = sum(stats['total_tokens'] for stats in phase_summary.values())
        total_cost = sum(stats['cost'] for stats in phase_summary.values())
        
        return {
            'phase_summary': phase_summary,
            'phase_columns': phase_columns,
            'total_phase_latency': round(total_latency, 3),
            'total_phase_tokens': total_tokens,
            'total_phase_cost': round(total_cost, 6),
            'phase_distribution': {
                'decision_ratio': phase_summary['decision']['latency_seconds'] / total_latency if total_latency > 0 else 0,
                'execute_ratio': phase_summary['execute']['latency_seconds'] / total_latency if total_latency > 0 else 0,
                'retry_ratio': phase_summary['retry']['latency_seconds'] / total_latency if total_latency > 0 else 0,
                'generate_ratio': phase_summary['generate']['latency_seconds'] / total_latency if total_latency > 0 else 0,
            }
        }
    
    def reset_state(self):
        """Reset agent state for new query processing."""
        self.tool_results.clear()
        self.completed_subtasks.clear()
        self.context_history.clear()
        self.phase_operation_counts = {'decision': 0, 'execute': 0, 'retry': 0, 'generate': 0}
    
    def cleanup(self):
        """Cleanup resources."""
        self._llm_cache.clear()
        self._sql_cache.clear()
        self._vector_cache.clear()
        self._schema_cache.clear()
