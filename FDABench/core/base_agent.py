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
    """Class for representing a subtask"""
    subtask_id: str
    tool: str
    input: Dict[str, Any]
    description: str
    expected_result: Optional[str] = None
    expected_SQL: Optional[str] = None


@dataclass 
class Query:
    """Flexible Query class that can handle additional fields from test data"""
    instance_id: str
    db: str
    database_type: str
    tools_available: List[str]
    gold_subtasks: List[Subtask]
    
    # Optional fields with defaults
    query: str = ""
    advanced_query: str = ""
    original_query: str = ""
    ground_truth_report: str = ""
    multiple_choice_questions: List[Dict[str, Any]] = None
    level: str = ""
    question_type: str = ""
    
    # Single choice question related fields
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
        
        # Use query as advanced_query if advanced_query is empty
        if not self.advanced_query and self.query:
            self.advanced_query = self.query
        
        # Use query as original_query if original_query is empty
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
        
        # Initialize core components
        self.token_tracker = TokenTracker()
        self.tool_registry = ToolRegistry()
        
        # Initialize database manager for SQL operations
        try:
            from ..utils.database_connection_manager import DatabaseConnectionManager
            self.db_manager = DatabaseConnectionManager()
            logger.info("Initialized DatabaseConnectionManager")
        except ImportError as e:
            logger.warning(f"Could not initialize DatabaseConnectionManager: {e}")
            self.db_manager = None
        
        # State tracking
        self.tool_results = {}
        self.completed_subtasks = set()
        self.context_history = []
        self.max_history_length = 10
        
        # Phase tracking
        self.phase_operation_counts = {
            'decision': 0, 'execute': 0, 'retry': 0, 'generate': 0
        }
        
        # Performance optimization caches
        self._llm_cache = {}
        self._sql_cache = {}
        self._vector_cache = {}
        self._domain_cache = {}
        self._schema_cache = {}
        
        # Setup API client
        self._setup_api_client()
        
        # Initialize basic tools
        self._register_basic_tools()
        
        logger.info(f"Initialized BaseAgent with model: {model}")
    
    def _setup_api_client(self):
        """Set up API client based on configuration."""
        # Use provided API key or fallback to environment variables
        if not self.api_key:
            if "openrouter.ai" in self.api_base:
                self.api_key = os.environ.get("OPENROUTER_API_KEY")
            elif "api.openai.com" in self.api_base:
                self.api_key = os.environ.get("OPENAI_API_KEY")
            else:
                self.api_key = os.environ.get("CUSTOM_API_KEY")
        
        # Initialize OpenAI-compatible client
        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key
        )
    
    def _register_basic_tools(self):
        """Register basic tools that all agents can use."""
        try:
            # Import and register SQL tools
            from ..tools.sql_tools import SQLGenerationTool, SQLExecutionTool, SQLOptimizationTool, SQLDebugTool
            from ..tools.schema_tools import SchemaInspectionTool
            from ..tools.search_tools import WebSearchTool, VectorSearchTool
            from ..tools.file_tools import FileSystemSearchTool, FileReaderTool, FileWriterTool
            from ..tools.context_tools import ContextHistoryTool
            from ..tools.optimization_tools import QueryOptimizationTool, PerformanceTool
            
            # Register SQL tools
            self.register_tool("generated_sql", SQLGenerationTool(llm_client=self, db_manager=self.db_manager), category="sql", description="Generate SQL queries from natural language")
            self.register_tool("execute_sql", SQLExecutionTool(db_manager=self.db_manager), category="sql", description="Execute SQL queries against databases")
            self.register_tool("sql_optimize", SQLOptimizationTool(llm_client=self, db_manager=self.db_manager), category="sql", description="Optimize SQL queries for better performance")
            self.register_tool("sql_debug", SQLDebugTool(llm_client=self, db_manager=self.db_manager), category="sql", description="Debug and fix SQL query issues")
            
            # Register schema tools
            self.register_tool("get_schema_info", SchemaInspectionTool(), category="schema", description="Get database schema information")
            self.register_tool("schema_understanding", SchemaInspectionTool(), category="schema", description="Understand database structure")
            
            # Register search tools
            self.register_tool("web_context_search", WebSearchTool(api_key=self.api_key), category="search", description="Search web for context information")
            self.register_tool("vectorDB_search", VectorSearchTool(), category="search", description="Search vector database for similar content")
            
            # Register file tools
            self.register_tool("file_system_search", FileSystemSearchTool(), category="file", description="Search file system for relevant files")
            self.register_tool("context_history", ContextHistoryTool(), category="context", description="Manage conversation context and history")
            
            # Register optimization tools
            self.register_tool("query_optimize", QueryOptimizationTool(), category="optimization", description="Optimize database queries")
            self.register_tool("performance_analyze", PerformanceTool(), category="optimization", description="Analyze query performance")
            
            logger.info(f"Registered {len(self.list_tools())} basic tools")
            
        except ImportError as e:
            logger.warning(f"Could not import some tools: {e}")
        except Exception as e:
            logger.error(f"Error registering tools: {e}")
    
    def estimate_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Estimate token count using proper tokenization.
        
        Args:
            text: Text to estimate tokens for
            model: Model name for tokenizer selection
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
            
        try:
            # Try to use tiktoken for OpenAI-compatible models
            import tiktoken
            
            # Select encoding based on model
            model_name = model or self.model
            if "gpt-4" in model_name.lower():
                encoding = tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in model_name.lower():
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                # Use cl100k_base for most modern models
                encoding = tiktoken.get_encoding("cl100k_base")
            
            return len(encoding.encode(text))
            
        except ImportError:
            # Fallback 1: Use transformers tokenizer if available
            try:
                from transformers import AutoTokenizer
                
                # Use a general tokenizer for estimation
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                return len(tokenizer.encode(text, add_special_tokens=False))
                
            except (ImportError, Exception):
                # Fallback 2: Improved character-based estimation
                # Based on empirical analysis of various tokenizers
                # Adjust ratio based on text characteristics
                
                # Basic character count
                char_count = len(text)
                
                # Apply heuristics for better estimation
                # - Longer words have better char:token ratio
                # - Code/technical text has worse ratio
                # - Natural language has better ratio
                
                words = text.split()
                if not words:
                    return max(1, char_count // 4)
                
                avg_word_length = char_count / len(words)
                
                if avg_word_length > 8:  # Likely technical/code content
                    ratio = 3.2  # Worse char:token ratio
                elif avg_word_length < 4:  # Short words
                    ratio = 3.8
                else:  # Normal text
                    ratio = 4.0
                
                return max(1, int(char_count / ratio))
        
        except Exception as e:
            logger.warning(f"Token estimation failed: {e}, using fallback")
            return max(1, len(text) // 4)
    
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
        
        # Create cache key
        input_text = " ".join([msg.get("content", "") for msg in messages])
        cache_key = f"{model}:{category}:{hash(input_text)}"
        
        # Check cache first
        if cache_key in self._llm_cache:
            cached_result = self._llm_cache[cache_key]
            # Track cached call for statistics
            self.token_tracker.track_call(
                category, 
                cached_result['input_tokens'], 
                cached_result['output_tokens'], 
                model,
                cached_result.get('cost'),
                phase
            )
            return cached_result['response']
        
        # Estimate input tokens
        input_tokens = self.estimate_tokens(input_text)
        
        try:
            # Make API call
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
            
            # Get actual usage from response if available
            usage = getattr(completion, 'usage', None)
            if usage:
                actual_input_tokens = getattr(usage, 'prompt_tokens', input_tokens)
                actual_output_tokens = getattr(usage, 'completion_tokens', self.estimate_tokens(response, model))
            else:
                # Estimate both input and output tokens properly when API doesn't provide usage
                actual_input_tokens = self.estimate_tokens(input_text, model)
                actual_output_tokens = self.estimate_tokens(response, model)
            
            # Get cost from response if available
            cost = getattr(completion, 'cost', None)
            
            # Track token usage
            self.token_tracker.track_call(category, actual_input_tokens, actual_output_tokens, model, cost, phase)
            
            # Cache the result
            self._llm_cache[cache_key] = {
                'response': response,
                'input_tokens': actual_input_tokens,
                'output_tokens': actual_output_tokens,
                'cost': cost
            }
            
            # Limit cache size
            if len(self._llm_cache) > 1000:
                keys_to_remove = list(self._llm_cache.keys())[:100]
                for key in keys_to_remove:
                    del self._llm_cache[key]
            
            return response
            
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            # Track failed call
            self.token_tracker.track_call(category, input_tokens, 0, model, 0.0, phase)
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
        
        # Calculate totals
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
        # Reset phase tracking for new query
        self.phase_operation_counts = {'decision': 0, 'execute': 0, 'retry': 0, 'generate': 0}
        # Note: We don't reset token tracker as we want to track across queries
    
    def cleanup(self):
        """Cleanup resources."""
        self._llm_cache.clear()
        self._sql_cache.clear()
        self._vector_cache.clear()
        self._domain_cache.clear()
        self._schema_cache.clear()

    def _get_domain_classification(self, query: str) -> str:
        """
        Get domain classification for the query.
        
        Args:
            query: Search query string
            
        Returns:
            Comma-separated domain string
        """
        # Define domain categories
        DOMAINS = [
            "Healthcare_Medical Systems", "Sports_Competition", "Transportation_Aviation", "Education Institution",
            "E commerce", "Government_Public Administration", "Weather_Environmental", "Software Development",
            "Genomics_Bioinformatics", "Estate_Property", "Music", "Telecommunications", "Food_Beverage Industry",
            "Manufacturing_Production", "Blockchain", "Social Media", "Geographic", "Database_Data",
            "Urban Planning", "Agriculture_Farming", "Marketing_Advertising", "Human Resources", "Energy_Utilities",
            "Culture", "Business Management", "Automotive Industry", "Pet Services", "Marine_Shipping",
            "Events_Planning", "Natural Language Processing", "Customer Relationship", "Scientific Research",
            "Political Science_Voting", "Document Management", "Finance_Investment", "Travel_Tourism",
            "Security Enforcement", "Supply Chain", "Gaming_Entertainment", "International Development",
            "Movie_Animation", "Philanthropy", "Books", "Trading", "Law_Legal", "Computer Vision",
            "Arts", "Psychology", "Material Science", "Applied Sciences"
        ]
        
        prompt = f"""Given the following query, classify it into one or more domains from this list:
{chr(10).join([f"{i + 1}. {domain}" for i, domain in enumerate(DOMAINS)])}

Query: {query}

Return exactly three most relevant domain names separated by semicolons. Example: "Trading; Finance_Investment; E-commerce".
Do not include any other text, only return the three domain names separated by semicolons."""

        try:
            response = self.call_llm([{"role": "user", "content": prompt}], category="domain_classification")
            domains = response.strip()
            
            # Clean up response if it doesn't match expected format
            if ";" not in domains:
                # Try to extract domain names from response
                found_domains = []
                for domain in DOMAINS:
                    if domain in domains:
                        found_domains.append(domain)
                        if len(found_domains) == 3:
                            break

                if len(found_domains) == 3:
                    domains = "; ".join(found_domains)
                else:
                    # Use default values if unable to extract three domains
                    domains = "E-commerce; Business Management; Customer Relationship"

            return domains
        except Exception as e:
            logger.error(f"Domain classification failed: {e}")
            # Return default domains on error
            return "E-commerce; Business Management; Database_Data"

    def _search_single_domain(self, domain: str, query: str) -> Optional[str]:
        """
        Search within a single domain.
        
        Args:
            domain: Domain to search in
            query: Search query
            
        Returns:
            Search results as string or None
        """
        try:
            # Simple mock search based on domain and query
            query_lower = query.lower()
            
            if domain == "E-commerce":
                if any(word in query_lower for word in ['product', 'order', 'customer', 'sales', 'purchase', 'buy', 'shop']):
                    return f"E-commerce domain: Found relevant information about {query}. This domain contains product catalogs, customer orders, and sales data."
            
            elif domain == "Business Management":
                if any(word in query_lower for word in ['business', 'management', 'company', 'organization', 'enterprise']):
                    return f"Business Management domain: Found relevant information about {query}. This domain contains organizational data, management processes, and business operations."
            
            elif domain == "Database_Data":
                if any(word in query_lower for word in ['database', 'data', 'table', 'query', 'sql', 'schema']):
                    return f"Database_Data domain: Found relevant information about {query}. This domain contains database schemas, data models, and query patterns."
            
            elif domain == "Analytics":
                if any(word in query_lower for word in ['analytics', 'report', 'metrics', 'kpi', 'dashboard']):
                    return f"Analytics domain: Found relevant information about {query}. This domain contains reporting data, metrics, and analytical insights."
            
            elif domain == "Financial":
                if any(word in query_lower for word in ['financial', 'finance', 'money', 'budget', 'cost', 'revenue']):
                    return f"Financial domain: Found relevant information about {query}. This domain contains financial data, budgets, and monetary information."
            
            elif domain == "Healthcare_Medical Systems":
                if any(word in query_lower for word in ['health', 'medical', 'patient', 'hospital', 'doctor', 'treatment']):
                    return f"Healthcare_Medical Systems domain: Found relevant information about {query}. This domain contains medical data, patient records, and healthcare information."
            
            elif domain == "Software Development":
                if any(word in query_lower for word in ['software', 'development', 'programming', 'code', 'application', 'system']):
                    return f"Software Development domain: Found relevant information about {query}. This domain contains software projects, development processes, and technical documentation."
            
            elif domain == "Marketing_Advertising":
                if any(word in query_lower for word in ['marketing', 'advertising', 'campaign', 'promotion', 'brand', 'customer']):
                    return f"Marketing_Advertising domain: Found relevant information about {query}. This domain contains marketing campaigns, advertising data, and customer engagement metrics."
            
            # Default response for any domain
            return f"{domain} domain: Found general information related to {query}. This domain may contain relevant context for your query."
            
        except Exception as e:
            logger.error(f"Error searching domain {domain}: {e}")
            return None