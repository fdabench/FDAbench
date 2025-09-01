#!/usr/bin/env python3
"""
LotusAgent - Custom Agent using Lotus for SQL generation and execution

This agent inherits from CustomAgentBase and implements the three task types:
- single_choice: Answer single choice questions
- multiple_choice: Answer multiple choice questions  
- report: Generate analytical reports

The agent uses Lotus for SQL generation and DatabaseConnectionManager for execution.
"""

import os
import sys
import json
import time
import traceback
from typing import Dict, List, Any, Optional

# Add necessary paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from .custom_agent import CustomAgentBase

# Lotus imports
try:
    import pandas as pd
    import lotus
    from lotus.models import LM
    from lotus.sem_ops import sem_map, sem_agg
    from lotus.data_connectors import DataConnector
except ImportError as e:
    print(f"Warning: Lotus imports failed: {e}")
    print("Please ensure Lotus is installed correctly")

# Database connection import
try:
    from ..utils.database_connection_manager import DatabaseConnectionManager
except ImportError as e:
    print(f"Warning: DatabaseConnectionManager import failed: {e}")
    # Fallback to lotus examples path
    try:
        sys.path.insert(0, '/home/ziting.wang/jhx/lotus/examples/db_examples/')
        from database_connection_manager import DatabaseConnectionManager
    except ImportError as e2:
        print(f"Warning: Fallback DatabaseConnectionManager import also failed: {e2}")


class LotusAgent(CustomAgentBase):
    """
    Custom agent that uses Lotus for SQL generation and execution.
    Supports single choice, multiple choice, and report generation tasks.
    """
    
    def __init__(self, 
                 model: str = "gpt-4o-mini",
                 api_base: str = "https://openrouter.ai/api/v1",
                 api_key: Optional[str] = None,
                 **kwargs):
        """
        Initialize LotusAgent with Lotus LM and database manager.
        
        Args:
            model: LLM model to use (default: gpt-4o-mini)
            api_base: API base URL
            api_key: API key for the service
            **kwargs: Additional configuration parameters
        """
        super().__init__(model=model, api_base=api_base, api_key=api_key, **kwargs)
        
        # Initialize Lotus LM
        try:
            self.lm = LM(model=model)
            lotus.settings.configure(lm=self.lm)
        except Exception as e:
            print(f"Warning: Failed to initialize Lotus LM: {e}")
            self.lm = None
        
        # Initialize database connection manager
        try:
            self.db_manager = DatabaseConnectionManager()
        except Exception as e:
            print(f"Warning: Failed to initialize DatabaseConnectionManager: {e}")
            self.db_manager = None
    
    def process_query_logic(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main query processing logic for Lotus agent.
        
        Args:
            query_data: Query data containing instance_id, db, query, task type, etc.
            
        Returns:
            Dictionary with processing results and metrics
        """
        start_time = time.time()
        
        try:
            # Extract key information
            instance_id = query_data.get("instance_id", "unknown")
            db_name = query_data.get("db", "")
            database_type = query_data.get("database_type", "bird")
            query = query_data.get("query", "")
            task_type = query_data.get("question_type", "")
            options = query_data.get("options", {})
            
            # Validate required components
            if not self.lm or not self.db_manager:
                return self._create_error_result(instance_id, "Lotus or DatabaseConnectionManager not initialized")
            
            if not query:
                return self._create_error_result(instance_id, "No query provided")
            
            # Get database configuration
            try:
                db_config = self.db_manager.get_database_config(
                    instance_id=instance_id,
                    db_name=db_name, 
                    database_type=database_type
                )
                schema_info = self.db_manager.get_schema_info(db_config)
            except Exception as e:
                return self._create_error_result(instance_id, f"Database configuration error: {str(e)}")
            
            # Generate SQL using Lotus patterns
            try:
                sql_query = self._generate_sql_with_lotus(query, schema_info)
            except Exception as e:
                return self._create_error_result(instance_id, f"SQL generation error: {str(e)}")
            
            # Execute SQL
            try:
                sql_result = self.db_manager.execute_sql(db_config, sql_query)
                if sql_result.get("status") != "success":
                    return self._create_error_result(instance_id, f"SQL execution error: {sql_result.get('error', 'Unknown error')}")
                
                result_data = sql_result["results"]["query_results"]
            except Exception as e:
                return self._create_error_result(instance_id, f"SQL execution error: {str(e)}")
            
            # Process based on task type
            if task_type == "single_choice":
                answer = self._process_single_choice(result_data, query, options)
            elif task_type == "multiple_choice":
                answer = self._process_multiple_choice(result_data, query, options)
            elif task_type == "report":
                answer = self._process_report(result_data, query)
            else:
                return self._create_error_result(instance_id, f"Unsupported task type: {task_type}")
            
            # Calculate metrics
            end_time = time.time()
            processing_time = round(end_time - start_time, 2)
            token_summary = self.token_tracker.get_token_summary()
            
            return {
                "instance_id": instance_id,
                "agent_type": "lotus",
                "model": self.model,
                "task_type": task_type,
                "generated_sql": sql_query,
                "sql_execution_result": result_data,
                "answer": answer,
                "metrics": {
                    "latency_seconds": processing_time,
                    "token_usage": token_summary,
                    "sql_generated": bool(sql_query),
                    "sql_executed_successfully": True
                },
                "processing_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
            }
            
        except Exception as e:
            return self._create_error_result(
                query_data.get("instance_id", "unknown"), 
                f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
            )
    
    def _generate_sql_with_lotus(self, query: str, schema_info: Dict[str, Any]) -> str:
        """
        Generate SQL using Lotus LLM patterns.
        
        Args:
            query: Natural language query
            schema_info: Database schema information
            
        Returns:
            Generated SQL query string
        """
        schema_text = self._format_schema_for_prompt(schema_info)
        
        prompt = f"""You are an SQL expert. Based on the database schema and user question, 
OUTPUT **ONLY** the SQL query - no comments or explanations.

IMPORTANT SQL RULES:
1. Use backticks (`) around column/table names that contain spaces or are SQL reserved words
2. Ensure column names match EXACTLY those in the schema (case-sensitive)
3. Use proper SQLite syntax

Database schema:
{schema_text}

User question:
{query}

SQL:"""
        
        # Use Lotus sem_map for SQL generation
        prompt_df = pd.DataFrame({"prompt": [prompt]})
        result_df = prompt_df.sem_map("{prompt}")
        
        # Extract SQL from result
        sql = str(result_df.iloc[0, -1]).strip()
        
        # Clean up SQL
        sql = sql.replace("```sql", "").replace("```", "").strip().strip('"\'')
        if not sql.endswith(";"):
            sql += ";"
            
        return sql
    
    def _format_schema_for_prompt(self, schema_info: Dict[str, Any]) -> str:
        """Format schema information for LLM prompt with SQL reserved keyword handling."""
        if "error" in schema_info:
            return f"Schema error: {schema_info['error']}"
        
        schema_parts = []
        tables = schema_info.get("tables", {}) if isinstance(schema_info.get("tables"), dict) else schema_info
        
        # SQL reserved keywords that need escaping
        sql_reserved_keywords = {
            'order', 'date', 'time', 'year', 'month', 'day', 'hour', 'minute', 'second',
            'user', 'table', 'column', 'index', 'key', 'value', 'group', 'having',
            'where', 'select', 'from', 'join', 'left', 'right', 'inner', 'outer',
            'union', 'intersect', 'except', 'case', 'when', 'then', 'else', 'end',
            'row', 'rank', 'dense_rank', 'partition', 'window', 'over'
        }
        
        for table_name, table_info in tables.items():
            if not isinstance(table_info, dict):
                continue
                
            columns = table_info.get("columns", {})
            if columns:
                column_specs = []
                for col, dtype in columns.items():
                    # Escape reserved keywords and spaces
                    if col.lower() in sql_reserved_keywords or ' ' in col:
                        escaped_col = f"`{col}`"
                    else:
                        escaped_col = col
                    column_specs.append(f"{escaped_col} ({dtype})")
                
                # Also escape table name if needed
                if table_name.lower() in sql_reserved_keywords or ' ' in table_name:
                    escaped_table = f"`{table_name}`"
                else:
                    escaped_table = table_name
                    
                schema_parts.append(f"Table {escaped_table}: {', '.join(column_specs)}")
            else:
                schema_parts.append(f"Table {table_name}: (no columns)")
        
        return "\n".join(schema_parts) if schema_parts else "No schema information available"
    
    def _process_single_choice(self, result_data: List[Dict], query: str, options: Dict[str, str]) -> Dict[str, Any]:
        """Process single choice question using Lotus."""
        if not result_data:
            return {"answer": "A", "reasoning": "No data available"}
        
        # Convert result to DataFrame for Lotus processing
        result_df = pd.DataFrame(result_data)
        
        # Use Lotus to judge each option
        judged_options = self._judge_options_with_lotus(result_df, query, options)
        
        # Find the option marked as correct
        correct_options = [opt for opt, keep in judged_options.items() if keep]
        
        if correct_options:
            selected_option = correct_options[0]
            return {
                "answer": selected_option,
                "reasoning": options[selected_option]
            }
        else:
            # Fallback to first option if no clear winner
            first_option = list(options.keys())[0]
            return {
                "answer": first_option,
                "reasoning": f"Selected {first_option} as fallback: {options[first_option]}"
            }
    
    def _process_multiple_choice(self, result_data: List[Dict], query: str, options: Dict[str, str]) -> Dict[str, Any]:
        """Process multiple choice question using Lotus."""
        if not result_data:
            return {"answers": ["A"], "explanation": ["No data available"]}
        
        # Convert result to DataFrame for Lotus processing
        result_df = pd.DataFrame(result_data)
        
        # Judge all options
        judged_options = self._judge_options_with_lotus(result_df, query, options)
        
        # Collect all correct options
        correct_options = [opt for opt, keep in judged_options.items() if keep]
        
        if correct_options:
            return {
                "answers": correct_options,
                "explanation": [options[opt] for opt in correct_options]
            }
        else:
            # Fallback to first option
            first_option = list(options.keys())[0]
            return {
                "answers": [first_option],
                "explanation": [f"Selected {first_option} as fallback: {options[first_option]}"]
            }
    
    def _process_report(self, result_data: List[Dict], query: str) -> str:
        """Generate report using Lotus."""
        if not result_data:
            return "## No Data Available\n\nNo data was returned from the database query."
        
        # Convert to DataFrame and limit for processing
        result_df = pd.DataFrame(result_data).head(8)  # Limit to 8 rows for efficiency
        
        # Generate report using Lotus sem_agg
        if len(result_df) > 0:
            columns = result_df.columns.tolist()
            placeholder = ", ".join(f"{{{col}}}" for col in columns)
            
            prompt = f"""Example row: {placeholder}

Based on the data, provide a comprehensive analysis for: {query}

Structure your response with:
1. Executive Summary
2. Data Analysis Results  
3. Key Insights
4. Conclusions

Format in Markdown."""

            try:
                report = result_df.sem_agg(prompt).iloc[0, -1]
                return str(report)
            except Exception as e:
                # Fallback report generation
                return self._generate_fallback_report(result_df, query)
        else:
            return self._generate_fallback_report(result_df, query)
    
    def _judge_options_with_lotus(self, result_df: pd.DataFrame, query: str, options: Dict[str, str]) -> Dict[str, bool]:
        """Judge options using Lotus semantic operations."""
        judged_options = {}
        
        # Limit result data for processing
        limited_df = result_df.head(8)
        
        for option_key, option_text in options.items():
            try:
                # Create prompt for option evaluation
                if len(limited_df) > 0:
                    # Convert to markdown table for context
                    data_context = limited_df.to_markdown(index=False)
                else:
                    data_context = "No data available"
                
                prompt = f"""Database query results:
{data_context}

Question: {query}
Option: {option_text}

Based on the data, is this option correct? Answer only YES or NO."""
                
                # Use Lotus to evaluate
                eval_df = pd.DataFrame({"eval_prompt": [prompt]})
                result = eval_df.sem_map("{eval_prompt}").iloc[0, -1]
                
                # Parse result
                result_str = str(result).strip().upper()
                judged_options[option_key] = "YES" in result_str
                
            except Exception as e:
                print(f"Warning: Failed to judge option {option_key}: {e}")
                # Default to False for failed evaluations
                judged_options[option_key] = False
        
        return judged_options
    
    def _generate_fallback_report(self, result_df: pd.DataFrame, query: str) -> str:
        """Generate a basic report when Lotus processing fails."""
        report_parts = ["# Analysis Report", ""]
        
        if len(result_df) > 0:
            report_parts.extend([
                "## Data Summary",
                f"- Total rows: {len(result_df)}",
                f"- Columns: {', '.join(result_df.columns.tolist())}",
                "",
                "## Sample Data",
                result_df.head().to_markdown(index=False),
                ""
            ])
        else:
            report_parts.extend([
                "## Data Summary", 
                "No data available from the query execution.",
                ""
            ])
        
        report_parts.extend([
            "## Query Context",
            f"Original question: {query}",
            "",
            "## Conclusion",
            "This is a basic report generated due to processing limitations. Please review the data manually for detailed insights.",
            ""
        ])
        
        return "\n".join(report_parts)
    
    def _create_error_result(self, instance_id: str, error_msg: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            "instance_id": instance_id,
            "agent_type": "lotus",
            "model": self.model,
            "error": error_msg,
            "processing_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def get_custom_tools(self) -> List[str]:
        """Return list of tools this agent supports."""
        return ["lotus_sql_generation", "database_execution", "semantic_analysis"]
    
    def validate_custom_config(self) -> bool:
        """Validate agent configuration."""
        if not self.lm:
            print("Warning: Lotus LM not initialized")
            return False
        if not self.db_manager:
            print("Warning: DatabaseConnectionManager not initialized")
            return False
        return True