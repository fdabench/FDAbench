"""
Enhanced SQL-related tools with DocETL operator selection.
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional
import os
import pandas as pd
from docetl import DSLRunner
import requests
import tempfile
import yaml

logger = logging.getLogger(__name__)

def extract_table_name_from_sql(sql_query: str) -> Optional[str]:
    """
    Extract the main table name from SQL query.
    This is a simple implementation focusing on common patterns.
    """
    try:
        # Clean the SQL query
        sql_clean = re.sub(r'\s+', ' ', sql_query.strip().upper())
        
        # Pattern to match FROM clause
        from_pattern = r'FROM\s+([^\s,\(\)]+)'
        match = re.search(from_pattern, sql_clean)
        
        if match:
            table_name = match.group(1).strip()
            # Remove schema prefixes if any (e.g., schema.table -> table)
            if '.' in table_name:
                table_name = table_name.split('.')[-1]
            return table_name.lower()
        
        return None
    except Exception as e:
        logger.error(f"Error extracting table name: {e}")
        return None

def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation: 1 token ~= 4 characters)"""
    return max(1, len(text) // 4)

def choose_docetl_operator_with_llm(natural_language_query: str, available_operators: List[str], api_key: str) -> tuple[str, dict]:
    """
    Use LLM to choose the most appropriate DocETL operator for the given query.
    Returns: (operator_name, token_stats)
    """
    token_stats = {
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0
    }
    
    try:
        # Available DocETL operators
        operators_to_consider = [op for op in available_operators if op in ["map", "filter"]]
        
        prompt = f"""
Given the natural language query and available DocETL operators, choose the most appropriate operator.

Natural Language Query: "{natural_language_query}"

Available DocETL Operators:
- map: Transform data by adding new columns or extracting information from each row (use only for explicit data transformation requests)
- filter: Filter rows based on conditions to keep only relevant data (use for analysis, queries, and data selection)

Guidelines:
1. Use "filter" for all analytical queries, questions, and data exploration tasks
2. Use "filter" for questions asking "what", "who", "how many", "which", "when", "where"
3. Use "filter" for summarization, insights, comparisons, finding patterns, or selecting data
4. Use "filter" for any query that involves examining, analyzing, or working with specific data
5. Use "map" ONLY when explicitly asked to "add columns", "transform data", "create new fields", or "modify existing data structure"

Return ONLY the operator name: either "map" or "filter". No explanation needed.
"""
        
        # Estimate input tokens
        token_stats['input_tokens'] = estimate_tokens(prompt)
        
        # Make OpenAI API call
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek/deepseek-chat-v3-0324",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,
            "temperature": 0
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            chosen_operator = result["choices"][0]["message"]["content"].strip().lower()
            
            # Get actual usage from API response
            if 'usage' in result:
                usage = result['usage']
                token_stats['input_tokens'] = usage.get('prompt_tokens', token_stats['input_tokens'])
                token_stats['output_tokens'] = usage.get('completion_tokens', token_stats['output_tokens'])
                token_stats['total_tokens'] = usage.get('total_tokens', 
                    token_stats['input_tokens'] + token_stats['output_tokens'])
            else:
                token_stats['output_tokens'] = estimate_tokens(chosen_operator)
                token_stats['total_tokens'] = token_stats['input_tokens'] + token_stats['output_tokens']
                
        else:
            logger.error(f"OpenAI API call failed with status {response.status_code}: {response.text}")
            chosen_operator = None
        
        # Validate and return the chosen operator
        if chosen_operator and chosen_operator in operators_to_consider:
            return chosen_operator, token_stats
        else:
            # Default fallback logic
            logger.warning(f"Invalid operator choice: {chosen_operator}, using fallback logic")
            query_lower = natural_language_query.lower()
            if any(word in query_lower for word in ["filter", "select only", "remove", "exclude", "show only"]):
                return "filter", token_stats
            else:
                return "map", token_stats  # Default for analysis
                
    except Exception as e:
        logger.error(f"Error choosing DocETL operator: {e}")
        return "map", token_stats  # Default fallback

def apply_docetl_operator(df: pd.DataFrame, operator: str, natural_language_query: str, 
                         columns: List[str], api_key: str, token_tracker=None) -> pd.DataFrame:
    """
    Apply the chosen DocETL operator to the DataFrame using DocETL pipeline.
    """
    try:
        # Get the number of data rows for tracking LLM calls
        data_rows_count = len(df)
        
        # Create temporary files for DocETL pipeline
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as data_file:
            # Convert DataFrame to JSON for DocETL
            data_json = df.to_dict('records')
            json.dump(data_json, data_file, indent=2)
            data_file_path = data_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as output_file:
            output_file_path = output_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as pipeline_file:
            pipeline_file_path = pipeline_file.name
        
        # Create DocETL pipeline configuration
        if operator == "map":
            # Map operation - analyze/transform data
            config = {
                "datasets": {
                    "input_data": {
                        "path": data_file_path,
                        "type": "file"
                    }
                },
                "default_model": "openrouter/deepseek/deepseek-chat-v3-0324",
                "operations": [
                    {
                        "name": "analyze_data",
                        "type": "map",
                        "prompt": f"""
Analyze this data record and answer the following query: "{natural_language_query}"

Record data: {{{{ input }}}}

Based on the query, provide a relevant analysis, summary, or extracted information.
If the query asks for specific information, extract and return that.
If the query asks for analysis or insights, provide those.
Return your response as a clear, concise answer.
                        """.strip(),
                        "output": {
                            "schema": {
                                "analysis_result": "str"
                            }
                        }
                    }
                ],
                "pipeline": {
                    "steps": [
                        {
                            "name": "process_data",
                            "input": "input_data", 
                            "operations": ["analyze_data"]
                        }
                    ],
                    "output": {
                        "type": "file",
                        "path": output_file_path
                    }
                }
            }
        
        elif operator == "filter":
            # Filter operation - select relevant rows
            config = {
                "datasets": {
                    "input_data": {
                        "path": data_file_path,
                        "type": "file"
                    }
                },
                "default_model": "openrouter/deepseek/deepseek-chat-v3-0324",
                "operations": [
                    {
                        "name": "filter_data",
                        "type": "filter",
                        "prompt": f"""
Evaluate this data record against the following criteria: "{natural_language_query}"

Record data: {{{{ input }}}}

Return true if this record matches the criteria or should be included based on the query.
Return false if this record should be filtered out or doesn't match the criteria.
                        """.strip(),
                        "output": {
                            "schema": {
                                "matches_criteria": "boolean"
                            }
                        }
                    }
                ],
                "pipeline": {
                    "steps": [
                        {
                            "name": "filter_data_step",
                            "input": "input_data",
                            "operations": ["filter_data"]
                        }
                    ],
                    "output": {
                        "type": "file", 
                        "path": output_file_path
                    }
                }
            }
        
        # Save pipeline configuration
        with open(pipeline_file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Set API key environment variable
        os.environ['OPENROUTER_API_KEY'] = api_key
        
        # Run DocETL pipeline
        logger.info(f"Running DocETL {operator} operation...")
        runner = DSLRunner.from_yaml(pipeline_file_path)
        runner.load_run_save()
        
        # Load results
        with open(output_file_path, 'r') as f:
            results = json.load(f)
        
        # Convert results back to DataFrame
        if results:
            result_df = pd.DataFrame(results)
            logger.info(f"DocETL {operator} operation completed successfully. Results shape: {result_df.shape}")
            
            # Track LLM calls - one for each data row processed by DocETL
            if token_tracker:
                for i in range(data_rows_count):
                    token_tracker.track_call(
                        category=f"docetl_{operator}_row_{i+1}",
                        input_tokens=100,  # estimated tokens per row for DocETL processing
                        output_tokens=50,  # estimated tokens per row for DocETL processing
                        model="deepseek/deepseek-chat-v3-0324",
                        cost=None
                    )
                logger.info(f"Tracked {data_rows_count} docetl {operator} operations (one per data row)")
            
            return result_df
        else:
            logger.warning(f"DocETL {operator} operation returned no results")
            
            # Still track the processing attempts even if no results
            if token_tracker:
                for i in range(data_rows_count):
                    token_tracker.track_call(
                        category=f"docetl_{operator}_empty_row_{i+1}",
                        input_tokens=100,  # estimated tokens per row
                        output_tokens=10,  # minimal output for empty results
                        model="deepseek/deepseek-chat-v3-0324",
                        cost=None
                    )
                logger.info(f"Tracked {data_rows_count} docetl {operator} operations (empty results, one per data row)")
            
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error applying DocETL {operator} operator: {e}")
        # Return original DataFrame if DocETL operation fails
        return df
    
    finally:
        # Clean up temporary files
        for temp_path in [data_file_path, output_file_path, pipeline_file_path]:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Could not clean up temp file {temp_path}: {e}")



class SQLExecutionTool:
    """Enhanced tool for executing SQL queries with DocETL operator selection"""
    
    def __init__(self, db_manager=None, token_tracker=None):
        self.db_manager = db_manager
        self.token_tracker = token_tracker
        self.available_operators = ["map", "filter"]
    
    def execute(self, sql_query: str = None, database_name: str = None,
                database_type: str = None, instance_id: str = None,
                natural_language_query: str = None, previous_results: Dict = None,
                **kwargs) -> Dict[str, Any]:
        """
        Execute SQL query on database with enhanced DocETL operator selection.
        
        Args:
            sql_query: SQL query to execute
            database_name: Target database
            database_type: Type of database (bird, spider2-lite, etc.)
            instance_id: Instance identifier for database connection
            natural_language_query: Original query (for DocETL processing)
            previous_results: Results from previous tools
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with status and results
        """
        try:
            # If no SQL provided, try to get from previous results
            if not sql_query and previous_results:
                for tool_name in ["sql_generate", "generated_sql"]:
                    if tool_name in previous_results:
                        result = previous_results[tool_name]
                        if isinstance(result, dict) and "sql_query" in result:
                            sql_query = result["sql_query"]
                            break
            
            if not sql_query:
                return {"status": "error", "error": "No SQL query provided"}
            
               # ENHANCEMENT 1: Extract table name and modify SQL to get whole table
            table_name = extract_table_name_from_sql(sql_query)
            if table_name:
                # Replace the complex SQL with simple SELECT * FROM table
                modified_sql = f"SELECT * FROM {table_name}"
                logger.info(f"Modified SQL from complex query to: {modified_sql}")
                sql_to_execute = modified_sql
            else:
                sql_to_execute = sql_query
            


            # Use database manager for SQL execution
            if self.db_manager and database_name and database_type:
                try:
                    # Import the DatabaseConfig class
                    try:
                        from ..utils.database_connection_manager import DatabaseConfig
                    except ImportError:
                        # Create a local DatabaseConfig if import fails
                        from dataclasses import dataclass
                        @dataclass
                        class DatabaseConfig:
                            database_type: str
                            instance_id: str
                            db_name: str
                            connection_params: Dict[str, Any] = None
                    
                    # Get database configuration
                    if hasattr(self.db_manager, 'get_database_config') and instance_id:
                        config = self.db_manager.get_database_config(instance_id, database_name, database_type)
                    else:
                        config = DatabaseConfig(
                            database_type=database_type,
                            instance_id=instance_id or "default",
                            db_name=database_name
                        )
                    
                    # Execute SQL using database manager
                    execution_result = self.db_manager.execute_sql(config, sql_to_execute)
                    
                    if execution_result["status"] == "success":
                        # Apply intelligent DocETL operator selection
                        if natural_language_query:
                            try:
                                # Check API key availability
                                api_key = os.environ.get('OPENROUTER_API_KEY')
                                if not api_key:
                                    logger.error("OPENROUTER_API_KEY not found, DocETL processing will fail")
                                    raise ValueError("Missing OPENROUTER_API_KEY")

                                # Convert execution_result to pandas DataFrame
                                query_results = execution_result["results"]["query_results"]
                                columns = execution_result["results"].get("columns", [])
                                
                                if query_results and columns:
                                    df = pd.DataFrame(query_results, columns=columns)
                                    logger.info(f"Created DataFrame with shape: {df.shape}")
                                    
                                    # Choose the right DocETL operator using LLM
                                    chosen_operator, tokens_choose_operator = choose_docetl_operator_with_llm(
                                        natural_language_query, 
                                        self.available_operators, 
                                        api_key
                                    )
                                    
                                    # Track the operator choice LLM call
                                    if self.token_tracker:
                                        self.token_tracker.track_call(
                                            category=f"docetl_operator_choice",
                                            input_tokens=tokens_choose_operator['input_tokens'],
                                            output_tokens=tokens_choose_operator['output_tokens'],
                                            model="deepseek/deepseek-chat-v3-0324",
                                            cost=None
                                        )
                                    
                                    logger.info(f"LLM chose DocETL operator: {chosen_operator}")
                                    logger.info(f"Operator choice tokens: {tokens_choose_operator}")
                                    
                                    # Apply the chosen DocETL operator
                                    processed_df = apply_docetl_operator(
                                        df, chosen_operator, natural_language_query, columns, api_key, self.token_tracker
                                    )
                                    
                                    # Track DocETL operation (estimate tokens based on data size)
                                    if self.token_tracker:
                                        input_text = natural_language_query + str(df.to_string())
                                        estimated_input_tokens = estimate_tokens(input_text)
                                        estimated_output_tokens = estimate_tokens(str(processed_df.to_string()))
                                        
                                        self.token_tracker.track_call(
                                            category=f"docetl_{chosen_operator}",
                                            input_tokens=estimated_input_tokens,
                                            output_tokens=estimated_output_tokens,
                                            model="deepseek/deepseek-chat-v3-0324",
                                            cost=None
                                        )
                                    
                                    # Convert back to the expected format
                                    if isinstance(processed_df, pd.DataFrame):
                                        processed_results = processed_df.to_dict('records')
                                    else:
                                        processed_results = processed_df
                                    
                                    logger.info(f"Processed results count: {len(processed_results) if isinstance(processed_results, list) else 'N/A'}")
                                    table_name = extract_table_name_from_sql(sql_query)

                                    return {
                                        "status": "success",
                                        "results": {
                                            "original_sql_query": sql_query,
                                            "executed_sql_query": sql_to_execute,
                                            "query_results": processed_results,
                                            "total_results_count": len(processed_results) if isinstance(processed_results, list) else 1,
                                            "original_count": execution_result["results"]["total_results_count"],
                                            "database_name": database_name,
                                            "database_type": database_type,
                                            "instance_id": instance_id,
                                            "columns": columns,
                                            "table_name": table_name,
                                            "docetl_operator_used": chosen_operator,
                                            "processed_by_docetl": True
                                        }
                                    }
                                        
                            except Exception as e:
                                logger.error(f"DocETL processing failed, returning original results: {e}")
                                # Fall back to original results if DocETL processing fails
                    
                    if execution_result["status"] == "success":
                        table_name = extract_table_name_from_sql(sql_query)
                        return {
                            "status": "success",
                            "results": {
                                "original_sql_query": sql_query,
                                "executed_sql_query": sql_to_execute,
                                "query_results": execution_result["results"]["query_results"],
                                "total_results_count": execution_result["results"]["total_results_count"],
                                "database_name": database_name,
                                "database_type": database_type,
                                "instance_id": instance_id,
                                "columns": execution_result["results"].get("columns", []),
                                "table_name": table_name,
                                "processed_by_docetl": False
                            }
                        }
                    else:
                        return {
                            "status": "error",
                            "error": f"Database execution failed: {execution_result.get('error', 'Unknown error')}"
                        }
                        
                except Exception as e:
                    logger.error(f"Database manager execution failed: {e}")
                    return {
                        "status": "error",
                        "error": f"Database execution failed: {str(e)}"
                    }
            
            # If required parameters are missing, return error
            error_msg = "SQL execution failed due to missing parameters"
            if not self.db_manager:
                error_msg = "Database manager is not available for SQL execution"
            elif not database_name:
                error_msg = "Database name is required for SQL execution"
            elif not database_type:
                error_msg = "Database type is required for SQL execution"
            
            logger.error(f"SQL execution failed: {error_msg}")
            return {
                "status": "error", 
                "error": error_msg
            }
            
        except Exception as e:
            logger.error(f"SQL execution failed: {str(e)}")
            return {"status": "error", "error": str(e)}



# Keep the other classes from the original file unchanged

class SQLGenerationTool:
    """Tool for generating SQL queries from natural language"""
    
    def __init__(self, llm_client=None, db_manager=None):
        self.llm_client = llm_client
        self.db_manager = db_manager
    
    def execute(self, natural_language_query: str, database_name: str = None, 
                database_type: str = None, instance_id: str = None,
                schema_info: Dict = None, **kwargs) -> Dict[str, Any]:
        """
        Generate SQL query from natural language.
        
        Args:
            natural_language_query: The query in natural language
            database_name: Target database name
            database_type: Type of database (bird, spider2-lite, etc.)
            instance_id: Instance identifier for database connection
            schema_info: Database schema information (optional)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with status and results
        """
        try:
            if not natural_language_query:
                return {"status": "error", "error": "No natural language query provided"}
            
            # Get schema info from database manager if available and not provided
            if not schema_info and self.db_manager and database_name:
                try:
                    # Import the DatabaseConfig class
                    try:
                        from ..utils.database_connection_manager import DatabaseConfig
                    except ImportError:
                        # Create a local DatabaseConfig if import fails
                        from dataclasses import dataclass
                        @dataclass
                        class DatabaseConfig:
                            database_type: str
                            instance_id: str
                            db_name: str
                            connection_params: Dict[str, Any] = None
                    
                    if database_type and instance_id:
                        # Use get_database_config to get proper connection params
                        if hasattr(self.db_manager, 'get_database_config'):
                            config = self.db_manager.get_database_config(instance_id, database_name, database_type)
                        else:
                            config = DatabaseConfig(
                                database_type=database_type,
                                instance_id=instance_id,
                                db_name=database_name
                            )
                        schema_result = self.db_manager.get_schema_info(config)
                        if "error" not in schema_result:
                            schema_info = schema_result
                except Exception as e:
                    logger.warning(f"Could not retrieve schema info: {e}")
            
            # Build prompt for SQL generation
            prompt_parts = [
                f'Given the following database schema for the {database_type or "unknown"} database "{database_name}":',
                ''
            ]
            
            if schema_info:
                prompt_parts.append(f"Schema:\n{json.dumps(schema_info, indent=2)}")
            else:
                prompt_parts.append("Schema: Not available")
            
            prompt_parts.extend([
            '',
            f'Based *only* on this schema and the user\'s request, generate a single, valid SQL query to answer the following:',
            f'User Request: "{natural_language_query}"',
            '',
            'STRICT OUTPUT RULES:',
            '- Output MUST be only one SQL query',
            '- Do NOT include explanations, comments, markdown formatting, or text outside SQL',
            '- Do NOT include triple backticks (```), just return raw SQL',
            '- If the query cannot be answered with the schema, output exactly: QUERY_IMPOSSIBLE',
            '',
            'SQL RULES:',
            '- Only use tables and columns listed in the schema',
            '- Ensure correct SQL syntax for the database type',
            '- For JSON fields in BigQuery, use proper JSON functions',
            '- For SQLite JSON fields, use json_extract() function',
            '- If database type is BigQuery, always use full path from bigquery-public-data',
            '- Generate queries that return comprehensive datasets (avoid unnecessary LIMIT clauses)',
            '- Include all relevant columns useful for analysis',
            '',
            'Final check: If your output contains anything other than pure SQL, replace it entirely with QUERY_IMPOSSIBLE.'
        ])
            
            prompt = "\n".join(prompt_parts)
            
            # Use LLM client if available, otherwise return error
            if self.llm_client and hasattr(self.llm_client, 'call_llm'):
                try:
                    generated_sql = self.llm_client.call_llm(
                        [{"role": "user", "content": prompt}], 
                        category="sql_generate"
                    ).strip()
                    generated_sql = generated_sql.replace('```sql', '').replace('```', '').strip()
                    
                    if generated_sql == "QUERY_IMPOSSIBLE":
                        return {"status": "error", 
                                "error": f"Query cannot be answered based on the provided {schema_info}."}
                    
                    mock_sql = generated_sql
                except Exception as e:
                    logger.error(f"LLM SQL generation failed: {e}")
                    return {"status": "error", "error": f"LLM SQL generation failed: {str(e)}"}
            else:
                return {"status": "error", "error": "LLM client not available or does not have call_llm method"}
            
            # Validate that we actually generated meaningful SQL
            if not mock_sql or mock_sql.strip() == "" or "ERROR" in mock_sql.upper():
                return {"status": "error", "error": "Failed to generate valid SQL query"}
            
            return {
                "status": "success",
                "results": {
                    "sql_query": mock_sql,
                    "database_name": database_name,
                    "database_type": database_type,
                    "instance_id": instance_id,
                    "original_query": natural_language_query,
                    "schema_used": schema_info is not None
                }
            }
            
        except Exception as e:
            logger.error(f"SQL generation failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    


class SQLOptimizationTool:
    """Tool for optimizing SQL queries"""
    
    def __init__(self, llm_client=None, db_manager=None):
        self.llm_client = llm_client
        self.db_manager = db_manager
    
    def execute(self, sql_query: str, db_name: str = None, 
                database_type: str = None, instance_id: str = None,
                schema_info: Dict = None, **kwargs) -> Dict[str, Any]:
        # ... (keeping the original implementation)
        try:
            if not sql_query:
                return {"status": "error", "error": "No SQL query provided"}
            
            # Get schema info if not provided
            if not schema_info and self.db_manager and db_name and database_type:
                try:
                    # Import the DatabaseConfig class
                    try:
                        from ..utils.database_connection_manager import DatabaseConfig
                    except ImportError:
                        # Create a local DatabaseConfig if import fails
                        from dataclasses import dataclass
                        @dataclass
                        class DatabaseConfig:
                            database_type: str
                            instance_id: str
                            db_name: str
                            connection_params: Dict[str, Any] = None
                    
                    if hasattr(self.db_manager, 'get_database_config') and instance_id:
                        config = self.db_manager.get_database_config(instance_id, db_name, database_type)
                    else:
                        config = DatabaseConfig(
                            database_type=database_type,
                            instance_id=instance_id or "default",
                            db_name=db_name
                        )
                    schema_result = self.db_manager.get_schema_info(config)
                    if "error" not in schema_result:
                        schema_info = schema_result
                except Exception as e:
                    logger.warning(f"Could not retrieve schema info for optimization: {e}")
            
            # Use LLM for optimization if available
            if self.llm_client and hasattr(self.llm_client, 'call_llm'):
                try:
                    prompt = f"""
Given the following database schema and SQL query, optimize the query for better performance:

Database Schema:
{json.dumps(schema_info, indent=2) if schema_info else "Not available"}

Original SQL Query:
{sql_query}

Please optimize this query considering:
1. Index usage
2. Join order
3. Subquery optimization
4. WHERE clause optimization
5. SELECT column optimization
6. Database-specific optimizations for {database_type or "generic SQL"}

Return ONLY the optimized SQL query. If no optimization is possible, return the original query.
"""
                    optimized_sql = self.llm_client.call_llm(
                        [{"role": "user", "content": prompt}], 
                        category="sql_optimize"
                    ).strip()
                    optimized_sql = optimized_sql.replace('```sql', '').replace('```', '').strip()
                    
                    optimization_notes = "Query optimized using LLM analysis"
                except Exception as e:
                    logger.error(f"LLM optimization failed: {e}")
                    return {"status": "error", "error": f"LLM optimization failed: {str(e)}"}
            else:
                return {"status": "error", "error": "LLM client not available for optimization"}
            
            return {
                "status": "success",
                "results": {
                    "original_query": sql_query,
                    "optimized_query": optimized_sql,
                    "optimization_notes": optimization_notes,
                    "database_type": database_type,
                    "instance_id": instance_id,
                    "schema_used": schema_info is not None
                }
            }
            
        except Exception as e:
            logger.error(f"SQL optimization failed: {str(e)}")
            return {"status": "error", "error": str(e)}


class SQLDebugTool:
    """Tool for debugging SQL queries"""
    
    def __init__(self, llm_client=None, db_manager=None):
        self.llm_client = llm_client
        self.db_manager = db_manager
    
    def execute(self, failed_sql: str, error: str, database_name: str = None,
                database_type: str = None, instance_id: str = None,
                natural_language_query: str = None, schema_info: Dict = None,
                **kwargs) -> Dict[str, Any]:
        # ... (keeping the original implementation from the original file)
        try:
            if not failed_sql:
                return {"status": "error", "error": "No SQL query provided for debugging"}
            
            # Get schema info if not provided
            if not schema_info and self.db_manager and database_name and database_type:
                try:
                    # Import the DatabaseConfig class
                    try:
                        from ..utils.database_connection_manager import DatabaseConfig
                    except ImportError:
                        # Create a local DatabaseConfig if import fails
                        from dataclasses import dataclass
                        @dataclass
                        class DatabaseConfig:
                            database_type: str
                            instance_id: str
                            db_name: str
                            connection_params: Dict[str, Any] = None
                    
                    if hasattr(self.db_manager, 'get_database_config') and instance_id:
                        config = self.db_manager.get_database_config(instance_id, database_name, database_type)
                    else:
                        config = DatabaseConfig(
                            database_type=database_type,
                            instance_id=instance_id or "default",
                            db_name=database_name
                        )
                    schema_result = self.db_manager.get_schema_info(config)
                    if "error" not in schema_result:
                        schema_info = schema_result
                except Exception as e:
                    logger.warning(f"Could not retrieve schema info for debugging: {e}")
            
            # Use LLM for debugging if available
            if self.llm_client and hasattr(self.llm_client, 'call_llm'):
                try:
                    prompt = f"""
Given the following information:
1. Database Schema:
{json.dumps(schema_info, indent=2) if schema_info else "Not available"}

2. Original Natural Language Query:
{natural_language_query or "Not provided"}

3. Failed SQL Query:
{failed_sql}

4. Error Message:
{error}

Please analyze the error and generate a corrected SQL query that will work correctly.
Consider:
1. Syntax errors
2. Table/column name mismatches
3. Data type issues
4. Missing or incorrect joins
5. Incorrect function usage
6. Database-specific syntax for {database_type or "generic SQL"}

Return ONLY the corrected SQL query. If you cannot fix the query, return "QUERY_UNFIXABLE".
"""
                    corrected_sql = self.llm_client.call_llm(
                        [{"role": "user", "content": prompt}], 
                        category="sql_debug"
                    ).strip()
                    corrected_sql = corrected_sql.replace('```sql', '').replace('```', '').strip()
                    
                    if corrected_sql == "QUERY_UNFIXABLE":
                        return {
                            "status": "error",
                            "error": "Unable to fix the SQL query based on the error message"
                        }
                    
                    # Try to execute the corrected SQL if db_manager is available
                    if self.db_manager and database_name and database_type:
                        try:
                            execution_tool = SQLExecutionTool(self.db_manager)
                            execution_result = execution_tool.execute(
                                sql_query=corrected_sql,
                                database_name=database_name,
                                database_type=database_type,
                                instance_id=instance_id
                            )
                            
                            if execution_result["status"] == "success":
                                return {
                                    "status": "success",
                                    "results": {
                                        "original_query": failed_sql,
                                        "corrected_query": corrected_sql,
                                        "error_analysis": f"LLM analyzed and fixed error: {error}",
                                        "fix_description": "Query corrected using LLM analysis",
                                        "execution_results": execution_result["results"],
                                        "database_type": database_type,
                                        "instance_id": instance_id
                                    }
                                }
                            else:
                                return {
                                    "status": "error",
                                    "error": f"Corrected query still failed: {execution_result.get('error', 'Unknown error')}"
                                }
                        except Exception as e:
                            logger.warning(f"Could not test corrected query: {e}")
                    
                    return {
                        "status": "success",
                        "results": {
                            "original_query": failed_sql,
                            "corrected_query": corrected_sql,
                            "error_analysis": f"LLM analyzed and fixed error: {error}",
                            "fix_description": "Query corrected using LLM analysis",
                            "database_type": database_type,
                            "instance_id": instance_id,
                            "note": "Corrected query not tested due to unavailable database manager"
                        }
                    }
                    
                except Exception as e:
                    logger.error(f"LLM debug failed: {e}")
                    return {"status": "error", "error": f"LLM debug failed: {str(e)}"}
            else:
                return {"status": "error", "error": "LLM client not available for debugging"}
            
        except Exception as e:
            logger.error(f"SQL debug failed: {str(e)}")
            return {"status": "error", "error": str(e)}