"""
SQL-related tools for FDABench Package.

These tools provide SQL generation, execution, optimization, and debugging capabilities.
They are designed to work with the tool registry system and integrate with the DatabaseConnectionManager.
"""
import json
import logging
from typing import Dict, List, Any, Optional
import os
import requests
import pandas as pd
import palimpzest as pz
from palimpzest.constants import Model
import numpy as np
import shutil
from dataclasses import dataclass


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

def choose_pz_operator_with_llm(natural_language_query: str, available_operators: List[str], api_key: str) -> tuple[str, dict]:
    """
    Use LLM to choose the most appropriate palimpzest operator for the given query.
    Returns: (operator_name, token_stats)
    """
    token_stats = {
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0
    }
    
    try:
        # Only include actually supported operators in palimpzest
        supported_operators = ["sem_filter", "sem_add_columns"]
        operators_to_consider = [op for op in supported_operators if op in available_operators]
        
        prompt = f"""
Given the natural language query and available palimpzest operators, choose the most appropriate operator.

Natural Language Query: "{natural_language_query}"

Available Palimpzest Operators:
- sem_filter: Filter data based on natural language conditions (keeps rows that match criteria)
- sem_add_columns: Extract/compute new fields from existing data (adds new columns with extracted information)

Important Rules:
1. Use "sem_filter" for most queries that involve:
   - Finding/selecting specific data ("what", "who", "which", "where" questions)
   - Filtering based on conditions 
   - Searching for relevant information
   - Data analysis that requires subset of data
   
2. Use "sem_add_columns" ONLY when explicitly asked to:
   - Extract specific new fields/columns from existing data
   - Add computed columns or derived information
   - Transform data structure by adding new attributes

3. Default to "sem_filter" when in doubt - it's the most versatile operator for data analysis

Return ONLY the operator name (e.g., "sem_filter"). No explanation needed.
"""
        
        # Estimate input tokens
        token_stats['input_tokens'] = estimate_tokens(prompt)
        
        # Make direct OpenRouter API call
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/j../FDABench",
            "X-Title": "FDABench"
        }
        
        payload = {
            "model": "deepseek/deepseek-chat-v3-0324",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            chosen_operator = result["choices"][0]["message"]["content"].strip().lower()
            
            # Estimate output tokens
            token_stats['output_tokens'] = estimate_tokens(chosen_operator)
            
            # Get actual usage if available from API response
            if 'usage' in result:
                usage = result['usage']
                token_stats['input_tokens'] = usage.get('prompt_tokens', token_stats['input_tokens'])
                token_stats['output_tokens'] = usage.get('completion_tokens', token_stats['output_tokens'])
                token_stats['total_tokens'] = usage.get('total_tokens', 
                    token_stats['input_tokens'] + token_stats['output_tokens'])
            else:
                token_stats['total_tokens'] = token_stats['input_tokens'] + token_stats['output_tokens']
                
        else:
            logger.error(f"OpenRouter API call failed with status {response.status_code}: {response.text}")
            chosen_operator = None
        
        # Validate the chosen operator
        if chosen_operator and chosen_operator in operators_to_consider:
            return chosen_operator, token_stats
        else:
            # Default fallback logic
            logger.info("Using fallback logic for operator selection")
            query_lower = natural_language_query.lower()
            if any(word in query_lower for word in ["extract", "add column", "compute", "generate field"]):
                return "sem_add_columns", token_stats
            else:
                return "sem_filter", token_stats  # Default for most queries
                
    except Exception as e:
        logger.error(f"Error choosing palimpzest operator: {e}")
        return "sem_filter", token_stats  # Default fallback

# --- REFACTORED FUNCTION ---
def apply_pz_operator(dataset, operator: str, natural_language_query: str, columns: List[str], 
                      token_tracker=None, estimated_input_tokens: int = 0, data_rows_count: int = 0):
    """
    Apply the chosen palimpzest operator to the dataset.
    Accepts pre-calculated token estimates and row counts for tracking.
    """
    try:
        filter_prompt = f"The data is relevant to the query: {natural_language_query}"

        # Determine the operation and result (all fall back to sem_filter for now)
        if operator == "sem_filter":
            result = dataset.sem_filter(filter_prompt)
            category_prefix = f"palimpzest_{operator}"
        elif operator == "sem_add_columns":
            logger.info(f"sem_add_columns requested but not fully implemented, using sem_filter as fallback")
            result = dataset.sem_filter(filter_prompt)
            category_prefix = f"palimpzest_{operator}_fallback"
        else:
            logger.warning(f"Unknown operator: {operator}, using sem_filter as fallback")
            result = dataset.sem_filter(filter_prompt)
            category_prefix = "palimpzest_unknown_fallback"

        # Track token usage if a token_tracker is provided and we have rows to process
        if token_tracker and data_rows_count > 0:
            # Estimate output tokens based on result size
            result_len = len(result) if hasattr(result, '__len__') else 0
            estimated_output_tokens = result_len * 5 if result_len > 0 else 100
            
            # Track multiple LLM calls - one for each data row, using the pre-calculated input tokens
            for i in range(data_rows_count):
                token_tracker.track_call(
                    category=f"{category_prefix}_row_{i+1}",
                    input_tokens=estimated_input_tokens // data_rows_count if data_rows_count > 0 else estimated_input_tokens,
                    output_tokens=estimated_output_tokens // data_rows_count if data_rows_count > 0 else 0,
                    model="palimpzest_semantic_operation",
                    cost=None
                )
            
            logger.info(f"Tracked {data_rows_count} palimpzest '{operator}' operations (one per data row)")
        
        return result
            
    except Exception as e:
        logger.error(f"Error applying palimpzest operator {operator}: {e}")
        # Return original dataset if operation fails
        return dataset

class SQLExecutionTool:
    """Enhanced tool for executing SQL queries with LLM-based palimpzest operator selection"""
    
    def __init__(self, db_manager=None, token_tracker=None):
        self.db_manager = db_manager
        self.token_tracker = token_tracker
        # Only include actually supported operators
        self.available_operators = ["sem_filter", "sem_add_columns"]
    
    def execute(self, sql_query: str = None, database_name: str = None,
                database_type: str = None, instance_id: str = None,
                natural_language_query: str = None, previous_results: Dict = None,
                **kwargs) -> Dict[str, Any]:
        """
        Execute SQL query on database with LLM-based palimpzest operator selection.
        
        Args:
            sql_query: SQL query to execute
            database_name: Target database
            database_type: Type of database (bird, spider2-lite, etc.)
            instance_id: Instance identifier for database connection
            natural_language_query: Original query (for generation if needed)
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
                sql_query = modified_sql
            else:
                sql_query = sql_query
            
            # Use database manager if available for real execution
            if self.db_manager and database_name and database_type:
                try:
                    # Import the DatabaseConfig class or create a local one if import fails
                    try:
                        from ..utils.database_connection_manager import DatabaseConfig
                    except ImportError:
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
                    execution_result = self.db_manager.execute_sql(config, sql_query)
                    
                    if execution_result["status"] == "success":
                        # Apply palimpzest processing if natural_language_query is provided
                        if natural_language_query:
                            try:
                                # Convert execution_result to pandas DataFrame
                                query_results = execution_result["results"]["query_results"]
                                columns = execution_result["results"].get("columns", [])
                                
                                if query_results and columns:
                                    df = pd.DataFrame(query_results, columns=columns)

                                    
                                    # --- NEW: ESTIMATE TOKENS USING THE DATAFRAME ---
                                    # This is now done here, where the DataFrame is easily accessible.
                                    pz_estimated_input_tokens = 0
                                    if not df.empty:
                                        filter_prompt_for_estimation = f"The data is relevant to the query: {natural_language_query}"
                                        # Estimate based on the prompt plus the content of the entire DataFrame
                                        pz_estimated_input_tokens = estimate_tokens(filter_prompt_for_estimation) + estimate_tokens(df.to_string())

                                    # Define file paths using relative path from project root
                                    # NOTE: Consider using a temporary directory for better file management
                                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                                    data_dir = os.path.join(project_root, "examples", "data")
                                    os.makedirs(data_dir, exist_ok=True)
                                    csv_file_path = os.path.join(data_dir, "temp_data.csv")
                                    output_dir = os.path.join(data_dir, "pz-data")
                                    
                                    # Cleanup previous run's files
                                    if os.path.exists(csv_file_path): os.remove(csv_file_path)
                                    if os.path.exists(output_dir): shutil.rmtree(output_dir)
                                    os.makedirs(output_dir, exist_ok=True)


                                    # Save to CSV and create text files for Palimpzest
                                    df.to_csv(csv_file_path, index=False)
                                    for i, row in df.iterrows():
                                        filename = f"pz_temp{i+1}.txt"
                                        filepath = os.path.join(output_dir, filename)
                                        content = "\n".join([f"{col}: {row[col]}" for col in df.columns]) + "\n"
                                        with open(filepath, 'w', encoding='utf-8') as f: f.write(content)

                                    print(f"Created {len(df)} text files in {output_dir}")
                                    dataset = pz.Dataset(output_dir)
                                    
                                    def infer_pz_type(dtype):
                                        if np.issubdtype(dtype, np.number): return int
                                        elif np.issubdtype(dtype, np.datetime64): return str
                                        else: return str

                                    # Define column schema for palimpzest
                                    pz_columns = [{"name": col, "type": infer_pz_type(df[col].dtype), "desc": f"Column {col}"} for col in columns]
                                    dataset = dataset.sem_add_columns(pz_columns)

                                    # Use LLM to choose the appropriate operator
                                    api_key = os.environ.get('OPENROUTER_API_KEY')
                                    if not api_key:
                                        raise ValueError("Missing OPENROUTER_API_KEY for Palimpzest operator selection")
                                    
                                    chosen_operator, token_stats = choose_pz_operator_with_llm(
                                        natural_language_query, self.available_operators, api_key
                                    )
                                    
                                    if self.token_tracker:
                                        self.token_tracker.track_call(
                                            category="pz_operator_selection",
                                            input_tokens=token_stats['input_tokens'],
                                            output_tokens=token_stats['output_tokens'],
                                            model="deepseek/deepseek-chat-v3-0324", 
                                            cost=None
                                        )
                                    logger.info(f"LLM chose operator: {chosen_operator}")

                                    # --- MODIFIED: CALL apply_pz_operator WITH NEW ARGS ---
                                    processed_dataset = apply_pz_operator(
                                        dataset=dataset, 
                                        operator=chosen_operator, 
                                        natural_language_query=natural_language_query, 
                                        columns=columns, 
                                        token_tracker=self.token_tracker,
                                        estimated_input_tokens=pz_estimated_input_tokens,  # Pass pre-calculated value
                                        data_rows_count=len(df)  # Pass the row count
                                    )

                                    # Execute with palimpzest using DeepSeek V3
                                    config = pz.QueryProcessorConfig(policy=pz.MinCost(), available_models=[Model.DEEPSEEK_V3], verbose=True)
                                    output = processed_dataset.run(config)
                                    
                                    # Convert back to expected format
                                    filtered_df = output.to_df(cols=columns)
                                    filtered_results = filtered_df.to_dict('records')
                                    
                                    logger.info(f"Palimpzest processed {len(filtered_results)} results (original: {len(df)})")
                                    
                                    return {
                                        "status": "success",
                                        "results": {
                                            "sql_query": sql_query,
                                            "query_results": filtered_results,
                                            "total_results_count": len(filtered_results),
                                            "original_count": execution_result["results"]["total_results_count"],
                                            "database_name": database_name,
                                            "database_type": database_type,
                                            "instance_id": instance_id,
                                            "columns": columns,
                                            "pz_operator_used": chosen_operator,
                                            "filtered_by_palimpzest": True
                                        }
                                    }
                                        
                            except Exception as e:
                                logger.error(f"Palimpzest processing failed, returning original SQL results: {e}")
                                # Fall back to original results if palimpzest processing fails
                    
                    # This block will be reached if Palimpzest fails or is skipped
                    return {
                        "status": "success",
                        "results": {
                            "sql_query": sql_query,
                            "query_results": execution_result["results"]["query_results"],
                            "total_results_count": execution_result["results"]["total_results_count"],
                            "database_name": database_name,
                            "database_type": database_type,
                            "instance_id": instance_id,
                            "columns": execution_result["results"].get("columns", []),
                            "filtered_by_palimpzest": False
                        }
                    }
                        
                except Exception as e:
                    logger.error(f"Database manager execution failed: {e}")
                    return {"status": "error", "error": f"Database execution failed: {str(e)}"}
            
            # If required parameters are missing for db_manager
            error_msg_map = {
                "db_manager": "Database manager is not available for SQL execution",
                "database_name": "Database name is required for SQL execution",
                "database_type": "Database type is required for SQL execution"
            }
            error_msg = next((msg for key, msg in error_msg_map.items() if not locals().get(key)), "SQL execution failed due to missing parameters")
            logger.error(f"SQL execution pre-condition failed: {error_msg}")
            return {"status": "error", "error": error_msg}
            
        except Exception as e:
            logger.error(f"An unexpected error occurred in SQLExecutionTool: {str(e)}", exc_info=True)
            return {"status": "error", "error": str(e)}


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
        """
        Optimize SQL query for better performance.
        
        Args:
            sql_query: SQL query to optimize
            db_name: Database name
            database_type: Type of database (bird, spider2-lite, etc.)
            instance_id: Instance identifier for database connection
            schema_info: Schema information for optimization context
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with status and results
        """
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
        """
        Debug and fix SQL query errors.
        
        Args:
            failed_sql: The SQL that failed
            error: Error message
            database_name: Target database
            database_type: Type of database (bird, spider2-lite, etc.)
            instance_id: Instance identifier for database connection
            natural_language_query: Original query
            schema_info: Schema information
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with status and results
        """
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