import json
import logging
import re
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
   
    try:
        sql_clean = re.sub(r'\s+', ' ', sql_query.strip().upper())
        
        from_pattern = r'FROM\s+([^\s,\(\)]+)'
        match = re.search(from_pattern, sql_clean)
        
        if match:
            table_name = match.group(1).strip()
            if '.' in table_name:
                table_name = table_name.split('.')[-1]
            return table_name.lower()
        
        return None
    except Exception as e:
        logger.error(f"Error extracting table name: {e}")
        return None

        
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
        
        # Token counts will be obtained from API response
        
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
            
            # Token counts already obtained from API response
            
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
        
        if chosen_operator and chosen_operator in operators_to_consider:
            return chosen_operator, token_stats
        else:
            logger.info("Using fallback logic for operator selection")
            query_lower = natural_language_query.lower()
            if any(word in query_lower for word in ["extract", "add column", "compute", "generate field"]):
                return "sem_add_columns", token_stats
            else:
                return "sem_filter", token_stats  
                
    except Exception as e:
        logger.error(f"Error choosing palimpzest operator: {e}")
        return "sem_filter", token_stats 

def apply_pz_operator(dataset, operator: str, natural_language_query: str, columns: List[str]):
    
    try:
        filter_prompt = f"The data is relevant to the query: {natural_language_query}"

        if operator == "sem_filter":
            result = dataset.sem_filter(filter_prompt)
        elif operator == "sem_add_columns":
            logger.info(f"sem_add_columns requested but not fully implemented, using sem_filter as fallback")
            result = dataset.sem_filter(filter_prompt)
        else:
            logger.warning(f"Unknown operator: {operator}, using sem_filter as fallback")
            result = dataset.sem_filter(filter_prompt)

        return result
            
    except Exception as e:
        logger.error(f"Error applying palimpzest operator {operator}: {e}")
        return dataset


def track_pz_tokens_with_fallback(output, token_tracker=None, operator: str = "unknown", 
                                  fallback_input_tokens: int = 0, fallback_output_tokens: int = 0, 
                                  data_rows_count: int = 0):
   
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    operation_details = []
    method_used = "none"
    
    try:
        native_stats_found = False
        
        if hasattr(output, 'execution_stats'):
            stats = output.execution_stats
            logger.info("Found execution_stats in Palimpzest output - using native tracking")
            native_stats_found = True
            method_used = "native_execution_stats"

            for op_stats in stats.get('operator_stats', []):
                op_input_tokens = op_stats.get('total_input_tokens', 0)
                op_output_tokens = op_stats.get('total_output_tokens', 0)
                op_cost = op_stats.get('cost_per_record', 0)
                op_name = op_stats.get('op_name', 'unknown')
                
                total_input_tokens += op_input_tokens
                total_output_tokens += op_output_tokens
                total_cost += op_cost
                
                operation_details.append({
                    'op_name': op_name,
                    'input_tokens': op_input_tokens,
                    'output_tokens': op_output_tokens,
                    'cost': op_cost
                })

        if hasattr(output, 'stats') and output.stats:
            logger.info("Found stats attribute in Palimpzest output - using native tracking")
            native_stats_found = True
            method_used = "native_record_stats"
            
            for stat in output.stats:
                op_input_tokens = getattr(stat, 'total_input_tokens', 0)
                op_output_tokens = getattr(stat, 'total_output_tokens', 0)
                op_cost = getattr(stat, 'cost_per_record', 0.0)
                op_name = getattr(stat, 'op_name', 'unknown')
                
                total_input_tokens += op_input_tokens
                total_output_tokens += op_output_tokens
                total_cost += op_cost
                
                operation_details.append({
                    'op_name': op_name,
                    'input_tokens': op_input_tokens,
                    'output_tokens': op_output_tokens,
                    'cost': op_cost
                })
                
                logger.info(f"Operation: {op_name}")
                logger.info(f"  Input tokens: {op_input_tokens}")
                logger.info(f"  Output tokens: {op_output_tokens}")
                logger.info(f"  Cost: ${op_cost:.6f}")
                
                if token_tracker:
                    token_tracker.track_call(
                        category=f"palimpzest_{op_name}",
                        input_tokens=op_input_tokens,
                        output_tokens=op_output_tokens,
                        model="palimpzest_native",
                        cost=op_cost
                    )

        if not native_stats_found or (total_input_tokens == 0 and total_output_tokens == 0):
            logger.warning("Palimpzest native token stats not found or empty - using fallback estimation")
            method_used = "fallback_estimation"
            
            total_input_tokens = fallback_input_tokens
            
            if fallback_output_tokens > 0:
                total_output_tokens = fallback_output_tokens
            else:
                result_len = len(output) if hasattr(output, '__len__') else 0
                if result_len > 0:
                    try:
                        result_content = str(output) if hasattr(output, '__str__') else f"{result_len} items"
                        total_output_tokens = 0  
                    except:
                        total_output_tokens = result_len * 5  
                else:
                    total_output_tokens = 0
            
            if token_tracker and data_rows_count > 0:
                token_tracker.track_call(
                    category=f"palimpzest_{operator}_fallback",
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    model="palimpzest_estimated",
                    cost=None
                )
                
                operation_details.append({
                    'op_name': f'{operator}_estimated',
                    'input_tokens': total_input_tokens,
                    'output_tokens': total_output_tokens,
                    'cost': 0.0
                })

        # Log summary statistics
        logger.info(f"Palimpzest Token Usage Summary (method: {method_used}):")
        logger.info(f"  Input tokens: {total_input_tokens}")
        logger.info(f"  Output tokens: {total_output_tokens}")
        logger.info(f"  Total cost: ${total_cost:.6f}")
        
        if token_tracker and (total_input_tokens > 0 or total_output_tokens > 0):
            token_tracker.track_call(
                category=f"palimpzest_{operator}_total",
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                model=f"palimpzest_{method_used}",
                cost=total_cost
            )
        
        return {
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_cost': total_cost,
            'operation_details': operation_details,
            'method_used': method_used,
            'native_stats_available': native_stats_found
        }
        
    except Exception as e:
        logger.error(f"Error in token tracking, using basic fallback: {e}")
        
        if token_tracker and (fallback_input_tokens > 0 or fallback_output_tokens > 0):
            token_tracker.track_call(
                category=f"palimpzest_{operator}_error_fallback",
                input_tokens=fallback_input_tokens,
                output_tokens=fallback_output_tokens,
                model="palimpzest_error_fallback",
                cost=None
            )
        
        return {
            'total_input_tokens': fallback_input_tokens,
            'total_output_tokens': fallback_output_tokens,
            'total_cost': 0.0,
            'operation_details': [],
            'method_used': 'error_fallback',
            'native_stats_available': False,
            'error': str(e)
        }

class SQLExecutionTool:
    """Enhanced tool for executing SQL queries with LLM-based palimpzest operator selection"""
    
    def __init__(self, db_manager=None, token_tracker=None):
        self.db_manager = db_manager
        self.token_tracker = token_tracker
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
            if not sql_query and previous_results:
                for tool_name in ["sql_generate", "generated_sql"]:
                    if tool_name in previous_results:
                        result = previous_results[tool_name]
                        if isinstance(result, dict) and "sql_query" in result:
                            sql_query = result["sql_query"]
                            break
            
            if not sql_query:
                return {"status": "error", "error": "No SQL query provided"}

            table_name = extract_table_name_from_sql(sql_query)
            if table_name:
                modified_sql = f"SELECT * FROM {table_name}"
                logger.info(f"Modified SQL from complex query to: {modified_sql}")
                sql_query = modified_sql
            else:
                sql_query = sql_query
            
            if self.db_manager and database_name and database_type:
                try:
                    try:
                        from ..utils.database_connection_manager import DatabaseConfig
                    except ImportError:
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
                    
                    execution_result = self.db_manager.execute_sql(config, sql_query)
                    
                    if execution_result["status"] == "success":
                        if natural_language_query:
                            try:
                                query_results = execution_result["results"]["query_results"]
                                columns = execution_result["results"].get("columns", [])
                                
                                if query_results and columns:
                                    df = pd.DataFrame(query_results, columns=columns)

                                    
                                    fallback_input_tokens = 0
                                    fallback_output_tokens = 0
                                    if not df.empty:
                                        filter_prompt_for_estimation = f"The data is relevant to the query: {natural_language_query}"
                                        fallback_input_tokens = 0  # No token estimation without API response

                                   
                                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                                    data_dir = os.path.join(project_root, "examples", "data")
                                    os.makedirs(data_dir, exist_ok=True)
                                    csv_file_path = os.path.join(data_dir, "temp_data.csv")
                                    output_dir = os.path.join(data_dir, "pz-data")
                                    
                                    if os.path.exists(csv_file_path): os.remove(csv_file_path)
                                    if os.path.exists(output_dir): shutil.rmtree(output_dir)
                                    os.makedirs(output_dir, exist_ok=True)


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

                                    pz_columns = [{"name": col, "type": infer_pz_type(df[col].dtype), "desc": f"Column {col}"} for col in columns]
                                    dataset = dataset.sem_add_columns(pz_columns)

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

                                    processed_dataset = apply_pz_operator(
                                        dataset=dataset, 
                                        operator=chosen_operator, 
                                        natural_language_query=natural_language_query, 
                                        columns=columns
                                    )

                                    config = pz.QueryProcessorConfig(policy=pz.MinCost(), available_models=[Model.DEEPSEEK_V3], verbose=True)
                                    output = processed_dataset.run(config)
                                    
                                    token_stats = track_pz_tokens_with_fallback(
                                        output=output,
                                        token_tracker=self.token_tracker,
                                        operator=chosen_operator,
                                        fallback_input_tokens=fallback_input_tokens,
                                        fallback_output_tokens=fallback_output_tokens,
                                        data_rows_count=len(df)
                                    )
                                    
                                    filtered_df = output.to_df(cols=columns)
                                    filtered_results = filtered_df.to_dict('records')
                                    
                                    logger.info(f"Palimpzest processed {len(filtered_results)} results (original: {len(df)})")
                                    logger.info(f"Token tracking method used: {token_stats.get('method_used', 'unknown')}")
                                    
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
                                            "filtered_by_palimpzest": True,
                                            "token_stats": {
                                                "total_input_tokens": token_stats.get('total_input_tokens', 0),
                                                "total_output_tokens": token_stats.get('total_output_tokens', 0),
                                                "total_cost": token_stats.get('total_cost', 0.0),
                                                "method_used": token_stats.get('method_used', 'unknown'),
                                                "native_stats_available": token_stats.get('native_stats_available', False),
                                                "operation_details": token_stats.get('operation_details', [])
                                            }
                                        }
                                    }
                                        
                            except Exception as e:
                                logger.error(f"Palimpzest processing failed, returning original SQL results: {e}")
                    
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
    
    def __init__(self, llm_client=None, db_manager=None):
        self.llm_client = llm_client
        self.db_manager = db_manager
    
    def execute(self, natural_language_query: str, database_name: str = None, 
                database_type: str = None, instance_id: str = None,
                schema_info: Dict = None, **kwargs) -> Dict[str, Any]:
        
        try:
            if not natural_language_query:
                return {"status": "error", "error": "No natural language query provided"}
            
            if not schema_info and self.db_manager and database_name:
                try:
                    try:
                        from ..utils.database_connection_manager import DatabaseConfig
                    except ImportError:
                        from dataclasses import dataclass
                        @dataclass
                        class DatabaseConfig:
                            database_type: str
                            instance_id: str
                            db_name: str
                            connection_params: Dict[str, Any] = None
                    
                    if database_type and instance_id:
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
   
    
    def __init__(self, llm_client=None, db_manager=None):
        self.llm_client = llm_client
        self.db_manager = db_manager
    
    def execute(self, sql_query: str, db_name: str = None, 
                database_type: str = None, instance_id: str = None,
                schema_info: Dict = None, **kwargs) -> Dict[str, Any]:
       
        try:
            if not sql_query:
                return {"status": "error", "error": "No SQL query provided"}
            
            if not schema_info and self.db_manager and db_name and database_type:
                try:
                    try:
                        from ..utils.database_connection_manager import DatabaseConfig
                    except ImportError:
                       
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
            
            if not schema_info and self.db_manager and database_name and database_type:
                try:
                    try:
                        from ..utils.database_connection_manager import DatabaseConfig
                    except ImportError:
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