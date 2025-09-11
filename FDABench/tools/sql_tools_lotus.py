
import json
import logging
import re
from typing import Dict, List, Any, Optional
import os
import pandas as pd
import lotus
from lotus.models import LM
import requests
from openai import OpenAI

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

    
def choose_lotus_operator_with_openrouter(natural_language_query: str, available_operators: List[str], api_key) -> tuple[str, dict]:
    token_stats = {
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0
    }
    
    try:
        test_operators = ["sem_filter", "sem_agg", "sem_map", "sem_search", "sem_topk"]
        operators_to_consider = [op for op in test_operators if op in available_operators]
        
        prompt = f"""
Given the natural language query and available lotus operators, choose the most appropriate operator.

Natural Language Query: "{natural_language_query}"

Available Operators:
- sem_filter: Filter rows based on conditions (returns subset of rows)
- sem_agg: Aggregate data across rows (returns summary/analysis)  
- sem_map: Transform/extract information from each row (returns new columns)
- sem_search: Search for similar content (returns ranked results)
- sem_topk: Find top K items based on criteria (returns top items)

Rules:
1. Default to "sem_agg" for most queries involving data analysis, questions, or information extraction
2. Use "sem_agg" for any query asking "what", "who", "how many", "which", "when", "where"
3. Use "sem_agg" for comparisons, summaries, insights, or analytical requests
4. Use "sem_agg" for finding patterns, trends, or relationships in data
5. Only use other operators in very specific cases:
  - "sem_filter" only when explicitly asking to "filter out" or "remove" specific rows
  - "sem_map" only when explicitly asking to "add new columns" or "transform each row"
  - "sem_search" only when explicitly asking for "similar" or "search-like" functionality
  - "sem_topk" only when explicitly asking for "top K" with a specific number

Return ONLY the operator name (e.g., "sem_agg"). No explanation needed.
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
            logger.error("Default fallback logic")
            query_lower = natural_language_query.lower()
            if any(word in query_lower for word in ["filter", "where", "select", "find records"]):
                return "sem_filter", token_stats
            elif any(word in query_lower for word in ["analyze", "summary", "who is", "longest", "report"]):
                return "sem_agg", token_stats
            elif any(word in query_lower for word in ["top", "best", "highest", "maximum"]):
                return "sem_topk", token_stats
            else:
                return "sem_agg", token_stats
                
    except Exception as e:
        logger.error(f"Error choosing lotus operator: {e}")
        return "sem_filter", token_stats
    

def apply_lotus_operator(df: pd.DataFrame, operator: str, natural_language_query: str, 
                        columns: List[str], lm=None, token_tracker=None) -> pd.DataFrame:
    try:
        data_rows_count = len(df)
        
        user_instruction = natural_language_query
        columns_found = []
        
        if not columns_found:
            column_refs = ", ".join([f"{{{col}}}" for col in columns]) 
            user_instruction = f"Based on {column_refs}, {natural_language_query}"
        
        if lm:
            estimated_input_tokens = lm.count_tokens(user_instruction + "\n" + df.to_string())
        else:
            estimated_input_tokens = 0  # No token estimation without LM
        
        logger.info(f"Applying {operator} with instruction: {user_instruction}")
        
        if operator == "sem_filter":
            initial_physical_tokens = lm.stats.physical_usage.total_tokens if lm else 0
            
            result = df.sem_filter(user_instruction)
            
            if lm:
                actual_total_tokens = lm.stats.physical_usage.total_tokens
                estimated_output_tokens = actual_total_tokens - initial_physical_tokens - estimated_input_tokens
                if estimated_output_tokens < 0:
                    estimated_output_tokens = 0  # Use 0 if negative
            else:
                estimated_output_tokens = 0  # No token estimation without LM
            
            if token_tracker:
                token_tracker.track_call(
                    category=f"lotus_{operator}_batch",
                    input_tokens=estimated_input_tokens,
                    output_tokens=estimated_output_tokens,
                    model="lotus_semantic_operation",
                    cost=lm.stats.physical_usage.total_cost if lm else None
                )
                logger.info(f"Tracked 1 lotus {operator} batch operation for {data_rows_count} rows")
                
                if lm:
                    logger.info(f"Lotus actual usage - Physical tokens: {lm.stats.physical_usage.total_tokens}, Virtual tokens: {lm.stats.virtual_usage.total_tokens}")
            
            return result
            
        elif operator == "sem_agg":
            initial_physical_tokens = lm.stats.physical_usage.total_tokens if lm else 0
            
            result = df.sem_agg(user_instruction, all_cols=True)
            
            if lm:
                actual_total_tokens = lm.stats.physical_usage.total_tokens
                estimated_output_tokens = actual_total_tokens - initial_physical_tokens - estimated_input_tokens
                if estimated_output_tokens < 0:
                    estimated_output_tokens = 0  # No token estimation without LM
            else:
                estimated_output_tokens = 0  # No token estimation without LM
            
            if token_tracker:
                token_tracker.track_call(
                    category=f"lotus_{operator}_batch",
                    input_tokens=estimated_input_tokens,
                    output_tokens=estimated_output_tokens,
                    model="lotus_semantic_operation",
                    cost=lm.stats.physical_usage.total_cost if lm else None
                )
                logger.info(f"Tracked 1 lotus {operator} batch operation for {data_rows_count} rows")
                
                if lm:
                    logger.info(f"Lotus actual usage - Physical tokens: {lm.stats.physical_usage.total_tokens}, Virtual tokens: {lm.stats.virtual_usage.total_tokens}")
            
            return result
            
        elif operator == "sem_map":
            initial_physical_tokens = lm.stats.physical_usage.total_tokens if lm else 0
            
            result = df.sem_map(user_instruction)
            
            if lm:
                actual_total_tokens = lm.stats.physical_usage.total_tokens
                estimated_output_tokens = actual_total_tokens - initial_physical_tokens - estimated_input_tokens
                if estimated_output_tokens < 0:
                    estimated_output_tokens = 0  # Use 0 if negative
            else:
                estimated_output_tokens = 0  # No token estimation without LM
            
            if token_tracker:
                token_tracker.track_call(
                    category=f"lotus_{operator}_batch",
                    input_tokens=estimated_input_tokens,
                    output_tokens=estimated_output_tokens,
                    model="lotus_semantic_operation",
                    cost=lm.stats.physical_usage.total_cost if lm else None
                )
                logger.info(f"Tracked 1 lotus {operator} batch operation for {data_rows_count} rows")
                
                if lm:
                    logger.info(f"Lotus actual usage - Physical tokens: {lm.stats.physical_usage.total_tokens}, Virtual tokens: {lm.stats.virtual_usage.total_tokens}")
            
            return result
            
        elif operator == "sem_search" and columns:
            initial_physical_tokens = lm.stats.physical_usage.total_tokens if lm else 0
            
            result = df.sem_search(columns[0], user_instruction, K=min(100, len(df)))
            
            if lm:
                actual_total_tokens = lm.stats.physical_usage.total_tokens
                estimated_output_tokens = actual_total_tokens - initial_physical_tokens - estimated_input_tokens
                if estimated_output_tokens < 0:
                    estimated_output_tokens = 0  # Use 0 if negative
            else:
                estimated_output_tokens = 0  # No token estimation without LM
            
            if token_tracker:
                token_tracker.track_call(
                    category=f"lotus_{operator}_batch",
                    input_tokens=estimated_input_tokens,
                    output_tokens=estimated_output_tokens,
                    model="lotus_semantic_operation",
                    cost=lm.stats.physical_usage.total_cost if lm else None
                )
                logger.info(f"Tracked 1 lotus {operator} batch operation for {data_rows_count} rows")
                
                if lm:
                    logger.info(f"Lotus actual usage - Physical tokens: {lm.stats.physical_usage.total_tokens}, Virtual tokens: {lm.stats.virtual_usage.total_tokens}")
            
            return result
            
        elif operator == "sem_topk":
            initial_physical_tokens = lm.stats.physical_usage.total_tokens if lm else 0
            
            result = df.sem_topk(user_instruction, K=10)
            
            if lm:
                actual_total_tokens = lm.stats.physical_usage.total_tokens
                estimated_output_tokens = actual_total_tokens - initial_physical_tokens - estimated_input_tokens
                if estimated_output_tokens < 0:
                    estimated_output_tokens = 0  # Use 0 if negative
            else:
                estimated_output_tokens = 0  # No token estimation without LM
            
            if token_tracker:
                token_tracker.track_call(
                    category=f"lotus_{operator}_batch",
                    input_tokens=estimated_input_tokens,
                    output_tokens=estimated_output_tokens,
                    model="lotus_semantic_operation",
                    cost=lm.stats.physical_usage.total_cost if lm else None
                )
                logger.info(f"Tracked 1 lotus {operator} batch operation for {data_rows_count} rows")
                
                if lm:
                    logger.info(f"Lotus actual usage - Physical tokens: {lm.stats.physical_usage.total_tokens}, Virtual tokens: {lm.stats.virtual_usage.total_tokens}")
            
            return result
        else:
            logger.warning(f"Unsupported operator: {operator}, using sem_agg as fallback")
            result = df.sem_agg(user_instruction, all_cols=True)
            
            estimated_output_tokens = 0  # No token estimation without LM
            
            if token_tracker:
                token_tracker.track_call(
                    category=f"lotus_{operator}_fallback_batch",
                    input_tokens=estimated_input_tokens,
                    output_tokens=estimated_output_tokens,
                    model="lotus_semantic_operation",
                    cost=None
                )
                logger.info(f"Tracked 1 lotus {operator} (fallback) batch operation for {data_rows_count} rows")
            
            return result
            
    except Exception as e:
        logger.error(f"Error applying lotus operator {operator}: {e}")
        return df

class SQLExecutionTool:
    
    def __init__(self, db_manager=None, token_tracker=None):
        self.db_manager = db_manager
        self.token_tracker = token_tracker
        self.available_operators = ["sem_filter", "sem_agg", "sem_map", "sem_search", "sem_topk", 
                                  "sem_join", "sem_extract", "sem_cluster_by", "sem_dedup"]
    
    def execute(self, sql_query: str = None, database_name: str = None,
                database_type: str = None, instance_id: str = None,
                natural_language_query: str = None, previous_results: Dict = None,
                **kwargs) -> Dict[str, Any]:
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
                sql_to_execute = modified_sql
            else:
                sql_to_execute = sql_query

            if self.db_manager and database_name and database_type:
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
                    
                    execution_result = self.db_manager.execute_sql(config, sql_to_execute)
                    
                    if execution_result["status"] == "success":
                        if natural_language_query:
                            try:
                                api_key = os.environ.get('OPENROUTER_API_KEY')
                                if not api_key:
                                    logger.error("OPENROUTER_API_KEY not found, lotus processing will fail")
                                    raise ValueError("Missing OPENROUTER_API_KEY")
                                
                                lm = LM(
                                    model="openrouter/deepseek/deepseek-chat-v3-0324",
                                    api_key=api_key,
                                    api_base="https://openrouter.ai/api/v1"
                                )
                                lotus.settings.configure(lm=lm)
                                logger.info("Lotus LM configured successfully")

                                query_results = execution_result["results"]["query_results"]
                                columns = execution_result["results"].get("columns", [])
                                
                                if query_results and columns:
                                    df = pd.DataFrame(query_results, columns=columns)
                                    logger.info(f"Created DataFrame with shape: {df.shape}")
                                    
                                    chosen_operator,tokens_choose_operator = choose_lotus_operator_with_openrouter(
                                        natural_language_query, 
                                        self.available_operators, 
                                        api_key
                                    )
                                    self.token_tracker.track_call(
                                        category=f"lotus_{chosen_operator}",
                                        input_tokens=tokens_choose_operator['input_tokens'],
                                        output_tokens=tokens_choose_operator['output_tokens'],
                                        model="deepseek/deepseek-chat-v3-0324",
                                        cost=None
                                    )
                                    logger.info(f"LLM chose tokens: {tokens_choose_operator}")
                                    logger.info(f"LLM chose operator: {chosen_operator}")
                                    if self.token_tracker:
                                        input_text = natural_language_query + str(df.head().to_string())
                                        estimated_input_tokens = lm.count_tokens(input_text)
                                    
                                    processed_df = apply_lotus_operator(
                                        df, chosen_operator, natural_language_query, columns, lm, self.token_tracker
                                    )
                                    
                                    if lm:
                                        lm.print_total_usage()
                                    
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
                                            "lotus_operator_used": chosen_operator,
                                            "processed_by_lotus": True
                                        }
                                    }
                                        
                            except Exception as e:
                                logger.error(f"Lotus processing failed, returning original results: {e}")
                    
                    if execution_result["status"] == "success":
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
                                "processed_by_lotus": False
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