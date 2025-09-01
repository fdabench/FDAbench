"""
SQL-related tools for FDABench Package.

These tools provide SQL generation, execution, optimization, and debugging capabilities.
They are designed to work with the tool registry system and integrate with the DatabaseConnectionManager.
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class SQLGenerationTool:
    """Tool for generating SQL queries from natural language"""
    
    def __init__(self, llm_client=None, db_manager=None):
        self.llm_client = llm_client
        self.db_manager = db_manager
    
    def _extract_sql_from_response(self, response: str) -> str:
        """Extract clean SQL query from LLM response that may contain explanatory text."""
        # Remove markdown code blocks
        response = response.replace('```sql', '').replace('```', '').strip()
        
        # Common patterns for SQL statements
        sql_keywords = ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']
        
        # Try to find SQL statement starting with common keywords
        lines = response.split('\n')
        sql_lines = []
        in_sql = False
        
        for line in lines:
            line = line.strip()
            if not line:
                if in_sql:
                    sql_lines.append('')  # Preserve empty lines in SQL
                continue
                
            # Check if line starts with SQL keyword
            if any(line.upper().startswith(keyword) for keyword in sql_keywords):
                in_sql = True
                sql_lines.append(line)
            elif in_sql:
                # Check if this looks like a continuation of SQL (starts with common SQL patterns)
                if (line.upper().startswith(('FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 
                                           'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN', 'OUTER JOIN',
                                           'AND', 'OR', 'UNION', 'INTERSECT', 'EXCEPT', 'LIMIT',
                                           'OFFSET', 'AS', 'ON')) or
                    line.startswith((')', '(', ',')) or
                    re.match(r'^\s*[\w\.\[\]]+\s*[=<>!]+', line) or  # condition patterns
                    re.match(r'^\s*[\w\.\[\]]+\s*,', line)):  # column lists
                    sql_lines.append(line)
                else:
                    # This looks like explanatory text, stop collecting SQL
                    break
        
        if sql_lines:
            sql_query = '\n'.join(sql_lines).strip()
            # Clean up any remaining explanatory text at the beginning
            if ':' in sql_query:
                # Split on colon and take the part that starts with SQL keyword
                parts = sql_query.split(':', 1)
                if len(parts) > 1:
                    potential_sql = parts[1].strip()
                    if any(potential_sql.upper().startswith(keyword) for keyword in sql_keywords):
                        sql_query = potential_sql
            return sql_query
        
        # Fallback: return original response if no clear SQL pattern found
        return response
    
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
                'Important Rules:',
                '- Only use tables and columns listed in the schema',
                '- Ensure correct SQL syntax for the database type',
                '- For JSON fields in BigQuery, use proper JSON functions',
                '- For SQLite JSON fields, use json_extract() function',
                '- Return ONLY the SQL query',
                '- If the request cannot be answered with the given schema, return "QUERY_IMPOSSIBLE"'
            ])
            
            prompt = "\n".join(prompt_parts)
            
            # Use LLM client if available, otherwise return error
            if self.llm_client and hasattr(self.llm_client, 'call_llm'):
                try:
                    generated_sql = self.llm_client.call_llm(
                        [{"role": "user", "content": prompt}], 
                        category="sql_generate"
                    ).strip()
                    generated_sql = self._extract_sql_from_response(generated_sql)
                    
                    if generated_sql == "QUERY_IMPOSSIBLE":
                        return {"status": "error", "error": "Query cannot be answered based on the provided schema."}
                    
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
    


class SQLExecutionTool:
    """Tool for executing SQL queries"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
    
    def execute(self, sql_query: str = None, database_name: str = None,
                database_type: str = None, instance_id: str = None,
                natural_language_query: str = None, previous_results: Dict = None,
                **kwargs) -> Dict[str, Any]:
        """
        Execute SQL query on database.
        
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
                # First check if SQL was passed directly as a parameter
                if "sql_query" in kwargs:
                    sql_query = kwargs["sql_query"]
                # Then check previous results for SQL from different tool names
                elif isinstance(previous_results, dict):
                    for tool_name in ["sql_generate", "generated_sql", "sql_gen"]:
                        if tool_name in previous_results:
                            result = previous_results[tool_name]
                            if isinstance(result, dict):
                                if "sql_query" in result:
                                    sql_query = result["sql_query"]
                                    break
                                elif "results" in result and isinstance(result["results"], dict):
                                    if "sql_query" in result["results"]:
                                        sql_query = result["results"]["sql_query"]
                                        break
            
            # Additional logging for debugging
            if sql_query:
                logger.info(f"✅ SQL query found for execution: {sql_query[:100]}...")
            else:
                logger.warning(f"⚠️ No SQL query found. Parameters: sql_query={sql_query}, previous_results keys: {list(previous_results.keys()) if previous_results else 'None'}")
            
            if not sql_query:
                return {"status": "error", "error": "No SQL query provided"}
            
            # Use database manager if available for real execution
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
                    execution_result = self.db_manager.execute_sql(config, sql_query)
                    
                    if execution_result["status"] == "success":
                        return {
                            "status": "success",
                            "results": {
                                "sql_query": sql_query,
                                "query_results": execution_result["results"]["query_results"],
                                "total_results_count": execution_result["results"]["total_results_count"],
                                "database_name": database_name,
                                "database_type": database_type,
                                "instance_id": instance_id,
                                "columns": execution_result["results"].get("columns", [])
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
            
            # If required parameters are missing, return error with detailed information
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