"""
SQL-related tools for FDABench Package.

These tools provide SQL generation, execution, optimization, and debugging capabilities.
They are designed to work with the tool registry system and integrate with the DatabaseConnectionManager.

This module serves as the base for semantic-operator-enhanced variants:
- sql_tools_docetl.py (DocETL operators)
- sql_tools_lotus.py (Lotus operators)
- sql_tools_pz.py (Palimpzest operators)
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "extract_table_name_from_sql",
    "get_database_config",
    "SQLGenerationTool",
    "SQLExecutionTool",
    "SQLOptimizationTool",
    "SQLDebugTool",
]


# =============================================================================
# Utility Functions
# =============================================================================

def extract_table_name_from_sql(sql_query: str) -> Optional[str]:
    """Extract the primary table name from a SQL query.

    Args:
        sql_query: SQL query string

    Returns:
        Lowercase table name or None if not found
    """
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


def get_database_config(db_manager, instance_id: str, database_name: str, database_type: str):
    """Get database configuration with fallback to local dataclass.

    Args:
        db_manager: Database connection manager instance
        instance_id: Instance identifier
        database_name: Database name
        database_type: Type of database

    Returns:
        DatabaseConfig instance
    """
    try:
        from ..utils.database_connection_manager import DatabaseConfig
    except ImportError:
        @dataclass
        class DatabaseConfig:
            database_type: str
            instance_id: str
            db_name: str
            connection_params: Dict[str, Any] = None

    if hasattr(db_manager, 'get_database_config') and instance_id:
        return db_manager.get_database_config(instance_id, database_name, database_type)

    return DatabaseConfig(
        database_type=database_type,
        instance_id=instance_id or "default",
        db_name=database_name
    )


def _get_schema_info(db_manager, database_name: str, database_type: str,
                     instance_id: str) -> Optional[Dict]:
    """Helper to retrieve schema info from database manager.

    Args:
        db_manager: Database connection manager
        database_name: Target database name
        database_type: Type of database
        instance_id: Instance identifier

    Returns:
        Schema info dict or None
    """
    if not db_manager or not database_name:
        return None

    try:
        if database_type and instance_id:
            config = get_database_config(db_manager, instance_id, database_name, database_type)
            schema_result = db_manager.get_schema_info(config)
            if "error" not in schema_result:
                return schema_result
    except Exception as e:
        logger.warning(f"Could not retrieve schema info: {e}")
    return None


# =============================================================================
# SQL Generation Tool
# =============================================================================

class SQLGenerationTool:
    """Tool for generating SQL queries from natural language."""

    def __init__(self, llm_client=None, db_manager=None):
        self.llm_client = llm_client
        self.db_manager = db_manager

    def _extract_sql_from_response(self, response: str) -> str:
        """Extract clean SQL query from LLM response that may contain explanatory text."""
        response = response.replace('```sql', '').replace('```', '').strip()

        sql_keywords = ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']

        lines = response.split('\n')
        sql_lines = []
        in_sql = False

        for line in lines:
            line = line.strip()
            if not line:
                if in_sql:
                    sql_lines.append('')
                continue

            if any(line.upper().startswith(keyword) for keyword in sql_keywords):
                in_sql = True
                sql_lines.append(line)
            elif in_sql:
                if line.endswith(';'):
                    sql_lines.append(line)
                    break

                non_sql_patterns = [
                    line.startswith('Note:'),
                    line.startswith('Explanation:'),
                    line.startswith('This query'),
                    line.startswith('The above'),
                    line.startswith('Result:'),
                    line.startswith('Output:'),
                    line.startswith('##'),
                    line.startswith('**'),
                    ':' in line and not any(kw in line.upper() for kw in ['SELECT', 'CASE', 'CAST']),
                ]

                if any(non_sql_patterns):
                    break

                sql_lines.append(line)

        if sql_lines:
            sql_query = '\n'.join(sql_lines).strip()
            if ':' in sql_query:
                parts = sql_query.split(':', 1)
                if len(parts) > 1:
                    potential_sql = parts[1].strip()
                    if any(potential_sql.upper().startswith(kw) for kw in sql_keywords):
                        sql_query = potential_sql
            return sql_query

        return response

    def execute(self, natural_language_query: str, database_name: str = None,
                database_type: str = None, instance_id: str = None,
                schema_info: Dict = None, **kwargs) -> Dict[str, Any]:
        """Generate SQL query from natural language.

        Args:
            natural_language_query: The query in natural language
            database_name: Target database name
            database_type: Type of database (bird, spider2-lite, etc.)
            instance_id: Instance identifier for database connection
            schema_info: Database schema information (optional)

        Returns:
            Dictionary with status and results
        """
        try:
            if not natural_language_query:
                return {"status": "error", "error": "No natural language query provided"}

            if not schema_info:
                schema_info = _get_schema_info(
                    self.db_manager, database_name, database_type, instance_id
                )

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
                'CRITICAL Rules:',
                '- Use EXACT table and column names from schema (case-sensitive!)',
                '- Table names like "Match" must be "Match", NOT "matches" or "match"',
                '- Column names must match schema exactly',
                '- Only use tables and columns that exist in the schema above',
                '- Ensure correct SQL syntax for the database type',
                '- For JSON fields in BigQuery, use proper JSON functions',
                '- For SQLite JSON fields, use json_extract() function',
                '- Return ONLY the SQL query - no explanation, no text',
                '- If the request cannot be answered with the given schema, return "QUERY_IMPOSSIBLE"'
            ])

            prompt = "\n".join(prompt_parts)

            if self.llm_client and hasattr(self.llm_client, 'call_llm'):
                try:
                    generated_sql = self.llm_client.call_llm(
                        [{"role": "user", "content": prompt}],
                        category="sql_generate"
                    ).strip()
                    generated_sql = self._extract_sql_from_response(generated_sql)

                    if generated_sql == "QUERY_IMPOSSIBLE":
                        return {"status": "error", "error": "Query cannot be answered based on the provided schema."}

                except Exception as e:
                    logger.error(f"LLM SQL generation failed: {e}")
                    return {"status": "error", "error": f"LLM SQL generation failed: {str(e)}"}
            else:
                return {"status": "error", "error": "LLM client not available or does not have call_llm method"}

            if not generated_sql or generated_sql.strip() == "" or "ERROR" in generated_sql.upper():
                return {"status": "error", "error": "Failed to generate valid SQL query"}

            return {
                "status": "success",
                "results": {
                    "sql_query": generated_sql,
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


# =============================================================================
# SQL Execution Tool
# =============================================================================

class SQLExecutionTool:
    """Tool for executing SQL queries.

    This is the base class. Semantic-operator-enhanced versions inherit from this
    and override the execute() method to add post-processing.
    """

    def __init__(self, db_manager=None):
        self.db_manager = db_manager

    def _resolve_sql_query(self, sql_query: Optional[str], previous_results: Optional[Dict],
                           **kwargs) -> Optional[str]:
        """Resolve SQL query from direct parameter or previous tool results.

        Args:
            sql_query: Direct SQL query parameter
            previous_results: Results from previous tools

        Returns:
            Resolved SQL query or None
        """
        if sql_query:
            return sql_query

        if "sql_query" in kwargs:
            return kwargs["sql_query"]

        if previous_results and isinstance(previous_results, dict):
            # Check all possible tool names for SQL generation results
            for tool_name in ["generate_sql", "sql_generate", "generated_sql", "sql_gen"]:
                if tool_name in previous_results:
                    result = previous_results[tool_name]
                    if isinstance(result, dict):
                        if "sql_query" in result:
                            return result["sql_query"]
                        if "results" in result and isinstance(result["results"], dict):
                            if "sql_query" in result["results"]:
                                return result["results"]["sql_query"]
                        # Also check for 'query' key
                        if "query" in result:
                            return result["query"]
        return None

    def _execute_sql_query(self, sql_query: str, database_name: str,
                           database_type: str, instance_id: str) -> Dict[str, Any]:
        """Execute SQL and return raw results. Used by subclasses for base execution.

        Args:
            sql_query: SQL query to execute
            database_name: Target database
            database_type: Type of database
            instance_id: Instance identifier

        Returns:
            Dictionary with status, results, and columns
        """
        if not self.db_manager:
            return {"status": "error", "error": "Database manager is not available"}
        if not database_name:
            return {"status": "error", "error": "Database name is required"}
        if not database_type:
            return {"status": "error", "error": "Database type is required"}

        try:
            config = get_database_config(self.db_manager, instance_id, database_name, database_type)
            execution_result = self.db_manager.execute_sql(config, sql_query)

            if execution_result["status"] == "success":
                return {
                    "status": "success",
                    "query_results": execution_result["results"]["query_results"],
                    "total_results_count": execution_result["results"]["total_results_count"],
                    "columns": execution_result["results"].get("columns", [])
                }
            else:
                return {
                    "status": "error",
                    "error": execution_result.get('error', 'Unknown error')
                }
        except Exception as e:
            logger.error(f"Database execution failed: {e}")
            return {"status": "error", "error": str(e)}

    def execute(self, sql_query: str = None, database_name: str = None,
                database_type: str = None, instance_id: str = None,
                natural_language_query: str = None, previous_results: Dict = None,
                **kwargs) -> Dict[str, Any]:
        """Execute SQL query on database.

        Args:
            sql_query: SQL query to execute
            database_name: Target database
            database_type: Type of database (bird, spider2-lite, etc.)
            instance_id: Instance identifier for database connection
            natural_language_query: Original query (for generation if needed)
            previous_results: Results from previous tools

        Returns:
            Dictionary with status and results
        """
        try:
            sql_query = self._resolve_sql_query(sql_query, previous_results, **kwargs)

            if sql_query:
                logger.info(f"SQL query found for execution: {sql_query[:100]}...")
            else:
                logger.warning(f"No SQL query found. previous_results keys: "
                             f"{list(previous_results.keys()) if previous_results else 'None'}")
                return {"status": "error", "error": "No SQL query provided"}

            result = self._execute_sql_query(sql_query, database_name, database_type, instance_id)

            if result["status"] != "success":
                return {"status": "error", "error": f"Database execution failed: {result.get('error')}"}

            return {
                "status": "success",
                "results": {
                    "sql_query": sql_query,
                    "query_results": result["query_results"],
                    "total_results_count": result["total_results_count"],
                    "database_name": database_name,
                    "database_type": database_type,
                    "instance_id": instance_id,
                    "columns": result["columns"]
                }
            }

        except Exception as e:
            logger.error(f"SQL execution failed: {str(e)}")
            return {"status": "error", "error": str(e)}


# =============================================================================
# SQL Optimization Tool
# =============================================================================

class SQLOptimizationTool:
    """Tool for optimizing SQL queries."""

    def __init__(self, llm_client=None, db_manager=None):
        self.llm_client = llm_client
        self.db_manager = db_manager

    def execute(self, sql_query: str, db_name: str = None,
                database_type: str = None, instance_id: str = None,
                schema_info: Dict = None, **kwargs) -> Dict[str, Any]:
        """Optimize SQL query for better performance.

        Args:
            sql_query: SQL query to optimize
            db_name: Database name
            database_type: Type of database (bird, spider2-lite, etc.)
            instance_id: Instance identifier for database connection
            schema_info: Schema information for optimization context

        Returns:
            Dictionary with status and results
        """
        try:
            if not sql_query:
                return {"status": "error", "error": "No SQL query provided"}

            if not schema_info:
                schema_info = _get_schema_info(
                    self.db_manager, db_name, database_type, instance_id
                )

            if not (self.llm_client and hasattr(self.llm_client, 'call_llm')):
                return {"status": "error", "error": "LLM client not available for optimization"}

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
            try:
                optimized_sql = self.llm_client.call_llm(
                    [{"role": "user", "content": prompt}],
                    category="sql_optimize"
                ).strip()
                optimized_sql = optimized_sql.replace('```sql', '').replace('```', '').strip()
            except Exception as e:
                logger.error(f"LLM optimization failed: {e}")
                return {"status": "error", "error": f"LLM optimization failed: {str(e)}"}

            return {
                "status": "success",
                "results": {
                    "original_query": sql_query,
                    "optimized_query": optimized_sql,
                    "optimization_notes": "Query optimized using LLM analysis",
                    "database_type": database_type,
                    "instance_id": instance_id,
                    "schema_used": schema_info is not None
                }
            }

        except Exception as e:
            logger.error(f"SQL optimization failed: {str(e)}")
            return {"status": "error", "error": str(e)}


# =============================================================================
# SQL Debug Tool
# =============================================================================

class SQLDebugTool:
    """Tool for debugging SQL queries."""

    def __init__(self, llm_client=None, db_manager=None):
        self.llm_client = llm_client
        self.db_manager = db_manager

    def execute(self, failed_sql: str, error: str, database_name: str = None,
                database_type: str = None, instance_id: str = None,
                natural_language_query: str = None, schema_info: Dict = None,
                **kwargs) -> Dict[str, Any]:
        """Debug and fix SQL query errors.

        Args:
            failed_sql: The SQL that failed
            error: Error message
            database_name: Target database
            database_type: Type of database (bird, spider2-lite, etc.)
            instance_id: Instance identifier for database connection
            natural_language_query: Original query
            schema_info: Schema information

        Returns:
            Dictionary with status and results
        """
        try:
            if not failed_sql:
                return {"status": "error", "error": "No SQL query provided for debugging"}

            if not schema_info:
                schema_info = _get_schema_info(
                    self.db_manager, database_name, database_type, instance_id
                )

            if not (self.llm_client and hasattr(self.llm_client, 'call_llm')):
                return {"status": "error", "error": "LLM client not available for debugging"}

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
            try:
                corrected_sql = self.llm_client.call_llm(
                    [{"role": "user", "content": prompt}],
                    category="sql_debug"
                ).strip()
                corrected_sql = corrected_sql.replace('```sql', '').replace('```', '').strip()
            except Exception as e:
                logger.error(f"LLM debug failed: {e}")
                return {"status": "error", "error": f"LLM debug failed: {str(e)}"}

            if corrected_sql == "QUERY_UNFIXABLE":
                return {"status": "error", "error": "Unable to fix the SQL query based on the error message"}

            # Try to execute the corrected SQL
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
            logger.error(f"SQL debug failed: {str(e)}")
            return {"status": "error", "error": str(e)}
