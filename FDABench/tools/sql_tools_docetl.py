"""
DocETL-enhanced SQL tools for FDABench.

This module provides SQL tools with DocETL semantic operator integration.
It inherits base functionality from sql_tools and adds DocETL-specific processing.
"""

import json
import logging
import os
import tempfile
from typing import Dict, List, Any, Tuple

import pandas as pd
import requests
import yaml
from docetl import DSLRunner

from .sql_tools import (
    SQLGenerationTool,
    SQLOptimizationTool,
    SQLDebugTool,
    SQLExecutionTool as _BaseSQLExecutionTool,
    extract_table_name_from_sql,
    get_database_config,
)

__all__ = [
    "SQLGenerationTool",
    "SQLExecutionTool",
    "SQLOptimizationTool",
    "SQLDebugTool",
    "choose_docetl_operator_with_llm",
    "apply_docetl_operator",
]

logger = logging.getLogger(__name__)


# =============================================================================
# DocETL-specific Helper Functions
# =============================================================================

def choose_docetl_operator_with_llm(
    natural_language_query: str,
    available_operators: List[str],
    api_key: str
) -> Tuple[str, Dict[str, int]]:
    """Use LLM to choose the most appropriate DocETL operator.

    Args:
        natural_language_query: The user's query
        available_operators: List of available operators
        api_key: API key for OpenRouter

    Returns:
        Tuple of (operator_name, token_stats)
    """
    token_stats = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}

    try:
        supported_ops = ["map", "filter", "reduce", "resolve", "gather", "unnest"]
        operators_to_consider = [op for op in supported_ops if op in available_operators]

        prompt = f"""
Given the natural language query and available DocETL operators, choose the most appropriate operator.

Natural Language Query: "{natural_language_query}"

Available DocETL Operators:
- map: Transform/extract information from each row (adds new columns, extracts fields, computes derived values)
- filter: Filter rows based on conditions (keeps rows that match criteria, removes irrelevant data)
- reduce: Aggregate data across multiple rows (summarize, combine, compute totals/averages)
- resolve: Deduplicate or merge similar records (entity resolution, data cleaning)
- gather: Collect and organize information from multiple sources
- unnest: Flatten nested data structures into individual rows

Rules:
1. Default to "filter" for most queries involving data analysis, questions, or information extraction
2. Use "filter" for any query asking "what", "who", "how many", "which", "when", "where"
3. Use "filter" for comparisons, finding patterns, or selecting specific data subsets
4. Use "map" when explicitly asked to "add columns", "transform data", "extract fields", or "compute new values"
5. Use "reduce" for aggregation tasks like "summarize", "total", "average", "combine all"
6. Use "resolve" for deduplication or entity matching tasks
7. Use "gather" for collecting information from multiple fields or sources
8. Use "unnest" when dealing with nested/array data that needs flattening

Return ONLY the operator name (e.g., "filter"). No explanation needed.
"""
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

            if 'usage' in result:
                usage = result['usage']
                token_stats['input_tokens'] = usage.get('prompt_tokens', 0)
                token_stats['output_tokens'] = usage.get('completion_tokens', 0)
                token_stats['total_tokens'] = usage.get('total_tokens',
                    token_stats['input_tokens'] + token_stats['output_tokens'])
        else:
            logger.error(f"OpenRouter API failed: {response.status_code}: {response.text}")
            chosen_operator = None

        if chosen_operator and chosen_operator in operators_to_consider:
            return chosen_operator, token_stats

        # Fallback logic
        logger.warning(f"Invalid operator choice: {chosen_operator}, using fallback")
        query_lower = natural_language_query.lower()
        if any(word in query_lower for word in ["filter", "select only", "remove", "exclude", "show only"]):
            return "filter", token_stats
        elif any(word in query_lower for word in ["summarize", "aggregate", "total", "average", "combine"]):
            return "reduce", token_stats
        elif any(word in query_lower for word in ["deduplicate", "merge", "resolve", "match entities"]):
            return "resolve", token_stats
        elif any(word in query_lower for word in ["flatten", "unnest", "expand"]):
            return "unnest", token_stats
        return "filter", token_stats  # Default to filter for analysis

    except Exception as e:
        logger.error(f"Error choosing DocETL operator: {e}")
        return "map", token_stats


def apply_docetl_operator(
    df: pd.DataFrame,
    operator: str,
    natural_language_query: str,
    columns: List[str],
    api_key: str,
    token_tracker=None
) -> pd.DataFrame:
    """Apply DocETL operator to DataFrame.

    Args:
        df: Input DataFrame
        operator: Operator type ("map" or "filter")
        natural_language_query: The user's query
        columns: List of column names
        api_key: API key for OpenRouter
        token_tracker: Optional token tracker

    Returns:
        Processed DataFrame
    """
    data_file_path = None
    output_file_path = None
    pipeline_file_path = None

    try:
        # Create temp files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(df.to_dict('records'), f, indent=2)
            data_file_path = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file_path = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            pipeline_file_path = f.name

        # Build config based on operator
        if operator == "map":
            operation_config = {
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
                "output": {"schema": {"analysis_result": "str"}}
            }
        else:  # filter
            operation_config = {
                "name": "filter_data",
                "type": "filter",
                "prompt": f"""
Evaluate this data record against the following criteria: "{natural_language_query}"

Record data: {{{{ input }}}}

Return true if this record matches the criteria or should be included based on the query.
Return false if this record should be filtered out or doesn't match the criteria.
""".strip(),
                "output": {"schema": {"matches_criteria": "boolean"}}
            }

        config = {
            "datasets": {
                "input_data": {"path": data_file_path, "type": "file"}
            },
            "default_model": "openrouter/deepseek/deepseek-chat-v3-0324",
            "operations": [operation_config],
            "pipeline": {
                "steps": [{
                    "name": f"{operator}_data_step",
                    "input": "input_data",
                    "operations": [operation_config["name"]]
                }],
                "output": {"type": "file", "path": output_file_path}
            }
        }

        with open(pipeline_file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        os.environ['OPENROUTER_API_KEY'] = api_key

        logger.info(f"Running DocETL {operator} operation...")
        runner = DSLRunner.from_yaml(pipeline_file_path)
        runner.load_run_save()

        with open(output_file_path, 'r') as f:
            results = json.load(f)

        if results:
            result_df = pd.DataFrame(results)
            logger.info(f"DocETL {operator} completed. Results shape: {result_df.shape}")
            return result_df

        logger.warning(f"DocETL {operator} returned no results")
        return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error applying DocETL {operator}: {e}")
        return df

    finally:
        for path in [data_file_path, output_file_path, pipeline_file_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception as e:
                    logger.warning(f"Could not clean up {path}: {e}")


# =============================================================================
# DocETL-enhanced SQL Execution Tool
# =============================================================================

class SQLExecutionTool(_BaseSQLExecutionTool):
    """SQL execution tool enhanced with DocETL semantic operators.

    Inherits base SQL execution and adds DocETL post-processing for
    semantic filtering and transformation of query results.
    """

    AVAILABLE_OPERATORS = ["map", "filter", "reduce", "resolve", "gather", "unnest"]

    def __init__(self, db_manager=None, token_tracker=None):
        super().__init__(db_manager)
        self.token_tracker = token_tracker

    def execute(
        self,
        sql_query: str = None,
        database_name: str = None,
        database_type: str = None,
        instance_id: str = None,
        natural_language_query: str = None,
        previous_results: Dict = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute SQL with optional DocETL semantic processing.

        Args:
            sql_query: SQL query to execute
            database_name: Target database
            database_type: Type of database
            instance_id: Instance identifier
            natural_language_query: Original NL query (triggers DocETL if provided)
            previous_results: Results from previous tools

        Returns:
            Dictionary with status and results
        """
        try:
            # Resolve SQL query
            sql_query = self._resolve_sql_query(sql_query, previous_results, **kwargs)
            if not sql_query:
                return {"status": "error", "error": "No SQL query provided"}

            # For DocETL, simplify complex queries to SELECT *
            table_name = extract_table_name_from_sql(sql_query)
            original_sql = sql_query
            if table_name:
                sql_query = f"SELECT * FROM {table_name}"
                logger.info(f"Simplified SQL for DocETL: {sql_query}")

            # Execute base SQL
            result = self._execute_sql_query(sql_query, database_name, database_type, instance_id)
            if result["status"] != "success":
                return {"status": "error", "error": f"Database execution failed: {result.get('error')}"}

            query_results = result["query_results"]
            columns = result["columns"]

            # Apply DocETL processing if NL query provided
            if natural_language_query and query_results and columns:
                try:
                    api_key = os.environ.get('OPENROUTER_API_KEY')
                    if not api_key:
                        raise ValueError("Missing OPENROUTER_API_KEY")

                    df = pd.DataFrame(query_results, columns=columns)
                    logger.info(f"Created DataFrame: {df.shape}")

                    chosen_operator, token_stats = choose_docetl_operator_with_llm(
                        natural_language_query, self.AVAILABLE_OPERATORS, api_key
                    )
                    logger.info(f"DocETL operator: {chosen_operator}")

                    processed_df = apply_docetl_operator(
                        df, chosen_operator, natural_language_query, columns, api_key, self.token_tracker
                    )

                    processed_results = (
                        processed_df.to_dict('records')
                        if isinstance(processed_df, pd.DataFrame)
                        else processed_df
                    )

                    return {
                        "status": "success",
                        "results": {
                            "original_sql_query": original_sql,
                            "executed_sql_query": sql_query,
                            "query_results": processed_results,
                            "total_results_count": len(processed_results) if isinstance(processed_results, list) else 1,
                            "original_count": result["total_results_count"],
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
                    logger.error(f"DocETL processing failed, returning original: {e}")

            # Return unprocessed results
            return {
                "status": "success",
                "results": {
                    "original_sql_query": original_sql,
                    "executed_sql_query": sql_query,
                    "query_results": query_results,
                    "total_results_count": result["total_results_count"],
                    "database_name": database_name,
                    "database_type": database_type,
                    "instance_id": instance_id,
                    "columns": columns,
                    "table_name": table_name,
                    "processed_by_docetl": False
                }
            }

        except Exception as e:
            logger.error(f"SQL execution failed: {str(e)}")
            return {"status": "error", "error": str(e)}
