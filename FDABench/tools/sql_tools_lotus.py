"""
Lotus-enhanced SQL tools for FDABench.

This module provides SQL tools with Lotus semantic operator integration.
It inherits base functionality from sql_tools and adds Lotus-specific processing.
"""

import logging
import os
from typing import Dict, List, Any, Tuple

import lotus
import pandas as pd
import requests
from lotus.models import LM

from .sql_tools import (
    SQLGenerationTool,
    SQLOptimizationTool,
    SQLDebugTool,
    SQLExecutionTool as _BaseSQLExecutionTool,
    extract_table_name_from_sql,
)

__all__ = [
    "SQLGenerationTool",
    "SQLExecutionTool",
    "SQLOptimizationTool",
    "SQLDebugTool",
    "choose_lotus_operator_with_openrouter",
    "apply_lotus_operator",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Lotus-specific Helper Functions
# =============================================================================

def choose_lotus_operator_with_openrouter(
    natural_language_query: str,
    available_operators: List[str],
    api_key: str
) -> Tuple[str, Dict[str, int]]:
    """Use LLM to choose the most appropriate Lotus operator.

    Args:
        natural_language_query: The user's query
        available_operators: List of available operators
        api_key: API key for OpenRouter

    Returns:
        Tuple of (operator_name, token_stats)
    """
    token_stats = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}

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
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/FDABench",
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
        logger.warning(f"Invalid operator: {chosen_operator}, using fallback")
        query_lower = natural_language_query.lower()
        if any(word in query_lower for word in ["filter", "where", "select", "find records"]):
            return "sem_filter", token_stats
        elif any(word in query_lower for word in ["analyze", "summary", "who is", "longest", "report"]):
            return "sem_agg", token_stats
        elif any(word in query_lower for word in ["top", "best", "highest", "maximum"]):
            return "sem_topk", token_stats
        return "sem_agg", token_stats

    except Exception as e:
        logger.error(f"Error choosing Lotus operator: {e}")
        return "sem_filter", token_stats


def apply_lotus_operator(
    df: pd.DataFrame,
    operator: str,
    natural_language_query: str,
    columns: List[str],
    lm=None,
    token_tracker=None
) -> pd.DataFrame:
    """Apply Lotus semantic operator to DataFrame.

    Args:
        df: Input DataFrame
        operator: Operator type
        natural_language_query: The user's query
        columns: List of column names
        lm: Lotus LM instance
        token_tracker: Optional token tracker

    Returns:
        Processed DataFrame or result
    """
    try:
        data_rows_count = len(df)

        # Build instruction with column references
        column_refs = ", ".join([f"{{{col}}}" for col in columns])
        user_instruction = f"Based on {column_refs}, {natural_language_query}"

        estimated_input_tokens = lm.count_tokens(user_instruction + "\n" + df.to_string()) if lm else 0

        logger.info(f"Applying {operator} with instruction: {user_instruction}")

        initial_physical_tokens = lm.stats.physical_usage.total_tokens if lm else 0

        # Execute operator
        if operator == "sem_filter":
            result = df.sem_filter(user_instruction)
        elif operator == "sem_agg":
            result = df.sem_agg(user_instruction, all_cols=True)
        elif operator == "sem_map":
            result = df.sem_map(user_instruction)
        elif operator == "sem_search" and columns:
            result = df.sem_search(columns[0], user_instruction, K=min(100, len(df)))
        elif operator == "sem_topk":
            result = df.sem_topk(user_instruction, K=10)
        else:
            logger.warning(f"Unsupported operator: {operator}, using sem_agg")
            result = df.sem_agg(user_instruction, all_cols=True)

        # Track tokens
        if lm and token_tracker:
            actual_total_tokens = lm.stats.physical_usage.total_tokens
            estimated_output_tokens = max(0, actual_total_tokens - initial_physical_tokens - estimated_input_tokens)

            token_tracker.track_call(
                category=f"lotus_{operator}_batch",
                input_tokens=estimated_input_tokens,
                output_tokens=estimated_output_tokens,
                model="lotus_semantic_operation",
                cost=lm.stats.physical_usage.total_cost if lm else None
            )
            logger.info(f"Tracked lotus {operator} batch for {data_rows_count} rows")
            logger.info(f"Lotus usage - Physical: {lm.stats.physical_usage.total_tokens}, "
                       f"Virtual: {lm.stats.virtual_usage.total_tokens}")

        return result

    except Exception as e:
        logger.error(f"Error applying Lotus operator {operator}: {e}")
        return df


# =============================================================================
# Lotus-enhanced SQL Execution Tool
# =============================================================================

class SQLExecutionTool(_BaseSQLExecutionTool):
    """SQL execution tool enhanced with Lotus semantic operators.

    Inherits base SQL execution and adds Lotus post-processing for
    semantic filtering, aggregation, and transformation of query results.
    """

    AVAILABLE_OPERATORS = [
        "sem_filter", "sem_agg", "sem_map", "sem_search", "sem_topk",
        "sem_join", "sem_extract", "sem_cluster_by", "sem_dedup"
    ]

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
        """Execute SQL with optional Lotus semantic processing.

        Args:
            sql_query: SQL query to execute
            database_name: Target database
            database_type: Type of database
            instance_id: Instance identifier
            natural_language_query: Original NL query (triggers Lotus if provided)
            previous_results: Results from previous tools

        Returns:
            Dictionary with status and results
        """
        try:
            # Resolve SQL query
            sql_query = self._resolve_sql_query(sql_query, previous_results, **kwargs)
            if not sql_query:
                return {"status": "error", "error": "No SQL query provided"}

            # For Lotus, simplify complex queries to SELECT *
            table_name = extract_table_name_from_sql(sql_query)
            original_sql = sql_query
            if table_name:
                sql_query = f"SELECT * FROM {table_name}"
                logger.info(f"Simplified SQL for Lotus: {sql_query}")

            # Execute base SQL
            result = self._execute_sql_query(sql_query, database_name, database_type, instance_id)
            if result["status"] != "success":
                return {"status": "error", "error": f"Database execution failed: {result.get('error')}"}

            query_results = result["query_results"]
            columns = result["columns"]

            # Apply Lotus processing if NL query provided
            if natural_language_query and query_results and columns:
                try:
                    api_key = os.environ.get('OPENROUTER_API_KEY')
                    if not api_key:
                        raise ValueError("Missing OPENROUTER_API_KEY")

                    # Configure Lotus LM
                    lm = LM(
                        model="openrouter/deepseek/deepseek-chat-v3-0324",
                        api_key=api_key,
                        api_base="https://openrouter.ai/api/v1"
                    )
                    lotus.settings.configure(lm=lm)
                    logger.info("Lotus LM configured successfully")

                    df = pd.DataFrame(query_results, columns=columns)
                    logger.info(f"Created DataFrame: {df.shape}")

                    # Choose operator
                    chosen_operator, token_stats = choose_lotus_operator_with_openrouter(
                        natural_language_query, self.AVAILABLE_OPERATORS, api_key
                    )

                    if self.token_tracker:
                        self.token_tracker.track_call(
                            category=f"lotus_{chosen_operator}_selection",
                            input_tokens=token_stats['input_tokens'],
                            output_tokens=token_stats['output_tokens'],
                            model="deepseek/deepseek-chat-v3-0324",
                            cost=None
                        )

                    logger.info(f"Lotus operator: {chosen_operator}")

                    # Apply operator
                    processed_df = apply_lotus_operator(
                        df, chosen_operator, natural_language_query, columns, lm, self.token_tracker
                    )

                    if lm:
                        lm.print_total_usage()

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
                            "lotus_operator_used": chosen_operator,
                            "processed_by_lotus": True
                        }
                    }

                except Exception as e:
                    logger.error(f"Lotus processing failed, returning original: {e}")

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
                    "processed_by_lotus": False
                }
            }

        except Exception as e:
            logger.error(f"SQL execution failed: {str(e)}")
            return {"status": "error", "error": str(e)}
