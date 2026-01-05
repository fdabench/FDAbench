"""
Palimpzest-enhanced SQL tools for FDABench.

This module provides SQL tools with Palimpzest semantic operator integration.
It inherits base functionality from sql_tools and adds Palimpzest-specific processing.
"""

import logging
import os
import shutil
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import palimpzest as pz
import requests
from palimpzest.constants import Model

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
    "choose_pz_operator_with_llm",
    "apply_pz_operator",
    "track_pz_tokens_with_fallback",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Palimpzest-specific Helper Functions
# =============================================================================

def choose_pz_operator_with_llm(
    natural_language_query: str,
    available_operators: List[str],
    api_key: str
) -> Tuple[str, Dict[str, int]]:
    """Use LLM to choose the most appropriate Palimpzest operator.

    Args:
        natural_language_query: The user's query
        available_operators: List of available operators
        api_key: API key for OpenRouter

    Returns:
        Tuple of (operator_name, token_stats)
    """
    token_stats = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}

    try:
        supported_operators = ["sem_filter", "sem_add_columns", "sem_agg", "sem_topk", "sem_sim_join"]
        operators_to_consider = [op for op in supported_operators if op in available_operators]

        prompt = f"""
Given the natural language query and available Palimpzest operators, choose the most appropriate operator.

Natural Language Query: "{natural_language_query}"

Available Palimpzest Operators:
- sem_filter: Filter data based on natural language conditions (keeps rows that match criteria)
- sem_add_columns: Extract/compute new fields from existing data (adds new columns with extracted information)
- sem_agg: Aggregate data across rows (summarize, analyze, compute statistics across dataset)
- sem_topk: Find top K items based on criteria (ranking, best matches, highest/lowest values)
- sem_sim_join: Join datasets based on semantic similarity (fuzzy matching, entity resolution)

Rules:
1. Default to "sem_filter" for most queries involving data analysis, questions, or information extraction
2. Use "sem_filter" for any query asking "what", "who", "how many", "which", "when", "where"
3. Use "sem_filter" for comparisons, finding patterns, or selecting specific data subsets
4. Use "sem_add_columns" when explicitly asked to "add columns", "extract fields", or "compute new values"
5. Use "sem_agg" for aggregation tasks like "summarize all", "analyze overall", "total across"
6. Use "sem_topk" when explicitly asking for "top K", "best N", "highest/lowest K items"
7. Use "sem_sim_join" for joining or matching similar records across datasets

Return ONLY the operator name (e.g., "sem_filter"). No explanation needed.
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
        logger.info("Using fallback logic for operator selection")
        query_lower = natural_language_query.lower()
        if any(word in query_lower for word in ["extract", "add column", "compute", "generate field"]):
            return "sem_add_columns", token_stats
        elif any(word in query_lower for word in ["summarize all", "aggregate", "total across", "analyze overall"]):
            return "sem_agg", token_stats
        elif any(word in query_lower for word in ["top", "best", "highest", "lowest", "rank"]):
            return "sem_topk", token_stats
        elif any(word in query_lower for word in ["match", "join similar", "fuzzy", "entity resolution"]):
            return "sem_sim_join", token_stats
        return "sem_filter", token_stats

    except Exception as e:
        logger.error(f"Error choosing Palimpzest operator: {e}")
        return "sem_filter", token_stats


def apply_pz_operator(
    dataset,
    operator: str,
    natural_language_query: str,
    columns: List[str]
):
    """Apply Palimpzest operator to dataset.

    Args:
        dataset: Palimpzest dataset
        operator: Operator type
        natural_language_query: The user's query
        columns: List of column names

    Returns:
        Processed dataset
    """
    try:
        filter_prompt = f"The data is relevant to the query: {natural_language_query}"

        if operator == "sem_filter":
            result = dataset.sem_filter(filter_prompt)
        elif operator == "sem_add_columns":
            logger.info("sem_add_columns not fully implemented, using sem_filter")
            result = dataset.sem_filter(filter_prompt)
        else:
            logger.warning(f"Unknown operator: {operator}, using sem_filter")
            result = dataset.sem_filter(filter_prompt)

        return result

    except Exception as e:
        logger.error(f"Error applying Palimpzest operator {operator}: {e}")
        return dataset


def track_pz_tokens_with_fallback(
    output,
    token_tracker=None,
    operator: str = "unknown",
    fallback_input_tokens: int = 0,
    fallback_output_tokens: int = 0,
    data_rows_count: int = 0
) -> Dict[str, Any]:
    """Track token usage from Palimpzest output with fallback estimation.

    Args:
        output: Palimpzest output object
        token_tracker: Optional token tracker
        operator: Operator name
        fallback_input_tokens: Fallback input token count
        fallback_output_tokens: Fallback output token count
        data_rows_count: Number of data rows processed

    Returns:
        Dictionary with token statistics
    """
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    operation_details = []
    method_used = "none"

    try:
        native_stats_found = False

        # Try execution_stats attribute
        if hasattr(output, 'execution_stats'):
            stats = output.execution_stats
            logger.info("Found execution_stats - using native tracking")
            native_stats_found = True
            method_used = "native_execution_stats"

            for op_stats in stats.get('operator_stats', []):
                op_input = op_stats.get('total_input_tokens', 0)
                op_output = op_stats.get('total_output_tokens', 0)
                op_cost = op_stats.get('cost_per_record', 0)
                op_name = op_stats.get('op_name', 'unknown')

                total_input_tokens += op_input
                total_output_tokens += op_output
                total_cost += op_cost

                operation_details.append({
                    'op_name': op_name,
                    'input_tokens': op_input,
                    'output_tokens': op_output,
                    'cost': op_cost
                })

        # Try stats attribute
        if hasattr(output, 'stats') and output.stats:
            logger.info("Found stats attribute - using native tracking")
            native_stats_found = True
            method_used = "native_record_stats"

            for stat in output.stats:
                op_input = getattr(stat, 'total_input_tokens', 0)
                op_output = getattr(stat, 'total_output_tokens', 0)
                op_cost = getattr(stat, 'cost_per_record', 0.0)
                op_name = getattr(stat, 'op_name', 'unknown')

                total_input_tokens += op_input
                total_output_tokens += op_output
                total_cost += op_cost

                operation_details.append({
                    'op_name': op_name,
                    'input_tokens': op_input,
                    'output_tokens': op_output,
                    'cost': op_cost
                })

                logger.info(f"Op: {op_name}, In: {op_input}, Out: {op_output}, Cost: ${op_cost:.6f}")

                if token_tracker:
                    token_tracker.track_call(
                        category=f"palimpzest_{op_name}",
                        input_tokens=op_input,
                        output_tokens=op_output,
                        model="palimpzest_native",
                        cost=op_cost
                    )

        # Use fallback if no native stats
        if not native_stats_found or (total_input_tokens == 0 and total_output_tokens == 0):
            logger.warning("No native stats - using fallback estimation")
            method_used = "fallback_estimation"

            total_input_tokens = fallback_input_tokens
            total_output_tokens = fallback_output_tokens if fallback_output_tokens > 0 else 0

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

        logger.info(f"Palimpzest Token Summary (method: {method_used}):")
        logger.info(f"  Input: {total_input_tokens}, Output: {total_output_tokens}, Cost: ${total_cost:.6f}")

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
        logger.error(f"Token tracking error, using fallback: {e}")

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


# =============================================================================
# Palimpzest-enhanced SQL Execution Tool
# =============================================================================

class SQLExecutionTool(_BaseSQLExecutionTool):
    """SQL execution tool enhanced with Palimpzest semantic operators.

    Inherits base SQL execution and adds Palimpzest post-processing for
    semantic filtering and column extraction.
    """

    AVAILABLE_OPERATORS = ["sem_filter", "sem_add_columns", "sem_agg", "sem_topk", "sem_sim_join"]

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
        """Execute SQL with optional Palimpzest semantic processing.

        Args:
            sql_query: SQL query to execute
            database_name: Target database
            database_type: Type of database
            instance_id: Instance identifier
            natural_language_query: Original NL query (triggers Palimpzest if provided)
            previous_results: Results from previous tools

        Returns:
            Dictionary with status and results
        """
        try:
            # Resolve SQL query
            sql_query = self._resolve_sql_query(sql_query, previous_results, **kwargs)
            if not sql_query:
                return {"status": "error", "error": "No SQL query provided"}

            # For Palimpzest, simplify complex queries to SELECT *
            table_name = extract_table_name_from_sql(sql_query)
            original_sql = sql_query
            if table_name:
                sql_query = f"SELECT * FROM {table_name}"
                logger.info(f"Simplified SQL for Palimpzest: {sql_query}")

            # Execute base SQL
            result = self._execute_sql_query(sql_query, database_name, database_type, instance_id)
            if result["status"] != "success":
                return {"status": "error", "error": f"Database execution failed: {result.get('error')}"}

            query_results = result["query_results"]
            columns = result["columns"]

            # Apply Palimpzest processing if NL query provided
            if natural_language_query and query_results and columns:
                try:
                    df = pd.DataFrame(query_results, columns=columns)

                    # Setup temp directories
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    data_dir = os.path.join(project_root, "examples", "data")
                    os.makedirs(data_dir, exist_ok=True)
                    csv_file_path = os.path.join(data_dir, "temp_data.csv")
                    output_dir = os.path.join(data_dir, "pz-data")

                    # Clean up existing files
                    if os.path.exists(csv_file_path):
                        os.remove(csv_file_path)
                    if os.path.exists(output_dir):
                        shutil.rmtree(output_dir)
                    os.makedirs(output_dir, exist_ok=True)

                    # Write data files
                    df.to_csv(csv_file_path, index=False)
                    for i, row in df.iterrows():
                        filepath = os.path.join(output_dir, f"pz_temp{i+1}.txt")
                        content = "\n".join([f"{col}: {row[col]}" for col in df.columns]) + "\n"
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content)

                    logger.info(f"Created {len(df)} text files in {output_dir}")

                    # Create dataset and add columns
                    dataset = pz.Dataset(output_dir)

                    def infer_pz_type(dtype):
                        if np.issubdtype(dtype, np.number):
                            return int
                        return str

                    pz_columns = [
                        {"name": col, "type": infer_pz_type(df[col].dtype), "desc": f"Column {col}"}
                        for col in columns
                    ]
                    dataset = dataset.sem_add_columns(pz_columns)

                    # Choose operator
                    api_key = os.environ.get('OPENROUTER_API_KEY')
                    if not api_key:
                        raise ValueError("Missing OPENROUTER_API_KEY")

                    chosen_operator, token_stats = choose_pz_operator_with_llm(
                        natural_language_query, self.AVAILABLE_OPERATORS, api_key
                    )

                    if self.token_tracker:
                        self.token_tracker.track_call(
                            category="pz_operator_selection",
                            input_tokens=token_stats['input_tokens'],
                            output_tokens=token_stats['output_tokens'],
                            model="deepseek/deepseek-chat-v3-0324",
                            cost=None
                        )

                    logger.info(f"Palimpzest operator: {chosen_operator}")

                    # Apply operator
                    processed_dataset = apply_pz_operator(
                        dataset, chosen_operator, natural_language_query, columns
                    )

                    # Run pipeline
                    config = pz.QueryProcessorConfig(
                        policy=pz.MinCost(),
                        available_models=[Model.DEEPSEEK_V3],
                        verbose=True
                    )
                    output = processed_dataset.run(config)

                    # Track tokens
                    pz_token_stats = track_pz_tokens_with_fallback(
                        output=output,
                        token_tracker=self.token_tracker,
                        operator=chosen_operator,
                        fallback_input_tokens=0,
                        fallback_output_tokens=0,
                        data_rows_count=len(df)
                    )

                    # Get results
                    filtered_df = output.to_df(cols=columns)
                    filtered_results = filtered_df.to_dict('records')

                    logger.info(f"Palimpzest processed {len(filtered_results)} results (original: {len(df)})")

                    return {
                        "status": "success",
                        "results": {
                            "sql_query": original_sql,
                            "query_results": filtered_results,
                            "total_results_count": len(filtered_results),
                            "original_count": result["total_results_count"],
                            "database_name": database_name,
                            "database_type": database_type,
                            "instance_id": instance_id,
                            "columns": columns,
                            "pz_operator_used": chosen_operator,
                            "filtered_by_palimpzest": True,
                            "token_stats": {
                                "total_input_tokens": pz_token_stats.get('total_input_tokens', 0),
                                "total_output_tokens": pz_token_stats.get('total_output_tokens', 0),
                                "total_cost": pz_token_stats.get('total_cost', 0.0),
                                "method_used": pz_token_stats.get('method_used', 'unknown'),
                                "native_stats_available": pz_token_stats.get('native_stats_available', False),
                                "operation_details": pz_token_stats.get('operation_details', [])
                            }
                        }
                    }

                except Exception as e:
                    logger.error(f"Palimpzest processing failed, returning original: {e}")

            # Return unprocessed results
            return {
                "status": "success",
                "results": {
                    "sql_query": original_sql,
                    "query_results": query_results,
                    "total_results_count": result["total_results_count"],
                    "database_name": database_name,
                    "database_type": database_type,
                    "instance_id": instance_id,
                    "columns": columns,
                    "filtered_by_palimpzest": False
                }
            }

        except Exception as e:
            logger.error(f"SQL execution failed: {str(e)}")
            return {"status": "error", "error": str(e)}
