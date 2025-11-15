#!/usr/bin/env python3
"""
Test Script: DeepAnalyze Adapter

This script tests DeepAnalyze Adapter performance on FDABench datasets.
The adapter integrates the DeepAnalyze model into FDABench's agent framework.
Based on test_new_tooluse_agent.py structure, adapted for DeepAnalyze model.
"""

import os
import sys
import logging
import time
import argparse
import json
import pandas as pd

# # Add package path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# # Import DeepAnalyze Adapter
from deepanalyze_adapter import DeepAnalyzeAdapter

# # Import FDABench components
from FDABench.evaluation import ReportEvaluator
from FDABench.utils.test_utils import (
    load_test_data,
    generate_task_name,
    create_query_row,
    store_in_duckdb,
    evaluate_agent_result,
    validate_agent_basic,
    print_summary,
    _calculate_tool_success_rate
)

# # Import tools
from FDABench.tools.sql_tools import SQLGenerationTool, SQLExecutionTool
from FDABench.tools.schema_tools import SchemaInfoTool
from FDABench.tools.search_tools import WebSearchTool, VectorSearchTool
from FDABench.tools.file_tools import FileSystemTool
from FDABench.tools.context_tools import ContextHistoryTool

# # Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.environ.get('DEEPANALYZE_MODEL_PATH')
DEFAULT_API_URL = os.environ.get('DEEPANALYZE_API_URL', 'http://localhost:8000/v1/chat/completions')
DEFAULT_MAX_AGENT_ROUNDS = 10
DEFAULT_MAX_DEEPANALYZE_ROUNDS = 30


def generate_duckdb_filename(model_name):
    """Generate DuckDB filename"""
    if not model_name:
        model_name = "deepanalyze"
    clean_model = model_name.replace('/', '_').replace('-', '_')
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, f"test_query_results_{clean_model}.duckdb")


def setup_tools(adapter):
    """Register all necessary tools for the DeepAnalyze Adapter"""

    db_manager = getattr(adapter, 'db_manager', None)
    if db_manager is None:
        logger.warning("Database manager is not available")
    else:
        logger.info("Database manager is available")

    # SQL tools (Note: DeepAnalyze uses CSV, these tools are mainly for other scenarios)
    adapter.register_tool(
        "get_schema_info",
        SchemaInfoTool(db_manager),
        category="database",
        description="Get database schema information"
    )

    # Search tools
    openrouter_api_key = os.environ.get('OPENROUTER_API_KEY')
    adapter.register_tool(
        "web_context_search",
        WebSearchTool(api_key=openrouter_api_key),
        category="search",
        description="Search web for context"
    )
    adapter.register_tool(
        "perplexity_search",
        WebSearchTool(api_key=openrouter_api_key),
        category="search",
        description="Perplexity web search"
    )
    adapter.register_tool(
        "vectorDB_search",
        VectorSearchTool(),
        category="search",
        description="Search vector database"
    )

    # File system tools
    adapter.register_tool(
        "file_system",
        FileSystemTool(),
        category="file",
        description="Search file system"
    )

    # Context tools
    adapter.register_tool(
        "context_history",
        ContextHistoryTool(),
        category="context",
        description="Manage context history"
    )

    logger.info(f"Registered {len(adapter.list_tools())} tools: {adapter.list_tools()}")


def load_test_cases(input_path=None, dataset_index=0):
    """
    Load test cases either from a JSON/JSONL file or HuggingFace dataset.

    Args:
        input_path: Optional path to JSON/JSONL file.
        dataset_index: Index for FDABench-Lite sample when no input file is given.

    Returns:
        List of test case dictionaries.
    """
    if input_path:
        logger.info(f"Loading test data from file: {input_path}")
        test_data_list = []

        if input_path.endswith('.jsonl'):
            with open(input_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        test_data_list.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.error(f"Line {line_num} rows JSON parse failed: {e}")
        else:
            with open(input_path, 'r', encoding='utf-8') as f:
                test_data_list = [json.load(f)]

        logger.info(f"Successfully loaded {len(test_data_list)} test data entries")
        return test_data_list

    logger.info(f"Using FDABench-Lite sample index: {dataset_index}")
    return [load_test_data(index=dataset_index)]


def test_deepanalyze_adapter(
    test_data=None,
    model_path=None,
    api_url=None,
    max_agent_rounds=DEFAULT_MAX_AGENT_ROUNDS,
    max_deepanalyze_rounds=DEFAULT_MAX_DEEPANALYZE_ROUNDS,
    skip_eval=False,
    duckdb_path=None,
    api_key=None
):
    """
    Test DeepAnalyze Adapter

    Args:
        test_data: Test data (optional)
        model_path: DeepAnalyze model path
        api_url: vLLM API URL
        max_agent_rounds: Max FDABench adapter outer loop rounds
        max_deepanalyze_rounds: Max internal DeepAnalyze reasoning rounds
        skip_eval: Disable ROUGE/F1 evaluation if True
        duckdb_path: Optional override for DuckDB output file
        api_key: Optional API key passed into BaseAgent
    """

    logger.info("=" * 80)
    logger.info("Test DeepAnalyze Adapter")
    logger.info("=" * 80)

    api_key = api_key or os.environ.get('OPENROUTER_API_KEY')
    model_path = model_path or DEFAULT_MODEL_PATH
    api_url = api_url or DEFAULT_API_URL
    duckdb_path = duckdb_path or generate_duckdb_filename("deepanalyze")

    if not model_path:
        raise ValueError(
            "Model path not provided. Pass --model_path or set DEEPANALYZE_MODEL_PATH."
        )

    if not api_url:
        raise ValueError(
            "API URL not provided. Pass --api_url or set DEEPANALYZE_API_URL."
        )

    logger.info("Configuration:")
    logger.info(f"  Model path: {model_path}")
    logger.info(f"  API URL: {api_url}")
    logger.info(f"  Max adapter rounds: {max_agent_rounds}")
    logger.info(f"  Max DeepAnalyze rounds: {max_deepanalyze_rounds}")
    logger.info(f"  Evaluation: {'enabled' if not skip_eval else 'skipped'}")

    adapter = DeepAnalyzeAdapter(
        model_path=model_path,
        api_url=api_url,
        max_agent_rounds=max_agent_rounds,
        max_deepanalyze_rounds=max_deepanalyze_rounds,
        api_key=api_key
    )

    setup_tools(adapter)

    try:
        if test_data is None:
            test_data = load_test_data()

        logger.info(f"Loading test case: {test_data.get('task_id', 'unknown')} - {test_data.get('instance_id', 'unknown')}")
        logger.info(f"Database: {test_data.get('db', 'unknown')} ({test_data.get('database_type', 'unknown')})")
        logger.info(f"Question type: {test_data.get('question_type', 'unknown')}")
        logger.info(f"Available tools: {test_data.get('tools_available', [])}")

        task_name = generate_task_name(test_data, 'deepanalyze')
        logger.info(f"Task name: {task_name}")

        evaluator = None
        evaluation_enabled = not skip_eval
        if evaluation_enabled:
            try:
                evaluator = ReportEvaluator()
                logger.info("ReportEvaluator initialized successfully")
            except Exception as e:
                logger.warning(f"Unable to initialize evaluator: {e}")
                evaluation_enabled = False

        ground_truth_report = test_data.get('ground_truth_report', '')

        logger.info("\n" + "=" * 80)
        logger.info("Starting query processing...")
        logger.info("=" * 80)

        start_time = time.time()
        result = adapter.process_query_from_json(test_data)
        processing_time = time.time() - start_time

        logger.info(f"\nProcessing complete, time taken: {processing_time:.2f}s")

        evaluation_scores, tool_recall_metrics = evaluate_agent_result(
            result, test_data, evaluator, ground_truth_report, evaluation_enabled, logger
        )

        query_row = create_query_row(
            result=result,
            test_data=test_data,
            evaluation_scores=evaluation_scores,
            tool_recall_metrics=tool_recall_metrics,
            task_name=task_name,
            pattern='deepanalyze',
            ground_truth_report=ground_truth_report
        )

        df = pd.DataFrame([query_row])

        from FDABench.utils.test_utils import ALL_COLUMNS
        if 'completed_tools' in df.columns:
            df['completed_tools'] = df['completed_tools'].apply(
                lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
            )
        for col in ALL_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[ALL_COLUMNS]

        logger.info(f"Creating DataFrame: {len(df)} rows x {len(df.columns)} columns")

        store_success = store_in_duckdb(df, duckdb_path, task_name)

        logger.info("\n" + "=" * 80)
        logger.info("Processing results:")
        logger.info("=" * 80)
        logger.info(f"Instance ID: {result.get('instance_id', 'N/A')}")
        logger.info(f"Database: {result.get('db_name', 'N/A')} ({result.get('database_type', 'N/A')})")
        logger.info(f"Status: {'SUCCESS' if 'error' not in result else 'ERROR'}")
        logger.info(f"Processing time: {processing_time:.2f}s")

        if 'error' in result:
            logger.error(f"Error: {result['error']}")
        else:
            metrics = result.get('metrics', {})
            logger.info(f"\nTotal latency: {metrics.get('latency_seconds', 0):.2f}s")
            logger.info(f"Tool execution time: {metrics.get('total_tool_execution_time', 0):.2f}s")
            logger.info(f"External latency: {metrics.get('external_latency', 0):.2f}s")
            logger.info(f"Executed tools: {metrics.get('tools_executed', [])}")
            logger.info(f"Success rate: {metrics.get('success_rate', 0):.2%}")
            logger.info(f"Agent rounds: {metrics.get('total_agent_rounds', 0)}")

            token_summary = metrics.get('token_summary', {})
            logger.info(f"Total tokens: {token_summary.get('total_tokens', 0)}")

            if result.get('report'):
                logger.info(f"\nReport:\n{result['report'][:500]}...")
            elif result.get('selected_answer'):
                logger.info(f"\nSelected answer: {result['selected_answer']}")
            elif result.get('response'):
                logger.info(f"\nResponse: {result['response'][:200]}...")

            if evaluation_scores:
                logger.info(f"\nROUGE-1: {evaluation_scores.get('rouge1', 0):.4f}")
                logger.info(f"ROUGE-2: {evaluation_scores.get('rouge2', 0):.4f}")
                logger.info(f"ROUGE-L: {evaluation_scores.get('rougeL', 0):.4f}")
                logger.info(f"F1: {evaluation_scores.get('f1', 0):.4f}")

            if tool_recall_metrics:
                logger.info(f"\nTool Recall: {tool_recall_metrics.get('tool_recall', 0):.3f}")
                logger.info(f"Tool Precision: {tool_recall_metrics.get('tool_precision', 0):.3f}")
                logger.info(f"Tool F1: {tool_recall_metrics.get('tool_f1', 0):.3f}")

        if store_success:
            logger.info(f"\nResults stored to DuckDB: {duckdb_path}")
        else:
            logger.warning("Failed to persist results to DuckDB.")

        summary = {
            "task_name": task_name,
            "total_queries": 1,
            "successful_queries": 0 if 'error' in result else 1,
            "failed_queries": 1 if 'error' in result else 0,
            "success_rate": _calculate_tool_success_rate(result),
            "total_processing_time_seconds": round(processing_time, 2),
            "evaluation_enabled": evaluation_enabled,
            "evaluated_queries": 1 if evaluation_scores else 0,
            "evaluation_summary": evaluation_scores,
            "tool_recall_summary": tool_recall_metrics,
            "duckdb_path": duckdb_path if store_success else None,
            "dataframe_rows": len(df),
            "dataframe_columns": len(df.columns)
        }

        return {
            "summary": summary,
            "result": result,
            "dataframe": df.to_dict('records')
        }

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Run the DeepAnalyze Adapter example against FDABench tasks.'
    )
    parser.add_argument('--input', type=str, help='Input JSON/JSONL file path')
    parser.add_argument('--index', type=int, default=0, help='FDABench-Lite sample index when no input file is provided')
    parser.add_argument('--model_path', type=str, help='DeepAnalyze model path (overrides DEEPANALYZE_MODEL_PATH)')
    parser.add_argument('--api_url', type=str, help='vLLM API URL (overrides DEEPANALYZE_API_URL)')
    parser.add_argument('--api_key', type=str, help='Optional API key passed into FDABench BaseAgent')
    parser.add_argument('--duckdb_path', type=str, default=None, help='Where to store DuckDB results')
    parser.add_argument('--skip_eval', action='store_true', help='Skip ROUGE/F1 evaluation to speed up runs')
    parser.add_argument('--max_agent_rounds', type=int, default=DEFAULT_MAX_AGENT_ROUNDS, help='Max FDABench adapter rounds')
    parser.add_argument('--max_deepanalyze_rounds', type=int, default=DEFAULT_MAX_DEEPANALYZE_ROUNDS, help='Max DeepAnalyze inner rounds')
    parser.add_argument('--max_tasks', type=int, default=None, help='Maximum tasks to process (for testing)')
    args = parser.parse_args()

    try:
        test_data_list = load_test_cases(args.input, args.index)

        # Limit processing count
        if args.max_tasks and args.max_tasks > 0:
            test_data_list = test_data_list[:args.max_tasks]
            logger.info(f"Limiting to first {args.max_tasks} tasks")

        # # Batch processing
        all_results = []
        all_summaries = []
        total_start_time = time.time()

        for idx, test_data in enumerate(test_data_list, 1):
            logger.info("\n" + "=" * 80)
            logger.info(f"Processing task {idx}/{len(test_data_list)}")
            logger.info("=" * 80)

            try:
                # Run test
                result_data = test_deepanalyze_adapter(
                    test_data=test_data,
                    model_path=args.model_path,
                    api_url=args.api_url,
                    max_agent_rounds=args.max_agent_rounds,
                    max_deepanalyze_rounds=args.max_deepanalyze_rounds,
                    skip_eval=args.skip_eval,
                    duckdb_path=args.duckdb_path,
                    api_key=args.api_key
                )

                all_results.append(result_data["result"])
                all_summaries.append(result_data["summary"])

                logger.info(f"✅ Task {idx} completed")

            except Exception as e:
                logger.error(f"❌ Task {idx} failed: {str(e)}")
                import traceback
                traceback.print_exc()
                # Continue to next task
                continue

        total_time = time.time() - total_start_time

        # # Aggregate statistics
        logger.info("\n" + "=" * 80)
        logger.info("# Batch processing completed")
        logger.info("=" * 80)
        logger.info(f"Total tasks: {len(test_data_list)}")
        logger.info(f"Successful tasks: {len(all_results)}")
        logger.info(f"Failed tasks: {len(test_data_list) - len(all_results)}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Average time per task: {total_time / len(test_data_list):.2f}s" if test_data_list else "N/A")

        # # Aggregate statistics
        if all_summaries:
            total_successful = sum(s.get('successful_queries', 0) for s in all_summaries)
            total_failed = sum(s.get('failed_queries', 0) for s in all_summaries)
            avg_success_rate = sum(s.get('success_rate', 0) for s in all_summaries) / len(all_summaries)

            logger.info(f"\nTotal successful queries: {total_successful}")
            logger.info(f"Total failed queries: {total_failed}")
            logger.info(f"Average success rate: {avg_success_rate:.2%}")

            # # Evaluation statistics
            evaluated = [s for s in all_summaries if s.get('evaluation_summary')]
            if evaluated:
                avg_rouge1 = sum(s['evaluation_summary'].get('rouge1', 0) for s in evaluated) / len(evaluated)
                avg_rouge2 = sum(s['evaluation_summary'].get('rouge2', 0) for s in evaluated) / len(evaluated)
                avg_rougeL = sum(s['evaluation_summary'].get('rougeL', 0) for s in evaluated) / len(evaluated)
                avg_f1 = sum(s['evaluation_summary'].get('f1', 0) for s in evaluated) / len(evaluated)

                logger.info(f"\nAverage ROUGE-1: {avg_rouge1:.4f}")
                logger.info(f"Average ROUGE-2: {avg_rouge2:.4f}")
                logger.info(f"Average ROUGE-L: {avg_rougeL:.4f}")
                logger.info(f"Average F1: {avg_f1:.4f}")

            # DuckDB path
            duckdb_paths = [s.get('duckdb_path') for s in all_summaries if s.get('duckdb_path')]
            if duckdb_paths:
                logger.info(f"\nAll results stored to: {duckdb_paths[0]}")

        return len(all_results) > 0

    except Exception as e:
        logger.error(f"# Batch processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
