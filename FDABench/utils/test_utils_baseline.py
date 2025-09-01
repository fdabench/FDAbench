"""
Common utility functions for testing database agents.

This module provides shared functions used across all agent pattern tests,
including data loading, task name generation, DuckDB operations, and result formatting.
"""

import os
import json
import logging
import time
import pandas as pd
import duckdb
import numpy as np
from typing import Dict, Any, List, Optional

# Set up logging
logger = logging.getLogger(__name__)


def calculate_ehv_metric(
    evaluation_scores: Dict[str, float],
    total_cost: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    q0: float = 0.6,
    e0: float = 0.0
) -> float:
    """
    Calculate the normalized Expected HyperVolume (n-EHV) metric for a single query.
    
    Args:
        evaluation_scores: Dictionary containing rouge1, f1, precision, recall scores
        total_cost: Total cost of the query execution
        alpha: Exponent for efficiency importance (default: 1.0)
        beta: Exponent for quality importance (default: 1.0)  
        q0: Minimum acceptable quality threshold (default: 0.6)
        e0: Minimum acceptable efficiency threshold (default: 0.0)
        
    Returns:
        float: EHV score in range [0, 1]
    """
    # Calculate composite quality score from evaluation metrics
    # Weight ROUGE-1 (40%), F1 (40%), Precision (10%), Recall (10%)
    rouge1 = evaluation_scores.get('rouge1', 0.0)
    f1 = evaluation_scores.get('f1', 0.0)
    precision = evaluation_scores.get('precision', 0.0)
    recall = evaluation_scores.get('recall', 0.0)
    
    quality_score = 0.4 * rouge1 + 0.4 * f1 + 0.1 * precision + 0.1 * recall
    quality_score = np.clip(quality_score, 0.0, 1.0)
    
    # Calculate efficiency (E = 1 / cost)
    if total_cost <= 0:
        efficiency = 1.0  # Free queries get maximum efficiency
    else:
        efficiency = 1.0 / total_cost
    
    # Apply reference point thresholds
    if quality_score <= q0 or efficiency <= e0:
        return 0.0  # Clip-to-zero enforcement
    
    # Calculate normalized factors (simplified for single query)
    # For single query, we use reasonable max values
    e_max = 100.0  # Assume max efficiency of 100 (cost = 0.01)
    
    e_normalized = ((efficiency - e0) / (e_max - e0)) ** alpha
    q_normalized = ((quality_score - q0) / (1.0 - q0)) ** beta
    
    # EHV score is the product of normalized factors
    ehv = e_normalized * q_normalized
    
    return float(np.clip(ehv, 0.0, 1.0))

import os
import json
from typing import Dict, Any, List

def load_test_data(
    test_file_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Load all tasks from a JSONL test dataset.

    Args:
        test_file_path: Path to the JSONL test data file.

    Returns:
        A list containing all data records from the file.

    Raises:
        FileNotFoundError: If the test file is not found.
        ValueError: If no data is found in the file.
    """
    # Handle default path - find the correct relative path to sample data
    if test_file_path is None:
        # Get the directory where this script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up to FDABench root directory and find sample data
        project_root = os.path.dirname(os.path.dirname(current_dir))  # ../..
        test_file_path = os.path.join(project_root, "sample", "sample_data.json")
    
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"Test file not found: {test_file_path}")

    all_data = []

    # Read the JSONL file line by line and load all data
    with open(test_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                all_data.append(record)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")
                continue

    if not all_data:
        raise ValueError(f"No data found in the file: {test_file_path}")

    print(f"Loaded {len(all_data)} records from {test_file_path}")
    return all_data


def generate_task_name(query_data: Dict[str, Any], pattern: str) -> str:
    """
    Generate task name based on query characteristics and agent pattern
    
    Args:
        query_data: Query data dictionary
        pattern: Agent pattern name (e.g., 'tooluse', 'planning', 'reflection', 'multiagent')
        
    Returns:
        Generated task name string
    """
    database_type = query_data.get("database_type", "unknown").lower()
    level = query_data.get("level", "unknown").lower()
    question_type = query_data.get("question_type", "unknown").lower()
    
    # Determine dataset name
    if "spider2-lite" in database_type:
        dataset_name = "spider2lite"
    elif "bird" in database_type:
        dataset_name = "bird"
    elif "spider2-snow" in database_type:
        dataset_name = "spider2snow"
    else:
        dataset_name = database_type.replace("-", "").replace("_", "")
    
    # Determine pattern suffix
    pattern_suffix = pattern
    if "multiple_choice" in question_type:
        pattern_suffix = f"{pattern}_mc"
    elif "report" in question_type:
        pattern_suffix = f"{pattern}_report"
    
    return f"{pattern_suffix}_{dataset_name}_{level}"


def create_query_row(
    result: Dict[str, Any],
    test_data: Dict[str, Any],
    evaluation_scores: Dict[str, float],
    tool_recall_metrics: Dict[str, Any],
    task_name: str,
    pattern: str,
    ground_truth_report: str = ""
) -> Dict[str, Any]:
    """
    Create a comprehensive row for pandas DataFrame with pattern-specific adaptations
    
    Args:
        result: Agent processing result
        test_data: Original test data
        evaluation_scores: Evaluation metrics
        tool_recall_metrics: Tool recall metrics
        task_name: Generated task name
        pattern: Agent pattern name
        ground_truth_report: Ground truth report text
        
    Returns:
        Dictionary representing a DataFrame row
    """
    # Get question type to determine which fields to include
    question_type = result.get('question_type', '')
    
    # Handle result content based on question type
    if question_type in ["multiple_choice", "single_choice"]:
        # For choice questions, use selected_answer and correct_answer
        report_content = ""
        selected_answer_content = result.get('selected_answer', '') if result.get('selected_answer') else ''
        correct_answer_content = json.dumps(result.get('correct_answer', [])) if result.get('correct_answer') else ''
    else:
        # For report questions, use report field
        report_content = result.get('report', '') if result.get('report') else ''
        selected_answer_content = ""
        correct_answer_content = ""
    
    # Pattern-specific metrics handling
    metrics = result.get('metrics', {})
    
    # Common metrics
    base_row = {
        # Basic information
        'instance_id': result.get('instance_id', ''),
        'task_id': test_data.get('task_id', ''),
        'database_type': result.get('database_type', ''),
        'db_name': result.get('db_name', ''),
        'query': result.get('query', ''),
        'level': result.get('level', ''),
        'question_type': question_type,
        'model': result.get('model', 'deepseek-chat-v3-0324'),
        'processing_time': result.get('processing_time', ''),
        'task_name': task_name,
        'design_pattern': pattern,
        
        # Performance metrics
        'latency_seconds': metrics.get('latency_seconds', 0),
        'success_rate': _calculate_tool_success_rate(result),
        
        # Token usage
        'total_input_tokens': metrics.get('token_summary', {}).get('token_usage', {}).get('input', 0),
        'total_output_tokens': metrics.get('token_summary', {}).get('token_usage', {}).get('output', 0),
        'total_tokens': metrics.get('token_summary', {}).get('total_tokens', 0),
        'total_llm_calls': metrics.get('token_summary', {}).get('total_calls', 0),
        
        # Cost metrics
        'input_cost': metrics.get('cost', {}).get('input_cost', 0.0),
        'output_cost': metrics.get('cost', {}).get('output_cost', 0.0),
        'total_cost': metrics.get('cost', {}).get('total_cost', 0.0),
        
        # Result content
        'report': report_content,
        'selected_answer': selected_answer_content,
        'correct_answer': correct_answer_content,
        'options': json.dumps(result.get('options', {})) if result.get('options') else '',
        'ground_truth_report': ground_truth_report,
        
        # Evaluation metrics (ROUGE scores)
        'rouge1': evaluation_scores.get('rouge1', 0.0),
        'rouge2': evaluation_scores.get('rouge2', 0.0),
        'rougeL': evaluation_scores.get('rougeL', 0.0),
        
        # Evaluation metrics (Precision, Recall, F1)
        'precision': evaluation_scores.get('precision', 0.0),
        'recall': evaluation_scores.get('recall', 0.0),
        'f1': evaluation_scores.get('f1', 0.0),
        
        # LLM Judge score
        'llm_judge_score': evaluation_scores.get('llm_judge_score', 0.0),
        
        # EHV metric
        'ehv': evaluation_scores.get('ehv', 0.0),
        
        # Evaluation metadata
        'evaluation_time_seconds': evaluation_scores.get('evaluation_time', 0),
        'generated_report_length': len(report_content),
        'ground_truth_report_length': len(ground_truth_report),
        'has_evaluation': len(evaluation_scores) > 0,
        
        # Tool recall metrics
        'tool_recall': tool_recall_metrics.get('tool_recall', 0.0),
        'tool_precision': tool_recall_metrics.get('tool_precision', 0.0),
        'tool_f1': tool_recall_metrics.get('tool_f1', 0.0),
        'tool_tp': tool_recall_metrics.get('TP', 0),
        'tool_fn': tool_recall_metrics.get('FN', 0),
        'tool_fp': tool_recall_metrics.get('FP', 0),
        'tool_missed': json.dumps(tool_recall_metrics.get('missed_tools', [])),
        'tool_extra': json.dumps(tool_recall_metrics.get('extra_tools', [])),
        
        # Error handling - Enhanced validation
        'has_error': _has_execution_errors(result),
        'error_message': _get_error_summary(result),
        
        # Timestamp
        'created_at': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Pattern-specific metrics
    if pattern == 'tooluse':
        base_row.update({
            'completed_tools': json.dumps(metrics.get('completed_tools', [])),
            'total_steps': metrics.get('total_steps', 0),
            'tools_executed': json.dumps(metrics.get('tools_executed', [])),
            'subtask_results': json.dumps(result.get('subtask_results', {}))
        })
    elif pattern == 'planning':
        base_row.update({
            'completed_tools': json.dumps(metrics.get('completed_subtasks', [])),
            'total_steps': metrics.get('total_subtasks', 0),
            'tools_executed': json.dumps(metrics.get('tools_executed', [])),
            'subtask_results': json.dumps(result.get('subtask_results', {}))
        })
    elif pattern == 'reflection':
        base_row.update({
            'completed_tools': json.dumps(metrics.get('completed_subtasks', [])),
            'total_steps': metrics.get('total_subtasks', 0),
            'tools_executed': json.dumps(metrics.get('tools_executed', [])),
            'reflection_steps': metrics.get('reflection_steps', 0),
            'total_reflections': metrics.get('total_reflections', 0),
            'subtask_results': json.dumps(result.get('subtask_results', {})),
            'reflection_summary': json.dumps(result.get('reflection_summary', []))
        })
    elif pattern == 'multi':
        base_row.update({
            'completed_tools': json.dumps(metrics.get('completed_tools', [])),
            'total_steps': metrics.get('total_expert_actions', 0),
            'tools_executed': json.dumps(metrics.get('completed_tools', [])),
            'total_expert_actions': metrics.get('total_expert_actions', 0),
            'successful_tools': metrics.get('successful_tools', 0),
            'coordination_steps': metrics.get('coordination_steps', 0),
            'expert_actions': json.dumps(result.get('expert_actions', [])),
            'coordination_results': json.dumps(result.get('coordination_results', []))
        })
    
    return base_row


def store_in_duckdb(df: pd.DataFrame, duckdb_path: str, task_name: str) -> bool:
    """
    Store DataFrame in DuckDB with error handling and statistics
    
    Args:
        df: DataFrame to store
        duckdb_path: Path to DuckDB file
        task_name: Task name for statistics
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Debug print
        print('store_in_duckdb: About to insert DataFrame:')
        print(df.head())
        print('store_in_duckdb: task_name:', task_name)
        print('store_in_duckdb: instance_id:', df['instance_id'].iloc[0] if 'instance_id' in df.columns else None)
        print('store_in_duckdb: design_pattern:', df['design_pattern'].iloc[0] if 'design_pattern' in df.columns else None)
        # Adapt for completed_tools as list
        if 'completed_tools' in df.columns:
            df['completed_tools'] = df['completed_tools'].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
        conn = duckdb.connect(duckdb_path)
        
        # Check if table exists
        table_exists = conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'query_results'").fetchone()[0] > 0
        
        if not table_exists:
            # Create table if it doesn't exist
            col_defs = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                if col == 'completed_tools':
                    col_type = 'VARCHAR'  # Store json string
                elif 'object' in dtype or 'string' in dtype:
                    col_type = 'VARCHAR'
                elif 'int' in dtype:
                    col_type = 'BIGINT'
                elif 'float' in dtype:
                    col_type = 'DOUBLE'
                elif 'bool' in dtype:
                    col_type = 'BOOLEAN'
                else:
                    col_type = 'VARCHAR'
                col_defs.append(f'{col} {col_type}')
            col_defs_str = ', '.join(col_defs)
            conn.execute(f'CREATE TABLE query_results ({col_defs_str})')
            print("Created new query_results table")
        
        # Get existing table column structure
        existing_cols = [desc[0] for desc in conn.execute("DESCRIBE query_results").fetchall()]
        print(f"Existing table columns: {len(existing_cols)}")
        
        # Fill missing columns in DataFrame
        for col in existing_cols:
            if col not in df.columns:
                df[col] = None
                print(f"Filling missing column: {col}")
        
        # Ensure DataFrame column order matches table
        df = df[existing_cols]
        print(f"DataFrame columns: {len(df.columns)}")
        
        # Register DataFrame with unique temporary table name
        tmp_table = f"tmp_df_{os.getpid()}_{int(time.time())}"
        conn.register(tmp_table, df)
        
        # Insert data - use all columns
        insert_sql = f"INSERT INTO query_results SELECT * FROM {tmp_table}"
        conn.execute(insert_sql)
        print(f"Successfully inserted data with {len(df.columns)} columns")
        
        # Force flush to disk
        conn.execute("CHECKPOINT")
        
        # Verify data
        row_count = conn.execute("SELECT COUNT(*) FROM query_results").fetchone()[0]
        print(f"Data stored in DuckDB: {duckdb_path} (total rows: {row_count})")
        
        # Get some sample statistics from DuckDB
        stats_query = """
        SELECT 
            COUNT(*) as total_rows,
            AVG(CASE WHEN has_error THEN 0 ELSE 1 END) as success_rate,
            AVG(latency_seconds) as avg_latency,
            AVG(total_cost) as avg_cost,
            AVG(CASE WHEN has_evaluation THEN rouge1 ELSE NULL END) as avg_rouge1,
            AVG(CASE WHEN has_evaluation THEN f1 ELSE NULL END) as avg_f1,
            COUNT(CASE WHEN has_evaluation THEN 1 END) as evaluated_count
        FROM query_results 
        WHERE task_name = ?
        """
        
        db_stats = conn.execute(stats_query, [task_name]).fetchone()
        print(f"DuckDB Stats - Success Rate: {db_stats[1] or 0:.2%}, Avg Latency: {db_stats[2] or 0:.2f}s, "
                   f"Avg ROUGE-1: {db_stats[4] or 0:.4f}, Evaluated: {db_stats[6]}")
        
        conn.close()
        
        # Wait for file system sync
        time.sleep(0.2)
        
        # Final verification
        conn = duckdb.connect(duckdb_path)
        final_count = conn.execute("SELECT COUNT(*) FROM query_results").fetchone()[0]
        conn.close()
        
        if final_count == row_count:
            print("✅ Data persistence verification successful")
        else:
            print(f"⚠️ Data persistence verification failed: expected {row_count} rows, actual {final_count} rows")
        
        return True
        
    except Exception as e:
        print(f"Failed to store data in DuckDB: {str(e)}")
        return False




def evaluate_agent_result(
    result: Dict[str, Any],
    test_data: Dict[str, Any],
    evaluator,
    ground_truth_report: str,
    EVALUATION_AVAILABLE: bool,
    logger_instance: logging.Logger
) -> tuple[Dict[str, float], Dict[str, Any]]:
    """
    Evaluate agent result with both report metrics and tool recall
    
    Args:
        result: Agent processing result
        test_data: Original test data
        evaluator: ReportEvaluator instance
        ground_truth_report: Ground truth report text
        EVALUATION_AVAILABLE: Whether evaluation is available
        logger_instance: Logger instance
        
    Returns:
        Tuple of (evaluation_scores, tool_recall_metrics)
    """
    evaluation_scores = {}
    evaluation_time = 0
    
    # Report evaluation
    if (EVALUATION_AVAILABLE and evaluator and 
        result.get('instance_id') and ground_truth_report and 
        result.get('report', '') and "error" not in result):
        
        eval_start_time = time.time()
        try:
            generated_report = result.get('report', '')
            
            evaluation_scores = evaluator.evaluate_reports(
                generated_report=generated_report,
                ground_truth_report=ground_truth_report
            )
            evaluation_time = time.time() - eval_start_time
            
            logger_instance.info(f"Evaluation completed: ROUGE-1: {evaluation_scores.get('rouge1', 0):.4f}, "
                      f"F1: {evaluation_scores.get('f1', 0):.4f}")
            
        except Exception as e:
            logger_instance.warning(f"Evaluation failed: {str(e)}")
            evaluation_scores = {}
            evaluation_time = time.time() - eval_start_time
    
    # Tool recall evaluation
    tool_recall_metrics = {}
    if test_data.get("gold_subtasks") and evaluator:
        try:
            gold_subtasks = test_data["gold_subtasks"]
            # Convert to list of dicts if needed
            if gold_subtasks and hasattr(gold_subtasks[0], '__dict__'):
                gold_subtasks = [vars(s) for s in gold_subtasks]
            
            # Get tools executed (pattern-specific)
            tools_executed = []
            if result.get("metrics", {}).get("tools_executed"):
                tools_executed = result["metrics"]["tools_executed"]
            elif result.get("metrics", {}).get("completed_tools"):
                tools_executed = result["metrics"]["completed_tools"]
            
            if tools_executed:
                # Calculate tool recall
                tool_recall_metrics = evaluator.evaluate_tool_recall(
                    gold_subtasks=gold_subtasks,
                    actual_tools_executed=tools_executed
                )
                
                logger_instance.info(f"Tool Recall: {tool_recall_metrics['tool_recall']:.3f} "
                          f"(TP:{tool_recall_metrics['TP']}, FN:{tool_recall_metrics['FN']}, "
                          f"Missed: {tool_recall_metrics['missed_tools']})")
                
        except Exception as tool_eval_error:
            logger_instance.warning(f"Failed to evaluate tool recall: {tool_eval_error}")
    
    # Add evaluation time to scores
    evaluation_scores['evaluation_time'] = evaluation_time
    
    # Calculate EHV metric if we have both evaluation scores and cost information
    if evaluation_scores and result.get('metrics', {}).get('cost', {}).get('total_cost') is not None:
        total_cost = result['metrics']['cost']['total_cost']
        try:
            ehv_score = calculate_ehv_metric(evaluation_scores, total_cost)
            evaluation_scores['ehv'] = ehv_score
            logger_instance.info(f"EHV Score: {ehv_score:.4f}")
        except Exception as e:
            logger_instance.warning(f"Failed to calculate EHV metric: {str(e)}")
            evaluation_scores['ehv'] = 0.0
    else:
        evaluation_scores['ehv'] = 0.0
    
    return evaluation_scores, tool_recall_metrics


def save_results(
    result: Dict[str, Any],
    df: pd.DataFrame,
    pattern: str,
    base_path: str = "/home/gong0092/wang/Downloads/feature_bench"
) -> tuple[str, str]:
    """
    Save results to JSON and CSV files
    
    Args:
        result: Agent processing result
        df: DataFrame with results
        pattern: Agent pattern name
        base_path: Base path for saving files
        
    Returns:
        Tuple of (json_file_path, csv_file_path)
    """
    # Save results to JSON
    json_file = os.path.join(base_path, f"test_{pattern}_results.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    # Save DataFrame to CSV
    csv_file = os.path.join(base_path, f"test_{pattern}_results.csv")
    df.to_csv(csv_file, index=False)
    
    return json_file, csv_file


def validate_agent_basic(
    result: Dict[str, Any],
    summary: Dict[str, Any],
    logger_instance: logging.Logger
) -> bool:
    """
    Perform basic validation checks common to all agent patterns
    
    Args:
        result: Agent processing result
        summary: Test summary
        logger_instance: Logger instance
        
    Returns:
        True if all validations pass, False otherwise
    """
    success = True
    
    # Basic validation
    if 'error' not in result:
        logger_instance.info("✅ Agent processed query without errors")
    else:
        logger_instance.warning("⚠️  Agent encountered errors during processing")
        success = False
        
    if result.get('report') or result.get('selected_answer') or result.get('response'):
        logger_instance.info("✅ Agent generated a response")
    else:
        logger_instance.warning("⚠️  No response generated")
        success = False
    
    # Evaluation validation
    if summary.get('evaluation_enabled'):
        if summary.get('evaluated_queries', 0) > 0:
            logger_instance.info("✅ Evaluation completed successfully")
        else:
            logger_instance.warning("⚠️  Evaluation was enabled but no queries were evaluated")
    else:
        logger_instance.info("ℹ️  Evaluation was not available or disabled")
    
    # DuckDB validation
    if summary.get('duckdb_path'):
        logger_instance.info("✅ Results stored in DuckDB successfully")
    else:
        logger_instance.warning("⚠️  Failed to store results in DuckDB")
        success = False
    
    return success


def print_summary(
    summary: Dict[str, Any],
    logger_instance: logging.Logger
):
    """
    Print test summary information
    
    Args:
        summary: Test summary dictionary
        logger_instance: Logger instance
    """
    logger_instance.info("=== SUMMARY ===")
    logger_instance.info(f"Task: {summary.get('task_name', 'unknown')}")
    logger_instance.info(f"Success Rate: {summary.get('success_rate', 0):.2%}")
    logger_instance.info(f"Processing Time: {summary.get('total_processing_time_seconds', 0):.2f}s")
    logger_instance.info(f"Evaluation: {summary.get('evaluated_queries', 0)}/{summary.get('total_queries', 0)} queries evaluated")


def _calculate_tool_success_rate(result: Dict[str, Any]) -> float:
    """Calculate actual tool success rate based on tool execution results"""
    # Check different result formats based on agent pattern
    
    # Planning Agent - check subtask_results
    if 'subtask_results' in result:
        subtask_results = result['subtask_results']
        if not subtask_results:
            return 0.0
        
        successful_tools = 0
        total_tools = len(subtask_results)
        
        for subtask_id, subtask_result in subtask_results.items():
            if subtask_result.get("status") == "success" and not subtask_result.get("error"):
                successful_tools += 1
        
        return successful_tools / total_tools if total_tools > 0 else 0.0
    
    # Multi-Agent - check coordination_results  
    elif 'coordination_results' in result:
        coordination_results = result['coordination_results']
        if not coordination_results:
            return 0.0
        
        successful_tools = 0
        total_tools = len(coordination_results)
        
        # Count successful tools based on status
        for coord_result in coordination_results:
            if coord_result.get("status") == "success":
                successful_tools += 1
        
        return successful_tools / total_tools if total_tools > 0 else 0.0
    
    # Tool Use Agent - check tool execution results
    elif 'tool_results' in result:
        tool_results = result['tool_results']
        if not tool_results:
            return 0.0
        
        successful_tools = 0
        total_tools = len(tool_results)
        
        for tool_name, tool_result in tool_results.items():
            if isinstance(tool_result, dict) and tool_result.get("status") == "success" and not tool_result.get("error"):
                successful_tools += 1
        
        return successful_tools / total_tools if total_tools > 0 else 0.0
    
    # Fallback to agent's reported success rate
    else:
        metrics = result.get('metrics', {})
        return metrics.get('success_rate', 0.0)


def _has_execution_errors(result: Dict[str, Any]) -> bool:
    """Enhanced error detection that checks tool execution status"""
    # Check for direct error in result
    if 'error' in result:
        return True
    
    # Check subtask execution results (Planning Agent)
    if 'subtask_results' in result:
        for subtask_id, subtask_result in result['subtask_results'].items():
            if subtask_result.get("status") != "success" or subtask_result.get("error"):
                return True
    
    # Check coordination results (Multi-Agent)
    if 'coordination_results' in result:
        for coord_result in result['coordination_results']:
            if coord_result.get("status") != "success" or coord_result.get("error"):
                return True
    
    # Check tool results (Tool Use Agent)
    if 'tool_results' in result:
        for tool_name, tool_result in result['tool_results'].items():
            if isinstance(tool_result, dict):
                if tool_result.get("status") != "success" or tool_result.get("error"):
                    return True
    
    # Check if success rate is 0 (indicates failures)
    success_rate = _calculate_tool_success_rate(result)
    if success_rate == 0.0:
        return True
    
    return False


def _get_error_summary(result: Dict[str, Any]) -> str:
    """Get a summary of all errors in the execution"""
    errors = []
    
    # Direct error
    if 'error' in result:
        errors.append(f"Main error: {result['error']}")
    
    # Subtask errors (Planning Agent)
    if 'subtask_results' in result:
        for subtask_id, subtask_result in result['subtask_results'].items():
            if subtask_result.get("error"):
                errors.append(f"Subtask {subtask_id}: {subtask_result['error']}")
            elif subtask_result.get("status") != "success":
                errors.append(f"Subtask {subtask_id}: {subtask_result.get('status', 'failed')}")
    
    # Coordination errors (Multi-Agent)
    if 'coordination_results' in result:
        for i, coord_result in enumerate(result['coordination_results']):
            if coord_result.get("error"):
                errors.append(f"Coordination step {i+1}: {coord_result['error']}")
            elif coord_result.get("status") != "success":
                errors.append(f"Coordination step {i+1}: {coord_result.get('status', 'failed')}")
    
    # Tool errors (Tool Use Agent)
    if 'tool_results' in result:
        for tool_name, tool_result in result['tool_results'].items():
            if isinstance(tool_result, dict):
                if tool_result.get("error"):
                    errors.append(f"Tool {tool_name}: {tool_result['error']}")
                elif tool_result.get("status") != "success":
                    errors.append(f"Tool {tool_name}: {tool_result.get('status', 'failed')}")
    
    return "; ".join(errors) if errors else ""


# Unify all field sets for all design patterns
ALL_COLUMNS = [
    'instance_id', 'task_id', 'database_type', 'db_name', 'query', 'level', 'question_type', 'model', 'processing_time',
    'task_name', 'design_pattern', 'latency_seconds', 'success_rate', 'total_input_tokens', 'total_output_tokens',
    'total_tokens', 'total_llm_calls', 'input_cost', 'output_cost', 'total_cost', 'report', 'selected_answer', 'correct_answer',
    'options', 'ground_truth_report', 'rouge1', 'rouge2', 'rougeL', 'precision', 'recall', 'f1', 'llm_judge_score',
    'ehv', 'evaluation_time_seconds', 'generated_report_length', 'ground_truth_report_length', 'has_evaluation',
    'tool_recall', 'tool_precision', 'tool_f1', 'tool_tp', 'tool_fn', 'tool_fp', 'tool_missed', 'tool_extra',
    'has_error', 'error_message', 'created_at', 'completed_tools', 'total_steps', 'tools_executed',
    'reflection_steps', 'total_reflections', 'subtask_results', 'reflection_summary',
    'coordination_results', 'coordination_steps', 'expert_actions', 'successful_tools', 'total_expert_actions'
]