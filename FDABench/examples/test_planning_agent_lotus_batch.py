

import os
import sys
import logging
import time
import pandas as pd
import argparse
import json

from FDABench.agents.planning_agent import PlanningAgent
from FDABench.evaluation import ReportEvaluator
from FDABench.utils.test_utils_baseline import (
    load_test_data,
    generate_task_name,
    create_query_row,
    store_in_duckdb,
    evaluate_agent_result,
    validate_agent_basic,
    print_summary,
    _calculate_tool_success_rate
)

from FDABench.tools.sql_tools_lotus import SQLGenerationTool, SQLExecutionTool, SQLOptimizationTool, SQLDebugTool
from FDABench.tools.schema_tools import SchemaInfoTool
from FDABench.tools.search_tools import WebSearchTool, VectorSearchTool
from FDABench.tools.file_tools import FileSystemTool
from FDABench.tools.context_tools import ContextHistoryTool

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EVALUATION_AVAILABLE = True


def setup_tools(agent):
    """Register all necessary tools with the agent."""
    
    db_manager = getattr(agent, 'db_manager', None)
    if db_manager is None:
        logger.warning("Database manager is not available - SQL tools will not function properly")
    else:
        logger.info("Database manager is available for SQL operations")
    
    agent.register_tool("get_schema_info", SchemaInfoTool(db_manager), 
                       category="database", description="Get database schema information")
    agent.register_tool("generated_sql", SQLGenerationTool(llm_client=agent, db_manager=db_manager), 
                       category="database", description="Generate SQL from natural language")
    agent.register_tool("execute_sql", SQLExecutionTool(db_manager, token_tracker=agent.token_tracker), 
                       category="database", description="Execute SQL queries")
    agent.register_tool("sql_optimize", SQLOptimizationTool(llm_client=agent, db_manager=db_manager), 
                       category="database", description="Optimize SQL queries")
    agent.register_tool("sql_debug", SQLDebugTool(llm_client=agent, db_manager=db_manager), 
                       category="database", description="Debug SQL query errors")
    
    openrouter_api_key = os.environ.get('OPENROUTER_API_KEY')
    agent.register_tool("web_context_search", WebSearchTool(api_key=openrouter_api_key), 
                       category="search", description="Search web for context")
    agent.register_tool("perplexity_search", WebSearchTool(api_key=openrouter_api_key), 
                       category="search", description="Perplexity web search")
    agent.register_tool("vectorDB_search", VectorSearchTool(), 
                       category="search", description="Search vector database")
    
    agent.register_tool("file_system", FileSystemTool(), 
                       category="file", description="Search file system")
    
    agent.register_tool("context_history", ContextHistoryTool(), 
                       category="context", description="Manage context history")
    
    logger.info(f"Registered {len(agent.list_tools())} tools: {agent.list_tools()}")

def test_planning_agent(test_data_list=None, subset="report", limit=None):
    """Test the new PlanningAgent with the provided test data list or default data"""
    
    logger.info("Testing New Refactored PlanningAgent")
    
    api_key = os.environ.get('OPENROUTER_API_KEY', os.environ.get('OPENROUTER_API_KEY'))
    
    try:
        if test_data_list is None:
            test_data_list = load_test_data(subset=subset, limit=limit)
        
        if not isinstance(test_data_list, list):
            test_data_list = [test_data_list]
        
        logger.info(f"Loaded {len(test_data_list)} test cases")
        
        evaluator = None
        global EVALUATION_AVAILABLE
        try:
            evaluator = ReportEvaluator()
            logger.info("Initialized ReportEvaluator for metrics calculation")
        except Exception as e:
            logger.warning(f"Failed to initialize evaluator: {e}")
            EVALUATION_AVAILABLE = False
        
        all_results = []
        all_query_rows = []
        total_processing_time = 0
        successful_queries = 0
        failed_queries = 0
        
        for i, test_data in enumerate(test_data_list, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing test case {i}/{len(test_data_list)}")
            logger.info(f"Task ID: {test_data.get('task_id', 'unknown')} - {test_data.get('instance_id', 'unknown')}")
            logger.info(f"Database: {test_data.get('db', 'unknown')} ({test_data.get('database_type', 'unknown')})")
            logger.info(f"Question type: {test_data.get('question_type', 'unknown')}")
            logger.info(f"Available tools: {test_data.get('tools_available', [])}")
            
            try:
                logger.info(f"Creating new agent instance for test case {i}")
                agent = PlanningAgent(
                    model="deepseek/deepseek-chat-v3-0324",
                    api_key=api_key,
                    max_planning_steps=10,
                    max_execution_time=60
                )
                
                setup_tools(agent)
                task_name = generate_task_name(test_data, 'planning')
                logger.info(f"Task name: {task_name}")
                
                ground_truth_report = test_data.get('ground_truth_report', '')
                
                logger.info("Processing query with new PlanningAgent...")
                start_time = time.time()
                result = agent.process_query_from_json(test_data)

                processing_time = time.time() - start_time
                total_processing_time += processing_time
                
                if 'error' in result:
                    failed_queries += 1
                    logger.error(f"Test case {i} failed: {result['error']}")
                else:
                    successful_queries += 1
                    logger.info(f"Test case {i} completed successfully")
                
                evaluation_scores, tool_recall_metrics = evaluate_agent_result(
                    result, test_data, evaluator, ground_truth_report, EVALUATION_AVAILABLE, logger
                )
                
                query_row = create_query_row(
                    result=result,
                    test_data=test_data,
                    evaluation_scores=evaluation_scores,
                    tool_recall_metrics=tool_recall_metrics,
                    task_name=task_name,
                    pattern='planning',
                    ground_truth_report=ground_truth_report
                )
                
                all_results.append(result)
                all_query_rows.append(query_row)
                
                single_df = pd.DataFrame([query_row])
                from FDABench.utils.test_utils import ALL_COLUMNS
                import json
                if 'completed_tools' in single_df.columns:
                    single_df['completed_tools'] = single_df['completed_tools'].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
                for col in ALL_COLUMNS:
                    if col not in single_df.columns:
                        single_df[col] = None
                single_df = single_df[ALL_COLUMNS]
                
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                results_dir = os.path.join(project_root, "results")
                os.makedirs(results_dir, exist_ok=True)
                duckdb_path = os.path.join(results_dir, "lotus.duckdb")
                store_success = store_in_duckdb(single_df, duckdb_path, f"{task_name}_case_{i}")
                if store_success:
                    logger.info(f"✅ Test case {i} results stored to DuckDB immediately")
                else:
                    logger.warning(f"⚠️ Failed to store test case {i} results to DuckDB")
                
                logger.info(f"Instance ID: {result.get('instance_id', 'N/A')}")
                logger.info(f"Status: {'SUCCESS' if 'error' not in result else 'ERROR'}")
                logger.info(f"Processing time: {processing_time:.2f}s")
                
                if 'error' not in result:
                    metrics = result.get('metrics', {})
                    logger.info(f"Success rate: {metrics.get('success_rate', 0):.2%}")
                    logger.info(f"Tools executed: {metrics.get('tools_executed', [])}")
                    
                    if evaluation_scores:
                        logger.info(f"ROUGE-L: {evaluation_scores.get('rougeL', 0):.4f}")
                        logger.info(f"F1: {evaluation_scores.get('f1', 0):.4f}")
                    
                    if tool_recall_metrics:
                        logger.info(f"Tool Recall: {tool_recall_metrics.get('tool_recall', 0):.3f}")
                
            except Exception as e:
                logger.error(f"Test case {i} failed with exception: {str(e)}")
                failed_queries += 1
                error_result = {
                    'instance_id': test_data.get('instance_id', 'unknown'),
                    'error': str(e),
                    'db_name': test_data.get('db', 'unknown'),
                    'database_type': test_data.get('database_type', 'unknown')
                }
                all_results.append(error_result)
                
                task_name = generate_task_name(test_data, 'planning')
                error_query_row = create_query_row(
                    result=error_result,
                    test_data=test_data,
                    evaluation_scores=None,
                    tool_recall_metrics=None,
                    task_name=task_name,
                    pattern='planning',
                    ground_truth_report=test_data.get('ground_truth_report', '')
                )
                all_query_rows.append(error_query_row)
                
                error_single_df = pd.DataFrame([error_query_row])
                from FDABench.utils.test_utils import ALL_COLUMNS
                import json
                if 'completed_tools' in error_single_df.columns:
                    error_single_df['completed_tools'] = error_single_df['completed_tools'].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
                for col in ALL_COLUMNS:
                    if col not in error_single_df.columns:
                        error_single_df[col] = None
                error_single_df = error_single_df[ALL_COLUMNS]
                
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                results_dir = os.path.join(project_root, "results")
                os.makedirs(results_dir, exist_ok=True)
                duckdb_path = os.path.join(results_dir, "lotus.duckdb")
                error_store_success = store_in_duckdb(error_single_df, duckdb_path, f"{task_name}_error_case_{i}")
                if error_store_success:
                    logger.info(f"✅ Test case {i} error results stored to DuckDB immediately")
                else:
                    logger.warning(f"⚠️ Failed to store test case {i} error results to DuckDB")
        
        df = pd.DataFrame(all_query_rows)
        from FDABench.utils.test_utils import ALL_COLUMNS
        import json
        if 'completed_tools' in df.columns:
            df['completed_tools'] = df['completed_tools'].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
        for col in ALL_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[ALL_COLUMNS]
        logger.info(f"\nCreated combined DataFrame with {len(df)} rows and {len(df.columns)} columns")
        
        batch_task_name = f"planning_lotus_batch_{int(time.time())}"
        
        logger.info(f"\n{'='*60}")
        logger.info("BATCH PROCESSING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total test cases: {len(test_data_list)}")
        logger.info(f"Successful queries: {successful_queries}")
        logger.info(f"Failed queries: {failed_queries}")
        logger.info(f"Success rate: {successful_queries/len(test_data_list):.2%}")
        logger.info(f"Total processing time: {total_processing_time:.2f}s")
        logger.info(f"Average processing time: {total_processing_time/len(test_data_list):.2f}s")
        
        logger.info(f"Results stored in DuckDB: {duckdb_path}")
        
        summary = {
            "task_name": batch_task_name,
            "total_queries": len(test_data_list),
            "successful_queries": successful_queries,
            "failed_queries": failed_queries,
            "success_rate": successful_queries / len(test_data_list),
            "total_processing_time_seconds": round(total_processing_time, 2),
            "average_processing_time_seconds": round(total_processing_time / len(test_data_list), 2),
            "evaluation_enabled": EVALUATION_AVAILABLE,
            "evaluated_queries": sum(1 for result in all_results if not result.get('error')),
            "duckdb_path": duckdb_path if store_success else None,
            "dataframe_rows": len(df),
            "dataframe_columns": len(df.columns)
        }
        
        return {
            "summary": summary,
            "results": all_results,
            "dataframe": df.to_dict('records')
        }
        
    except Exception as e:
        logger.error(f"Batch test failed: {str(e)}")
        raise

def test_planning_functionality():
    """Test specific planning agent functionality"""
    logger.info("Testing Planning Agent Specific Features")
    
    api_key = os.environ.get('OPENROUTER_API_KEY')
    agent = PlanningAgent(
        model="deepseek/deepseek-chat-v3-0324",
        api_key=api_key,
        max_planning_steps=3,
        max_execution_time=30
    )
    
    setup_tools(agent)
    
    test_query_data = {
        "instance_id": "planning_test_001",
        "db": "test_db",
        "level": "medium",
        "database_type": "bird",
        "question_type": "report",
        "tools_available": ["get_schema_info", "generated_sql", "execute_sql"],
        "query": "Find the total number of customers in the database.",
        "advanced_query": "Analyze customer data and provide insights on total customer count with breakdown by regions."
    }
    
    from FDABench.core.base_agent import Query
    known_fields = {
        'instance_id', 'db', 'database_type', 'tools_available',
        'query', 'advanced_query', 'level', 'question_type'
    }
    filtered_data = {k: v for k, v in test_query_data.items() if k in known_fields}
    filtered_data['gold_subtasks'] = []
    query = Query(**filtered_data)
    
    logger.info("Testing automatic planning...")
    subtasks = agent.plan_tasks(query)
    
    logger.info(f"Generated {len(subtasks)} subtasks:")
    for i, subtask in enumerate(subtasks, 1):
        logger.info(f"  {i}. {subtask.tool}: {subtask.description}")
        logger.info(f"     Input: {subtask.input}")
    
    logger.info("Testing task name generation...")
    task_name = agent.generate_task_name([test_query_data])
    logger.info(f"Generated task name: {task_name}")
    
    return True
def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(description='Test PlanningAgent with HuggingFace dataset for batch processing')
    parser.add_argument('--subset', type=str, default='report', choices=['report', 'single', 'multiple'],
                        help='Which subset to load from HuggingFace (default: report)')
    parser.add_argument('--limit', type=int, help='Limit number of samples to process')
    args = parser.parse_args()
    
    try:
        logger.info(f"Loading from HuggingFace dataset (subset: {args.subset}, limit: {args.limit})")
        
        result_data = test_planning_agent(None, subset=args.subset, limit=args.limit)
        
        logger.info("Batch test completed successfully")
        
        summary = result_data["summary"]
        results = result_data["results"]
        
        first_success_result = next((r for r in results if 'error' not in r), None)
        if first_success_result:
            validate_agent_basic(first_success_result, summary, logger)
        
        tools_executed_count = sum(1 for r in results if r.get('metrics', {}).get('tools_executed'))
        if tools_executed_count > 0:
            logger.info(f"✅ {tools_executed_count}/{len(results)} test cases executed tools successfully")
        else:
            logger.warning("⚠️  No test cases executed tools")
        
        print_summary(summary, logger)
        
        return True
        
    except Exception as e:
        logger.error(f"Batch test failed with exception: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)