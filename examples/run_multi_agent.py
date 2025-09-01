#!/home/wang/miniconda3/envs/bench/bin/python
"""
Test script for the new refactored MultiAgent.

This script tests the refactored package by processing the first task from test_report.json
using the new clean package structure with multi-agent coordination.
"""

import os
import sys
import logging
import time
import pandas as pd
import argparse
import json

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# Import from the new package
from FDABench.agents.multi_agent import MultiAgent
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

# Import tools
from FDABench.tools.sql_tools import SQLGenerationTool, SQLExecutionTool, SQLOptimizationTool, SQLDebugTool
from FDABench.tools.schema_tools import SchemaInfoTool
from FDABench.tools.search_tools import WebSearchTool, VectorSearchTool
from FDABench.tools.file_tools import FileSystemTool
from FDABench.tools.context_tools import ContextHistoryTool

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global evaluation availability
EVALUATION_AVAILABLE = True


def generate_duckdb_filename(model_name):
    """Generate DuckDB filename based on model name"""
    if not model_name:
        model_name = "default"
    # Clean model name for filename (replace / with _)
    clean_model = model_name.replace('/', '_').replace('-', '_')
    # Use relative path from project root to results directory
    project_root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, f"test_query_results_{clean_model}.duckdb")


def setup_tools(agent):
    """Register all necessary tools with the agent."""
    
    # Get database manager from agent
    db_manager = getattr(agent, 'db_manager', None)
    if db_manager is None:
        logger.warning("Database manager is not available - SQL tools will not function properly")
    else:
        logger.info("Database manager is available for SQL operations")
    
    # SQL tools
    agent.register_tool("get_schema_info", SchemaInfoTool(db_manager), 
                       category="database", description="Get database schema information")
    agent.register_tool("generated_sql", SQLGenerationTool(llm_client=agent, db_manager=db_manager), 
                       category="database", description="Generate SQL from natural language")
    agent.register_tool("execute_sql", SQLExecutionTool(db_manager), 
                       category="database", description="Execute SQL queries")
    agent.register_tool("sql_optimize", SQLOptimizationTool(llm_client=agent, db_manager=db_manager), 
                       category="database", description="Optimize SQL queries")
    agent.register_tool("sql_debug", SQLDebugTool(llm_client=agent, db_manager=db_manager), 
                       category="database", description="Debug SQL query errors")
    
    # Search tools
    # Get OpenRouter API key for perplexity search
    openrouter_api_key = os.environ.get('OPENROUTER_API_KEY')
    agent.register_tool("web_context_search", WebSearchTool(api_key=openrouter_api_key), 
                       category="search", description="Search web for context")
    agent.register_tool("perplexity_search", WebSearchTool(api_key=openrouter_api_key), 
                       category="search", description="Perplexity web search")
    agent.register_tool("vectorDB_search", VectorSearchTool(), 
                       category="search", description="Search vector database")
    
    # File system tools
    agent.register_tool("file_system", FileSystemTool(), 
                       category="file", description="Search file system")
    
    # Context tools
    agent.register_tool("context_history", ContextHistoryTool(), 
                       category="context", description="Manage context history")
    
    logger.info(f"Registered {len(agent.list_tools())} tools: {agent.list_tools()}")


def test_multi_agent(test_data=None, model=None):
    """Test the new MultiAgent with the provided test data or default data"""
    
    logger.info("Testing New Refactored MultiAgent")
    
    # Create agent instance
    api_key = os.environ.get('OPENROUTER_API_KEY', os.environ.get('OPENROUTER_API_KEY'))
    model_name = model or "deepseek/deepseek-chat-v3-0324"
    agent = MultiAgent(
        model=model_name,
        api_key=api_key,
        max_experts=10,
        max_execution_time=120  # Increase from 30 to 120 seconds (2 minutes)
    )
    
    # Setup tools
    setup_tools(agent)
    
    # Load test data using common function or provided data
    try:
        if test_data is None:
            test_data = load_test_data()
        logger.info(f"Loaded test case: {test_data.get('task_id', 'unknown')} - {test_data.get('instance_id', 'unknown')}")
        logger.info(f"Database: {test_data.get('db', 'unknown')} ({test_data.get('database_type', 'unknown')})")
        logger.info(f"Question type: {test_data.get('question_type', 'unknown')}")
        logger.info(f"Available tools: {test_data.get('tools_available', [])}")
        
        # Generate task name using common function
        task_name = generate_task_name(test_data, 'multi')
        logger.info(f"Task name: {task_name}")
        
        # Initialize evaluator
        evaluator = None
        global EVALUATION_AVAILABLE
        try:
            evaluator = ReportEvaluator()
            logger.info("Initialized ReportEvaluator for metrics calculation")
        except Exception as e:
            logger.warning(f"Failed to initialize evaluator: {e}")
            EVALUATION_AVAILABLE = False
        
        # Get ground truth report
        ground_truth_report = test_data.get('ground_truth_report', '')
        
        # Process the query
        logger.info("Processing query with new MultiAgent...")
        start_time = time.time()
        result = agent.process_query_from_json(test_data)
        processing_time = time.time() - start_time
        
        # Evaluate results using common function
        evaluation_scores, tool_recall_metrics = evaluate_agent_result(
            result, test_data, evaluator, ground_truth_report, EVALUATION_AVAILABLE, logger
        )
        
        # Create DataFrame row using common function
        query_row = create_query_row(
            result=result,
            test_data=test_data,
            evaluation_scores=evaluation_scores,
            tool_recall_metrics=tool_recall_metrics,
            task_name=task_name,
            pattern='multi',
            ground_truth_report=ground_truth_report
        )
        
        # Create DataFrame
        df = pd.DataFrame([query_row])
        # Fill all fields
        from FDABench.utils.test_utils import ALL_COLUMNS
        import json
        if 'completed_tools' in df.columns:
            df['completed_tools'] = df['completed_tools'].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
        for col in ALL_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[ALL_COLUMNS]
        logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        
        # Store in DuckDB using common function
        duckdb_path = generate_duckdb_filename(model_name)
        store_success = store_in_duckdb(df, duckdb_path, task_name)
        
        # Display results
        logger.info("Processing results:")
        logger.info(f"Instance ID: {result.get('instance_id', 'N/A')}")
        logger.info(f"Database: {result.get('db_name', 'N/A')} ({result.get('database_type', 'N/A')})")
        logger.info(f"Status: {'SUCCESS' if 'error' not in result else 'ERROR'}")
        logger.info(f"Processing time: {processing_time:.2f}s")
        
        if 'error' in result:
            logger.error(f"Error: {result['error']}")
        else:
            # Metrics
            metrics = result.get('metrics', {})
            logger.info(f"Total Latency: {metrics.get('latency_seconds', 0):.2f}s")
            logger.info(f"Tool Execution Time: {metrics.get('total_tool_execution_time', 0):.2f}s")
            logger.info(f"External Latency: {metrics.get('external_latency', 0):.2f}s")
            logger.info(f"Completed tools: {metrics.get('completed_tools', [])}")
            logger.info(f"Total expert actions: {metrics.get('total_expert_actions', 0)}")
            logger.info(f"Success rate: {metrics.get('success_rate', 0):.2%}")
            logger.info(f"Coordination steps: {metrics.get('coordination_steps', 0)}")
            
            # Token usage
            token_summary = metrics.get('token_summary', {})
            logger.info(f"Total tokens: {token_summary.get('total_tokens', 0)}")
            
            # Response
            if result.get('report'):
                logger.info(f"Report: {result['report'][:200]}...")
            elif result.get('selected_answer'):
                logger.info(f"Selected answer: {result['selected_answer']}")
            elif result.get('response'):
                logger.info(f"Response: {result['response'][:200]}...")
            
            # Evaluation results
            if evaluation_scores:
                logger.info(f"ROUGE-1: {evaluation_scores.get('rouge1', 0):.4f}")
                logger.info(f"ROUGE-2: {evaluation_scores.get('rouge2', 0):.4f}")
                logger.info(f"ROUGE-L: {evaluation_scores.get('rougeL', 0):.4f}")
                logger.info(f"F1: {evaluation_scores.get('f1', 0):.4f}")
                logger.info(f"Precision: {evaluation_scores.get('precision', 0):.4f}")
                logger.info(f"Recall: {evaluation_scores.get('recall', 0):.4f}")
            
            # Tool recall results
            if tool_recall_metrics:
                logger.info(f"Tool Recall: {tool_recall_metrics.get('tool_recall', 0):.3f}")
                logger.info(f"Tool Precision: {tool_recall_metrics.get('tool_precision', 0):.3f}")
                logger.info(f"Tool F1: {tool_recall_metrics.get('tool_f1', 0):.3f}")
                logger.info(f"TP: {tool_recall_metrics.get('TP', 0)}, FN: {tool_recall_metrics.get('FN', 0)}, FP: {tool_recall_metrics.get('FP', 0)}")
            
            # Expert actions summary
            expert_actions = result.get('expert_actions', [])
            if expert_actions:
                for i, action in enumerate(expert_actions):
                    logger.info(f"  Expert {i+1}: {action.get('expert_type', 'unknown')} - {action.get('tool_name', 'unknown')}")
                    logger.info(f"    Priority: {action.get('priority', 'unknown')}")
                    logger.info(f"    Reasoning: {action.get('reasoning', 'N/A')[:100]}...")
        
        # Store results in DuckDB only
        logger.info(f"Results stored in DuckDB: {duckdb_path}")
        
        # Validate agent functionality
        agent_info = agent.get_agent_info()
        logger.info(f"Agent type: {agent_info['agent_type']}")
        logger.info(f"Model: {agent_info['model']}")
        logger.info(f"Registered tools: {len(agent_info['registered_tools'])}")
        
        # Create summary
        summary = {
            "task_name": task_name,
            "total_queries": 1,
            "successful_queries": 0 if 'error' in result else 1,
            "failed_queries": 1 if 'error' in result else 0,
            "success_rate": _calculate_tool_success_rate(result),
            "total_processing_time_seconds": round(processing_time, 2),
            "evaluation_enabled": EVALUATION_AVAILABLE,
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
        raise


def test_multi_agent_coordination(test_data):
    """Test specific multi-agent coordination functionality using loaded test data"""
    logger.info("Testing Multi-Agent Coordination Features")
    
    # Create agent instance
    api_key = os.environ.get('OPENROUTER_API_KEY', os.environ.get('OPENROUTER_API_KEY'))
    agent = MultiAgent(
        model="deepseek/deepseek-chat-v3-0324",
        api_key=api_key,
        max_experts=4,
        max_execution_time=30
    )
    
    # Setup minimal tools for testing
    setup_tools(agent)
    
    # Convert test_data to Query object for coordination testing
    from FDABench.core.base_agent import Query, Subtask
    known_fields = {
        'instance_id', 'db', 'database_type', 'tools_available', 'gold_subtasks',
        'query', 'advanced_query', 'level', 'question_type'
    }
    filtered_data = {k: v for k, v in test_data.items() if k in known_fields}
    
    # Convert gold_subtasks to Subtask objects if they exist and are valid
    if 'gold_subtasks' in filtered_data and filtered_data['gold_subtasks']:
        try:
            subtasks = []
            for subtask_data in filtered_data['gold_subtasks']:
                # Ensure all required fields are present
                if all(field in subtask_data for field in ['subtask_id', 'tool', 'input', 'description']):
                    subtasks.append(Subtask(**subtask_data))
                else:
                    logger.warning(f"Skipping invalid subtask: {subtask_data}")
            filtered_data['gold_subtasks'] = subtasks
        except Exception as e:
            logger.warning(f"Failed to convert gold_subtasks: {e}")
            # If conversion fails, create empty list to avoid errors
            filtered_data['gold_subtasks'] = []
    else:
        # If no gold_subtasks or empty, create empty list
        filtered_data['gold_subtasks'] = []
    
    query = Query(**filtered_data)
    
    # Test coordinator agent
    logger.info("Testing coordinator agent...")
    try:
        expert_actions = agent._coordinator_agent(query)
        logger.info(f"Coordinator planned {len(expert_actions)} expert actions:")
        for i, action in enumerate(expert_actions, 1):
            logger.info(f"  {i}. {action.expert_type} expert: {action.tool_name} (priority: {action.priority})")
            logger.info(f"     Reasoning: {action.reasoning}")
    except Exception as e:
        logger.warning(f"Coordinator test failed: {e}")
    
    # Test expert parameter preparation
    logger.info("Testing expert parameter preparation...")
    from FDABench.agents.multi_agent import ExpertAction
    
    # Use tools from test data instead of hardcoded ones
    available_tools = test_data.get('tools_available', [])
    test_actions = []
    
    # Create test actions based on available tools
    tool_to_expert_mapping = {
        'get_schema_info': ('database', 'sql'),
        'generated_sql': ('database', 'sql'), 
        'sql_generate': ('database', 'sql'),
        'execute_sql': ('database', 'sql'),
        'web_context_search': ('search', 'web'),
        'perplexity_search': ('search', 'web'),
        'vectorDB_search': ('search', 'vector'),
        'file_system_search': ('file', 'file'),
        'context_history': ('context', 'context')
    }
    
    for i, tool in enumerate(available_tools[:3]):  # Test first 3 tools
        if tool in tool_to_expert_mapping:
            category, expert_type = tool_to_expert_mapping[tool]
            test_actions.append(ExpertAction(expert_type, tool, i+1, f"Test {tool}"))
    
    for action in test_actions:
        try:
            params = agent._prepare_expert_parameters(action, query)
            logger.info(f"Parameters for {action.expert_type} expert ({action.tool_name}): {list(params.keys())}")
        except Exception as e:
            logger.warning(f"Parameter preparation failed for {action.expert_type}: {e}")
    
    # Test expert types mapping
    logger.info("Testing expert types mapping...")
    for expert_type, tools in agent.expert_types.items():
        logger.info(f"  {expert_type} expert handles: {tools}")
    
    return True


def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(description='Test MultiAgent with optional input file')
    parser.add_argument('--input', type=str, help='Input JSON file path')
    parser.add_argument('--model', type=str, help='Model to use for the agent')
    args = parser.parse_args()
    
    try:
        # Load test data from file if provided, otherwise use default
        test_data = None
        if args.input:
            logger.info(f"Loading test data from: {args.input}")
            with open(args.input, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
        else:
            logger.info("Using default test data")
        
        result_data = test_multi_agent(test_data, args.model)
        
        logger.info("Test completed successfully")
        
        summary = result_data["summary"]
        result = result_data["result"]
        
        # Basic validation using common function
        validate_agent_basic(result, summary, logger)
        
        # Multi-agent specific validation
        if result.get('expert_actions'):
            logger.info("✅ Multi-agent coordination worked successfully")
        else:
            logger.warning("⚠️  No expert actions were performed")
        
        if result.get('metrics', {}).get('completed_tools'):
            logger.info("✅ Agent executed tools successfully")
        else:
            logger.warning("⚠️  No tools were executed")
        
        # Print summary using common function
        print_summary(summary, logger)
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with exception: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)