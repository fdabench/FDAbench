"""
Reflection Agent implementation for FDABench Package.

This agent uses a reflection-based approach to iteratively improve its performance
by analyzing and reflecting on its actions and results.

Supports DAG-based execution through DAGExecutionMixin for flexible task graph structures.
"""

import os
import time
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from types import SimpleNamespace

from ..core.base_agent import BaseAgent, Query, Subtask
from ..core.dag_execution_mixin import DAGExecutionMixin

logger = logging.getLogger(__name__)


@dataclass
class ReflectionResult:
    """Result of reflection process"""
    original_action: str
    reflection: str
    improved_action: Optional[str] = None
    status: str = "completed"  # "completed", "improved", "failed"


class ReflectionAgent(DAGExecutionMixin, BaseAgent):
    """
    Reflection Agent that uses iterative reflection to improve performance.

    Core Pattern: Reflection-based improvement through:
    1. Execute action
    2. Observe result
    3. Reflect on the action and result
    4. Improve action if needed
    5. Continue until task completion

    This agent maintains a history of actions and reflections to learn
    from past decisions and improve future performance.

    Supports DAG-based execution for flexible task graph structures.
    """

    def __init__(self,
                 model: str = "claude-sonnet-4-20250514",
                 api_base: str = "https://openrouter.ai/api/v1",
                 api_key: str = None,
                 max_steps: int = 7,
                 max_reflections: int = 1,
                 enable_dag: bool = False):
        """
        Initialize ReflectionAgent.

        Args:
            model: LLM model to use
            api_key: API key for the LLM service
            max_steps: Maximum number of steps to execute
            max_reflections: Maximum number of reflections per action
            enable_dag: Enable DAG-based execution mode
        """
        super().__init__(model, api_base, api_key)

        self.max_steps = max_steps
        self.max_reflections = max_reflections
        self.enable_dag = enable_dag

        # Reflection tracking
        self.history = []
        self.actions = []
        self.observations = []
        self.reflections = []
        self.completed_subtasks = set()
        self.tools_executed = []
        self.current_query = None
        self.reflection_count_per_query = 0  # Track reflections per query
        self.max_reflections_per_query = 1   # Limit reflections per query to 1 for extreme token efficiency

        # Initialize DAG execution mixin
        self.init_dag_execution()

        # Simple cache for repeated actions
        self.action_cache = {}
    
    def reflect_on_action(self, action: SimpleNamespace, observation: SimpleNamespace, query: Query) -> tuple:
        """
        Reflect on the action and observation, and provide suggestions for improvement.
        Optimized reflection logic: reflect on failures OR after 2 consecutive successes (not every step).
        """
        # Count recent successful actions
        successful_count = 0
        if observation.status == "success":
            successful_count = 1
            # Count previous successful observations
            for obs in reversed(self.observations):
                if getattr(obs, 'status', None) == "success":
                    successful_count += 1
                else:
                    break
        
        # Reflect if the action failed OR after exactly 2 consecutive successes
        # BUT only if we haven't exceeded the max reflections per query
        should_reflect_by_condition = (observation.status == "error") or (successful_count == 2)
        can_reflect_by_limit = self.reflection_count_per_query < self.max_reflections_per_query
        do_reflect = should_reflect_by_condition and can_reflect_by_limit
        
        # Log reflection decision
        if should_reflect_by_condition:
            if observation.status == "error":
                reason = f"failed action: {action.action_type}"
            else:
                reason = f"2 consecutive successes (current: {action.action_type})"
            
            if can_reflect_by_limit:
                logger.info(f"Reflecting due to {reason} (reflection {self.reflection_count_per_query + 1}/{self.max_reflections_per_query})")
            else:
                logger.info(f"Would reflect due to {reason}, but max reflections ({self.max_reflections_per_query}) reached for this query")
        else:
            logger.info(f"Skipping reflection - success count: {successful_count}, status: {observation.status}")
        
        if not do_reflect:
            return action, observation

        # Ultra-simplified reflection prompt
        reflection_prompt = f"""Action: {action.action_type} -> {observation.status}
Query: {query.advanced_query}

Better approach? Reply "OK" or suggest tool from: {query.tools_available}"""

        reflection = self.call_llm_with_phase(
            [{"role": "user", "content": reflection_prompt}], 
            phase="retry", 
            category="reflection"
        )
        # Increment reflection count after successful reflection
        self.reflection_count_per_query += 1
        
        self.reflections.append(ReflectionResult(
            original_action=action.action_type,
            reflection=reflection,
            status="completed"
        ))

        # Try to parse the suggested action
        try:
            import re
            json_match = re.search(r'\{.*\}', reflection, re.DOTALL)
            if json_match:
                action_dict = json.loads(json_match.group())
                new_action = SimpleNamespace(**action_dict)
                new_observation = self._execute_action(new_action)
                self.reflections[-1].improved_action = new_action.action_type
                self.reflections[-1].status = "improved"
                return new_action, new_observation
            else:
                for tool in query.tools_available:
                    if tool.lower() in reflection.lower():
                        new_action = SimpleNamespace(
                            action_type=tool,
                            parameters={"query": query.advanced_query}
                        )
                        new_observation = self._execute_action(new_action)
                        self.reflections[-1].improved_action = new_action.action_type
                        self.reflections[-1].status = "improved"
                        return new_action, new_observation
        except Exception:
            pass
        return action, observation
    
    def _decide_next_action(self, query: Query) -> SimpleNamespace:
        """
        Decide the next action based on current state and reflection history.
        
        Args:
            query: Current query being processed
            
        Returns:
            SimpleNamespace with action_type and parameters
        """
        # Build the prompt and call the LLM to decide the next action
        prompt = self._build_decision_prompt(query)
        response = self.call_llm_with_phase(
            [{"role": "user", "content": prompt}], 
            phase="decision", 
            category="action_decision"
        )
        
        try:
            # Clean up response - remove markdown formatting
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            # Additional debugging for empty responses
            if not cleaned_response:
                logger.error(f"Empty response after cleaning. Original response: '{response}'")
                return SimpleNamespace(action_type="terminate", parameters={"error": "Empty LLM response"})
            
            action_dict = json.loads(cleaned_response)
            
            # Robustly map LLM output fields to action_type/parameters
            action_type = self._extract_action_type(action_dict)
            parameters = self._extract_parameters(action_dict)
            
            if action_type is None:
                logger.warning(f"Could not determine action_type from LLM response: {action_dict}")
                action_type = "terminate"
                parameters = {"error": "Failed to parse action_type from LLM response"}
            
            # Validate action against available tools
            allowed_tools = query.tools_available
            if not self._is_action_allowed(action_type, allowed_tools):
                logger.warning(f"Action type '{action_type}' not in allowed tools {allowed_tools}")
                action_type, parameters = self._get_fallback_action(query, allowed_tools)
            
            return SimpleNamespace(action_type=action_type, parameters=parameters)
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.error(f"Original response: '{response}'")
            logger.error(f"Cleaned response: '{cleaned_response}' (length: {len(cleaned_response) if 'cleaned_response' in locals() else 'N/A'})")
            return SimpleNamespace(action_type="terminate", parameters={"error": f"Failed to parse LLM response: {str(e)}"})
    
    def _build_decision_prompt(self, query: Query) -> str:
        """Build the prompt for decision making"""
        # Only keep the last 2 steps of history to reduce token usage
        history_summary = []
        for i, (action, observation) in enumerate(list(zip(self.actions, self.observations))[-2:]):
            history_summary.append(f"Step {i+1}:")
            history_summary.append(f"Action: {action.action_type}")
            history_summary.append(f"Result: {observation.status}")
            if hasattr(observation, 'error') and observation.error:
                history_summary.append(f"Error: {observation.error}")
            history_summary.append("---")
        
        allowed_tools = query.tools_available
        
        # FIX: Improved prompt with better tool selection logic
        used_tools = list(self.tool_results.keys()) if self.tool_results else []
        
        # Build context about what's been accomplished
        context_info = []
        if "get_schema_info" in used_tools:
            context_info.append("✅ Database schema obtained")
        if "generated_sql" in used_tools:
            context_info.append("✅ SQL query generated")
        if "execute_sql" in used_tools:
            context_info.append("✅ SQL executed")
        
        context_str = " | ".join(context_info) if context_info else "No tools executed yet"
        
        # Suggest next logical step
        next_step_suggestion = ""
        if "get_schema_info" not in used_tools:
            next_step_suggestion = "Start with get_schema_info to understand the database structure"
        elif "generated_sql" not in used_tools:
            next_step_suggestion = "Use generated_sql to create SQL query based on schema"
        elif "execute_sql" not in used_tools:
            next_step_suggestion = "Use execute_sql to run the generated SQL query"
        else:
            next_step_suggestion = "All core tools completed, consider web_context_search or vectorDB_search for additional context"
        
        prompt = f"""Query: {query.advanced_query}
Database: {query.db}
Available Tools: {allowed_tools}
Context: {context_str}
Next Step: {next_step_suggestion}

Return JSON only:
{{"action_type": "TOOL_NAME", "parameters": {{"database_name": "{query.db}"}}}}

Choose the most logical next tool from: {allowed_tools + ["terminate"]}"""
        
        return prompt

    def _extract_action_type(self, action_dict: dict) -> Optional[str]:
        """Extract action_type from action dictionary"""
        if 'action_type' in action_dict:
            return action_dict['action_type']
        elif 'tool' in action_dict:
            return action_dict['tool']
        elif 'action' in action_dict:
            return action_dict['action']
        elif 'id' in action_dict:
            return action_dict['id']
        return None
    
    def _extract_parameters(self, action_dict: dict) -> dict:
        """Extract parameters from action dictionary"""
        if 'parameters' in action_dict:
            return action_dict['parameters']
        elif 'input' in action_dict:
            return action_dict['input']
        else:
            # Fallback: use all other keys except action_type/tool/action/id as parameters
            return {k: v for k, v in action_dict.items() if k not in ['action_type', 'tool', 'action', 'id']}
    
    def _is_action_allowed(self, action_type: str, allowed_tools: List[str]) -> bool:
        """Check if action is allowed based on available tools"""
        # Map tool names to action types for validation
        tool_to_action_map = {
            'get_schema_info': 'schema_understanding',
            'generated_sql': 'sql_generate', 
            'execute_sql': 'sql_execute',
            'perplexity_search': 'web_context_search'
        }
        
        # Check if action is allowed (either direct match or mapped)
        if action_type in allowed_tools:
            return True
        elif action_type in tool_to_action_map.values():
            # Check if any of the mapped tools are in allowed_tools
            for tool, mapped_action in tool_to_action_map.items():
                if mapped_action == action_type and tool in allowed_tools:
                    return True
        
        return False
    
    def _get_fallback_action(self, query: Query, allowed_tools: List[str]) -> tuple:
        """Get fallback action when current action is not allowed"""
        # Try to use any available tool that hasn't been used much
        if allowed_tools:
            tool_usage_count = {}
            for action in self.actions:
                tool_usage_count[action.action_type] = tool_usage_count.get(action.action_type, 0) + 1
            
            # Find the least used available tool
            least_used_tool = min(allowed_tools, key=lambda t: tool_usage_count.get(t, 0))
            return least_used_tool, {"query": query.advanced_query}
        
        return "terminate", {"error": "No more tools available"}
    
    def _execute_action(self, action: SimpleNamespace) -> SimpleNamespace:
        """Execute an action using the appropriate tool"""
        action_type = action.action_type
        parameters = action.parameters if hasattr(action, 'parameters') else {}
        
        # Create cache key for this action
        cache_key = f"{action_type}_{hash(str(sorted(parameters.items())))}"
        
        # Check cache first
        if cache_key in self.action_cache:
            logger.info(f"Using cached result for {action_type}")
            cached_obs = self.action_cache[cache_key]
            # Add execution time to cached result
            if not hasattr(cached_obs, 'execution_time'):
                cached_obs.execution_time = 0.0  # Cached results have no execution time
            return cached_obs
        
        # Record start time for execution
        start_time = time.time()
        
        try:
            # Map action types to tool names and prepare parameters
            tool_name = self._map_action_to_tool(action_type)
            tool_params = self._prepare_tool_parameters(tool_name, parameters, self.current_query)
            
            # Execute tool via registry
            if self.tool_registry.get_tool(tool_name):
                result = self.tool_registry.execute_tool(tool_name, **tool_params)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Create observation with execution time
                observation = SimpleNamespace(
                    status=result.get("status", "error"),
                    results=result.get("results", {}),
                    error=result.get("error", None),
                    execution_time=execution_time
                )
                
                # Cache successful results
                if observation.status == "success":
                    self.action_cache[cache_key] = observation
                    
                return observation
            else:
                # Handle terminate action
                if action_type == "terminate":
                    return SimpleNamespace(
                        status="success",
                        results={"message": "Execution terminated"},
                        error=None
                    )
                else:
                    return SimpleNamespace(
                        status="error",
                        results={},
                        error=f"Tool '{tool_name}' not registered"
                    )
                    
        except Exception as e:
            logger.error(f"Error executing action {action_type}: {str(e)}")
            return SimpleNamespace(
                status="error",
                results={},
                error=str(e)
            )
    
    def _map_action_to_tool(self, action_type: str) -> str:
        """Map action type to tool name"""
        action_to_tool_map = {
            # SQL related
            "sql_generate": "generated_sql",
            "generated_sql": "generated_sql",
            "generate_sql": "generate_sql",
            "sql_execute": "execute_sql",
            "execute_sql": "execute_sql",

            # Schema related
            "schema_understanding": "get_schema_info",
            "get_schema_info": "get_schema_info",

            # Search related
            "web_context_search": "web_context_search",
            "perplexity_search": "web_context_search",
            "web_search": "web_search",
            "vectorDB_search": "vectorDB_search",
            "vector_search": "vector_search",
            
            # Other tools
            "sql_optimize": "sql_optimize",
            "sql_debug": "sql_debug",
            "file_system_search": "file_system",
            "context_history": "context_history"
        }
        
        return action_to_tool_map.get(action_type, action_type)
    
    def _prepare_tool_parameters(self, tool_name: str, parameters: dict, query: Query) -> dict:
        """Prepare parameters for tool execution"""
        # Add previous results as context
        params = dict(parameters) if parameters else {}
        params["instance_id"] = query.instance_id  # Ensure all tools can get instance_id
        params["database_name"] = query.db  # Add database name
        params["database_type"] = query.database_type  # Add database type
        
        # FIX: Add tool-specific parameter preparation for SQL tools
        if tool_name in ["generated_sql", "sql_generate", "generate_sql"]:
            params["natural_language_query"] = query.advanced_query or query.query or query.original_query
            params["database_name"] = query.db
            # Pass schema info if available from previous tools
            if "get_schema_info" in self.tool_results:
                params["schema_info"] = self.tool_results["get_schema_info"]
            elif "schema_understanding" in self.tool_results:
                params["schema_info"] = self.tool_results["schema_understanding"]
                
        elif tool_name in ["execute_sql", "sql_execute"]:
            params["database_name"] = query.db
            params["natural_language_query"] = query.advanced_query or query.query or query.original_query
            # FIX: Pass generated SQL if available - check both action types and handle nested results
            sql_query = None
            
            # First try to get from tool_results using the exact action type
            for action_type in ["generated_sql", "sql_generate"]:
                if action_type in self.tool_results:
                    sql_result = self.tool_results[action_type]
                    if isinstance(sql_result, dict):
                        if "sql_query" in sql_result:
                            sql_query = sql_result["sql_query"]
                            break
                        elif "results" in sql_result and isinstance(sql_result["results"], dict):
                            if "sql_query" in sql_result["results"]:
                                sql_query = sql_result["results"]["sql_query"]
                                break
            
            # If we found SQL query, pass it directly
            if sql_query:
                params["sql_query"] = sql_query
                logger.info(f"✅ Successfully passed SQL query to execute_sql: {sql_query[:50]}...")
            else:
                logger.warning(f"⚠️ No SQL query found in tool results for execute_sql. Available keys: {list(self.tool_results.keys())}")
                # Pass the entire tool_results as previous_results for the tool to handle
                params["previous_results"] = self.tool_results.copy()
        
        elif tool_name in ["web_context_search", "perplexity_search", "web_search"]:
            params["query"] = query.advanced_query or query.query or query.original_query
            params["expected_query"] = query.advanced_query or query.query or query.original_query

        elif tool_name in ["vectorDB_search", "vector_search"]:
            params["query"] = query.advanced_query or query.query or query.original_query
            params["expected_query"] = query.advanced_query or query.query or query.original_query
            
        elif tool_name == "sql_optimize":
            # Pass SQL to optimize from previous results
            for action_type in ["generated_sql", "sql_generate"]:
                if action_type in self.tool_results:
                    sql_result = self.tool_results[action_type]
                    if isinstance(sql_result, dict):
                        if "sql_query" in sql_result:
                            params["sql_query"] = sql_result["sql_query"]
                            break
                        elif "results" in sql_result and isinstance(sql_result["results"], dict):
                            if "sql_query" in sql_result["results"]:
                                params["sql_query"] = sql_result["results"]["sql_query"]
                                break
            params["db_name"] = query.db
            
        elif tool_name == "file_system_search":
            params["pattern"] = "*"
            params["search_params"] = {"pattern": "*"}
        elif tool_name == "context_history":
            params["action"] = "get"
            params["data"] = {"limit": 10}
        elif tool_name == "sql_debug":
            # Get failed SQL and error from previous tool results
            failed_sql = None
            error = None
            
            # Look for failed SQL execution results
            for tool_name_prev, result in self.tool_results.items():
                if tool_name_prev in ["execute_sql", "sql_execute"] and isinstance(result, dict):
                    if result.get("status") == "error":
                        # Get the SQL that failed
                        if "sql_query" in result:
                            failed_sql = result["sql_query"]
                        elif "previous_results" in result:
                            # Try to get from previous results
                            for prev_tool, prev_result in result["previous_results"].items():
                                if prev_tool in ["sql_generate", "generated_sql"] and isinstance(prev_result, dict):
                                    if "sql_query" in prev_result:
                                        failed_sql = prev_result["sql_query"]
                                        break
                        
                        # Get the error message
                        error = result.get("error", "Unknown SQL execution error")
                        break
            
            # If no failed SQL found, use defaults
            if not failed_sql:
                failed_sql = "SELECT * FROM unknown_table"
            if not error:
                error = "Table 'unknown_table' doesn't exist"
            
            params["failed_sql"] = failed_sql
            params["error"] = error
            params["natural_language_query"] = query.advanced_query or query.query or query.original_query
        
        # Pass all previous tool results as context
        params["previous_results"] = self.tool_results.copy()
        
        return params
    
    def process_query_from_json(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a query using the reflection pattern.
        
        Args:
            query_data: Query data from JSON
            
        Returns:
            Processing results with metrics
        """
        start_time = time.time()
        
        try:
            # Prepare gold_subtasks for Query object (required for initialization)
            gold_subtasks = []
            if 'gold_subtasks' in query_data:
                # Ensure each subtask has input field
                for subtask in query_data['gold_subtasks']:
                    if 'input' not in subtask:
                        subtask['input'] = {}
                
                gold_subtasks = [Subtask(**s) for s in query_data['gold_subtasks']]
            
            # Filter known fields (exclude gold_subtasks from agent access)
            known_fields = {
                'instance_id', 'db', 'database_type', 'tools_available',
                'query', 'advanced_query', 'original_query', 'ground_truth_report', 
                'multiple_choice_questions', 'level', 'question_type', 'options', 
                'correct_answer', 'explanation'
            }
            
            filtered_data = {k: v for k, v in query_data.items() if k in known_fields}
            # Add gold_subtasks to filtered_data for Query initialization (but agent won't use it)
            filtered_data['gold_subtasks'] = gold_subtasks
            query = Query(**filtered_data)
            
            # Reset state
            self.reset_state()
            self.history = []
            self.actions = []
            self.observations = []
            self.reflections = []
            self.tools_executed = []
            self.tools_attempted = []
            self.current_query = query
            self.reflection_count_per_query = 0  # Reset reflection count for new query
            
            # === DECISION + EXECUTE + RETRY 阶段循环 ===
            # Execute with reflection loop
            step = 0
            # Remove consecutive_errors tracking - let agent continue even with errors
            # This makes it different from the original but keeps reflection pattern
            
            while step < self.max_steps:
                logger.info(f"Reflection Step {step + 1}: Deciding next action...")
                
                # Remove early termination conditions - let agent continue like ToolUseAgent
                # But keep the reflection pattern distinct
                
                # === DECISION 阶段：决策下一个行动 ===
                with self.phase_timing('decision', f'action_decision_step_{step+1}'):
                    action = self._decide_next_action(query)
                    logger.info(f"Action: {action.action_type}")
                
                # === EXECUTE 阶段：执行行动 ===
                with self.phase_timing('execute', f'action_execution_step_{step+1}'):
                    observation = self._execute_action(action)
                    logger.info(f"Observation: {observation.status}")
                
                # === RETRY 阶段：反思和改进 ===
                with self.phase_timing('retry', f'reflection_step_{step+1}'):
                    # Enhanced reflection step (core of Reflection Agent)
                    # This is what makes ReflectionAgent unique
                    action, observation = self.reflect_on_action(action, observation, query)
                
                # Store history
                self.actions.append(action)
                self.observations.append(observation)
                self.history.append({"action": action.__dict__, "observation": observation.__dict__})
                
                # Record tool results and track completion (like ToolUseAgent for consistency)
                if action.action_type != "terminate":
                    # Track attempted tools (every attempt, like ToolUseAgent)
                    actual_tool_name = self._get_actual_tool_name(action.action_type)
                    self.tools_attempted.append(actual_tool_name)
                    
                    # Record successful tool results (only on success, like ToolUseAgent)
                    if observation.status == "success":
                        self.tool_results[action.action_type] = observation.results
                        self.tools_executed.append(actual_tool_name)
                        logger.info(f"✅ Tool {actual_tool_name}: success")
                    else:
                        logger.warning(f"❌ Tool {actual_tool_name}: {getattr(observation, 'error', 'unknown error')}")
                
                if action.action_type == "terminate":
                    logger.info("Terminating execution.")
                    break
                
                step += 1
            
            # === GENERATE 阶段：生成最终答案 ===
            with self.phase_timing('generate', 'final_answer_generation'):
                # Generate final response based on query type
                if query.question_type == "multiple_choice":
                    response = self._generate_multiple_choice_answer(query)
                elif query.question_type == "single_choice":
                    response = self._generate_single_choice_answer(query)
                else:
                    response = self.generate_report(query)
            
            # Calculate metrics - include complete end-to-end latency
            end_time = time.time()
            total_latency = end_time - start_time
            
            # Calculate total tool execution time
            total_tool_execution_time = sum(
                getattr(obs, 'execution_time', 0.0) 
                for obs in self.observations 
                if hasattr(obs, 'execution_time')
            )
            
            # Calculate network and external service latency
            external_latency = total_latency - total_tool_execution_time
            
            token_summary = self.token_tracker.get_token_summary()
            
            # Build tool execution results for evaluation
            tool_execution_results = []
            for action, observation in zip(self.actions, self.observations):
                if action.action_type != "terminate":
                    # Get actual execution time from observation if available
                    execution_time = getattr(observation, 'execution_time', 0.0)
                    tool_execution_results.append({
                        "tool": self._get_actual_tool_name(action.action_type),
                        "status": observation.status,
                        "execution_time": execution_time,
                        "error": getattr(observation, 'error', None)
                    })
            
            return {
                "instance_id": query.instance_id,
                "database_type": query.database_type,
                "db_name": query.db,
                "query": query.advanced_query or query.query,
                "level": getattr(query, 'level', ''),
                "question_type": getattr(query, 'question_type', ''),
                "model": self.model,
                "response": response,
                "selected_answer": response if query.question_type in ["multiple_choice", "single_choice"] else None,
                "report": response if query.question_type not in ["multiple_choice", "single_choice"] else None,
                "correct_answer": getattr(query, 'correct_answer', None),
                "tool_execution_results": tool_execution_results,
                "reflection_summary": [
                    {
                        "original_action": r.original_action,
                        "status": r.status,
                        "improved_action": r.improved_action
                    } for r in self.reflections
                ],
                "metrics": {
                    "latency_seconds": round(total_latency, 2),
                    "total_tool_execution_time": round(total_tool_execution_time, 2),
                    "external_latency": round(external_latency, 2),
                    "tools_executed": self.tools_executed,
                    "tools_attempted": self.tools_attempted,
                    "success_rate": len(self.tools_executed) / len(self.tools_attempted) if self.tools_attempted else 0,
                    "reflection_steps": step + 1,
                    "total_reflections": len(self.reflections),
                    "token_summary": token_summary,
                    # 添加四阶段统计
                    **self.get_phase_results()['phase_columns']
                },
                # 详细的阶段统计
                "phase_statistics": self.get_phase_results()['phase_summary'],
                "phase_distribution": self.get_phase_results()['phase_distribution'],
                "processing_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "instance_id": query_data.get("instance_id", "unknown"),
                "model": self.model,
                "error": str(e),
                "processing_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
    

    
    def generate_report(self, query: Query) -> str:
        """Generate report based on tool results, following the same format as ground truth reports"""
        
        # Determine if we have tool results
        has_tool_results = bool(self.tool_results)
        
        # Optimized prompt with essential format requirements
        if has_tool_results:
            prompt = f"""Query: {query.advanced_query or query.query}

Tool Results: {json.dumps(self.tool_results, indent=1)}

Generate markdown report:
## Executive Summary
## Data Analysis Results  
## External Context & Insights
## Key Connections
## Conclusions

Each section 2-3 sentences. Be factual and analytical."""
        else:
            prompt = f"""Query: {query.advanced_query or query.query}

No tool results. Generate analytical report:
## Executive Summary
## Data Analysis Results
## External Context & Insights  
## Key Connections
## Conclusions

Each section 2-3 sentences. Focus on methodology and domain insights."""
        
        try:
            response = self.call_llm_with_phase(
                [{"role": "user", "content": prompt}],
                phase="generate",
                category="report_generation"
            )
            
            # Validate that the response contains the expected sections
            required_sections = ["## Executive Summary", "## Data Analysis Results", "## External Context & Insights", "## Key Connections", "## Conclusions"]
            missing_sections = [section for section in required_sections if section not in response]
            
            if missing_sections:
                # If sections are missing, try to generate a fallback report
                logger.warning(f"Generated report missing sections: {missing_sections}")
                return self._generate_fallback_report(query, has_tool_results)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return self._generate_fallback_report(query, has_tool_results)
    
    def _generate_fallback_report(self, query: Query, has_tool_results: bool) -> str:
        """Generate a fallback report when the main generation fails"""
        
        query_text = query.advanced_query or query.query
        
        if has_tool_results:
            # Try to extract some basic information from tool results
            tool_summary = "Tool execution was performed but detailed results are not available."
        else:
            tool_summary = "No tool execution results are available."
        
        fallback_report = f"""## Executive Summary
Analysis of the query "{query_text}" requires comprehensive data examination and contextual understanding. {tool_summary} The query appears to seek analytical insights that would benefit from structured data analysis.

## Data Analysis Results
To properly address this query, data analysis would involve examining relevant datasets, performing statistical calculations, and identifying key patterns and trends. The specific analytical approach depends on the nature of the query and available data sources.

## External Context & Insights
This type of analytical query typically involves understanding domain-specific context, industry trends, and relevant external factors that could influence the interpretation of results. Contextual knowledge is essential for meaningful analysis.

## Key Connections
The connection between data analysis and external context is crucial for comprehensive understanding. Statistical findings should be interpreted within the broader domain context to provide actionable insights.

## Conclusions
Based on the query requirements, a thorough analytical approach combining data examination with contextual understanding would be necessary to provide meaningful conclusions. The analysis should focus on actionable insights that directly address the query objectives."""
        
        return fallback_report
    
    def _generate_multiple_choice_answer(self, query: Query) -> str:
        """Generate multiple choice answer"""
        if not query.options:
            return "No options available"
        
        prompt = f"""
Based on the database results, answer this multiple-choice question by selecting all correct options.

Query: {query.advanced_query or query.query}

Options:
{json.dumps(query.options, indent=2)}

Database Results:
{json.dumps(self.tool_results, indent=2)}

Select all correct options. Output ONLY the option letters separated by commas.
DO NOT include any explanations, text, or other content.
Example format: A, B, D

Answer:"""
        
        response = self.call_llm_with_phase(
            [{"role": "user", "content": prompt}],
            phase="generate",
            category="multiple_choice"
        )
        return response
    
    def _generate_single_choice_answer(self, query: Query) -> str:
        """Generate single choice answer"""
        if not query.options:
            return "No options available"
        
        # Extract available options from the query
        available_options = list(query.options.keys()) if query.options else ['A', 'B', 'C', 'D']
        options_text = ', '.join(available_options)
        
        prompt = f"""
        Query: {query.advanced_query or query.query}
        Options: {json.dumps(query.options)}
        Results: {json.dumps(self.tool_results, indent=1)}
        
        You must select EXACTLY ONE answer from the available options: {options_text}
        
        Answer:"""
        
        response = self.call_llm_with_phase(
            [{"role": "user", "content": prompt}],
            phase="generate",
            category="single_choice"
        )
        
        # Extract answer - only accept valid options from the question
        import re
        valid_pattern = f'[{"".join(available_options)}]'
        answer = re.search(valid_pattern, response.upper())
        result = answer.group() if answer else "Unable to determine"
        
        # Additional validation - ensure the answer is in available options
        if result != "Unable to determine" and result not in available_options:
            return "Unable to determine"
        
        return result
    
    def _get_actual_tool_name(self, action_type: str) -> str:
        """Map action_type to actual tool name for tools_executed tracking"""
        tool_mapping = {
            # SQL related
            "sql_generate": "generated_sql",
            "generated_sql": "generated_sql",
            "generate_sql": "generate_sql",
            "sql_execute": "execute_sql",
            "execute_sql": "execute_sql",

            # Schema related
            "schema_understanding": "get_schema_info",
            "get_schema_info": "get_schema_info",

            # Search related
            "web_context_search": "web_context_search",
            "perplexity_search": "web_context_search",
            "web_search": "web_search",
            "vectorDB_search": "vectorDB_search",
            "vector_search": "vector_search",
            
            # Other tools
            "sql_optimize": "sql_optimize",
            "sql_debug": "sql_debug",
            "file_system_search": "file_system",
            "context_history": "context_history"
        }
        return tool_mapping.get(action_type, action_type)


def main():
    """Main function to test ReflectionAgent"""
    # Note: Tools need to be registered externally before using the agent
    agent = ReflectionAgent()
    
    # Test query
    test_query = {
        "instance_id": "test_001",
        "db": "test_db",
        "database_type": "SQLite",
        "question_type": "report",
        "tools_available": ["get_schema_info", "generated_sql", "execute_sql"],
        "gold_subtasks": [
            {"subtask_id": "schema", "tool": "get_schema_info", "input": {"database_name": "test_db"}},
            {"subtask_id": "generate", "tool": "generated_sql", "input": {"natural_language_query": "Test query"}},
            {"subtask_id": "execute", "tool": "execute_sql", "input": {"database_name": "test_db"}}
        ],
        "query": "Test query",
        "advanced_query": "Test advanced query",
        "original_query": "Test original query"
    }
    
    result = agent.process_query_from_json(test_query)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()