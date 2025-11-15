#!/usr/bin/env python3
"""
DeepAnalyze Adapter - Integrating DeepAnalyze Model into FDABench Framework

This adapter integrates the DeepAnalyze model into FDABench's agent framework.
It acts as a bridge between DeepAnalyze's native <Code>/<Execute> capabilities
and FDABench's tool calling system, enabling seamless interoperability.

Key responsibilities:
- Wraps DeepAnalyze model (vLLM-based) as a FDABench-compatible agent
- Translates between DeepAnalyze's code execution format and FDABench's tool registry
- Manages workspace for CSV data and code execution
- Tracks implicit tool usage (SQL generation/execution via <Code>/<Execute>)
"""

import os
import sys
import time
import json
import logging
import re
from typing import Dict, List, Any, Optional
import pandas as pd

# Add FDABench to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Import FDABench core components
from FDABench.core.base_agent import BaseAgent
from FDABench.tools.schema_tools import SchemaInfoTool

# Import DeepAnalyze
try:
    # DeepAnalyze is in the DeepAnalyze folder at project root
    # /home/wang/Downloads/feature_bench/FDABench/FDABench/examples/deepanalyze_adapter.py
    # -> /home/wang/Downloads/feature_bench/DeepAnalyze
    current_file = os.path.abspath(__file__)
    feature_bench_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
    deepanalyze_path = os.path.join(feature_bench_root, 'DeepAnalyze')

    if not os.path.exists(deepanalyze_path):
        raise ImportError(f"DeepAnalyze path does not exist: {deepanalyze_path}")

    sys.path.insert(0, deepanalyze_path)
    from deepanalyze import DeepAnalyzeVLLM

    logging.info(f"✅ DeepAnalyze Import successful, path: {deepanalyze_path}")
except ImportError as e:
    logging.error(f"❌ Unable to import DeepAnalyze: {e}")
    logging.error(f"Current file path: {os.path.abspath(__file__)}")
    logging.error(f"Attempted import path: {deepanalyze_path if 'deepanalyze_path' in locals() else 'N/A'}")
    DeepAnalyzeVLLM = None

logger = logging.getLogger(__name__)


class DeepAnalyzeAdapter(BaseAgent):
    """
    DeepAnalyze Adapter - FDABench Adapter for DeepAnalyze Model

    This class adapts the DeepAnalyze model to work within FDABench's agent framework.
    It inherits from BaseAgent to leverage FDABench's infrastructure (tool registry,
    token tracking, database management) while preserving DeepAnalyze's unique capabilities.

    Architecture:
    - DeepAnalyze model: Handles reasoning, decision-making, code generation, data analysis
    - Adapter layer (this class): Tool call parsing, tool execution, result integration, answer extraction
    - Maintains DeepAnalyze's native <Code>/<Execute> mechanism
    - Extends with FDABench tools via JSON format tool calls

    Usage:
        adapter = DeepAnalyzeAdapter(
            model_path="/path/to/DeepAnalyze-8B",
            api_url="http://localhost:8000/v1/chat/completions"
        )
        result = adapter.process_query_from_json(query_data)
    """

    def __init__(
        self,
        model_path: str,
        api_url: str = "http://localhost:8000/v1/chat/completions",
        max_agent_rounds: int = 10,
        max_deepanalyze_rounds: int = 30,
        workspace_base: str = "/tmp/fdabench_workspace",
        **kwargs
    ):
        """
        Initialize DeepAnalyze Adapter

        Args:
            model_path: Path to DeepAnalyze model (e.g., "/path/to/DeepAnalyze-8B")
            api_url: vLLM API endpoint URL (default: http://localhost:8000/v1/chat/completions)
            max_agent_rounds: Maximum outer adapter loop rounds (default: 10)
            max_deepanalyze_rounds: Maximum internal DeepAnalyze reasoning rounds (default: 30)
            workspace_base: Base directory for task workspaces (default: /tmp/fdabench_workspace)
            **kwargs: Additional arguments passed to BaseAgent
        """
        # # Ensure API key is available (BaseAgent requires it even if not using OpenAI)
        if 'api_key' not in kwargs or not kwargs['api_key']:
            kwargs['api_key'] = os.environ.get('OPENROUTER_API_KEY') or os.environ.get('OPENAI_API_KEY') or 'dummy_key_for_deepanalyze'

        # # Initialize BaseAgent (will set up db_manager, token_tracker, etc.)
        super().__init__(**kwargs)

        self.model_path = model_path
        self.api_url = api_url
        self.max_agent_rounds = max_agent_rounds
        self.max_deepanalyze_rounds = max_deepanalyze_rounds
        self.workspace_base = workspace_base

        # # Initialize DeepAnalyze
        if DeepAnalyzeVLLM is None:
            raise ImportError("DeepAnalyze not imported correctly")

        self.deepanalyze = DeepAnalyzeVLLM(
            model_name=model_path,
            api_url=api_url,
            max_rounds=max_deepanalyze_rounds
        )

        # # State tracking
        self.workspace_dir = None
        self.tools_executed = []
        self.tools_attempted = []

        logger.info(f"DeepAnalyze Adapter initialization complete: {model_path}")

    def extract_tool_call_json(self, reasoning_output: str) -> Optional[Dict]:
        """
        Extract tool call JSON from DeepAnalyze output

        Supports two formats:
        1. Markdown JSON code block: ```json ... ```
        2. Plain JSON object
        """
        # # Method 1: Match JSON code block
        json_match = re.search(
            r'```json\s*(.*?)\s*```',
            reasoning_output,
            re.DOTALL
        )

        if json_match:
            try:
                tool_json = json.loads(json_match.group(1))
                if tool_json.get("action") == "call_tool":
                    logger.info(f"Detected JSON code block tool call: {tool_json.get('tool_name')}")
                    return tool_json
            except json.JSONDecodeError as e:
                logger.warning(f"Unable to parse tool call JSON: {e}")

        # # Method 2: Match plain JSON (as fallback)
        json_pattern = r'\{\s*"action"\s*:\s*"call_tool".*?\}'
        json_match2 = re.search(json_pattern, reasoning_output, re.DOTALL)

        if json_match2:
            try:
                # # Try to find complete JSON object (including nesting)
                start_pos = json_match2.start()
                brace_count = 0
                end_pos = start_pos

                for i, char in enumerate(reasoning_output[start_pos:], start=start_pos):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break

                json_str = reasoning_output[start_pos:end_pos]
                tool_json = json.loads(json_str)

                if tool_json.get("action") == "call_tool":
                    logger.info(f"Detected plain JSON tool call: {tool_json.get('tool_name')}")
                    return tool_json
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Fallback JSON parsing failed: {e}")

        return None

    def execute_fdabench_tool(self, tool_json: Dict, query_data: Dict) -> Dict:
        """
        Execute FDABench tool and return results

        Args:
            tool_json: Tool call JSON
            query_data: Query data

        Returns:
            Tool execution result
        """
        tool_name = tool_json["tool_name"]
        tool_params = tool_json.get("tool_params", {})

        # # Record tool call
        self.tools_attempted.append(tool_name)
        logger.info(f"Calling tool: {tool_name}, parameters: {tool_params}")

        start_time = time.time()

        try:
            # # Special handling: schema_understanding requires CSV preparation
            if tool_name == "schema_understanding":
                result = self.prepare_database_csv(
                    query=query_data.get("query"),
                    database_name=query_data.get("db"),
                    database_type=query_data.get("database_type"),
                    instance_id=query_data.get("instance_id")
                )
            else:
                # # Execute other tools
                # # Add necessary parameters
                tool_params["database_name"] = query_data.get("db")
                tool_params["database_type"] = query_data.get("database_type")
                tool_params["instance_id"] = query_data.get("instance_id")

                result = self.tool_registry.execute_tool(tool_name, **tool_params)

            execution_time = time.time() - start_time

            if result.get("status") == "success":
                self.tools_executed.append(tool_name)
                logger.info(f"✅ Tool {tool_name} succeeded, took {execution_time:.2f}s")
            else:
                logger.warning(f"❌ Tool {tool_name} failed: {result.get('error')}")

            # # Add execution time
            result["execution_time"] = execution_time
            return result

        except Exception as e:
            logger.error(f"Tool {tool_name} execution exception: {e}")
            return {
                "status": "error",
                "error": str(e),
                "execution_time": time.time() - start_time
            }

    def track_deepanalyze_implicit_tools(self, reasoning_output: str):
        """
        Detect implicit tool usage from DeepAnalyze output (SQL-related)

        # Unlike ToolUseAgent, DeepAnalyze uses <Code> and <Execute> tags
        # to implicitly execute SQL operations. To maintain consistent tool tracking, we need to detect these tags
        # and record them as corresponding tool calls.

        Detection rules:
        1. generated_sql: Detect <Code> tag (DeepAnalyze generated code)
        2. execute_sql: Detect <Execute> tag and check for errors

        Args:
            reasoning_output: DeepAnalyze reasoning output text
        """

        # 1. Detect SQL generation (<Code> tag)
        # # If DeepAnalyze generated code, consider SQL generation complete
        code_pattern = r'<Code>(.*?)</Code>'
        code_matches = re.findall(code_pattern, reasoning_output, re.DOTALL | re.IGNORECASE)

        if code_matches and "generated_sql" not in self.tools_attempted:
            logger.info("🔍 Detected <Code> tag - record implicit SQL generation")
            self.tools_attempted.append("generated_sql")
            self.tools_executed.append("generated_sql")  # # Having Code counts as success

        # 2. Detect SQL execution (<Execute> tag)
        execute_pattern = r'<Execute>(.*?)</Execute>'
        execute_matches = re.findall(execute_pattern, reasoning_output, re.DOTALL | re.IGNORECASE)

        if execute_matches:
            for idx, execute_block in enumerate(execute_matches):
                logger.info(f"🔍 Detected <Execute> tag #{idx+1} - record implicit SQL execution")

                # # Record once per Execute block
                self.tools_attempted.append("execute_sql")

                # # Check for errors
                # # Common Python error indicators
                error_indicators = [
                    "error:", "exception:", "traceback",
                    "failed", "errno", "valueerror", "syntaxerror",
                    "keyerror", "attributeerror", "typeerror",
                    "indexerror", "nameerror", "zerodivisionerror"
                ]

                has_error = any(
                    indicator in execute_block.lower()
                    for indicator in error_indicators
                )

                if not has_error:
                    self.tools_executed.append("execute_sql")
                    logger.info(f"✅ SQL execution succeeded #{idx+1}（no error）")
                else:
                    logger.warning(f"❌ SQL execution failed #{idx+1}，Detected error")
                    logger.debug(f"Error content: {execute_block[:200]}...")

    def prepare_database_csv(
        self,
        query: str,
        database_name: str,
        database_type: str,
        instance_id: str
    ) -> Dict:
        """
        Prepare database tables as CSV for DeepAnalyze workspace use
        # Similar to the method in sql_tools_lotus.py

        Args:
            query: User query
            database_name: Database name
            database_type: Database type
            instance_id: Instance ID

        Returns:
            Result containing CSV path and table information
        """
        try:
            logger.info(f"Preparing database CSV: {database_name} ({database_type})")

            # Step 1: Get schema information
            schema_tool = SchemaInfoTool(db_manager=self.db_manager)
            schema_result = schema_tool.execute(
                database_name=database_name,
                database_type=database_type,
                instance_id=instance_id
            )

            # # Enhanced error checking
            if not isinstance(schema_result, dict):
                error_msg = f"SchemaInfoTool returned non-dictionary type: {type(schema_result).__name__} = {repr(schema_result)}"
                logger.error(error_msg)
                return {"status": "error", "error": error_msg}

            if schema_result.get("status") != "success":
                error_msg = schema_result.get("error", f"Unknown schema error: {schema_result}")
                logger.error(f"Schema retrieval failed: {error_msg}")
                return {"status": "error", "error": error_msg}

            # Step 2: Select main table (simplified: select first table)
            # Note: 'tables' is a dictionary with table names as keys, not a list
            tables = schema_result.get("results", {}).get("tables", {})
            if not tables:
                logger.error(f"Schema results: {schema_result.get('results')}")
                return {"status": "error", "error": "No database tables found"}

            # # Select the first table as the main table
            # Get the first table name from the dictionary keys
            main_table = list(tables.keys())[0]
            logger.info(f"Select main table: {main_table}")
            logger.info(f"Available tables: {list(tables.keys())}")

            # Step 3: Execute SELECT * FROM table to get all data
            sql_query = f"SELECT * FROM {main_table}"

            config = self.db_manager.get_database_config(
                instance_id, database_name, database_type
            )
            execution_result = self.db_manager.execute_sql(config, sql_query)

            if execution_result["status"] != "success":
                return execution_result

            # Step 4: Convert to DataFrame
            df = pd.DataFrame(
                execution_result["results"]["query_results"],
                columns=execution_result["results"]["columns"]
            )

            logger.info(f"Retrieved data: {len(df)} rows x {len(df.columns)} columns")

            # Step 5: Save to workspace as CSV
            csv_filename = f"{main_table}.csv"
            csv_path = os.path.join(self.workspace_dir, csv_filename)
            df.to_csv(csv_path, index=False)

            logger.info(f"CSV saved: {csv_path}")

            return {
                "status": "success",
                "results": {
                    "table_name": main_table,
                    "csv_filename": csv_filename,
                    "csv_path": csv_path,
                    "rows": len(df),
                    "columns": list(df.columns),
                    "schema_info": schema_result.get("results")
                }
            }

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"CSV preparation failed: {e}")
            logger.error(f"Detailed error: {error_trace}")
            return {"status": "error", "error": f"{str(e)}\n{error_trace}"}

    def construct_prompt(self, query_data: Dict) -> str:
        """# Construct initial prompt"""

        query = query_data.get("advanced_query") or query_data.get("query")
        question_type = query_data.get("question_type", "report")
        options = query_data.get("options", {})
        db = query_data.get("db")
        database_type = query_data.get("database_type")

        prompt = f"""# Task: Database analysis and question answering

## Question
{query}

## Database information
- - Database name: {db}
- Database type: {database_type}

## Question type
{question_type}
"""

        if question_type in ["single_choice", "multiple_choice"]:
            prompt += f"""
## ## Options
{json.dumps(options, indent=2, ensure_ascii=False)}
"""

        prompt += f"""
## Available tools and operations

You have two ways to complete the task:

### ### Method 1: Call FDABench Tools (output JSON)

If you need external tools (database access, web search, etc.), output JSON code block:

```json
{{
  "action": "call_tool",
  "tool_name": "tool_name",
  "tool_params": {{
    "parameter_name": "parameter_value"
  }}
}}
```

**Available tools**:

1. **schema_understanding**: Get database schema and prepare table data as CSV
   ```json
   {{
     "action": "call_tool",
     "tool_name": "schema_understanding",
     "tool_params": {{
       "database_name": "{db}",
       "query": "description of your analysis goal"
     }}
   }}
   ```
   After calling, table data will be saved as CSV files to the current workspace, you can read and analyze with pandas.

2. **web_context_search**: Search web to get context information
   ```json
   {{
     "action": "call_tool",
     "tool_name": "web_context_search",
     "tool_params": {{
       "query": "search content"
     }}
   }}
   ```

3. **vectorDB_search**: Search vector database to get relevant documents
   ```json
   {{
     "action": "call_tool",
     "tool_name": "vectorDB_search",
     "tool_params": {{
       "query": "search content"
     }}
   }}
   ```

### ### Method 2: Data Analysis (using <Code>)

For data analysis tasks, directly use <Code> tags to write Python code:

<Code>
import pandas as pd
df = pd.read_csv('your_table.csv')
result = df.head()
print(result)
</Code>

Code will execute automatically, results will be returned in <Execute> tags.

## Important rules

1. **Do one thing at a time**: Don't include both JSON tool calls and <Code> in the same output
2. **Get data first**: If you need database data, call schema_understanding tool first
3. **Wait after tool call**: After outputting JSON, wait for tool execution results before continuing
4. **Code can be multi-round**: When using <Code>, you can reason through multiple rounds, each execution returns results
5. **Finally give answer**: After completing all analysis, provide the final answer in <Answer>...</Answer>

## Output format
"""

        if question_type == "single_choice":
            prompt += """- - Only return one option letter (A, B, C, etc.)
- Example: <Answer>B</Answer>
"""
        elif question_type == "multiple_choice":
            prompt += """- - Return comma-separated option letters
- Example: <Answer>A, C, D</Answer>
"""
        else:  # report
            prompt += """- Generate a comprehensive report with the following sections:
  ## Executive Summary (Executive Summary)
  ## Data Analysis Results (Data Analysis Results)
  ## External Context & Insights (External Context & Insights)
  ## Key Connections (Key Connections)
  ## Conclusions (Conclusions)
- Each section should be 2-3 sentences
"""

        return prompt

    def construct_followup_prompt(
        self,
        original_query: str,
        tool_name: str,
        tool_result: Dict,
        conversation_history: List[Dict]
    ) -> str:
        """Construct follow-up prompt with tool results"""

        prompt = f"""# Continue task

## Original question
{original_query}

## tool execution result
Tool: **{tool_name}**

Result:
```json
{json.dumps(tool_result, indent=2, ensure_ascii=False)}
```

## Next action
Based on the above tool results, continue analysis. You can:

1. **If you need more tools**: output JSON format tool call
2. **If you can analyze data**: use <Code>...</Code> to write Python code
3. **If you have enough information**: output <Answer>...</Answer> to give the final answer

Now please continue:
"""

        return prompt

    def construct_continuation_prompt(
        self,
        original_query: str,
        conversation_history: List[Dict]
    ) -> str:
        """Construct prompt asking if more operations are needed"""

        prompt = f"""# Continue task

## Original question
{original_query}

## Current progress
You have completed some analysis.

## Next step
If you need:
- **Calling tool**: output JSON format tool call
- **Further analysis**: use <Code>...</Code>
- **Give answer**: output <Answer>...</Answer>

Please continue your work:
"""

        return prompt

    def extract_answer(
        self,
        reasoning: str,
        question_type: str,
        options: Dict
    ) -> str:
        """Extract final answer from DeepAnalyze reasoning output"""

        # Look for <Answer>...</Answer> tag
        answer_match = re.search(
            r"<Answer>(.*?)</Answer>",
            reasoning,
            re.DOTALL | re.IGNORECASE
        )

        if not answer_match:
            logger.warning("Not found <Answer> tag，using fallback answer")
            return self._generate_fallback_answer(question_type, options)

        raw_answer = answer_match.group(1).strip()
        logger.info(f"Extracted raw answer: {raw_answer[:100]}...")

        if question_type == "single_choice":
            # # Extract single option letter
            valid_options = list(options.keys()) if options else ['A', 'B', 'C', 'D']
            pattern = f"[{''.join(valid_options)}]"
            match = re.search(pattern, raw_answer.upper())
            return match.group(0) if match else "Unable to determine"

        elif question_type == "multiple_choice":
            # # Extract multiple option letters
            valid_options = list(options.keys()) if options else ['A', 'B', 'C', 'D']
            pattern = f"[{''.join(valid_options)}]"
            matches = re.findall(pattern, raw_answer.upper())
            return ", ".join(sorted(set(matches))) if matches else "Unable to determine"

        else:  # report
            # # Return complete answer text
            return raw_answer

    def _generate_fallback_answer(self, question_type: str, options: Dict) -> str:
        """Generate fallback answer when extraction fails"""

        if question_type == "single_choice":
            return list(options.keys())[0] if options else "A"
        elif question_type == "multiple_choice":
            return list(options.keys())[0] if options else "A"
        else:  # report
            return """## Executive Summary
Unable to generate complete report due to processing error.

## Data Analysis Results
Analysis cannot be completed.

## External Context & Insights
Context information not available.

## Key Connections
Unable to establish connections.

## Conclusions
Further investigation needed."""

    def process_query_from_json(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing loop - core implementation

        Args:
            query_data: Query data (JSON format)

        Returns:
            Processing result (including answer, metrics, etc.)
        """
        start_time = time.time()

        logger.info("="*80)
        logger.info("process_query_from_json started")
        logger.info(f"Instance ID: {query_data.get('instance_id')}")
        logger.info(f"Question Type: {query_data.get('question_type')}")
        logger.info("="*80)

        # # Check if DeepAnalyze is available
        logger.info(f"Checking DeepAnalyze availability...")
        logger.info(f"  DeepAnalyzeVLLM class: {DeepAnalyzeVLLM is not None}")
        logger.info(f"  self.deepanalyze: {self.deepanalyze is not None}")

        if DeepAnalyzeVLLM is None or self.deepanalyze is None:
            error_msg = "DeepAnalyze failed to initialize successfully, unable to process query"
            logger.error(error_msg)
            return {
                "instance_id": query_data.get("instance_id"),
                "database_type": query_data.get("database_type"),
                "db_name": query_data.get("db"),
                "query": query_data.get("query"),
                "level": query_data.get("level"),
                "question_type": query_data.get("question_type"),
                "model": "DeepAnalyze-8B",
                "error": error_msg,
                "has_error": True,
                "error_message": error_msg,
                "response": self._generate_fallback_answer(
                    query_data.get("question_type"),
                    query_data.get("options")
                ),
                "selected_answer": None,
                "report": None,
                "correct_answer": query_data.get("correct_answer"),
                "tool_execution_results": [],
                "metrics": {
                    "latency_seconds": round(time.time() - start_time, 2),
                    "total_tool_execution_time": 0,
                    "external_latency": 0,
                    "tools_executed": [],
                    "tools_attempted": [],
                    "success_rate": 0,
                    "total_agent_rounds": 0,
                    "token_summary": {}
                },
                "conversation_history": [],
                "processing_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
            }

        # 1. # Set up workspace
        instance_id = query_data.get("instance_id", "default")
        workspace_dir = os.path.join(self.workspace_base, instance_id)
        os.makedirs(workspace_dir, exist_ok=True)
        self.workspace_dir = workspace_dir

        logger.info(f"Workspace: {workspace_dir}")

        # 2. # Construct initial prompt
        current_prompt = self.construct_prompt(query_data)

        # 3. # Initialize tracking
        conversation_history = []
        tool_results = []
        final_answer = None
        agent_round = 0

        # # Reset state
        self.tools_executed = []
        self.tools_attempted = []

        # 4. # Main loop
        logger.info(f"Starting main loop, max rounds: {self.max_agent_rounds}")

        for agent_round in range(self.max_agent_rounds):

            logger.info(f"\n{'='*60}")
            logger.info(f"Agent Round {agent_round + 1}/{self.max_agent_rounds}")
            logger.info(f"{'='*60}")

            try:
                # Calling DeepAnalyze
                logger.info(f"Calling DeepAnalyze (API: {self.api_url})...")
                logger.debug(f"Prompt first 500 characters: {current_prompt[:500]}")

                result = self.deepanalyze.generate(
                    prompt=current_prompt,
                    workspace=workspace_dir,
                    temperature=0.5,
                    max_tokens=8192
                )

                reasoning_output = result.get("reasoning", "")
                logger.info(f"✅ DeepAnalyze returned, output length: {len(reasoning_output)} characters")

                # # Track token usage
                # # DeepAnalyze may make multiple internal API calls (when processing <Code>)
                # # We need to track each API call for correct calculation of generate_llm_calls
                usage = result.get("usage", {})
                api_calls = result.get("api_calls", [])
                total_deepanalyze_rounds = result.get("total_rounds", 0)

                # # Track each API call (not per agent round, but per DeepAnalyze internal API call)
                if api_calls:
                    for call_idx, call_info in enumerate(api_calls, 1):
                        self.token_tracker.track_call(
                            category=f"deepanalyze_agent_round_{agent_round + 1}_api_call_{call_idx}",
                            input_tokens=call_info.get("prompt_tokens", 0),
                            output_tokens=call_info.get("completion_tokens", 0),
                            model="DeepAnalyze-8B",
                            cost=None,  # # vLLM does not provide cost information
                            phase="decision"  # # DeepAnalyze reasoning and code execution are all counted as decision phase
                        )

                    logger.info(f"DeepAnalyze made {len(api_calls)} API calls internally")
                    logger.info(f"Token usage summary: {usage.get('prompt_tokens', 0)} input + {usage.get('completion_tokens', 0)} output = {usage.get('total_tokens', 0)} total")

                elif usage:
                    # # Fallback: if no detailed api_calls，at least track summary
                    self.token_tracker.track_call(
                        category=f"deepanalyze_round_{agent_round + 1}",
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                        model="DeepAnalyze-8B",
                        cost=None,
                        phase="decision"
                    )
                    logger.info(f"Token usage: {usage.get('prompt_tokens', 0)} input + {usage.get('completion_tokens', 0)} output = {usage.get('total_tokens', 0)} total")

                conversation_history.append({
                    "round": agent_round + 1,
                    "prompt": current_prompt[:500] + "..." if len(current_prompt) > 500 else current_prompt,
                    "output": reasoning_output,
                    "usage": usage,
                    "deepanalyze_rounds": total_deepanalyze_rounds,
                    "api_calls": len(api_calls)
                })

                logger.info(f"DeepAnalyze output length: {len(reasoning_output)} characters")

                # ===== Critical: Detect implicit tool usage (SQL-related)=====
                # # Check in each round DeepAnalyze's output, track <Code> and <Execute> tags
                self.track_deepanalyze_implicit_tools(reasoning_output)

                # 5. # Check if there is a final answer
                if "<Answer>" in reasoning_output:
                    logger.info("✅ Detected <Answer> tag")
                    final_answer = self.extract_answer(
                        reasoning_output,
                        query_data.get("question_type"),
                        query_data.get("options")
                    )
                    logger.info(f"Final answer: {final_answer}")
                    break

                # 6. # Check if FDABench tools were called
                tool_call_json = self.extract_tool_call_json(reasoning_output)

                if tool_call_json:
                    # # Tool call branch
                    logger.info(f"🔧 Detected tool call: {tool_call_json['tool_name']}")

                    tool_result = self.execute_fdabench_tool(tool_call_json, query_data)
                    tool_results.append({
                        "tool": tool_call_json["tool_name"],
                        "status": tool_result.get("status"),
                        "execution_time": tool_result.get("execution_time", 0),
                        "result": tool_result
                    })

                    # # Construct next round prompt with tool results
                    current_prompt = self.construct_followup_prompt(
                        original_query=query_data.get("query"),
                        tool_name=tool_call_json["tool_name"],
                        tool_result=tool_result,
                        conversation_history=conversation_history
                    )

                else:
                    # # No tool call and no answer
                    logger.info("ℹ️  No tool call or answer detected, asking if more operations are needed")

                    # # Construct inquiry prompt
                    current_prompt = self.construct_continuation_prompt(
                        original_query=query_data.get("query"),
                        conversation_history=conversation_history
                    )

            except Exception as e:
                logger.error(f"Round {agent_round + 1} error occurred: {e}")
                import traceback
                logger.error(traceback.format_exc())

                # # Record error and return
                total_latency = time.time() - start_time
                total_tool_execution_time = sum(r.get("execution_time", 0) for r in tool_results)
                return {
                    "instance_id": query_data.get("instance_id"),
                    "database_type": query_data.get("database_type"),
                    "db_name": query_data.get("db"),
                    "query": query_data.get("query"),
                    "level": query_data.get("level"),
                    "question_type": query_data.get("question_type"),
                    "model": "DeepAnalyze-8B",
                    "error": str(e),
                    "has_error": True,
                    "error_message": f"Round {agent_round + 1} error: {str(e)}",
                    "response": self._generate_fallback_answer(
                        query_data.get("question_type"),
                        query_data.get("options")
                    ),
                    "selected_answer": None,
                    "report": None,
                    "correct_answer": query_data.get("correct_answer"),
                    "tool_execution_results": [
                        {
                            "tool": r.get("tool"),
                            "status": r.get("status"),
                            "execution_time": r.get("execution_time", 0),
                            "error": r.get("error")
                        } for r in tool_results
                    ],
                    "metrics": {
                        "latency_seconds": round(total_latency, 2),
                        "total_tool_execution_time": round(total_tool_execution_time, 2),
                        "external_latency": round(total_latency - total_tool_execution_time, 2),
                        "tools_executed": self.tools_executed,
                        "tools_attempted": self.tools_attempted,
                        "success_rate": len(self.tools_executed) / len(self.tools_attempted) if self.tools_attempted else 0,
                        "total_steps": agent_round + 1,
                        "total_agent_rounds": agent_round + 1,
                        "token_summary": self.token_tracker.get_token_summary() if hasattr(self, 'token_tracker') else {},
                        **self.get_phase_results()['phase_columns']
                    },
                    "phase_statistics": self.get_phase_results()['phase_summary'],
                    "phase_distribution": self.get_phase_results()['phase_distribution'],
                    "conversation_history": conversation_history,
                    "processing_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
                }

        # 7. # If no answer obtained, force extraction
        if final_answer is None:
            logger.warning("⚠️  Reached maximum rounds without answer, trying to extract from last output")
            if conversation_history:
                last_output = conversation_history[-1]["output"]
                final_answer = self.extract_answer(
                    last_output,
                    query_data.get("question_type"),
                    query_data.get("options")
                )
            else:
                final_answer = self._generate_fallback_answer(
                    query_data.get("question_type"),
                    query_data.get("options")
                )

        # 8. # Calculate metrics
        total_latency = time.time() - start_time
        total_tool_execution_time = sum(
            r.get("execution_time", 0) for r in tool_results
        )
        external_latency = total_latency - total_tool_execution_time

        # # # Calculate total DeepAnalyze API call count
        total_deepanalyze_api_calls = sum(
            h.get("api_calls", 0) for h in conversation_history
        )

        # 9. # Return result (consistent with ToolUseAgent format)
        return {
            "instance_id": query_data.get("instance_id"),
            "database_type": query_data.get("database_type"),
            "db_name": query_data.get("db"),
            "query": query_data.get("query"),
            "level": query_data.get("level"),
            "question_type": query_data.get("question_type"),
            "model": "DeepAnalyze-8B",
            "response": final_answer,
            "selected_answer": final_answer if query_data.get("question_type") in ["single_choice", "multiple_choice"] else None,
            "report": final_answer if query_data.get("question_type") == "report" else None,
            "correct_answer": query_data.get("correct_answer"),
            "tool_execution_results": [
                {
                    "tool": r.get("tool"),
                    "status": r.get("status"),
                    "execution_time": r.get("execution_time", 0),
                    "error": r.get("error")
                } for r in tool_results
            ],
            "metrics": {
                "latency_seconds": round(total_latency, 2),
                "total_tool_execution_time": round(total_tool_execution_time, 2),
                "external_latency": round(external_latency, 2),
                "tools_executed": self.tools_executed,
                "tools_attempted": self.tools_attempted,
                "success_rate": len(self.tools_executed) / len(self.tools_attempted) if self.tools_attempted else 0,
                "total_steps": agent_round + 1,  # Corresponds to step in ToolUseAgent
                "total_agent_rounds": agent_round + 1,
                "total_deepanalyze_api_calls": total_deepanalyze_api_calls,  # DeepAnalyze-specific
                "token_summary": self.token_tracker.get_token_summary() if hasattr(self, 'token_tracker') else {},
                # # Add four-phase statistics (consistent with ToolUseAgent)
                **self.get_phase_results()['phase_columns']
            },
            # # Detailed phase statistics (consistent with ToolUseAgent)
            "phase_statistics": self.get_phase_results()['phase_summary'],
            "phase_distribution": self.get_phase_results()['phase_distribution'],
            "conversation_history": conversation_history,  # # DeepAnalyze-specific: complete reasoning trajectory
            "processing_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        }


if __name__ == "__main__":
    # # Simple test
    logging.basicConfig(level=logging.INFO)

    print("DeepAnalyze Adapter loaded")
    print(f"DeepAnalyze model available: {DeepAnalyzeVLLM is not None}")
