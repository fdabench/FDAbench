"""DAG Annotator - Build DAG, rubric, gold_subtasks from exploration trace."""

import json
import logging
import requests
import os
from typing import Dict, List, Tuple

from dotenv import load_dotenv
load_dotenv()

from PUDDING.models.tree_models import ToolAction, TerminalPath
from PUDDING.tools.tool_executor import ToolExecutor

logger = logging.getLogger(__name__)

# Tool name standardization
TOOL_NAME_MAPPING = {
    "generated_sql": "generate_sql",
    "perplexity_search": "web_search",
    "vectorDB_search": "vector_search",
    "web_context_search": "web_search",
}

TOOL_TO_NODE_TYPE = {
    "get_schema_info": "SQL_QUERY",
    "generate_sql": "SQL_QUERY",
    "execute_sql": "SQL_QUERY",
    "web_search": "RETRIEVE_DOC",
    "vector_search": "RETRIEVE_DOC",
    "file_search": "RETRIEVE_DOC",
    "db_explore": "SQL_QUERY",
}

STANDARD_TOOLS_AVAILABLE = [
    "generate_sql", "web_search", "execute_sql", "vector_search",
    "context_history", "sql_optimize", "sql_debug", "file_system",
    "get_schema_info",
]


def standardize_tool_name(tool: str) -> str:
    return TOOL_NAME_MAPPING.get(tool, tool)


def infer_db_type(instance_id: str) -> str:
    if instance_id.startswith("bq") or instance_id.startswith("sf_bq"):
        return "bigquery"
    return "sqlite"


class DAGAnnotator:
    """Build DAG, rubric, gold_subtasks from exploration trace."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')

    def _call_llm(self, prompt: str, max_tokens: int = 800) -> str:
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": "anthropic/claude-opus-4.5",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens
                },
                timeout=60
            )
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
        return ""

    def annotate(
        self,
        record: Dict,
        terminal_path: TerminalPath,
        sql_result: str,
        sql_statement: str,
        enhanced_query: str,
        report: str,
    ) -> Dict:
        """Build all annotation artifacts from exploration trace.

        Returns dict with: dag, rubric, gold_subtasks, level, tools_available,
                          frozen_web_search, frozen_vector_search, db_type
        """
        db_name = record.get("db", "")
        task_id = record.get("task_id", "unknown")
        instance_id = record.get("instance_id", "")
        query = record.get("instruction", record.get("query", ""))

        # Build all components
        gold_subtasks = self._build_gold_subtasks(db_name, enhanced_query, sql_statement, sql_result, terminal_path)
        dag = self._build_dag(task_id, db_name, enhanced_query, sql_statement, sql_result, terminal_path)
        level = self._compute_difficulty(
            terminal_path, enhanced_query, sql_statement, query
        )
        rubric = self._build_rubric(enhanced_query, sql_result, terminal_path, report, level)

        frozen_web = ToolExecutor.build_frozen_web_search(terminal_path.actions)
        frozen_vec = ToolExecutor.build_frozen_vector_search(terminal_path.actions)

        return {
            "dag": dag,
            "rubric": rubric,
            "gold_subtasks": gold_subtasks,
            "level": level,
            "tools_available": list(STANDARD_TOOLS_AVAILABLE),
            "frozen_web_search": frozen_web,
            "frozen_vector_search": frozen_vec,
            "db_type": infer_db_type(instance_id),
        }

    def _build_gold_subtasks(
        self,
        db_name: str,
        query: str,
        sql_statement: str,
        sql_result: str,
        terminal_path: TerminalPath,
    ) -> List[Dict]:
        """Build gold_subtasks with proper depends_on chain."""
        subtasks = [
            {
                "subtask_id": "get_schema_info",
                "tool": "get_schema_info",
                "input": {"database_name": db_name},
                "description": f"Get schema information for database {db_name}",
                "depends_on": [],
            },
            {
                "subtask_id": "generate_sql",
                "tool": "generate_sql",
                "input": {"natural_language_query": query, "database_name": db_name},
                "expected_SQL": sql_statement,
                "description": f"Generate SQL to answer: {query[:100]}...",
                "depends_on": ["get_schema_info"],
            },
            {
                "subtask_id": "execute_sql",
                "tool": "execute_sql",
                "input": {"database_name": db_name},
                "expected_result": sql_result,
                "description": "Execute the generated SQL query",
                "depends_on": ["generate_sql"],
            },
        ]

        # Add search steps from terminal path
        prev_step = "execute_sql"
        seen_tools = {}  # tool_name -> count for deduplication
        for action in terminal_path.actions:
            tool_name = standardize_tool_name(action.tool_name)
            seen_tools[tool_name] = seen_tools.get(tool_name, 0) + 1
            count = seen_tools[tool_name]
            subtask_id = tool_name if count == 1 else f"{tool_name}_{count}"

            subtasks.append({
                "subtask_id": subtask_id,
                "tool": tool_name,
                "input": {"query": action.input_params.get("query", "")},
                "description": action.rationale or f"Search for: {action.input_params.get('query', '')[:80]}...",
                "depends_on": [prev_step],
            })
            prev_step = subtask_id

        return subtasks

    def _build_dag(
        self,
        task_id: str,
        db_name: str,
        query: str,
        sql_statement: str,
        sql_result: str,
        terminal_path: TerminalPath,
    ) -> Dict:
        """Build DAG structure matching FDAbench-Full format."""
        nodes = {}

        # SQL core nodes (always present)
        nodes["get_schema_info"] = {
            "node_id": "get_schema_info",
            "node_type": "SQL_QUERY",
            "tool": "get_schema_info",
            "input": {"database_name": db_name},
            "description": f"Get schema information for database {db_name}",
            "is_required": True,
            "is_critical_path": True,
            "alt_group_id": None,
            "expected_result": None,
            "expected_sql": None,
        }

        nodes["generate_sql"] = {
            "node_id": "generate_sql",
            "node_type": "SQL_QUERY",
            "tool": "generate_sql",
            "input": {"natural_language_query": query, "database_name": db_name},
            "description": f"Generate SQL to answer: {query[:100]}...",
            "is_required": True,
            "is_critical_path": True,
            "alt_group_id": None,
            "expected_result": None,
            "expected_sql": sql_statement,
        }

        nodes["execute_sql"] = {
            "node_id": "execute_sql",
            "node_type": "SQL_QUERY",
            "tool": "execute_sql",
            "input": {"database_name": db_name},
            "description": "Execute the generated SQL query",
            "is_required": True,
            "is_critical_path": True,
            "alt_group_id": None,
            "expected_result": sql_result,
            "expected_sql": None,
        }

        # Search nodes from terminal path
        edges = [
            {"source_id": "get_schema_info", "target_id": "generate_sql", "edge_type": "HARD_DEP"},
            {"source_id": "generate_sql", "target_id": "execute_sql", "edge_type": "HARD_DEP"},
        ]

        prev_node = "execute_sql"
        seen_tools = {}
        for action in terminal_path.actions:
            tool_name = standardize_tool_name(action.tool_name)
            seen_tools[tool_name] = seen_tools.get(tool_name, 0) + 1
            count = seen_tools[tool_name]
            node_id = tool_name if count == 1 else f"{tool_name}_{count}"

            nodes[node_id] = {
                "node_id": node_id,
                "node_type": TOOL_TO_NODE_TYPE.get(tool_name, "RETRIEVE_DOC"),
                "tool": tool_name,
                "input": {"query": action.input_params.get("query", "")},
                "description": action.rationale or f"Search for: {action.input_params.get('query', '')[:80]}...",
                "is_required": False,
                "is_critical_path": False,
                "alt_group_id": None,
                "expected_result": None,
                "expected_sql": None,
            }

            # All search edges are SOFT_DEP
            edges.append({
                "source_id": prev_node,
                "target_id": node_id,
                "edge_type": "SOFT_DEP",
            })
            prev_node = node_id

        # Entry/exit nodes
        all_node_ids = set(nodes.keys())
        targets = set(e["target_id"] for e in edges)
        sources = set(e["source_id"] for e in edges)
        entry_nodes = list(all_node_ids - targets)
        exit_nodes = list(all_node_ids - sources)

        return {
            "dag_id": task_id,
            "nodes": nodes,
            "edges": edges,
            "alt_groups": {},
            "entry_nodes": entry_nodes,
            "exit_nodes": exit_nodes,
            "critical_path": ["get_schema_info", "generate_sql", "execute_sql"],
            "metadata": {"db_name": db_name, "query": query},
        }

    def _compute_difficulty(
        self,
        terminal_path: TerminalPath,
        enhanced_query: str = "",
        sql_statement: str = "",
        original_query: str = "",
    ) -> str:
        """Compute difficulty using LLM-based 4-dimension BIRD scoring.

        Dimensions:
        1. SQL_COMPLEXITY (1-5)
        2. SOURCE_DIVERSITY (1-5)
        3. REASONING_DEPTH (1-5)
        4. DOMAIN_KNOWLEDGE (1-5)

        Average → easy(≤2.0) / medium(2.1-3.5) / hard(≥3.6).
        Falls back to source-count heuristic on LLM failure.
        """
        # Summarize exploration steps
        step_summary = ""
        for action in terminal_path.actions:
            prov = action.provenance or {}
            status = "OK" if prov.get("success", False) else "FAIL"
            step_summary += f"- [{action.tool_name}] ({status}): {action.input_params.get('query', '')[:80]}\n"

        prompt = f"""Rate the difficulty of this analytical task on 4 dimensions (1=trivial, 5=very complex).

QUERY: {(enhanced_query or original_query)[:400]}
SQL STATEMENT: {sql_statement[:300]}

EXPLORATION STEPS:
{step_summary[:600]}

Rate each dimension:
1. SQL_COMPLEXITY: How complex is the SQL? (1=simple SELECT, 5=multi-join subqueries with aggregation)
2. SOURCE_DIVERSITY: How many distinct source types are needed? (1=SQL only, 3=SQL+web, 5=SQL+web+vector+file+db)
3. REASONING_DEPTH: How many reasoning steps to connect sources? (1=direct lookup, 5=multi-step chain with inference)
4. DOMAIN_KNOWLEDGE: How much domain expertise is needed? (1=general, 5=highly specialized)

Output JSON only:
{{"SQL_COMPLEXITY": N, "SOURCE_DIVERSITY": N, "REASONING_DEPTH": N, "DOMAIN_KNOWLEDGE": N}}"""

        content = self._call_llm(prompt, 300)
        try:
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            scores = json.loads(content)
            dims = ["SQL_COMPLEXITY", "SOURCE_DIVERSITY", "REASONING_DEPTH", "DOMAIN_KNOWLEDGE"]
            values = [float(scores.get(d, 3)) for d in dims]
            avg = sum(values) / len(values)

            if avg <= 2.0:
                level = "easy"
            elif avg <= 3.5:
                level = "medium"
            else:
                level = "hard"

            logger.info(
                f"LLM difficulty: {dict(zip(dims, values))}, avg={avg:.2f} -> {level}"
            )
            return level
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"LLM difficulty scoring failed ({e}), falling back to heuristic")
            return self._compute_difficulty_heuristic(terminal_path)

    def _compute_difficulty_heuristic(self, terminal_path: TerminalPath) -> str:
        """Fallback: compute difficulty based on number of distinct external sources."""
        source_types = set()
        for action in terminal_path.actions:
            prov = action.provenance or {}
            if prov.get("success", False):
                source_types.add(action.tool_name)

        n_sources = len(source_types)
        if n_sources <= 1:
            return "easy"
        elif n_sources == 2:
            return "medium"
        else:
            return "hard"

    def _build_rubric(
        self,
        query: str,
        sql_result: str,
        terminal_path: TerminalPath,
        report: str,
        level: str,
    ) -> Dict:
        """Build rubric with LLM-generated criteria referencing actual data."""
        n_steps = len(terminal_path.actions)

        # Determine sources used
        sources_used = ["sql_execution"]
        for action in terminal_path.actions:
            prov = action.provenance or {}
            if prov.get("success", False):
                sources_used.append(action.tool_name)
        sources_used = list(set(sources_used))

        # Weights by level
        weights = {
            "easy": {"SQL_ACCURACY": 0.4, "EXTERNAL_INTEGRATION": 0.2, "LOGICAL_REASONING": 0.2, "COMPLETENESS": 0.2},
            "medium": {"SQL_ACCURACY": 0.3, "EXTERNAL_INTEGRATION": 0.25, "LOGICAL_REASONING": 0.25, "COMPLETENESS": 0.2},
            "hard": {"SQL_ACCURACY": 0.25, "EXTERNAL_INTEGRATION": 0.25, "LOGICAL_REASONING": 0.25, "COMPLETENESS": 0.25},
        }.get(level, {"SQL_ACCURACY": 0.25, "EXTERNAL_INTEGRATION": 0.25, "LOGICAL_REASONING": 0.25, "COMPLETENESS": 0.25})

        # LLM-generate specific criteria
        criteria = self._generate_criteria(query, sql_result, terminal_path, report)

        # Build chain_validation from actual steps
        chain_validation = []
        for action in terminal_path.actions:
            chain_validation.append({
                "step": len(chain_validation) + 1,
                "tool": standardize_tool_name(action.tool_name),
                "rationale": action.rationale[:150] if action.rationale else "",
            })

        return {
            "task_classification": {
                "type": level[0].upper(),
                "rationale": f"{level.capitalize()} task requiring SQL analysis + {n_steps}-step external knowledge synthesis",
                "sources_required": sources_used,
            },
            "evaluation_dimensions": {
                "SQL_ACCURACY": {
                    "weight": weights["SQL_ACCURACY"],
                    "criteria": criteria.get("SQL_ACCURACY", "Correctly executes SQL and interprets the result"),
                    "verification": "exact_match",
                },
                "EXTERNAL_INTEGRATION": {
                    "weight": weights["EXTERNAL_INTEGRATION"],
                    "criteria": criteria.get("EXTERNAL_INTEGRATION", "Integrates web/vector search findings with SQL data"),
                    "verification": "llm_judge",
                },
                "LOGICAL_REASONING": {
                    "weight": weights["LOGICAL_REASONING"],
                    "criteria": criteria.get("LOGICAL_REASONING", "Follows logical chain from SQL to context to insight"),
                    "verification": "llm_judge",
                },
                "COMPLETENESS": {
                    "weight": weights["COMPLETENESS"],
                    "criteria": criteria.get("COMPLETENESS", "Addresses all aspects of the query"),
                    "verification": "report_check",
                },
            },
            "chain_validation": chain_validation,
        }

    def _generate_criteria(
        self,
        query: str,
        sql_result: str,
        terminal_path: TerminalPath,
        report: str,
    ) -> Dict[str, str]:
        """Use LLM to generate specific, data-grounded evaluation criteria."""
        # Summarize search findings
        search_summary = ""
        for action in terminal_path.actions:
            if action.output and len(action.output) > 50:
                search_summary += f"[{action.tool_name}]: {action.output[:200]}...\n"

        prompt = f"""Generate specific evaluation criteria for an analytical report task. The criteria must reference the ACTUAL data.

QUERY: {query[:300]}
SQL RESULT: {sql_result[:300]}
SEARCH FINDINGS: {search_summary[:500]}

Generate criteria for these 4 dimensions. Each criterion must be 1-2 sentences and reference specific data values or findings.

Output JSON only:
{{
  "SQL_ACCURACY": "specific criterion referencing actual SQL result values...",
  "EXTERNAL_INTEGRATION": "specific criterion about what external findings should be integrated...",
  "LOGICAL_REASONING": "specific criterion about the reasoning chain from data to insight...",
  "COMPLETENESS": "specific criterion about what aspects must be covered..."
}}"""

        content = self._call_llm(prompt, 600)
        try:
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            return json.loads(content)
        except (json.JSONDecodeError, KeyError):
            logger.warning("Failed to generate LLM criteria, using defaults")
            return {
                "SQL_ACCURACY": f"Correctly executes SQL and identifies key values from result: {sql_result[:100]}",
                "EXTERNAL_INTEGRATION": "Integrates web/vector search findings with SQL data",
                "LOGICAL_REASONING": "Follows logical chain from SQL → context → insight",
                "COMPLETENESS": "Addresses all aspects of the query",
            }
