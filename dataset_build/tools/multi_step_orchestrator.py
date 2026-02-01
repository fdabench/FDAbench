# -*- coding: utf-8 -*-
"""Multi-Step Tool Orchestrator - Dynamic chain generation with SQL/Web/Vector integration."""

import os
import json
import logging
import time
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ToolType(Enum):
    SQL_EXECUTE = "sql_execute"
    WEB_SEARCH = "web_search"
    VECTOR_SEARCH = "vector_search"


@dataclass
class ToolStep:
    step_id: int
    tool_type: ToolType
    query: str
    result: str = ""
    success: bool = False
    urls: List[str] = field(default_factory=list)
    chunks: List[Dict] = field(default_factory=list)
    rationale: str = ""


@dataclass
class ChainContext:
    sql_result: str = ""
    sql_statement: str = ""
    steps: List[ToolStep] = field(default_factory=list)

    def add(self, step: ToolStep):
        self.steps.append(step)

    def get_context(self) -> str:
        parts = [f"SQL Result: {self.sql_result}"]
        for s in self.steps:
            if s.success:
                parts.append(f"[Step {s.step_id} - {s.tool_type.value}]: {s.result[:500]}")
        return "\n".join(parts)[:2000]

    def summary(self) -> str:
        return self.get_context()


class MultiStepOrchestrator:
    def __init__(self, vector_index_path: str = "./storage_faiss", api_key: str = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.vector_index_path = vector_index_path
        self._web_tool = None
        self._vector_index = None
        self._embedder = None

    def _call_llm(self, prompt: str, max_tokens: int = 1000) -> str:
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": "anthropic/claude-opus-4.5", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
            )
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
        return ""

    def _get_web_tool(self):
        if self._web_tool is None:
            from .external_tools import WebSearchTool
            self._web_tool = WebSearchTool(self.api_key)
        return self._web_tool

    def _get_vector_index(self):
        if self._vector_index is None:
            try:
                from FDABench.utils.vector_index_builder import FAISSVectorIndex, OpenAIEmbedder
                if os.path.exists(os.path.join(self.vector_index_path, "faiss.index")):
                    self._vector_index = FAISSVectorIndex.load(self.vector_index_path)
                    self._embedder = OpenAIEmbedder(api_key=self.openai_key)
            except Exception as e:
                logger.warning(f"Vector index load failed: {e}")
        return self._vector_index

    def plan_chain(self, query: str, sql_result: str, sql_statement: str) -> List[Dict]:
        prompt = f"""Given this analytical query and SQL result, plan a logical tool chain to build a complete answer.

QUERY: {query}
SQL: {sql_statement}
SQL RESULT: {sql_result}

Available tools:
- web_search: Get current real-world data, news, statistics, regulations
- vector_search: Get domain knowledge, academic research, methodology, definitions

Requirements:
1. Each step must build on SQL result or previous steps
2. Steps should be NON-REDUNDANT - each provides unique information
3. Chain should be LOGICAL - explain WHY each step is needed
4. Number of steps depends on query complexity (typically 2-5 steps)

Output JSON array with rationale for each step:
[
  {{"step": 1, "tool": "web_search", "query": "specific search query", "rationale": "why this step is needed after SQL"}},
  {{"step": 2, "tool": "vector_search", "query": "specific search query", "rationale": "what this adds to step 1"}}
]

Plan the minimal steps needed - don't add steps just to reach a number."""

        content = self._call_llm(prompt, 1200)
        try:
            if "```" in content:
                content = content.split("```")[1].replace("json", "").strip()
            return json.loads(content)
        except:
            return self._default_chain(query, sql_result)

    def _default_chain(self, query: str, sql_result: str) -> List[Dict]:
        return [
            {"step": 1, "tool": "web_search", "query": f"context explanation for {query}",
             "rationale": "Get real-world context to interpret SQL result"},
            {"step": 2, "tool": "vector_search", "query": f"domain knowledge {query}",
             "rationale": "Get academic/domain knowledge for deeper analysis"},
        ]

    def web_search(self, query: str, context: str = "") -> ToolStep:
        step = ToolStep(step_id=0, tool_type=ToolType.WEB_SEARCH, query=query)
        try:
            tool = self._get_web_tool()
            q = f"{query}\n\nContext:\n{context}" if context else query

            # Use search_with_urls to get both content and URLs
            result_data = tool.search_with_urls(q)
            step.result = result_data.get("content", "")
            step.urls = result_data.get("urls", [])

            # If no URLs from API, try to extract from content
            if not step.urls:
                import re
                step.urls = re.findall(r'https?://[^\s\)\]\"\'<>]+', step.result)[:10]

            step.success = True
        except Exception as e:
            step.result = str(e)
            step.success = False
        return step

    def vector_search(self, query: str, top_k: int = 5) -> ToolStep:
        step = ToolStep(step_id=0, tool_type=ToolType.VECTOR_SEARCH, query=query)
        try:
            index = self._get_vector_index()
            if not index:
                step.result = "Vector index not available"
                step.success = False
                return step

            emb = self._embedder.embed_query(query)
            results = index.search(emb, top_k=top_k)

            chunks, parts = [], []
            for r in results:
                chunk = r['chunk']
                chunks.append({
                    "chunk_id": chunk.get('id', ''),
                    "score": round(r['score'], 4),
                    "category": chunk.get('metadata', {}).get('category', ''),
                    "text_preview": chunk.get('text', '')[:300]
                })
                parts.append(f"[{chunk.get('metadata', {}).get('category', '')}] {chunk.get('text', '')[:400]}")

            step.result = "\n\n".join(parts)
            step.chunks = chunks
            step.success = True
        except Exception as e:
            step.result = str(e)
            step.success = False
        return step

    def execute_chain(self, query: str, sql_result: str, sql_statement: str) -> ChainContext:
        ctx = ChainContext(sql_result=sql_result, sql_statement=sql_statement)
        plan = self.plan_chain(query, sql_result, sql_statement)

        logger.info(f"Planned {len(plan)} steps for query")

        for item in plan:
            if item['tool'] == "web_search":
                step = self.web_search(item['query'], ctx.get_context())
            else:
                step = self.vector_search(item['query'])

            step.step_id = item['step']
            step.rationale = item.get('rationale', '')
            ctx.add(step)
            logger.info(f"Step {step.step_id} [{item['tool']}]: {'OK' if step.success else 'FAIL'}")

        return ctx

    def build_frozen_results(self, ctx: ChainContext) -> Dict:
        web_results = {"searches": []}
        vector_results = {"searches": []}

        for step in ctx.steps:
            if step.tool_type == ToolType.WEB_SEARCH and step.success:
                web_results["searches"].append({
                    "step": step.step_id,
                    "query": step.query,
                    "rationale": step.rationale,
                    "urls": step.urls,
                    "context_summary": step.result[:800]
                })
            elif step.tool_type == ToolType.VECTOR_SEARCH and step.success:
                vector_results["searches"].append({
                    "step": step.step_id,
                    "query": step.query,
                    "rationale": step.rationale,
                    "results": step.chunks,
                    "context_summary": step.result[:500]
                })

        return {"frozen_web_search": web_results, "frozen_vector_search": vector_results}

    def build_subtasks(self, db: str, query: str, sql: str, result: str, ctx: ChainContext) -> List[Dict]:
        subtasks = [
            {"subtask_id": "get_schema_info", "tool": "get_schema_info",
             "input": {"database_name": db}},
            {"subtask_id": "generate_sql", "tool": "generate_sql",
             "input": {"natural_language_query": query, "database_name": db},
             "expected_output": sql},
            {"subtask_id": "execute_sql", "tool": "execute_sql",
             "input": {"sql": sql, "database_name": db},
             "expected_output": result},
        ]

        for step in ctx.steps:
            tool_name = "web_search" if step.tool_type == ToolType.WEB_SEARCH else "vector_search"
            subtasks.append({
                "subtask_id": f"{tool_name}_{step.step_id}",
                "tool": tool_name,
                "input": {"query": step.query},
                "rationale": step.rationale,
                "depends_on": ["execute_sql"] if step.step_id == 1 else [f"{ctx.steps[step.step_id-2].tool_type.value.split('_')[0]}_search_{step.step_id-1}"]
            })

        return subtasks

    def generate_enhanced_query(self, original_query: str, ctx: ChainContext) -> str:
        prompt = f"""Transform this query into a harder analytical question that requires multi-step reasoning.

ORIGINAL QUERY: {original_query}

The enhanced query should:
1. NOT reveal any data or results - the agent must discover them
2. Require executing SQL on the database to get quantitative data
3. Require web search for current context/real-world information
4. Require vector search for domain knowledge/methodology
5. Require synthesizing all sources to reach a conclusion
6. Be 2-3 sentences, complete and grammatically correct

Bad example: "Given that 18.55% of teams..." (reveals data)
Good example: "Analyze the playoff qualification patterns for top-ranked NBA teams, explaining the quantitative trends from the database and the factors that drive these patterns based on current league structures and sports analytics research."

Output only the enhanced query text."""

        enhanced = self._call_llm(prompt, 400)
        if not enhanced or len(enhanced) < 50:
            enhanced = f"Analyze {original_query} by first querying the database for relevant statistics, then researching current context and domain knowledge to explain the patterns and their underlying factors."
        return enhanced

    def build_rubric(self, query: str, ctx: ChainContext) -> Dict:
        n_steps = len(ctx.steps)
        return {
            "task_classification": {
                "type": "H",
                "rationale": f"Requires SQL analysis + {n_steps}-step external knowledge synthesis",
                "sources_required": ["sql_execution", "web_search", "vector_search"]
            },
            "evaluation_dimensions": {
                "SQL_ACCURACY": {
                    "weight": 0.25,
                    "criteria": "Correctly executes SQL and interprets the result",
                    "verification": "exact_match"
                },
                "EXTERNAL_INTEGRATION": {
                    "weight": 0.25,
                    "criteria": "Integrates web/vector search findings with SQL data",
                    "verification": "llm_judge"
                },
                "LOGICAL_REASONING": {
                    "weight": 0.25,
                    "criteria": "Follows logical chain from SQL → context → insight",
                    "verification": "llm_judge"
                },
                "COMPLETENESS": {
                    "weight": 0.25,
                    "criteria": "Addresses all aspects of the query",
                    "verification": "report_check"
                }
            },
            "chain_validation": [
                {"step": s.step_id, "tool": s.tool_type.value, "rationale": s.rationale}
                for s in ctx.steps
            ]
        }
