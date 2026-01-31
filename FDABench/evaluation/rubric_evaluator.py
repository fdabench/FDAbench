# -*- coding: utf-8 -*-
"""Rubric-based evaluator for multi-source analytical tasks."""

import os
import re
import json
import requests
from typing import Dict, List, Any, Optional


class RubricEvaluator:
    def __init__(self, api_key: str = None, model: str = "anthropic/claude-opus-4.5"):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.model = model

    def _call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
            )
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"LLM call failed: {e}")
        return ""

    def evaluate_sql_accuracy(self, agent_report: str, gold_sql_result: str) -> Dict[str, Any]:
        """Check if agent correctly executed SQL and interpreted the result."""
        gold_value = gold_sql_result.strip()

        # Extract numeric value from gold result
        match = re.search(r'[\d.]+', gold_value)
        gold_num = match.group() if match else gold_value

        # Check if the value appears in agent report
        found = gold_num in agent_report or gold_value in agent_report

        if not found:
            # Use LLM to check semantic equivalence
            prompt = f"""Does this report correctly state or interpret the SQL result?

SQL Result: {gold_sql_result}
Report excerpt (first 1000 chars): {agent_report[:1000]}

Answer with:
- "YES" if the report mentions or correctly interprets this value
- "NO" if the report misses or misinterprets this value

Answer (YES/NO):"""
            response = self._call_llm(prompt, 50)
            found = "YES" in response.upper()

        return {
            "dimension": "SQL_ACCURACY",
            "score": 1.0 if found else 0.0,
            "gold_value": gold_sql_result,
            "found_in_report": found
        }

    def evaluate_external_integration(self, agent_report: str, frozen_web: Dict, frozen_vector: Dict) -> Dict[str, Any]:
        """Check if agent integrated external search findings with SQL data."""
        web_summaries = [s.get("context_summary", "")[:300] for s in frozen_web.get("searches", [])]
        vector_summaries = [s.get("context_summary", "")[:300] for s in frozen_vector.get("searches", [])]

        prompt = f"""Evaluate if this report integrates external knowledge with database findings.

Report (first 1500 chars):
{agent_report[:1500]}

Expected external knowledge sources:
Web searches: {json.dumps(web_summaries, ensure_ascii=False)[:800]}
Vector searches: {json.dumps(vector_summaries, ensure_ascii=False)[:800]}

Score 0-1 based on:
- 0.0: No external knowledge integration
- 0.5: Mentions some external context but doesn't connect to data
- 1.0: Fully integrates external knowledge with SQL findings

Respond with just a number (0.0, 0.5, or 1.0):"""

        response = self._call_llm(prompt, 50)
        match = re.search(r'[01]\.?[05]?', response)
        score = float(match.group()) if match else 0.5

        return {
            "dimension": "EXTERNAL_INTEGRATION",
            "score": min(max(score, 0), 1),
            "web_sources": len(web_summaries),
            "vector_sources": len(vector_summaries)
        }

    def evaluate_logical_reasoning(self, agent_report: str, chain_validation: List[Dict]) -> Dict[str, Any]:
        """Check if report follows logical chain from SQL → context → insight."""
        chain_desc = "\n".join([f"Step {c['step']}: {c['tool']} - {c['rationale'][:100]}" for c in chain_validation])

        prompt = f"""Evaluate the logical reasoning flow in this report.

Expected reasoning chain:
{chain_desc}

Report (first 1500 chars):
{agent_report[:1500]}

Score 0-1 based on:
- 0.0: No logical flow, random statements
- 0.5: Some logical structure but gaps in reasoning
- 1.0: Clear progression from data → context → insight

Respond with just a number (0.0, 0.5, or 1.0):"""

        response = self._call_llm(prompt, 50)
        match = re.search(r'[01]\.?[05]?', response)
        score = float(match.group()) if match else 0.5

        return {
            "dimension": "LOGICAL_REASONING",
            "score": min(max(score, 0), 1),
            "expected_steps": len(chain_validation)
        }

    def evaluate_completeness(self, agent_report: str, enhanced_query: str) -> Dict[str, Any]:
        """Check if report addresses all aspects of the query."""
        prompt = f"""Evaluate if this report completely addresses the query.

Query: {enhanced_query}

Report (first 1500 chars):
{agent_report[:1500]}

Score 0-1 based on:
- 0.0: Fails to address the query
- 0.5: Partially addresses the query
- 1.0: Fully addresses all aspects of the query

Respond with just a number (0.0, 0.5, or 1.0):"""

        response = self._call_llm(prompt, 50)
        match = re.search(r'[01]\.?[05]?', response)
        score = float(match.group()) if match else 0.5

        return {
            "dimension": "COMPLETENESS",
            "score": min(max(score, 0), 1)
        }

    def evaluate_tool_usage(self, gold_subtasks: List[Dict], agent_tools: List[str]) -> Dict[str, Any]:
        """Check tool usage recall and precision."""
        expected = set(s.get("tool", "") for s in gold_subtasks if s.get("tool"))
        actual = set(agent_tools)

        tp = len(expected & actual)
        fn = len(expected - actual)
        fp = len(actual - expected)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "tool_recall": recall,
            "tool_precision": precision,
            "tool_f1": f1,
            "expected_tools": list(expected),
            "actual_tools": list(actual),
            "missed": list(expected - actual),
            "extra": list(actual - expected)
        }

    def evaluate(self, test_case: Dict, agent_report: str, agent_tools: List[str] = None) -> Dict[str, Any]:
        """Full rubric-based evaluation."""
        rubric = test_case.get("rubric", {})
        dimensions = rubric.get("evaluation_dimensions", {})

        results = {"dimensions": {}, "tool_usage": {}, "final_score": 0.0}

        # SQL Accuracy
        if "SQL_ACCURACY" in dimensions:
            sql_result = test_case.get("sql_result", "")
            sql_eval = self.evaluate_sql_accuracy(agent_report, sql_result)
            results["dimensions"]["SQL_ACCURACY"] = sql_eval

        # External Integration
        if "EXTERNAL_INTEGRATION" in dimensions:
            frozen_web = test_case.get("frozen_web_search", {})
            frozen_vector = test_case.get("frozen_vector_search", {})
            ext_eval = self.evaluate_external_integration(agent_report, frozen_web, frozen_vector)
            results["dimensions"]["EXTERNAL_INTEGRATION"] = ext_eval

        # Logical Reasoning
        if "LOGICAL_REASONING" in dimensions:
            chain = rubric.get("chain_validation", [])
            logic_eval = self.evaluate_logical_reasoning(agent_report, chain)
            results["dimensions"]["LOGICAL_REASONING"] = logic_eval

        # Completeness
        if "COMPLETENESS" in dimensions:
            query = test_case.get("enhanced_query", test_case.get("original_question", ""))
            comp_eval = self.evaluate_completeness(agent_report, query)
            results["dimensions"]["COMPLETENESS"] = comp_eval

        # Calculate weighted final score
        total_weight = 0
        weighted_sum = 0
        for dim_name, dim_result in results["dimensions"].items():
            weight = dimensions.get(dim_name, {}).get("weight", 0.25)
            weighted_sum += dim_result["score"] * weight
            total_weight += weight

        results["final_score"] = weighted_sum / total_weight if total_weight > 0 else 0

        # Tool usage evaluation
        if agent_tools:
            gold_subtasks = test_case.get("gold_subtasks", [])
            results["tool_usage"] = self.evaluate_tool_usage(gold_subtasks, agent_tools)

        return results
