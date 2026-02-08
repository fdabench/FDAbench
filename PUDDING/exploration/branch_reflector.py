"""SelfReflect: Evaluate exploration branches for PRUNE/CONTINUE/SUFFICIENT decisions."""

import json
import logging
import requests
import os
from typing import Tuple

from dotenv import load_dotenv
load_dotenv()

from PUDDING.models.tree_models import ToolAction, ExplorationNode, BranchDecision

logger = logging.getLogger(__name__)


class BranchReflector:
    """LLM-based self-reflection for exploration branch decisions."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')

    def _call_llm(self, prompt: str, max_tokens: int = 500) -> str:
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

    def reflect(self, node: ExplorationNode, query: str, sql_result: str) -> Tuple[BranchDecision, str]:
        """Evaluate a node's cumulative context and decide PRUNE/CONTINUE/SUFFICIENT.

        Criteria:
        1. Evidence diversity - spans structured + unstructured?
        2. Relevance - artifact relates to query?
        3. Sufficiency - enough cross-source evidence for multi-source report?
        4. Redundancy - duplicates existing evidence?

        Returns:
            (decision, rationale) tuple
        """
        context_summary = self._build_context_summary(node)

        prompt = f"""Evaluate the following exploration context for building an analytical report.

QUERY: {query[:500]}
SQL RESULT: {sql_result[:300]}

ACCUMULATED EVIDENCE ({len(node.cumulative_actions)} steps):
{context_summary}

Evaluate against these criteria:
1. EVIDENCE DIVERSITY: Does the evidence span both structured data (SQL) and unstructured sources (web/vector)?
2. RELEVANCE: Is the latest retrieved artifact relevant to the query?
3. SUFFICIENCY: Is there enough cross-source evidence to write a comprehensive multi-source report?
4. REDUNDANCY: Does the latest step duplicate information already available?

Based on your evaluation, decide ONE of:
- SUFFICIENT: We have enough diverse, relevant evidence for a high-quality report (typically need at least 1 web + 1 vector search result alongside SQL)
- CONTINUE: Evidence is promising but more sources are needed for completeness
- PRUNE: Latest retrieval is irrelevant, redundant, or failed; this branch is unproductive

Output JSON only (no markdown):
{{"decision": "SUFFICIENT|CONTINUE|PRUNE", "rationale": "brief explanation"}}"""

        content = self._call_llm(prompt, 300)
        return self._parse_decision(content, node)

    def _build_context_summary(self, node: ExplorationNode) -> str:
        """Build summary of all accumulated evidence."""
        parts = []
        for i, action in enumerate(node.cumulative_actions):
            success = bool(action.output and len(action.output) > 50)
            status = "SUCCESS" if success else "FAILED/EMPTY"
            parts.append(f"Step {i+1} [{action.tool_name}] ({status}):")
            parts.append(f"  Query: {action.input_params.get('query', '')[:100]}")
            if success:
                parts.append(f"  Result: {action.output[:300]}...")
            else:
                parts.append(f"  Result: {action.output[:100]}")

            # Show provenance highlights
            prov = action.provenance or {}
            if prov.get("urls"):
                parts.append(f"  URLs: {len(prov['urls'])} found")
            if prov.get("chunks"):
                parts.append(f"  Chunks: {len(prov['chunks'])} found")

        return "\n".join(parts) if parts else "No evidence collected yet."

    def _parse_decision(self, content: str, node: ExplorationNode) -> Tuple[BranchDecision, str]:
        """Parse LLM decision response."""
        try:
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)
            decision_str = data.get("decision", "CONTINUE").upper()
            rationale = data.get("rationale", "")

            if decision_str == "SUFFICIENT":
                return BranchDecision.SUFFICIENT, rationale
            elif decision_str == "PRUNE":
                return BranchDecision.PRUNE, rationale
            else:
                return BranchDecision.CONTINUE, rationale

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse reflection decision: {e}")
            # Fallback heuristic: if we have 2+ successful actions, mark sufficient
            successful = sum(1 for a in node.cumulative_actions if a.output and len(a.output) > 50)
            if successful >= 2:
                return BranchDecision.SUFFICIENT, "Fallback: sufficient evidence accumulated"
            elif successful == 0 and len(node.cumulative_actions) > 0:
                return BranchDecision.PRUNE, "Fallback: no successful retrievals"
            else:
                return BranchDecision.CONTINUE, "Fallback: need more evidence"
