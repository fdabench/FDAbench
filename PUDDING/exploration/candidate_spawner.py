"""SpawnCandidates: Propose exploration branches from a frontier node."""

import json
import logging
import requests
import os
from typing import List

from dotenv import load_dotenv
load_dotenv()

from PUDDING.models.tree_models import ToolAction, ExplorationNode

logger = logging.getLogger(__name__)


class CandidateSpawner:
    """LLM-based candidate branch spawner for tree exploration."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')

    def _call_llm(self, prompt: str, max_tokens: int = 1000) -> str:
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

    def spawn_candidates(self, node: ExplorationNode, query: str, sql_result: str, revision_feedback: str = "") -> List[ToolAction]:
        """Propose 1-4 candidate exploration branches from a frontier node.

        Args:
            node: The frontier node to expand from
            query: Original analytical query
            sql_result: SQL execution result
            revision_feedback: Optional expert feedback requesting additional evidence

        Returns:
            List of ToolAction candidates (not yet executed)
        """
        # Build context summary from cumulative actions
        existing_context = self._summarize_context(node.cumulative_actions)

        feedback_block = ""
        if revision_feedback:
            feedback_block = f"""
EXPERT REVISION FEEDBACK (address these evidence gaps):
{revision_feedback[:500]}
"""

        prompt = f"""You are exploring information sources to build a comprehensive analytical report.

QUERY: {query[:500]}
SQL RESULT: {sql_result[:300]}

ALREADY RETRIEVED:
{existing_context}
{feedback_block}
Available tools:
- web_search: Get current real-world data, news, statistics, market information
- vector_search: Get domain knowledge, academic research, methodology, technical documentation
- file_search: Search local file system for domain-specific documents and reference materials
- db_explore: Explore database schema, table relationships, column metadata, sample data

Propose 1-4 NEW search actions that would add non-redundant value. Each should target different information needs not already covered.

Requirements:
1. Each action must ADD NEW information (not duplicate existing)
2. Queries should be specific and targeted
3. Consider what's still needed for a complete analytical report
4. If existing context is already sufficient, propose fewer actions

Output JSON array only (no markdown, no explanation):
[
  {{"tool": "web_search", "query": "specific search query here", "rationale": "what new information this adds"}},
  {{"tool": "vector_search", "query": "specific search query here", "rationale": "what domain knowledge this provides"}}
]"""

        content = self._call_llm(prompt, 800)
        return self._parse_candidates(content, query)

    def _summarize_context(self, actions: List[ToolAction]) -> str:
        """Summarize what has already been retrieved."""
        if not actions:
            return "Nothing retrieved yet (starting fresh)."

        parts = []
        for i, action in enumerate(actions):
            status = "OK" if action.output and len(action.output) > 50 else "FAILED/EMPTY"
            parts.append(f"  Step {i+1} [{action.tool_name}]: {action.input_params.get('query', '')[:80]} -> {status}")
            if action.output and len(action.output) > 50:
                parts.append(f"    Summary: {action.output[:200]}...")
        return "\n".join(parts)

    def _parse_candidates(self, content: str, query: str) -> List[ToolAction]:
        """Parse LLM response into ToolAction list."""
        try:
            # Clean up response
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            items = json.loads(content)
            candidates = []
            for item in items[:4]:  # Max 4 candidates
                tool_name = item.get("tool", "web_search")
                if tool_name not in ("web_search", "vector_search", "file_search", "db_explore"):
                    tool_name = "web_search"
                candidates.append(ToolAction(
                    tool_name=tool_name,
                    input_params={"query": item.get("query", "")},
                    rationale=item.get("rationale", ""),
                ))
            return candidates if candidates else self._fallback_candidates(query)

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse candidates: {e}")
            return self._fallback_candidates(query)

    def _fallback_candidates(self, query: str) -> List[ToolAction]:
        """Fallback candidates if LLM parsing fails."""
        return [
            ToolAction(
                tool_name="web_search",
                input_params={"query": f"current trends statistics {query[:80]}"},
                rationale="Get real-world context and current data",
            ),
            ToolAction(
                tool_name="vector_search",
                input_params={"query": f"methodology analysis {query[:80]}"},
                rationale="Get domain knowledge and research findings",
            ),
        ]
