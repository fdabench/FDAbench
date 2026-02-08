"""Single source validator - reject tasks solvable from a single source."""

import json
import logging
import requests
import os

from dotenv import load_dotenv
load_dotenv()

from PUDDING.models.tree_models import TerminalPath

logger = logging.getLogger(__name__)


class SingleSourceValidator:
    """LLM-based check: can the task be answered from a single source alone?"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')

    def _call_llm(self, prompt: str, max_tokens: int = 300) -> str:
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

    def is_multi_source(self, query: str, sql_result: str, report: str, terminal_path: TerminalPath) -> bool:
        """Check if the task truly requires multiple sources.

        Symmetric check: reject if EITHER SQL alone OR external knowledge alone
        can answer the query.

        Returns:
            True if multi-source (keep the task), False if single-source (reject).
        """
        # Count distinct successful source types
        source_types = set()
        source_types.add("sql")  # SQL is always a source

        for action in terminal_path.actions:
            prov = action.provenance or {}
            if prov.get("success", False):
                source_types.add(action.tool_name)

        # Quick check: if only SQL data was used (no successful searches), reject
        if len(source_types) < 2:
            logger.info("Single source check: only SQL data, rejecting")
            return False

        # Summarize external evidence
        external_summary = ""
        for action in terminal_path.actions:
            if action.output and len(action.output) > 50:
                external_summary += f"[{action.tool_name}]: {action.output[:300]}\n"

        # LLM symmetric verification: check both directions
        prompt = f"""Analyze whether this analytical report truly requires BOTH SQL data AND external knowledge, or could be answered from a single source alone.

QUERY: {query[:300]}
SQL RESULT: {sql_result[:200]}

EXTERNAL EVIDENCE SUMMARY:
{external_summary[:600]}

REPORT EXCERPT:
{report[:500]}

Answer TWO questions:
1. Could the report's conclusions be drawn from SQL data alone WITHOUT the external search results?
2. Could the report's conclusions be drawn from external knowledge alone WITHOUT the SQL data?

If EITHER answer is "yes", the task is single-source and should be rejected.

Output JSON only:
{{"sql_alone_sufficient": true/false, "external_alone_sufficient": true/false, "rationale": "brief explanation"}}"""

        content = self._call_llm(prompt)
        try:
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            data = json.loads(content)
            sql_alone = data.get("sql_alone_sufficient", False)
            ext_alone = data.get("external_alone_sufficient", False)
            rationale = data.get("rationale", "")
            logger.info(
                f"Single source check: sql_alone_sufficient={sql_alone}, "
                f"external_alone_sufficient={ext_alone}, reason={rationale[:100]}"
            )
            # Reject if either source alone is sufficient
            if sql_alone or ext_alone:
                return False
            return True
        except (json.JSONDecodeError, KeyError):
            # Default to keeping the task if LLM parsing fails
            logger.warning("Failed to parse single source check, defaulting to multi-source")
            return len(source_types) >= 2
