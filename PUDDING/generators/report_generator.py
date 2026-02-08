"""Report task generation - enhanced queries and ground truth reports."""

import logging
import requests
import os
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from PUDDING.models.tree_models import ToolAction, TerminalPath

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate enhanced queries and ground truth reports from exploration results."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.conversation_history: List[dict] = []

    def _call_llm(self, prompt: str, max_tokens: int = 1000) -> str:
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": "anthropic/claude-opus-4.5",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                },
                timeout=60
            )
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
        return ""

    def _call_llm_with_messages(self, messages: List[dict], max_tokens: int = 2000) -> str:
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": "anthropic/claude-opus-4.5",
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                },
                timeout=90
            )
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
        return ""

    def generate_enhanced_query(self, original_query: str, terminal_path: TerminalPath) -> str:
        """Generate an enhanced analytical query that doesn't reveal data.

        Args:
            original_query: The original database query/question
            terminal_path: The exploration path with all retrieved context

        Returns:
            Enhanced query string (2-3 sentences)
        """
        # Summarize what the path discovered
        path_summary = self._summarize_path(terminal_path)

        prompt = f"""Transform this database query into a harder analytical question requiring multi-step reasoning.

ORIGINAL QUERY: {original_query[:400]}

CONTEXT DISCOVERED (do NOT reveal these specifics in the new query):
{path_summary}

The enhanced query should:
1. NOT reveal any specific data values - the agent must discover them through SQL + search
2. Require SQL execution + web search + vector search to fully answer
3. Be 2-3 sentences long
4. Ask for analysis that integrates database results with external context
5. Reference the need for industry trends, domain knowledge, or comparative analysis

Output only the enhanced query text, nothing else."""

        enhanced = self._call_llm(prompt, 300)
        if not enhanced or len(enhanced) < 50:
            enhanced = (
                f"Analyze {original_query[:200]}... by querying the database, "
                f"researching current context and domain knowledge."
            )
        return enhanced

    def generate_report(
        self,
        enhanced_query: str,
        sql_result: str,
        terminal_path: TerminalPath,
    ) -> str:
        """Generate ground truth report from enhanced query and exploration results.

        Returns:
            Markdown report with sections: Executive Summary, Data Analysis Results,
            External Context & Insights, Key Connections, Conclusions
        """
        # Gather search context from path
        web_context = ""
        vector_context = ""
        for action in terminal_path.actions:
            if action.tool_name == "web_search" and action.output and len(action.output) > 50:
                web_context += f"\n{action.output[:1500]}\n"
            elif action.tool_name == "vector_search" and action.output and len(action.output) > 50:
                vector_context += f"\n{action.output[:1000]}\n"

        prompt = f"""You are a data analyst writing a comprehensive analytical report. Generate a report in the EXACT format shown below.

QUERY TO ANSWER:
{enhanced_query}

SQL DATABASE RESULT:
{sql_result[:2000]}

WEB SEARCH FINDINGS:
{web_context[:3000]}

DOMAIN KNOWLEDGE (Vector Search):
{vector_context[:2000]}

Write the report using this EXACT markdown structure:

## Executive Summary
[2-3 sentences summarizing the key finding from SQL data and its significance]

## Data Analysis Results
[Detailed analysis of the SQL query results with specific numbers and statistics]

## External Context & Insights
[Integration of web search findings showing industry trends, market data, and real-world context. Mention "External knowledge (web summary)" and "External knowledge (vector database)" to cite sources]

## Key Connections
[How the SQL data connects to broader market trends, industry patterns, and the external research findings]

## Conclusions
[Final conclusions with actionable insights and implications]

REQUIREMENTS:
- Use specific numbers and statistics from the SQL result
- Reference external sources naturally (e.g., "External knowledge indicates...", "Industry research shows...")
- Keep each section 2-4 sentences
- Total length: 400-600 words
- Output ONLY the report content starting with "## Executive Summary" """

        # Initialize conversation for potential revision
        self.conversation_history = [
            {"role": "user", "content": prompt}
        ]

        report = self._call_llm_with_messages(self.conversation_history, 1800)

        if report:
            self.conversation_history.append({"role": "assistant", "content": report})

        if not report or len(report) < 200 or "## Executive Summary" not in report:
            report = self._fallback_report(enhanced_query, sql_result)

        return report

    def reflect_on_report(self, report: str, query: str, sql_result: str, terminal_path: TerminalPath) -> str:
        """Automated quality self-reflection on generated report.

        Returns:
            Reflection text with quality assessment and suggestions
        """
        prompt = f"""Evaluate this analytical report for quality.

QUERY: {query[:300]}
SQL RESULT: {sql_result[:200]}

REPORT:
{report[:2000]}

Evaluate on:
1. DATA GROUNDING: Does it use specific numbers from SQL results?
2. EXTERNAL INTEGRATION: Does it meaningfully incorporate web/vector search findings?
3. LOGICAL FLOW: Does the analysis follow logically from data to insight?
4. COMPLETENESS: Does it address all aspects of the query?
5. CROSS-SOURCE SYNTHESIS: Does it connect SQL data with external context?

Provide a brief assessment (2-3 sentences) and note any specific improvements needed.
If the report is high quality, say "QUALITY: HIGH" at the start.
If it needs improvement, say "QUALITY: NEEDS_IMPROVEMENT" and list specific issues."""

        return self._call_llm(prompt, 400)

    def revise_report(self, feedback: str, original_query: str) -> str:
        """Revise report based on expert feedback using conversation history.

        Args:
            feedback: Expert feedback text
            original_query: Original query for context

        Returns:
            Revised report
        """
        revision_prompt = f"""Please revise the report based on this feedback:

FEEDBACK: {feedback}

Requirements:
- Address all feedback points
- Maintain the exact same section structure (## Executive Summary, ## Data Analysis Results, etc.)
- Keep using specific numbers from the data
- Output ONLY the revised report starting with "## Executive Summary" """

        self.conversation_history.append({"role": "user", "content": revision_prompt})

        revised = self._call_llm_with_messages(self.conversation_history, 1800)

        if revised:
            self.conversation_history.append({"role": "assistant", "content": revised})

        if not revised or len(revised) < 200:
            # Return the last good version from history
            for msg in reversed(self.conversation_history):
                if msg["role"] == "assistant" and "## Executive Summary" in msg.get("content", ""):
                    return msg["content"]
            return revised or ""

        return revised

    def reset_conversation(self):
        """Reset conversation history for a new generation."""
        self.conversation_history = []

    def _summarize_path(self, path: TerminalPath) -> str:
        """Summarize what a terminal path discovered."""
        parts = []
        for i, action in enumerate(path.actions):
            if action.output and len(action.output) > 50:
                parts.append(f"- [{action.tool_name}] {action.output[:200]}...")
        return "\n".join(parts) if parts else "No significant findings."

    def _fallback_report(self, query: str, sql_result: str) -> str:
        """Generate fallback report when LLM fails."""
        return f"""## Executive Summary
Analysis of the query reveals key insights from the database results combined with external research.

## Data Analysis Results
The SQL query results show: {sql_result[:300]}...

## External Context & Insights
External knowledge provides additional context for interpreting these findings.

## Key Connections
The data patterns align with broader industry trends identified through external research.

## Conclusions
Further analysis is recommended to fully understand the implications of these findings."""
