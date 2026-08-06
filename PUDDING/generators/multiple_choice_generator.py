# -*- coding: utf-8 -*-
"""Multiple Choice Generator - LLM-based multiple-choice task generation."""

import os
import json
import random
import logging
import re
import requests
import statistics
from typing import Dict, List, Optional, Any, Callable, Set

from ..models.data_models import SubtaskResult

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]

_DISMISSIVE = re.compile(
    r"\b(only|never|simply|solely|merely|purely|always|cannot|"
    r"dismiss(?:ed|es|ing)?|unreliable|at face value|without any|"
    r"no need|irrelevant|meaningless)\b", re.I)
_MECHANICAL = re.compile(
    r"\b(sql|quer(?:y|ies)|join(?:ed|s)?|rows?|gold (?:output|result)|"
    r"database output|ordinal|schema|table|column|aggregation)\b", re.I)
_COUNT_HINT = re.compile(r"exactly \w+ (?:options|answers|are correct)|select exactly", re.I)


class MultipleChoiceGenerator:
    """Generates 8-option multiple-choice tasks with human feedback support."""

    def __init__(self, api_key: Optional[str] = None, model: str = "anthropic/claude-opus-4.5"):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.conversation_history: List[Dict] = []

    def call_llm(self, prompt: str) -> str:
        return self._call_llm_with_messages([{"role": "user", "content": prompt}])

    def _call_llm_with_messages(self, messages: List[Dict]) -> str:
        try:
            if not self.api_key:
                self.logger.warning("Missing OPENROUTER_API_KEY, returning empty response")
                return ""
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": m["role"], "content": m["content"]}
                        for m in messages if isinstance(m, dict) and "content" in m
                    ],
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_tokens": 4000,
                },
                timeout=(15, 300),
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('choices'):
                    return data['choices'][0]['message']['content'].strip()
            return ""
        except Exception as e:
            self.logger.error(f"LLM call error: {e}")
            return ""

    def generate_multiple_choice_content(
        self,
        original_query: str,
        subtask_results: Dict[str, SubtaskResult],
        max_attempts: int = 3,
    ) -> Dict[str, Any]:
        gold_result = subtask_results['gold_result'].result
        external_result = subtask_results['external_search'].result
        n_correct = random.choices([3, 4, 5], weights=[0.25, 0.5, 0.25])[0]
        early_pair_share = random.random() < 0.5

        prompt = self._build_prompt(
            original_query, gold_result, external_result, n_correct, early_pair_share)
        self.conversation_history.append({"role": "user", "content": prompt})

        question = None
        for attempt in range(1, max_attempts + 1):
            response = self._call_llm_with_messages(self.conversation_history)
            self.conversation_history.append({"role": "assistant", "content": response})

            candidate = self._safe_json_parse(response)
            problems = self.validate(candidate, gold_result) if candidate else \
                ["output was not valid JSON with the required fields"]
            if not problems:
                question = candidate
                break

            self.logger.info(f"attempt {attempt}: {len(problems)} problems -> revising")
            self.conversation_history.append({
                "role": "user",
                "content": (
                    "Your previous output was REJECTED by the mechanical validator. "
                    "Fix exactly the following and return the full corrected JSON:\n- "
                    + "\n- ".join(problems)
                ),
            })

        if question is None:
            return self._create_fallback(original_query, gold_result, external_result)

        return {
            "question_data": question,
            "advanced_query": question.get("question", original_query),
            "success": True,
            "conversation_history": self.conversation_history.copy(),
        }

    def revise_content(self, feedback: str) -> Dict[str, Any]:
        self.conversation_history.append({
            "role": "user",
            "content": (
                f"Expert feedback on the task above:\n{feedback}\n\n"
                "Revise the task accordingly. Keep every construction rule from the "
                "original instructions and return the full corrected JSON."
            ),
        })
        response = self._call_llm_with_messages(self.conversation_history)
        self.conversation_history.append({"role": "assistant", "content": response})
        candidate = self._safe_json_parse(response)
        if candidate:
            return {"question_data": candidate, "success": True,
                    "conversation_history": self.conversation_history.copy()}
        return {"question_data": None, "success": False,
                "conversation_history": self.conversation_history.copy()}

    def reset_conversation(self):
        self.conversation_history = []

    def _build_prompt(self, original_query: str, gold_result: str,
                      external_result: str, n_correct: int,
                      early_pair_share: bool) -> str:
        position_rule = (
            "in most pairs put the CORRECT member on the earlier letter"
            if early_pair_share else
            "in most pairs put the CORRECT member on the later letter")
        return (
            "=== INPUT DATA ===\n\n"
            f"## Original Query (Natural Language)\n{original_query}\n\n"
            f"## SQL Execution Result\n```\n{str(gold_result)[:1200]}\n```\n"
            "(only the first 1200 characters are shown; a solver will see exactly "
            "this window, so every database fact you use must be visible in it)\n\n"
            "## External Knowledge Context\n"
            f"{external_result[:2500]}{'...[truncated]' if len(external_result) > 2500 else ''}\n\n"

            "=== TASK: Generate a Cross-Source Multiple-Choice Task ===\n\n"
            "The task must be answerable ONLY by combining the SQL result with the "
            "external context. A solver holding either source alone must fail.\n\n"

            "## CONSTRUCTION RULES ##\n\n"
            f"1) Eight options A-H; exactly {n_correct} are correct. The question must "
            "NOT state or hint how many options are correct. End the question with "
            "'Select all that apply.'\n\n"

            "2) Organise the options as flipped pairs: two statements identical except "
            "for one factual direction (X above Y vs X below Y; leader P vs leader Q). "
            "Exactly one member of each pair is correct. Pair members must sit on "
            f"NON-ADJACENT letters, and {position_rule}.\n\n"

            "3) At least TWO pairs must be decided by a DATABASE fact. A database fact "
            "qualifies only if the query itself produces it: which entity leads in the "
            "computed column, a comparison between two named rows of this result, a "
            "derived ratio or share, which group holds the extreme of the derived "
            "measure. BANNED: facts that are true for almost any dataset ('a maximum "
            "exists', 'the total is positive'), and raw public facts a model could "
            "recall without the result (famous rankings, well-known statistics) - the "
            "underlying databases are public, so build on what the QUERY computes, not "
            "on what the world knows.\n\n"

            "4) At least ONE pair must be decided by the external context, quoting its "
            "distinctive wording so the deciding sentence can be found by keyword. Name "
            "the source inside the option ('the retrieved delivery study').\n\n"

            "5) Never write a pivot's numeric value inside the question or any option; "
            "refer to it by role ('the level the retrieved study reports'). Oppositions "
            "must be symmetric - two named entities, two directions, two periods - never "
            "affirm-vs-negate or careful-vs-careless, which lean one way a priori.\n\n"

            "6) Style, enforced by a mechanical validator after you answer:\n"
            "- mean length of correct options within 10% of mean distractor length\n"
            "- the single LONGEST option must be a distractor\n"
            "- no dismissive/absolute wording anywhere (only, never, simply, merely, "
            "always, cannot, unreliable, dismissed, 'at face value', ...)\n"
            "- correct options must not read more interpretive or hedged than "
            "distractors; truth is factual, never virtuous\n"
            "- no database vocabulary in the question (say 'our order records', not "
            "'the SQL result')\n\n"

            "## OUTPUT FORMAT ##\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "question": "... Select all that apply.",\n'
            '  "options": {"A": "...", "B": "...", "C": "...", "D": "...",\n'
            '              "E": "...", "F": "...", "G": "...", "H": "..."},\n'
            f'  "correct_answer": [/* exactly {n_correct} letters */],\n'
            '  "db_pairs": [{"correct_letter": "B", "distractor_letter": "F",\n'
            '                "deciding_fact": "what in the result decides it",\n'
            '                "where_in_result": "the visible fragment"}],\n'
            '  "doc_pairs": [{"correct_letter": "C", "distractor_letter": "G",\n'
            '                 "deciding_quote": "verbatim from the context"}],\n'
            '  "explanation": "why each correct option holds and each distractor fails"\n'
            "}\n"
        )

    def validate(self, q: Dict[str, Any], gold_result: str) -> List[str]:
        probs: List[str] = []
        options = q.get("options") or {}
        if sorted(options.keys()) != LETTERS:
            return ["options must be exactly the eight letters A-H"]
        key = q.get("correct_answer") or []
        if not isinstance(key, list) or not 3 <= len(set(key)) <= 6 \
                or not set(key) <= set(LETTERS):
            probs.append("correct_answer must be 3-6 distinct letters from A-H")
        cs = set(key)

        question = q.get("question", "")
        if not question.strip():
            probs.append("question is empty")
        if _COUNT_HINT.search(question):
            probs.append("question must not disclose how many options are correct")
        if _MECHANICAL.search(question):
            probs.append("database vocabulary leaked into the question "
                         "(say 'our order records', not 'the SQL result')")

        lc = [len(options[k]) for k in options if k in cs]
        ld = [len(options[k]) for k in options if k not in cs]
        if lc and ld:
            ratio = statistics.mean(lc) / statistics.mean(ld)
            if not 0.90 <= ratio <= 1.10:
                probs.append(f"length ratio correct/distractor is {ratio:.2f}, "
                             f"must be within 0.90-1.10")
            if max(options, key=lambda k: len(options[k])) in cs:
                probs.append("the longest option is correct; make a distractor longest")
            k_len = len(cs)
            by_len = sorted(options, key=lambda k: len(options[k]), reverse=True)
            if set(by_len[:k_len]) == cs or set(by_len[-k_len:]) == cs:
                probs.append("the K longest (or K shortest) options reproduce the key")
        bad = [k for k in options if _DISMISSIVE.search(options[k])]
        if bad:
            probs.append(f"dismissive/absolute wording in options {bad}")

        db_pairs = q.get("db_pairs") or []
        if len(db_pairs) < 2:
            probs.append("declare at least TWO db_pairs decided by the query result")
        seen: Set[str] = set()
        for p in db_pairs + (q.get("doc_pairs") or []):
            cl, dl = p.get("correct_letter"), p.get("distractor_letter")
            if not (cl in LETTERS and dl in LETTERS):
                probs.append("every pair needs correct_letter and distractor_letter A-H")
                continue
            if cl in seen or dl in seen:
                probs.append("an option may belong to at most one pair")
            seen.update((cl, dl))
            if cl not in cs:
                probs.append(f"pair says {cl} is correct but {cl} is not in the key")
            if dl in cs:
                probs.append(f"pair says {dl} is the distractor but {dl} is in the key "
                             f"- contradictory members must not both be correct")
            if abs(LETTERS.index(cl) - LETTERS.index(dl)) <= 1:
                probs.append(f"pair {cl}/{dl} sits on adjacent letters")

        alnum = lambda x: re.sub(r"[^A-Za-z0-9]", "", str(x)).lower()
        window = alnum(str(gold_result)[:1200])
        for p in db_pairs:
            frag = alnum(p.get("where_in_result", ""))
            if frag and len(frag) > 12 and frag[:48] not in window:
                probs.append("a db_pair cites a result fragment outside the first 1200 "
                             "characters - the solver never sees it")
        return probs

    def _safe_json_parse(self, content: str) -> Optional[Dict[str, Any]]:
        try:
            cleaned = content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            data = json.loads(cleaned.strip())
            if isinstance(data, dict) and "options" in data:
                return data
            return None
        except (json.JSONDecodeError, TypeError):
            return None

    def _create_fallback(self, original_query: str, gold_result: str,
                         external_result: str) -> Dict[str, Any]:
        self.logger.warning("multiple-choice generation failed after all attempts")
        return {
            "question_data": None,
            "advanced_query": original_query,
            "success": False,
            "error": "generation did not pass mechanical validation",
            "conversation_history": self.conversation_history.copy(),
        }


class AblationGate:
    """Measured acceptance gate over question-only / documents-only / full evidence."""

    def __init__(self, solve_fn: Callable[[str], str], n_samples: int = 4,
                 max_blind: int = 1, max_docs_only: int = 2, min_full: int = 3):
        self.solve_fn = solve_fn
        self.n = n_samples
        self.max_blind = max_blind
        self.max_docs_only = max_docs_only
        self.min_full = min_full

    @staticmethod
    def _prompt(question: Dict[str, Any], evidence: str) -> str:
        opts = "\n".join(f"{k}. {v}" for k, v in question["options"].items())
        s = (f"Multiple-choice. Select ALL correct options.\n\n"
             f"Question: {question['question']}\n\nOptions:\n{opts}\n")
        if evidence:
            s += f"\nEvidence:\n{evidence}\n"
        return s + "\nRespond with only the correct letters, comma separated, prefixed by 'ANSWER:'."

    @staticmethod
    def _parse(text: str) -> Set[str]:
        m = re.search(r"(?:ANSWER|FINAL)[^A-H]*([A-H][^\n]*)", (text or "").upper())
        return set(re.findall(r"\b([A-H])\b", m.group(1) if m else (text or "").upper()))

    def evaluate(self, question: Dict[str, Any], sql_result: str,
                 documents: str) -> Dict[str, Any]:
        key = set(question["correct_answer"])
        db_text = "DATABASE RESULT: " + str(sql_result)[:1200]
        conditions = {
            "question_only": "",
            "documents_only": documents,
            "full": db_text + "\n\n" + documents,
        }
        hits = {}
        for name, evidence in conditions.items():
            correct = 0
            for _ in range(self.n):
                reply = self.solve_fn(self._prompt(question, evidence))
                if self._parse(reply) == key:
                    correct += 1
            hits[name] = correct
        passed = (hits["question_only"] <= self.max_blind
                  and hits["documents_only"] <= self.max_docs_only
                  and hits["full"] >= self.min_full)
        return {"passed": passed, "hits": hits, "n_samples": self.n}
