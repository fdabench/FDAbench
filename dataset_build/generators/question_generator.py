# -*- coding: utf-8 -*-
"""
Question Generator - LLM-based single choice question generation.
Extracted from main.py.
"""

import os
import json
import random
import logging
import requests
from typing import Dict, List, Optional, Any

from ..models.data_models import SubtaskResult


class SingleChoiceGenerator:
    """Handles single choice question generation with human feedback support."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.logger = logging.getLogger(__name__)
        self.conversation_history: List[Dict] = []

    def call_llm(self, prompt: str) -> str:
        """Call LLM with a single prompt."""
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
                    "model": "anthropic/claude-sonnet-4",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_tokens": 3000
                }
            )

            if response.status_code == 200:
                response_data = response.json()
                if 'choices' in response_data and response_data['choices']:
                    return response_data['choices'][0]['message']['content'].strip()

            return ""

        except Exception as e:
            self.logger.error(f"LLM call error: {e}")
            return ""

    def _call_llm_with_messages(self, messages: List[Dict]) -> str:
        """Call LLM with message history."""
        try:
            if not self.api_key:
                self.logger.warning("Missing OPENROUTER_API_KEY, returning empty response")
                return ""

            openai_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    if "role" in msg and "content" in msg:
                        openai_messages.append({"role": msg["role"], "content": msg["content"]})
                else:
                    openai_messages.append({"role": "user", "content": str(msg)})

            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "anthropic/claude-sonnet-4",
                    "messages": openai_messages,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_tokens": 3000
                }
            )

            if response.status_code == 200:
                response_data = response.json()
                if 'choices' in response_data and response_data['choices']:
                    return response_data['choices'][0]['message']['content'].strip()

            return ""

        except Exception as e:
            self.logger.error(f"LLM call with messages error: {e}")
            return ""

    def generate_single_choice_content(
        self,
        original_query: str,
        subtask_results: Dict[str, SubtaskResult]
    ) -> Dict[str, Any]:
        """Generate single choice questions using subtask results with random correct answer."""
        gold_result = subtask_results['gold_result'].result
        external_result = subtask_results['external_search'].result
        correct_options = ['A', 'B', 'C', 'D']
        random_correct_answer = random.choice(correct_options)

        # Structure the input data clearly for the LLM
        prompt_parts = [
            "=== INPUT DATA ===\n\n",
            f"## Original Query (Natural Language)\n{original_query}\n\n",
            f"## SQL Execution Result\n```\n{gold_result}\n```\n\n",
            "## External Knowledge Context\n",
            f"{external_result[:2000]}{'...[truncated]' if len(external_result) > 2000 else ''}\n\n",
            "## Data Quality Notes\n",
            "- Review if the SQL result actually answers the original query\n",
            "- Check if any filters (e.g., gender, date range) might be missing\n",
            "- Consider what the count represents: unique entities or instances?\n\n",
        ]

        unified_prompt = "".join(prompt_parts) + self._get_prompt_template()
        self.conversation_history.append({"role": "user", "content": unified_prompt})

        response = self.call_llm(unified_prompt)
        self.conversation_history.append({"role": "assistant", "content": response})

        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            result = json.loads(cleaned_response)
            questions = result.get("single_choice_questions", [])

            if questions and len(questions) > 0:
                question = questions[0]
                if all(key in question for key in ["question", "options", "correct_answer", "explanation"]):
                    # Validate the correct answer is one of A, B, C, D
                    correct_ans = question.get("correct_answer", [])
                    if isinstance(correct_ans, list) and len(correct_ans) == 1:
                        if correct_ans[0] in ['A', 'B', 'C', 'D']:
                            # Accept the LLM's choice - it knows which answer matches the explanation
                            self.logger.info(f"LLM selected correct answer: {correct_ans}")
                        else:
                            # Invalid answer, use random
                            self.logger.warning(f"Invalid answer {correct_ans}, using {random_correct_answer}")
                            question["correct_answer"] = [random_correct_answer]
                    else:
                        # Fix format if needed
                        if isinstance(correct_ans, str) and correct_ans in ['A', 'B', 'C', 'D']:
                            question["correct_answer"] = [correct_ans]
                        else:
                            question["correct_answer"] = [random_correct_answer]

                    return {
                        "question_data": question,
                        "advanced_query": question.get("question", original_query),
                        "success": True,
                        "conversation_history": self.conversation_history.copy()
                    }

            return self._create_fallback_single_choice(original_query, gold_result, external_result, random_correct_answer)

        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parsing failed: {e}, using fallback")
            return self._create_fallback_single_choice(original_query, gold_result, external_result, random_correct_answer)

    def _get_prompt_template(self, correct_answer: str = None) -> str:
        """Get the prompt template for single choice generation - optimized for data grounding."""
        # Note: correct_answer parameter kept for backward compatibility but no longer used
        return (
            "=== TASK: Generate a Data-Grounded Single Choice Question ===\n\n"

            "You are creating a question that tests the ability to integrate structured data (SQL results) "
            "with external knowledge (web/domain context) to reach a correct conclusion.\n\n"

            "## CRITICAL REQUIREMENTS ##\n\n"

            "1) DATA GROUNDING (Most Important)\n"
            "- The CORRECT answer MUST be verifiable from the provided SQL result + external knowledge\n"
            "- Do NOT make claims that cannot be supported by the given data\n"
            "- If the SQL result has limitations (e.g., missing filters, ambiguous counts), acknowledge this\n"
            "- The question should test data interpretation skills, NOT domain expertise\n\n"

            "2) QUESTION DESIGN\n"
            "- Keep the question direct and practical - avoid fictional analyst scenarios\n"
            "- Focus on: What does this data tell us? What are its limitations? How should we interpret it?\n"
            "- The question should require BOTH the SQL result AND external context to answer correctly\n"
            "- Do not reveal the exact numeric answer in the question\n\n"

            "3) ANSWER OPTIONS\n"
            "- Create exactly 4 options (A, B, C, D)\n"
            "- ONE option must be clearly CORRECT based on the data + external context\n"
            "- THREE options must be plausible but WRONG for specific reasons:\n"
            "  * One wrong option: misreads or misinterprets the data\n"
            "  * One wrong option: ignores the external context\n"
            "  * One wrong option: overclaims or makes unsupported conclusions\n"
            "- Place the correct answer randomly among the options\n"
            "- The correct answer must be VERIFIABLE from the given information\n\n"

            "4) COMMON PITFALLS TO AVOID\n"
            "- Do NOT claim 'significant progress' or trends without comparative data\n"
            "- Do NOT assume the count represents unique individuals vs. role instances\n"
            "- Do NOT add fictional context (analysts, reports, committees)\n"
            "- Do NOT make the correct answer obvious by using superlatives\n"
            "- ENSURE the correct answer can be logically derived from the given information\n\n"

            "5) EXPLANATION REQUIREMENTS\n"
            "- Clearly state which data points support the correct answer\n"
            "- Explain why each wrong option fails (what data/logic contradicts it)\n"
            "- Be specific about what the SQL result shows and what the external context adds\n\n"

            "## OUTPUT FORMAT ##\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "single_choice_questions": [{\n'
            '    "question": "A clear, direct question about interpreting the data...",\n'
            '    "options": {\n'
            '      "A": "First option...",\n'
            '      "B": "Second option...",\n'
            '      "C": "Third option...",\n'
            '      "D": "Fourth option..."\n'
            '    },\n'
            '    "correct_answer": ["X"],  // X = A, B, C, or D - whichever is correct based on data\n'
            '    "explanation": "The correct answer is X because [specific data points from SQL result and external context]. '
            'Option Y is wrong because [specific reason]. Option Z is wrong because [specific reason]..."\n'
            "  }]\n"
            "}\n\n"

            "Return only the JSON, no other text."
        )

    def _get_hard_prompt_template(self) -> str:
        """Prompt for hard difficulty with multi-step reasoning chains."""
        return (
            "=== TASK: Generate a HARD Multi-Step Reasoning Question ===\n\n"

            "You have access to MULTIPLE data sources gathered through a chain of tool calls.\n"
            "Create a question that REQUIRES integrating ALL sources to answer correctly.\n\n"

            "## DIFFICULTY: HARD ##\n"
            "This question must require:\n"
            "1. Multi-step reasoning across 3+ data sources\n"
            "2. Resolving apparent contradictions between sources\n"
            "3. Synthesizing quantitative (SQL) and qualitative (web/vector) evidence\n"
            "4. Critical evaluation of source reliability and limitations\n\n"

            "## DATA SOURCES PROVIDED ##\n"
            "- SQL Result: Precise quantitative data from database\n"
            "- Web Search: Current events, news, real-world context\n"
            "- Vector DB: Domain knowledge, theoretical frameworks, documentation\n\n"

            "## QUESTION DESIGN ##\n"
            "- The answer should NOT be derivable from any single source\n"
            "- Wrong answers should each fail because they ignore one data source\n"
            "- The correct answer must explicitly synthesize multiple sources\n"
            "- Test analytical reasoning, not factual recall\n\n"

            "## ANSWER OPTIONS ##\n"
            "- A: Plausible but only uses SQL data (ignores external context)\n"
            "- B: Plausible but only uses web context (ignores database facts)\n"
            "- C: Plausible but only uses vector knowledge (ignores current data)\n"
            "- D: Correct synthesis of ALL sources with proper reasoning\n"
            "- Shuffle positions so correct answer is random\n\n"

            "## OUTPUT FORMAT ##\n"
            "{\n"
            '  "single_choice_questions": [{\n'
            '    "question": "Complex question requiring multi-source synthesis...",\n'
            '    "options": {"A": "...", "B": "...", "C": "...", "D": "..."},\n'
            '    "correct_answer": ["X"],\n'
            '    "explanation": "X is correct because it synthesizes: (1) SQL shows..., '
            '(2) Web context reveals..., (3) Domain knowledge indicates... '
            'Other options fail because they only consider partial evidence."\n'
            "  }]\n"
            "}\n\n"
            "Return only JSON."
        )

    def generate_hard_content(self, original_query: str, chain_results: str, sql_result: str) -> Dict[str, Any]:
        """Generate hard difficulty question from multi-step chain results."""
        prompt = (
            "=== INPUT DATA (Multi-Step Chain) ===\n\n"
            f"## Original Query\n{original_query}\n\n"
            f"## SQL Result\n```\n{sql_result}\n```\n\n"
            f"## Multi-Step Tool Chain Results\n{chain_results}\n\n"
            + self._get_hard_prompt_template()
        )

        self.conversation_history = [{"role": "user", "content": prompt}]
        response = self.call_llm(prompt)
        self.conversation_history.append({"role": "assistant", "content": response})

        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

            result = json.loads(cleaned.strip())
            questions = result.get("single_choice_questions", [])

            if questions:
                q = questions[0]
                if all(k in q for k in ["question", "options", "correct_answer", "explanation"]):
                    return {
                        "question_data": q,
                        "advanced_query": q.get("question", original_query),
                        "success": True,
                        "conversation_history": self.conversation_history.copy()
                    }
        except Exception as e:
            self.logger.warning(f"Hard content generation failed: {e}")

        return self._create_fallback_single_choice(original_query, sql_result, chain_results, 'D')

    def _create_fallback_single_choice(
        self,
        original_query: str,
        gold_result: str,
        external_result: str,
        correct_answer: str = 'B'
    ) -> Dict[str, Any]:
        """Create fallback single choice question when generation fails."""
        options = {
            "A": "Option A - Initial analysis approach",
            "B": "Option B - Alternative analytical approach",
            "C": "Option C - Alternative interpretation",
            "D": "Option D - Incorrect methodology"
        }

        options[correct_answer] = options[correct_answer].replace("Alternative", "Correct") + " (Correct)"

        fallback_question = {
            "question": f"{original_query} Based on the data analysis, what can be inferred?",
            "options": options,
            "correct_answer": [correct_answer],
            "explanation": f"Fallback explanation - option {correct_answer} is correct as it requires combining database results with external context insights."
        }

        return {
            "question_data": fallback_question,
            "advanced_query": f"{original_query} (Enhanced with analytical extensions)",
            "success": False,
            "conversation_history": self.conversation_history.copy()
        }

    def revise_content(
        self,
        feedback: str,
        conversation_history: List[Dict],
        current_question_data: Dict,
        original_query: str,
        gold_result: str,
        external_result: str
    ) -> Dict[str, Any]:
        """Revise content based on feedback."""
        revision_prompt = (
            f"Based on the feedback: {feedback}\n\n"
            "Please revise the previous response to address the feedback. "
            "Maintain the same JSON format. Use 'single_choice_questions' key. "
            "Improve all components based on the suggestions while keeping the structure requirements."
        )

        messages = conversation_history + [{"role": "user", "content": revision_prompt}]
        response = self._call_llm_with_messages(messages)
        parsed_json, error_msg = self._safe_json_parse(response, "revise_content")

        if parsed_json is not None:
            try:
                single_choices = parsed_json.get("single_choice_questions", [])
                advanced_query = parsed_json.get("advanced_query", "")
                validated_choices = self._validate_single_choices(single_choices)
                updated_history = messages + [{"role": "assistant", "content": response}]

                result: Dict[str, Any] = {
                    "conversation_history": updated_history,
                    "success": True
                }

                if validated_choices:
                    question_data = validated_choices[0]
                    result.update({
                        "question_data": question_data,
                        "advanced_query": question_data.get("question", advanced_query or original_query),
                        "single_choice_questions": validated_choices
                    })
                else:
                    result.update({
                        "question_data": current_question_data,
                        "advanced_query": current_question_data.get("question", ""),
                        "success": False
                    })

                return result

            except Exception as e:
                self.logger.error(f"Error in revision: {e}")
                return self._create_revision_fallback(current_question_data, conversation_history, feedback)
        else:
            self.logger.warning(f"Failed to parse revised response: {error_msg}")
            return self._create_revision_fallback(current_question_data, conversation_history, feedback)

    def _safe_json_parse(self, content: str, function_name: str = ""):
        """Safe JSON parsing with detailed error handling."""
        try:
            content = content.strip()

            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            parsed_json = json.loads(content)
            return parsed_json, None

        except json.JSONDecodeError as e:
            error_msg = f"JSON parse error in {function_name}: {e}"
            self.logger.error(error_msg)
            return None, error_msg
        except Exception as e:
            error_msg = f"Other parsing error in {function_name}: {e}"
            self.logger.error(error_msg)
            return None, error_msg

    def _validate_single_choices(self, choices) -> List[Dict]:
        """Validate and clean single choice questions."""
        if not isinstance(choices, list):
            return [self._get_default_single_choice()]

        validated = []
        for choice in choices:
            if isinstance(choice, dict) and all(key in choice for key in ["question", "options", "correct_answer", "explanation"]):
                options = choice.get("options", {})
                if isinstance(options, dict) and all(key in options for key in ["A", "B", "C", "D"]):
                    correct_answer = choice.get("correct_answer", [])
                    if isinstance(correct_answer, list) and len(correct_answer) == 1:
                        validated.append(choice)
                    else:
                        fixed_choice = dict(choice)
                        if isinstance(correct_answer, list) and len(correct_answer) > 1:
                            fixed_choice["correct_answer"] = [correct_answer[0]]
                        elif not isinstance(correct_answer, list):
                            fixed_choice["correct_answer"] = [str(correct_answer)]
                        else:
                            fixed_choice["correct_answer"] = [random.choice(['A', 'B', 'C', 'D'])]
                        validated.append(fixed_choice)
                else:
                    validated.append(self._get_default_single_choice())
            else:
                validated.append(self._get_default_single_choice())

        if len(validated) == 0:
            validated.append(self._get_default_single_choice())

        return validated[:1]

    def _get_default_single_choice(self) -> Dict[str, Any]:
        """Get a default single choice structure with random correct answer."""
        random_correct_answer = random.choice(['A', 'B', 'C', 'D'])

        options = {
            "A": "Option A",
            "B": "Option B",
            "C": "Option C",
            "D": "Option D"
        }

        options[random_correct_answer] += " (Correct)"

        return {
            "question": "Default single choice question due to validation failure",
            "options": options,
            "correct_answer": [random_correct_answer],
            "explanation": f"Default explanation due to validation failure - option {random_correct_answer} is the correct answer"
        }

    def _create_revision_fallback(
        self,
        current_question_data: Dict,
        conversation_history: List[Dict],
        feedback: str
    ) -> Dict[str, Any]:
        """Create fallback result when revision fails."""
        updated_history = conversation_history + [
            {"role": "user", "content": f"User requested revision with feedback: {feedback}"},
            {"role": "assistant", "content": "Revision failed, keeping original content"}
        ]

        return {
            "question_data": current_question_data,
            "advanced_query": current_question_data.get("question", ""),
            "conversation_history": updated_history,
            "success": False
        }

    def reflect_on_content(
        self,
        question_data: Dict,
        original_query: str,
        gold_result: str,
        external_result: str
    ) -> str:
        """Generate reflection on the quality of generated content."""
        reflection_prompt = (
            f"Please review the following single choice question for quality:\n\n"
            f"Original query: {original_query}\n"
            f"Generated question: {question_data.get('question', '')}\n"
            f"Options: {json.dumps(question_data.get('options', {}), indent=2)}\n"
            f"Correct answer: {question_data.get('correct_answer', [])}\n"
            f"Explanation: {question_data.get('explanation', '')}\n\n"
            "Evaluate based on these criteria:\n"
            "1. Accuracy (does it match the original intent and data?)\n"
            "2. Completeness (does it fully test understanding?)\n"
            "3. Clarity (is it clear and logical?)\n"
            "4. Difficulty (appropriate analytical challenge?)\n\n"
            "Respond in this exact format:\n"
            "Decision: [Acceptable/Needs Revision]\n"
            "Reason: [Your reasoning in 1-2 sentences]\n"
            "Suggestions: [Specific suggestions if Decision is 'Needs Revision', otherwise write 'None']"
        )

        return self.call_llm(reflection_prompt)

    def reset_conversation(self):
        """Reset conversation history for a new query."""
        self.conversation_history = []
