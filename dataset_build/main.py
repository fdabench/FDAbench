#!/home/wang/miniconda3/envs/bench/bin/python
# -*- coding: utf-8 -*-
"""
Refactored Bird Dataset Build Script - Main Application
Split from original bird_med_websearch_build_oop.py and adapted for Bird dataset
"""

import os
import sys
import json
import logging
import time
import argparse
import glob
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import external tools
from external_tools import (
    ExternalToolManager, 
    ExternalToolType,
    SubtaskResult
)

class GoldSubtaskManager:
    """Manages gold subtasks including SQL tools and external search tools"""
    
    def __init__(self, file_system_path: Optional[str] = None):
        # Pass file system path to external tool manager
        self.external_tool_manager = ExternalToolManager(file_system_path=file_system_path)
        self.logger = logging.getLogger(__name__)
    
    def get_gold_result_for_instance(self, instance_id: str, gold_result_dir: str) -> str:
        """Get gold result for a specific instance"""
        try:
            # Try exact match first
            gold_file_path = os.path.join(gold_result_dir, f"{instance_id}.csv")
            if os.path.exists(gold_file_path):
                with open(gold_file_path, "r", encoding="utf-8") as f:
                    return f.read()
            
            # Try with suffix patterns
            pattern = os.path.join(gold_result_dir, f"{instance_id}_*.csv")
            matching_files = glob.glob(pattern)
            
            if matching_files:
                matching_files.sort()
                with open(matching_files[0], "r", encoding="utf-8") as f:
                    return f.read()
            
            self.logger.warning(f"No gold result found for instance: {instance_id}")
            return "N/A"
            
        except Exception as e:
            self.logger.error(f"Error loading gold result for {instance_id}: {e}")
            return f"Error: {e}"
    
    def get_sql_for_instance(self, instance_id: str, sql_dir: str) -> str:
        """Get SQL statement for a specific instance"""
        try:
            # Try exact match first
            sql_file_path = os.path.join(sql_dir, f"{instance_id}.sql")
            if os.path.exists(sql_file_path):
                with open(sql_file_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            
            # Try with suffix patterns
            pattern = os.path.join(sql_dir, f"{instance_id}_*.sql")
            matching_files = glob.glob(pattern)
            
            if matching_files:
                matching_files.sort()
                with open(matching_files[0], "r", encoding="utf-8") as f:
                    return f.read().strip()
            
            self.logger.warning(f"No SQL found for instance: {instance_id}")
            return "N/A"
            
        except Exception as e:
            self.logger.error(f"Error loading SQL for {instance_id}: {e}")
            return f"Error: {e}"
    
    def execute_subtasks_concurrent(self, instance_id: str, original_query: str, 
                                   gold_result_dir: str, sql_dir: str) -> Dict[str, SubtaskResult]:
        """Execute all subtasks concurrently and return results"""
        self.logger.info("📊 Gathering data from subtasks concurrently...")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_gold = executor.submit(self.get_gold_result_for_instance, instance_id, gold_result_dir)
            future_sql = executor.submit(self.get_sql_for_instance, instance_id, sql_dir)
            future_external = executor.submit(self.external_tool_manager.search, original_query, original_query)
            
            futures = {
                'get_schema_info': None,  # Placeholder for schema info
                'gold_result': future_gold,
                'sql_statement': future_sql,
                'external_search': future_external
            }
            
            results = {}
            for future in as_completed([f for f in futures.values() if f is not None]):
                for name, fut in futures.items():
                    if fut == future:
                        try:
                            if name == 'external_search':
                                result = future.result()  # This is already a SubtaskResult
                                results[name] = result
                            else:
                                result = future.result()
                                results[name] = SubtaskResult(name, result, success=True)
                            
                            elapsed = time.time() - start_time
                            self.logger.info(f"✅ {name} completed at {elapsed:.2f}s")
                        except Exception as e:
                            self.logger.error(f"❌ {name} failed: {e}")
                            results[name] = SubtaskResult(name, f"Error: {e}", success=False, error=str(e))
                        break
        
        # Add schema info placeholder
        results['get_schema_info'] = SubtaskResult('get_schema_info', 'Schema info placeholder', success=True)
        
        end_time = time.time()
        self.logger.info(f"📊 All subtasks completed: {end_time - start_time:.2f}s")
        
        return results
    
    def build_gold_subtasks(self, db: str, original_query: str, sql_statement: str, 
                           gold_result: str, query: str, selected_tool_type: str = None) -> List[Dict]:
        """Build gold subtasks following original script structure with dynamic external tool selection"""
        subtasks = [
            {
                "subtask_id": "get_schema_info",
                "tool": "get_schema_info",
                "input": {"database_name": db},
                "description": f"Provide schema information about the database"
            },
            {
                "subtask_id": "generated_sql",
                "tool": "generated_sql",
                "input": {
                    "natural_language_query": original_query,
                    "database_name": db
                },
                "expected_SQL": sql_statement,
                "description": f"Provide SQL to answer: {original_query}"  
            },
            {
                "subtask_id": "execute_sql",
                "tool": "execute_sql",
                "input": {
                    "database_name": db
                },
                "expected_result": gold_result,
                "description": f"Execute SQL to answer: {original_query}"  
            }
        ]
        
        # Add dynamic fourth subtask based on selected tool type
        if selected_tool_type == "VECTOR_SEARCH":
            subtasks.append({
                "subtask_id": "vectorDB_search",
                "tool": "vectorDB_search",
                "description": f"Retrieve relevant context for: {query}"
            })
        elif selected_tool_type == "FILE_SYSTEM":
            subtasks.append({
                "subtask_id": "file_system", 
                "tool": "file_system",
                "input": {
                    "natural_language_query": original_query
                },
                "description": f"Provide file information to answer: {query}"
            })
        else:  # Default to perplexity_search
            subtasks.append({
                "subtask_id": "web_context_search",
                "tool": "perplexity_search",
                "description": f"Retrieve relevant external context for: {query}"
            })
        
        return subtasks

class SingleChoiceGenerator:
    """Handles single choice question generation with human feedback support"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.logger = logging.getLogger(__name__)
        self.conversation_history = []

    def call_llm(self, prompt: str) -> str:
        try:
            if not self.api_key:
                self.logger.warning("Missing OPENROUTER_API_KEY, returning empty response")
                return ""

            import requests
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": "anthropic/claude-sonnet-4",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_tokens": 3000
                })
            )

            if response.status_code == 200:
                response_data = response.json()
                if 'choices' in response_data and response_data['choices']:
                    return response_data['choices'][0]['message']['content'].strip()

            return ""

        except Exception as e:
            self.logger.error(f"LLM call error: {e}")
            return ""
    
    def generate_single_choice_content(self, original_query: str, subtask_results: Dict[str, SubtaskResult]) -> Dict:
        """Generate single choice questions using subtask results with random correct answer"""
        import random

        gold_result = subtask_results['gold_result'].result
        external_result = subtask_results['external_search'].result
        correct_options = ['A', 'B', 'C', 'D']
        random_correct_answer = random.choice(correct_options)
        
        prompt_parts = [
            f"Original query: {original_query}\n",
            f"SQL result: {gold_result}\n",
            f"Web summary: {external_result}\n",
            "\n"
        ]

        unified_prompt = "".join(prompt_parts) + self._get_prompt_template(random_correct_answer)
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
                    if question.get("correct_answer") != [random_correct_answer]:
                        self.logger.info(f"Correcting answer from {question.get('correct_answer')} to [{random_correct_answer}]")
                        question["correct_answer"] = [random_correct_answer]
                    
                    return {
                        "question_data": question,
                        "advanced_query": question.get("question", original_query),
                        "success": True,
                        "conversation_history": self.conversation_history.copy()
                    }
            
            # Fallback if parsing succeeds but structure is invalid
            return self._create_fallback_single_choice(original_query, gold_result, external_result, random_correct_answer)
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parsing failed: {e}, using fallback")
            return self._create_fallback_single_choice(original_query, gold_result, external_result, random_correct_answer)
    
    def _get_prompt_template(self, correct_answer: str = 'B') -> str:
        """Get the prompt template for single choice generation based on TechReport specifications"""
        return (
            "Task Requirements (Based on FDABench Technical Report):\n\n"

            "1) Query Construction\n"
            "• Preserve the original analytical intent while extending scope to require heterogeneous data integration.\n"
            "• Request specific database metrics (fields, aggregations, temporal/categorical breakdowns) needed for quantitative evidence.\n"
            "• Specify external context requirements (theoretical frameworks, policy background, domain knowledge, validation sources).\n"
            "• Design questions such that database facts alone are insufficient without interpretive context from unstructured sources.\n"
            "• Avoid revealing numerical answers or conclusions within the question itself to prevent information leakage.\n\n"

            "2) Heterogeneous Integration Requirements\n"
            "• Design the task so that it CANNOT be solved from any single data source. Each source must provide essential, non-redundant information.\n"
            "• Structured Database (SQL): Provides quantitative evidence, precise metrics, temporal/categorical breakdowns, and statistical patterns.\n"
            "• Vector Database: Supplies theoretical frameworks, domain knowledge, conceptual models, technical specifications, and interpretive guidelines.\n"
            "• Web Search: Provides current events, regulatory updates, policy context, external validation, and real-world corroboration.\n"
            "• Ensure complementary roles: SQL answers 'what/how much,' Vector explains 'why/how to interpret,' Web validates 'current context/applicability.'\n\n"

            "3) Multi-Source Reasoning Chain\n"
            "1. SQL Execution: Identify required database tables, fields, aggregations, and constraints. Extract quantitative patterns without revealing numeric answers in the question.\n"
            "2. Vector Retrieval: Surface relevant theoretical frameworks, domain concepts, and interpretive guidelines that contextualize database findings.\n"
            "3. Web Search: Retrieve current events, policy updates, and external validation sources. Identify corroborating or contradictory information requiring resolution.\n"
            "4. Cross-Source Integration: Establish connections between SQL quantitative patterns, Vector conceptual frameworks, and Web contextual validation.\n"
            "5. Framework Application: Apply retrieved domain methodologies to reconcile information conflicts and establish causal interpretations.\n"
            "6. Synthesis: Produce evidence-backed conclusions integrating all three sources, with clear attribution and reasoning transparency.\n\n"

            "4) Answer Format (Single-choice)\n"
            "• 4 options (A, B, C, D) with exactly 1 correct answer\n"
            "• Distractors must be plausible and mutually exclusive\n"
            f"• The correct answer must be option {correct_answer}\n\n"

            "Quality Checklist:\n"
            "☐ Multi-source requirement verified (SQL + Web + Vector all necessary)\n"
            "☐ Integration necessity confirmed (single-source solution is insufficient by design)\n"
            "☐ Answer deterministic and verifiable (clear evidence path; reproducible from sources)\n"
            "☐ Realistic scenario (domain-appropriate assumptions; no artificial constraints)\n"
            "☐ No information leakage (prompt does not expose numeric results or final conclusions)\n"
            "☐ Complete reasoning chain (all steps specified and source-aligned)\n\n"

            "Output format: Return JSON with structure:\n"
            "{\n"
            '  "single_choice_questions": [{\n'
            '    "question": "...",\n'
            '    "options": {"A": "...", "B": "...", "C": "...", "D": "..."},\n'
            f'    "correct_answer": ["{correct_answer}"],\n'
            '    "explanation": "..."\n'
            "  }]\n"
            "}\n\n"
            "Return only valid JSON, no other text or explanation."
        )
    
    def _create_fallback_single_choice(self, original_query: str, gold_result: str, external_result: str, correct_answer: str = 'B') -> Dict:
        """Create fallback single choice question when generation fails with random correct answer"""
        options = {
            "A": "Option A - Initial analysis approach",
            "B": "Option B - Alternative analytical approach",  
            "C": "Option C - Alternative interpretation",
            "D": "Option D - Incorrect methodology"
        }
        
        # Mark the correct option
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
    
    def revise_content(self, feedback: str, conversation_history: List[Dict], current_question_data: Dict, original_query: str, gold_result: str, external_result: str) -> Dict:
        """Revise content based on feedback using unified approach like content_generators.py"""
        revision_prompt = (
            f"Based on the feedback: {feedback}\n\n"
            "Please revise the previous response to address the feedback. "
            "Maintain the same JSON format. Use 'single_choice_questions' key. "
            "Improve all components based on the suggestions while keeping the structure requirements."
        )
        
        # Add feedback to conversation history following reference pattern  
        messages = conversation_history + [{"role": "user", "content": revision_prompt}]
        
        # Call LLM with full conversation history (like content_generators.py)
        response = self._call_llm_with_messages(messages)
        
        # Parse the revised response using safe parsing (like content_generators.py)
        parsed_json, error_msg = self._safe_json_parse(response, "revise_content")
        
        if parsed_json is not None:
            try:
                # Extract single choice questions
                single_choices = parsed_json.get("single_choice_questions", [])
                advanced_query = parsed_json.get("advanced_query", "")
                
                # Validate single choices following reference pattern
                validated_choices = self._validate_single_choices(single_choices)
                
                # Update conversation history
                updated_history = messages + [{"role": "assistant", "content": response}]
                
                # Build result following content_generators.py pattern
                result = {
                    "conversation_history": updated_history,
                    "success": True
                }
                
                if validated_choices:
                    question_data = validated_choices[0]  # Get first validated question
                    result.update({
                        "question_data": question_data,
                        "advanced_query": question_data.get("question", advanced_query or original_query),
                        "single_choice_questions": validated_choices
                    })
                else:
                    # Fallback if no valid choices
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
    
    def _call_llm_with_messages(self, messages: List[Dict]) -> str:
        """Call LLM with message history (adapted from content_generators.py)"""
        try:
            if not self.api_key:
                self.logger.warning("Missing OPENROUTER_API_KEY, returning empty response")
                return ""

            import requests

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
                data=json.dumps({
                    "model": "anthropic/claude-sonnet-4",
                    "messages": openai_messages,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_tokens": 3000
                })
            )

            if response.status_code == 200:
                response_data = response.json()
                if 'choices' in response_data and response_data['choices']:
                    return response_data['choices'][0]['message']['content'].strip()

            return ""

        except Exception as e:
            self.logger.error(f"LLM call with messages error: {e}")
            return ""
    
    def _safe_json_parse(self, content: str, function_name: str = ""):
        """Safe JSON parsing with detailed error handling (from content_generators.py)"""
        try:
            # Clean response content
            content = content.strip()
            
            # Remove markdown code blocks
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # Try to parse JSON
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
    
    def _validate_single_choices(self, choices):
        """Validate and clean single choice questions (from content_generators.py)"""
        if not isinstance(choices, list):
            return [self._get_default_single_choice()]
        
        validated = []
        for choice in choices:
            if isinstance(choice, dict) and all(key in choice for key in ["question", "options", "correct_answer", "explanation"]):
                # Validate options structure
                options = choice.get("options", {})
                if isinstance(options, dict) and all(key in options for key in ["A", "B", "C", "D"]):
                    # Ensure single choice - correct_answer should have exactly one element
                    correct_answer = choice.get("correct_answer", [])
                    if isinstance(correct_answer, list) and len(correct_answer) == 1:
                        validated.append(choice)
                    else:
                        # Fix to single choice if multiple answers provided
                        fixed_choice = dict(choice)
                        if isinstance(correct_answer, list) and len(correct_answer) > 1:
                            fixed_choice["correct_answer"] = [correct_answer[0]]  # Take first answer
                        elif not isinstance(correct_answer, list):
                            fixed_choice["correct_answer"] = [str(correct_answer)]
                        else:
                            import random
                            default_options = ['A', 'B', 'C', 'D']
                            fixed_choice["correct_answer"] = [random.choice(default_options)]  # Random default single choice
                        validated.append(fixed_choice)
                else:
                    validated.append(self._get_default_single_choice())
            else:
                validated.append(self._get_default_single_choice())
        
        if len(validated) == 0:
            validated.append(self._get_default_single_choice())
        
        return validated[:1]  # Return only first question
    
    def _get_default_single_choice(self):
        """Get a default single choice structure with random correct answer"""
        import random
        
        # Randomly select correct answer from all options
        correct_options = ['A', 'B', 'C', 'D']
        random_correct_answer = random.choice(correct_options)
        
        options = {
            "A": "Option A",
            "B": "Option B",
            "C": "Option C", 
            "D": "Option D"
        }
        
        # Mark the correct option
        options[random_correct_answer] += " (Correct)"
        
        return {
            "question": "Default single choice question due to validation failure",
            "options": options,
            "correct_answer": [random_correct_answer],  # Random single choice
            "explanation": f"Default explanation due to validation failure - option {random_correct_answer} is the correct answer"
        }
    
    def _create_revision_fallback(self, current_question_data: Dict, conversation_history: List[Dict], feedback: str):
        """Create fallback result when revision fails"""
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
    
    def reflect_on_content(self, question_data: Dict, original_query: str, gold_result: str, external_result: str) -> str:
        """Generate reflection on the quality of generated content"""
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

class DatasetEntry:
    """Represents a single dataset entry"""

    def __init__(self, instance_id: str, db: str, level: str = None,
                 database_type: str = "bird"):
        self.instance_id = instance_id
        self.db = db
        self.level = level
        self.database_type = database_type
        self.timestamp = time.time()
    
    def to_dict(self, question_data: Dict, gold_subtasks: List[Dict]) -> Dict:
        """Convert to dictionary format following original script structure"""
        choice_question = question_data.get("question", "")
        
        return {
            "instance_id": self.instance_id,
            "db": self.db,
            "level": self.level,
            "database_type": self.database_type,
            "question_type": "single_choice",
            "tools_available": ["get_schema_info", "generated_sql", "execute_sql", "perplexity_search", "vectorDB_search", "sql_optimize", "file_system", "context_history", "sql_debug"],
            "gold_subtasks": gold_subtasks,  
            "query": choice_question,
            "options": question_data.get("options", {}),
            "correct_answer": question_data.get("correct_answer", []),
            "explanation": question_data.get("explanation", "")
        }

class BirdDatasetBuilder:
    """Main class for building Bird dataset with smart tool selection and human feedback"""
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        # Pass file system path from config
        file_system_path = config.get("file_system_path")
        self.subtask_manager = GoldSubtaskManager(file_system_path)
        self.question_generator = SingleChoiceGenerator()
        self.logger = self._setup_logging()
        
        # Human feedback settings
        self.interactive_mode = config.get('interactive', True)  # Default to True
        self.max_revisions = config.get('max_revisions', 3)
        
        # Statistics
        self.processed_count = 0
        self.error_count = 0
        self.accepted_count = 0
        self.revised_count = 0
        self.disposed_count = 0
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_path = self.config.get("log_path", "/tmp/bird_build.log")
        log_dir = os.path.dirname(log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        if root_logger.handlers:
            for handler in root_logger.handlers:
                root_logger.removeHandler(handler)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        return logging.getLogger(__name__)
    
    def display_results(self, original_query: str, content_result: Dict, subtask_results: Dict[str, SubtaskResult], reflection: str = ""):
        """Display generated content and results to user"""
        print("\n" + "=" * 80)
        print("📋 GENERATED CONTENT REVIEW")
        print("=" * 80)
        
        # Show original query
        print(f"\n📝 Original Query:")
        print(f"   {original_query}")
        
        # Show subtask results summary
        print(f"\n🔍 Data Sources:")
        gold_result = subtask_results.get('gold_result', SubtaskResult('gold_result', 'N/A')).result
        external_result = subtask_results.get('external_search', SubtaskResult('external_search', 'N/A')).result
        
        print(f"   • Gold Result: {gold_result[:100]}{'...' if len(gold_result) > 100 else ''}")
        print(f"   • External Knowledge: {external_result[:100]}{'...' if len(external_result) > 100 else ''}")
        
        # Show generated question
        question_data = content_result.get("question_data", {})
        print(f"\n❓ Generated Question:")
        print(f"   {question_data.get('question', 'N/A')}")
        
        # Show options
        print(f"\n📋 Options:")
        options = question_data.get('options', {})
        for key in ['A', 'B', 'C', 'D']:
            if key in options:
                print(f"   {key}. {options[key]}")
        
        print(f"\n✅ Correct Answer: {question_data.get('correct_answer', [])}")
        print(f"\n💡 Explanation:")
        print(f"   {question_data.get('explanation', 'N/A')}")
        
        # Show reflection if available
        if reflection:
            print(f"\n🤔 Quality Assessment:")
            print(f"   {reflection}")
        
        print("\n" + "=" * 80)
    
    def get_difficulty_vote(self) -> str:
        """Get human expert difficulty vote"""
        import sys

        if not sys.stdin.isatty():
            self.logger.warning("⚠️ Non-interactive environment detected, defaulting to medium difficulty")
            return "medium"

        print("\n📊 Difficulty Assessment:")
        print("Based on FDABench criteria, please vote on the difficulty level:")
        print("   (e) Easy - Straightforward question with clear intent, no external knowledge needed")
        print("   (m) Medium - Requires interpretation and domain context")
        print("   (h) Hard - Highly ambiguous, extensive theoretical frameworks needed")

        while True:
            try:
                user_input = input("\n👤 Difficulty vote (e/m/h): ").strip().lower()
                if user_input == 'e':
                    return "easy"
                elif user_input == 'm':
                    return "medium"
                elif user_input == 'h':
                    return "hard"
                else:
                    print("❌ Invalid input. Please enter 'e', 'm', or 'h'.")
            except (EOFError, KeyboardInterrupt):
                self.logger.warning("⚠️ Input interrupted, defaulting to medium difficulty")
                return "medium"

    def get_user_feedback(self) -> tuple:
        """Get user feedback and choice"""
        import sys
        
        # Check if we're in an interactive environment
        if not sys.stdin.isatty():
            # Non-interactive environment - auto-accept for demonstration
            self.logger.warning("⚠️ Non-interactive environment detected, auto-accepting content")
            print("\n⚠️ Non-interactive environment - auto-accepting content")
            return 'a', ""
        
        print("\n🎯 Review Options:")
        print("   (a) Accept - Save this content and continue")
        print("   (d) Dispose - Skip this item completely") 
        print("   (r) Revise - Provide feedback for improvement")
        
        while True:
            try:
                user_input = input("\n👤 Your choice (a/d/r): ").strip().lower()
                if user_input in ['a', 'd', 'r']:
                    feedback = ""
                    if user_input == 'r':
                        print("\n📝 Please provide specific feedback for improvement:")
                        feedback = input("💬 Your feedback: ").strip()
                        if not feedback:
                            print("❌ Feedback cannot be empty. Please try again.")
                            continue
                    return user_input, feedback
                else:
                    print("❌ Invalid input. Please enter 'a', 'd', or 'r'.")
            except (EOFError, KeyboardInterrupt):
                # Handle non-interactive or interrupted input
                self.logger.warning("⚠️ Input interrupted or unavailable, auto-accepting content")
                print("\n⚠️ Input interrupted - auto-accepting content")
                return 'a', ""
    
    def load_queries(self, query_path: str) -> List[Dict[str, Any]]:
        """Load queries from a JSONL file"""
        with open(query_path, "r", encoding="utf-8") as f:
            queries = []
            line_num = 0
            for line in f:
                line_num += 1
                if line.strip():
                    try:
                        data = json.loads(line)
                        queries.append(data)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON parse error in file {query_path} line {line_num}: {e}")
                        continue
            return queries
    
    def time_monitor(self, func_name=""):
        """Time monitoring decorator with logging"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    elapsed = end_time - start_time
                    self.logger.info(f"⏱️  {func_name or func.__name__}: {elapsed:.2f}s")
                    return result
                except Exception as e:
                    end_time = time.time()
                    elapsed = end_time - start_time
                    self.logger.error(f"❌ {func_name or func.__name__}: {elapsed:.2f}s (failed: {e})")
                    raise
            return wrapper
        return decorator
    
    def process_single_query(self, item: Dict, idx: int) -> bool:
        """Process a single query with human feedback support"""
        instance_id = item.get("instance_id") or item.get("id") or str(idx)
        db = item.get("db_id") or item.get("db") or "SQLite"
        original_query = item.get("instruction") or item.get("question") or item.get("query") or ""

        self.logger.info(f"🔄 Processing query {idx+1}: {instance_id}")
        total_start_time = time.time()

        try:
            subtask_results = self.subtask_manager.execute_subtasks_concurrent(
                instance_id, original_query, self.config["gold_result_dir"], self.config["sql_path"]
            )
            
            if self.interactive_mode:
                return self._process_interactive_mode(item, idx, subtask_results, total_start_time)
            else:
                return self._process_auto_mode(item, idx, subtask_results, total_start_time)
            
        except Exception as e:
            total_end_time = time.time()
            self.logger.error(f"❌ Error processing query {idx+1}: {e}")
            self.logger.error(f"🎯 Total processing time: {total_end_time - total_start_time:.2f}s")
            self.logger.error("=" * 60)
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            self.error_count += 1
            return False
    
    def _process_interactive_mode(self, item: Dict, idx: int, subtask_results: Dict[str, SubtaskResult], start_time: float) -> bool:
        """Process query in interactive mode with human feedback"""
        instance_id = item.get("instance_id") or item.get("id") or str(idx)
        db = item.get("db_id") or item.get("db") or "SQLite"
        original_query = item.get("instruction") or item.get("question") or item.get("query") or ""
        
        revision_count = 0
        content_result = None
        
        while revision_count <= self.max_revisions:
            # Generate single choice content only on first iteration
            if revision_count == 0:
                content_result = self.question_generator.generate_single_choice_content(
                    original_query, subtask_results
                )
            
            # Generate reflection
            if content_result["success"] and content_result.get("question_data"):
                reflection = self.question_generator.reflect_on_content(
                    content_result["question_data"],
                    original_query,
                    subtask_results['gold_result'].result,
                    subtask_results['external_search'].result
                )
            else:
                reflection = "Content generation failed or used fallback."
            
            # Display results to user
            self.display_results(original_query, content_result, subtask_results, reflection)
            
            user_choice, feedback = self.get_user_feedback()

            if user_choice == 'a':
                difficulty = self.get_difficulty_vote()
                self.accepted_count += 1
                return self._save_entry(item, content_result, subtask_results, start_time, difficulty)
                
            elif user_choice == 'd':
                self.disposed_count += 1
                total_end_time = time.time()
                self.logger.info(f"🗑️  Query {idx+1} disposed by user")
                self.logger.info(f"🎯 Total processing time: {total_end_time - start_time:.2f}s")
                print(f"\n🗑️  Query {idx+1} disposed and skipped.\n")
                return True
                
            elif user_choice == 'r':
                if revision_count >= self.max_revisions:
                    print(f"\n⚠️  Maximum revisions ({self.max_revisions}) reached. Accepting current version.\n")
                    difficulty = self.get_difficulty_vote()
                    self.accepted_count += 1
                    return self._save_entry(item, content_result, subtask_results, start_time, difficulty)
                
                print(f"\n🔄 Revising content (attempt {revision_count + 1}/{self.max_revisions})...")
                
                # Get conversation history and current data
                conversation_history = content_result.get("conversation_history", [])
                current_question_data = content_result.get("question_data", {})
                
                # Get data sources for context
                gold_result = subtask_results['gold_result'].result
                external_result = subtask_results['external_search'].result
                
                # Attempt revision with full context
                revised_result = self.question_generator.revise_content(
                    feedback, conversation_history, current_question_data, 
                    original_query, gold_result, external_result
                )
                
                if revised_result["success"]:
                    content_result = revised_result
                    self.revised_count += 1
                    print("✅ Content revised successfully!")
                else:
                    print("❌ Revision failed, keeping original content.")
                
                revision_count += 1
        
        # If we exit the loop without accepting/disposing, save the last version
        difficulty = self.get_difficulty_vote()
        self.accepted_count += 1
        return self._save_entry(item, content_result, subtask_results, start_time, difficulty)
    
    def _process_auto_mode(self, item: Dict, idx: int, subtask_results: Dict[str, SubtaskResult], start_time: float) -> bool:
        """Process query in automatic mode without human feedback"""
        # Generate single choice content
        content_result = self.question_generator.generate_single_choice_content(
            item.get("instruction") or item.get("question") or item.get("query") or "", subtask_results
        )
        
        # Log generated question for debugging
        if content_result["success"]:
            self.logger.info(f"✅ Generated single choice question for {item.get('instance_id', idx)}")
        else:
            self.logger.warning(f"⚠️ Used fallback question for {item.get('instance_id', idx)}")
        
        self.accepted_count += 1
        return self._save_entry(item, content_result, subtask_results, start_time)
    
    def _save_entry(self, item: Dict, content_result: Dict, subtask_results: Dict[str, SubtaskResult], start_time: float, difficulty: str = None) -> bool:
        """Save the final entry to output"""
        try:
            instance_id = item.get("instance_id") or item.get("id") or "unknown"
            db = item.get("db_id") or item.get("db") or "SQLite"
            original_query = item.get("instruction") or item.get("question") or item.get("query") or ""

            dataset_entry = DatasetEntry(instance_id, db, level=difficulty)
            selected_tool_type = getattr(subtask_results['external_search'], 'selected_tool_type', None)
            
            gold_subtasks = self.subtask_manager.build_gold_subtasks(
                db, original_query, 
                subtask_results['sql_statement'].result,
                subtask_results['gold_result'].result, 
                content_result["question_data"].get("question", ""),
                selected_tool_type
            )
            
            entry_dict = dataset_entry.to_dict(content_result["question_data"], gold_subtasks)
            
            # Save to output
            self._append_to_output(entry_dict)
            
            total_end_time = time.time()
            self.logger.info(f"✅ Successfully processed and saved entry")
            self.logger.info(f"🎯 Total processing time: {total_end_time - start_time:.2f}s")
            
            if self.interactive_mode:
                print(f"\n💾 Entry saved successfully!\n")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving entry: {e}")
            return False
    
    def _append_to_output(self, entry: Dict):
        """Append entry to output file"""
        output_path = self.config["output_path"]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    def build_dataset(self):
        """Main method to build the dataset"""
        mode_text = "Interactive Human Review" if self.interactive_mode else "Automatic"
        self.logger.info(f"🚀 Starting Bird Medium Single Choice Dataset Build ({mode_text} Mode)")
        self.logger.info(f"📁 Output path: {self.config['output_path']}")
        self.logger.info("🧠 Using LLM-based smart tool selection")

        if self.interactive_mode:
            self.logger.info(f"👤 Interactive mode enabled - Human review required for each item")
            self.logger.info(f"🔄 Maximum revisions per item: {self.max_revisions}")
            print(f"\n🎯 Welcome to Interactive Dataset Building!")
            print(f"   You will review each generated question and can:")
            print(f"   • Accept good questions")
            print(f"   • Request revisions with specific feedback")
            print(f"   • Dispose of unsuitable items")
            print(f"   Maximum {self.max_revisions} revisions per item.\n")
        
        overall_start_time = time.time()
        
        # Load queries
        self.logger.info(f"📂 Loading queries from: {self.config['bird_path']}")
        
        if not os.path.exists(self.config["bird_path"]):
            self.logger.error(f"❌ Bird path does not exist: {self.config['bird_path']}")
            return
        
        queries = self.load_queries(self.config["bird_path"])
        self.logger.info(f"📊 Loaded {len(queries)} queries")
        
        # Process queries
        for idx, item in enumerate(queries):
            if self.process_single_query(item, idx):
                self.processed_count += 1
            else:
                self.error_count += 1
        
        overall_end_time = time.time()
        total_time = overall_end_time - overall_start_time
        
        # Display comprehensive statistics
        self.logger.info("=" * 80)
        self.logger.info("🏁 BIRD DATASET BUILD COMPLETED")
        self.logger.info(f"📊 Total queries loaded: {len(queries)}")
        self.logger.info(f"✅ Successfully processed: {self.processed_count}")
        self.logger.info(f"❌ Errors encountered: {self.error_count}")
        
        if self.interactive_mode:
            self.logger.info("=" * 50)
            self.logger.info("👤 Human Review Statistics:")
            self.logger.info(f"   ✅ Accepted: {self.accepted_count}")
            self.logger.info(f"   🔄 Revised: {self.revised_count}")
            self.logger.info(f"   🗑️  Disposed: {self.disposed_count}")
            if self.processed_count > 0:
                acceptance_rate = (self.accepted_count / self.processed_count) * 100
                self.logger.info(f"   📈 Acceptance rate: {acceptance_rate:.1f}%")
        
        self.logger.info(f"⏱️  Total time: {total_time:.2f}s")
        if self.processed_count > 0:
            avg_time = total_time / self.processed_count
            self.logger.info(f"⚡ Average time per query: {avg_time:.2f}s")
        self.logger.info(f"💾 Output saved to: {self.config['output_path']}")
        self.logger.info("🧠 Used intelligent LLM-based tool selection")
        self.logger.info("=" * 80)

def setup_environment():
    """Setup environment variables"""
    load_dotenv()

def main():
    """Main function"""
    # Setup environment
    setup_environment()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Bird Smart Tool Selection Dataset Build Script with Human Review')
    parser.add_argument('--auto', action='store_true', help='Run in automatic mode without human review (default is interactive)')
    parser.add_argument('--max-revisions', type=int, default=3, help='Maximum number of revisions per item (default: 3)')
    args = parser.parse_args()
    
    # Configuration - using relative paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config = {
        "bird_path": os.path.join(current_dir, "input_data", "original_data", "bird.jsonl"),
        "gold_result_dir": os.path.join(current_dir, "input_data", "gold_sql_result"),
        "sql_path": os.path.join(current_dir, "input_data", "gold_sql_query"),
        "output_path": os.path.join(current_dir, "output_data", "medium_singlechoice.json"),
        "log_path": os.path.join(current_dir, "log", "bird_singlechoice.log"),
        "file_system_path": os.path.join(current_dir, "input_data", "file_system"),
        "database_type": "bird",
        "use_smart_tool_selection": True,  # Use smart tool selection for external search
        "interactive": not args.auto,  # Default to interactive mode (True) unless --auto is specified
        "max_revisions": args.max_revisions
    }
    
    # Create and run dataset builder
    builder = BirdDatasetBuilder(config)
    builder.build_dataset()

if __name__ == "__main__":
    main()