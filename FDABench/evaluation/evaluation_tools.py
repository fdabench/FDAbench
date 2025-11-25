import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics import precision_score, recall_score, f1_score
import re
from typing import Dict, List, Tuple, Union, Any
import json
import openai
import os
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Get OpenRouter API key from environment variable
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

class ReportEvaluator:
    def __init__(self, openai_api_key: str = None):
        # Initialize ROUGE scorer (use_stemmer=False for speed)
        self.latency= None
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

        # Initialize OpenRouter API
        if openai_api_key:
            self.openrouter_api_key = openai_api_key
            self.has_openai = True
        elif OPENROUTER_API_KEY:
            self.openrouter_api_key = OPENROUTER_API_KEY
            self.has_openai = True
        else:
            print("Warning: OpenRouter API key not provided. LLM judge functionality will not be available.")
            self.has_openai = False

    def _get_openrouter_client(self):
        """Initialize OpenRouter client"""
        return openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.openrouter_api_key,
        )

    def _call_openrouter_llm(self, messages, model="deepseek/deepseek-chat-v3-0324"):
        """Call OpenRouter LLM with messages"""
        try:
            client = self._get_openrouter_client()
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://github.com/wa../FDABench",
                    "X-Title": "Official First Round Report Evaluator for FDABenchmark",
                },
                model=model,
                messages=messages,
                temperature=0.8,
                max_tokens=10,
                presence_penalty=0.2,
                frequency_penalty=0.2
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenRouter API: {e}")
            return "0.0"  # Return default score on error

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for evaluation."""
        # Remove special characters and extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization."""
        # Remove punctuation and split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()

    def _calculate_rouge_scores(self, generated: str, ground_truth: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores between generated and ground truth text.
        
        ROUGE-N Formula:
        ROUGE-N = (Number of overlapping n-grams) / (Number of n-grams in reference)
        
        ROUGE-L Formula:
        ROUGE-L = LCS(generated, reference) / len(reference)
        where LCS is the Longest Common Subsequence
        
        Reference: Lin, C. Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries.
        """
        scores = self.rouge_scorer.score(ground_truth, generated)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }

    def _calculate_precision_recall_f1(self, generated: str, ground_truth: str) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score based on word overlap.
        
        Precision = |generated ∩ reference| / |generated|
        Recall = |generated ∩ reference| / |reference|
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        
        Reference: Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval.
        """
        generated_words = set(self._tokenize(generated))
        ground_truth_words = set(self._tokenize(ground_truth))
        
        # Calculate word overlap
        overlap = generated_words.intersection(ground_truth_words)
        
        if len(generated_words) == 0:
            precision = 0.0
        else:
            precision = len(overlap) / len(generated_words)
            
        if len(ground_truth_words) == 0:
            recall = 0.0
        else:
            recall = len(overlap) / len(ground_truth_words)
            
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    @staticmethod
    def _to_float(val):
        if isinstance(val, list):
            if len(val) > 0:
                return float(val[0])
            else:
                return 0.0
        try:
            return float(val)
        except Exception:
            return 0.0

    # def _calculate_ragas_score(self, generated: str, ground_truth: str) -> Dict[str, float]:
    #     """
    #     Calculate RAGAS scores using the ragas library.
    #     
    #     Formula:
    #     RAGAS = 0.4 * Faithfulness + 0.3 * Answer_Relevance + 0.3 * Context_Relevance
    #     
    #     Reference: Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation.
    #     """
    #     # RAGAS evaluation has been disabled to avoid LLM dependency
    #     return {}

    def _get_llm_judge_score(self, generated: str, ground_truth: str) -> float:
        """
        Get LLM's evaluation score for the generated report by evaluating each dimension separately.
        
        Formula:
        LLM_Score = 0.4 * Content_Accuracy + 0.2 * Logical_Flow + 0.3 * Factual_Consistency + 0.1 * Clarity
        
        Reference: Liu, Y., et al. (2023). A Survey of LLM Evaluation.
        """
        if not self.has_openai:
            print("Warning: OpenRouter API key not set. LLM judge score will be 0.0.")
            return 0.0
            
        try:
            # Evaluate each dimension separately
            dimensions = {
                "content_accuracy": {
                    "weight": 0.4,
                    "prompt": "Evaluate the content accuracy and completeness (0-1):\nDoes it cover all key points from the ground truth? Are the facts and information accurate?"
                },
                "logical_flow": {
                    "weight": 0.2,
                    "prompt": "Evaluate the logical flow and structure (0-1):\nIs the information well-organized? Does it follow a clear logical progression?"
                },
                "factual_consistency": {
                    "weight": 0.3,
                    "prompt": "Evaluate the factual consistency (0-1):\nAre the facts consistent with the ground truth? Are there any contradictions or inaccuracies?"
                },
                "clarity": {
                    "weight": 0.1,
                    "prompt": "Evaluate the clarity and readability (0-1):\nIs the language clear and professional? Is it easy to understand?"
                }
            }
            
            final_score = 0.0
            
            for dim_name, dim_info in dimensions.items():
                prompt = f"""You are an expert evaluator of reports. Please evaluate the following aspect of the generated report compared to the ground truth report.

Ground Truth Report:
{ground_truth}

Generated Report:
{generated}

{dim_info['prompt']}

Provide your score as a single number between 0 and 1, where:
0 = Completely inaccurate or irrelevant
1 = Perfect match with ground truth in terms of this aspect

Score:"""

                messages = [
                    {"role": "system", "content": f"You are an expert evaluator of reports. Evaluate this specific aspect independently and provide only a numerical score between 0 and 1."},
                    {"role": "user", "content": prompt}
                ]
                
                response = self._call_openrouter_llm(messages, model="deepseek/deepseek-chat-v3-0324")
                try:
                    score_match = re.search(r'0\.\d+|1\.0|0|1', response)
                    if score_match:
                        dim_score = float(score_match.group())
                        final_score += dim_score * dim_info['weight']
                    else:
                        print(f"Warning: Could not find a valid score for {dim_name} in response: {response}")
                except ValueError:
                    print(f"Warning: Could not parse score for {dim_name} from response: {response}")
            
            return min(max(final_score, 0), 1)  # Ensure final score is between 0 and 1
                
        except Exception as e:
            print(f"Error getting LLM judge score: {str(e)}")
            return 0.0

    def _calculate_bleu(self, generated: str, ground_truth: str) -> float:
        """Calculate BLEU score between generated and ground truth text."""
        reference = [self._tokenize(ground_truth)]
        hypothesis = self._tokenize(generated)
        if not hypothesis or not reference[0]:
            return 0.0
        smoothie = SmoothingFunction().method4
        return sentence_bleu(reference, hypothesis, smoothing_function=smoothie)

    def evaluate_reports(self, generated_report: str, ground_truth_report: str) -> Dict[str, float]:
        """Evaluate a generated report against a ground truth report using multiple metrics."""
        # Preprocess texts
        generated = self._preprocess_text(generated_report)
        ground_truth = self._preprocess_text(ground_truth_report)

        # Calculate all metrics (excluding RAGAS and LLM judge for speed)
        rouge_scores = self._calculate_rouge_scores(generated, ground_truth)
        precision_recall_f1 = self._calculate_precision_recall_f1(generated, ground_truth)
        # llm_score = self._get_llm_judge_score(generated, ground_truth)  # 注释掉LLM judge以加速
        bleu_score = self._calculate_bleu(generated, ground_truth)

        # Combine all scores (excluding RAGAS metrics and LLM judge)
        evaluation_results = {
            **rouge_scores,
            **precision_recall_f1,
            # 'llm_judge_score': llm_score,  # 注释掉LLM judge以加速
            'bleu': bleu_score
        }

        return evaluation_results

    def evaluate_batch(self, generated_reports: List[str], ground_truth_reports: List[str]) -> Dict[str, float]:
        """
        Evaluate multiple generated reports against their ground truth reports.
        
        Args:
            generated_reports (List[str]): List of generated reports
            ground_truth_reports (List[str]): List of corresponding ground truth reports
            
        Returns:
            Dict[str, float]: Dictionary containing average scores for all evaluation metrics
        """
        if len(generated_reports) != len(ground_truth_reports):
            raise ValueError("Number of generated reports must match number of ground truth reports")
        
        all_scores = []
        for gen, truth in zip(generated_reports, ground_truth_reports):
            scores = self.evaluate_reports(gen, truth)
            all_scores.append(scores)
        
        # Calculate average scores
        avg_scores = {}
        for metric in all_scores[0].keys():
            avg_scores[metric] = np.mean([score[metric] for score in all_scores])
        
        return avg_scores

    def evaluate_tool_recall(self, gold_subtasks: List[Dict], actual_tools_executed: List[str]) -> Dict[str, Any]:
        """
        Calculate tool call recall for agent evaluation.
        
        Formula:
        Recall = TP / (TP + FN)
        where:
        - TP (True Positive): Tools that should be called and were actually called
        - FN (False Negative): Tools that should be called but were not called (missed)
        
        Args:
            gold_subtasks: List of ground truth subtasks with tool information
            actual_tools_executed: List of tools actually executed by the agent
            
        Returns:
            Dict containing recall metrics and detailed breakdown
        """
        # Extract expected tools from gold_subtasks
        expected_tools = set()
        subtask_tool_mapping = {}
        
        for subtask in gold_subtasks:
            if isinstance(subtask, dict) and 'tool' in subtask:
                tool_name = subtask['tool']
                expected_tools.add(tool_name)
                subtask_id = subtask.get('subtask_id', 'unknown')
                subtask_tool_mapping[subtask_id] = tool_name
        
        # Convert actual_tools_executed to set for comparison
        actual_tools = set(actual_tools_executed)
        
        # Calculate recall metrics
        TP = len(expected_tools & actual_tools)  # True Positives: correctly called tools
        FN = len(expected_tools - actual_tools)  # False Negatives: missed tools
        FP = len(actual_tools - expected_tools)  # False Positives: extra tools called
        
        # Calculate recall
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        
        # Calculate precision
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Identify missed and extra tools
        missed_tools = list(expected_tools - actual_tools)
        extra_tools = list(actual_tools - expected_tools)
        correctly_called_tools = list(expected_tools & actual_tools)
        
        return {
            'tool_recall': recall,
            'tool_precision': precision,
            'tool_f1': f1,
            'expected_tools': list(expected_tools),
            'actual_tools': list(actual_tools),
            'correctly_called_tools': correctly_called_tools,
            'missed_tools': missed_tools,
            'extra_tools': extra_tools,
            'TP': TP,
            'FN': FN,
            'FP': FP,
            'total_expected': len(expected_tools),
            'total_actual': len(actual_tools),
            'subtask_tool_mapping': subtask_tool_mapping
        }

    def evaluate_single_query(self, ground_truth_report: str, generated_content: str) -> Dict[str, float]:
        """
        Evaluate a single query result (compatibility method for existing code).
        This method is used by some agents that call evaluate_single_query instead of evaluate_reports.
        """
        return self.evaluate_reports(generated_content, ground_truth_report) 