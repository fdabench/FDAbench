# -*- coding: utf-8 -*-
"""
Display utilities - Formatting and displaying results.
Extracted from main.py.
"""

from typing import Dict, Any, Optional

from ..models.data_models import SubtaskResult


def display_results(
    original_query: str,
    content_result: Dict[str, Any],
    subtask_results: Dict[str, SubtaskResult],
    reflection: str = ""
) -> None:
    """Display generated content and results to user."""
    print("\n" + "=" * 80)
    print("GENERATED CONTENT REVIEW")
    print("=" * 80)

    # Show original query
    print(f"\nOriginal Query:")
    print(f"   {original_query}")

    # Show subtask results summary
    print(f"\nData Sources:")
    gold_result = subtask_results.get('gold_result', SubtaskResult('gold_result', 'N/A')).result
    external_result = subtask_results.get('external_search', SubtaskResult('external_search', 'N/A')).result

    print(f"   - Gold Result: {gold_result[:100]}{'...' if len(gold_result) > 100 else ''}")
    print(f"   - External Knowledge: {external_result[:100]}{'...' if len(external_result) > 100 else ''}")

    # Show generated question
    question_data = content_result.get("question_data", {})
    print(f"\nGenerated Question:")
    print(f"   {question_data.get('question', 'N/A')}")

    # Show options
    print(f"\nOptions:")
    options = question_data.get('options', {})
    for key in ['A', 'B', 'C', 'D']:
        if key in options:
            print(f"   {key}. {options[key]}")

    print(f"\nCorrect Answer: {question_data.get('correct_answer', [])}")
    print(f"\nExplanation:")
    print(f"   {question_data.get('explanation', 'N/A')}")

    # Show reflection if available
    if reflection:
        print(f"\nQuality Assessment:")
        print(f"   {reflection}")

    print("\n" + "=" * 80)


def display_welcome_message(max_revisions: int) -> None:
    """Display welcome message for interactive mode."""
    print(f"\nWelcome to Interactive Dataset Building!")
    print(f"   You will review each generated question and can:")
    print(f"   - Accept good questions")
    print(f"   - Request revisions with specific feedback")
    print(f"   - Dispose of unsuitable items")
    print(f"   Maximum {max_revisions} revisions per item.\n")


def display_statistics(
    total_queries: int,
    processed_count: int,
    error_count: int,
    accepted_count: int,
    revised_count: int,
    disposed_count: int,
    total_time: float,
    interactive_mode: bool = True
) -> None:
    """Display comprehensive statistics at the end of build."""
    print("=" * 80)
    print("BIRD DATASET BUILD COMPLETED")
    print(f"Total queries loaded: {total_queries}")
    print(f"Successfully processed: {processed_count}")
    print(f"Errors encountered: {error_count}")

    if interactive_mode:
        print("=" * 50)
        print("Human Review Statistics:")
        print(f"   Accepted: {accepted_count}")
        print(f"   Revised: {revised_count}")
        print(f"   Disposed: {disposed_count}")
        if processed_count > 0:
            acceptance_rate = (accepted_count / processed_count) * 100
            print(f"   Acceptance rate: {acceptance_rate:.1f}%")

    print(f"Total time: {total_time:.2f}s")
    if processed_count > 0:
        avg_time = total_time / processed_count
        print(f"Average time per query: {avg_time:.2f}s")
    print("=" * 80)


def display_review_options() -> None:
    """Display review options for human feedback."""
    print("\nReview Options:")
    print("   (a) Accept - Save this content and continue")
    print("   (d) Dispose - Skip this item completely")
    print("   (r) Revise - Provide feedback for improvement")


def display_difficulty_options() -> None:
    """Display difficulty voting options."""
    print("\nDifficulty Assessment:")
    print("Based on FDABench criteria, please vote on the difficulty level:")
    print("   (e) Easy - Straightforward question with clear intent, no external knowledge needed")
    print("   (m) Medium - Requires interpretation and domain context")
    print("   (h) Hard - Highly ambiguous, extensive theoretical frameworks needed")
