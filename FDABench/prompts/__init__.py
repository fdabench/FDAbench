"""
Prompt templates for FDABench.

This module contains all prompt templates organized in a single file
for easy access and management across different agent patterns.
"""

from .prompts import PROMPTS, get_prompt, list_prompts, add_prompt, remove_prompt

__all__ = [
    "PROMPTS",
    "get_prompt", 
    "list_prompts",
    "add_prompt",
    "remove_prompt",
]