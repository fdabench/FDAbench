"""
Evaluation module for FDABench package.

This module provides comprehensive evaluation tools for database agents,
including ROUGE scores, RAGAS metrics, LLM judge scores, and tool recall evaluation.
"""

from .evaluation_tools import ReportEvaluator

__all__ = ['ReportEvaluator']