# -*- coding: utf-8 -*-
"""
I/O utilities - File operations for dataset building.
Extracted from main.py.
"""

import os
import json
import logging
from typing import List, Dict, Any


logger = logging.getLogger(__name__)


def load_queries(query_path: str) -> List[Dict[str, Any]]:
    """Load queries from a JSONL file."""
    queries: List[Dict[str, Any]] = []
    line_num = 0

    with open(query_path, "r", encoding="utf-8") as f:
        for line in f:
            line_num += 1
            if line.strip():
                try:
                    data = json.loads(line)
                    queries.append(data)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error in file {query_path} line {line_num}: {e}")
                    continue

    return queries


def append_to_output(entry: Dict[str, Any], output_path: str) -> None:
    """Append entry to output file."""
    ensure_output_dir(output_path)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def ensure_output_dir(output_path: str) -> None:
    """Ensure output directory exists."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def save_checkpoint_metadata(
    checkpoint_path: str,
    thread_id: str,
    current_idx: int,
    total_queries: int,
    statistics: Dict[str, int]
) -> None:
    """Save checkpoint metadata for session recovery."""
    metadata = {
        "thread_id": thread_id,
        "current_idx": current_idx,
        "total_queries": total_queries,
        "statistics": statistics
    }

    ensure_output_dir(checkpoint_path)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_checkpoint_metadata(checkpoint_path: str) -> Dict[str, Any]:
    """Load checkpoint metadata for session recovery."""
    if not os.path.exists(checkpoint_path):
        return {}

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        return json.load(f)
