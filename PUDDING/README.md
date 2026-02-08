# PUDDING

An agentic dataset construction framework that leverages LLM generation with iterative expert validation for reliable and scalable benchmark construction.

## Overview

PUDDING addresses the trade-off between manual annotation (high quality, low scalability) and fully automated LLM generation (scalable but prone to hallucination) by combining automation for efficiency with expert oversight for reliability.

## Three-Phase Workflow

### Phase 1: Tree-Structured Exploration
Given a candidate query with database instance and gold SQL:
- Extract database schema and execute gold SQL for quantitative results
- Tree-structured context grounding with per-branch self-reflection (PRUNE/CONTINUE/SUFFICIENT)
- Collect external context via web search, vector search, file search, and database exploration
- Candidates spawned from frontier nodes, executed, and reflected upon iteratively

### Phase 2: Report Generation and Expert Review
Iterative agent-expert collaboration:
- Generate enhanced query and ground truth report from terminal paths
- Automated quality self-reflection
- Expert review: **Accept**, **Revise** (re-enters Phase 1 for more evidence), or **Reject**

### Phase 3: Validation and Finalization
Quality validation and difficulty classification:
- Symmetric single-source check (reject if SQL alone OR external knowledge alone suffices)
- LLM-based difficulty scoring on 4 dimensions (SQL complexity, source diversity, reasoning depth, domain knowledge)
- DAG and rubric annotation from exploration trace

## Quick Start

```bash
python -m PUDDING.main          # Interactive mode (with expert review)
python -m PUDDING.main --auto   # Automatic mode
python -m PUDDING.main --limit 10
python -m PUDDING.main --resume <thread_id>
python -m PUDDING.main --input path/to/queries.jsonl --output path/to/output.json
```

## Configuration

Set your API key in `.env`:
```
OPENROUTER_API_KEY=your-api-key
```

## Module Structure

```
PUDDING/
├── main.py                 # Entry point
├── graph/                  # LangGraph workflow
│   ├── builder.py          # Graph construction
│   ├── nodes.py            # Processing nodes
│   ├── edges.py            # Conditional routing
│   ├── runner.py            # Workflow executor with interrupt handling
│   └── state.py            # State definitions
├── exploration/            # Tree exploration
│   ├── candidate_spawner.py # LLM-based branch proposal
│   └── branch_reflector.py  # Per-branch self-reflection
├── generators/             # Report generation
├── validation/             # Quality checks
│   ├── single_source_validator.py  # Symmetric single-source rejection
│   └── dag_annotator.py            # DAG, rubric, difficulty annotation
├── models/                 # Data models (ExplorationTree, ToolAction, TerminalPath)
├── tools/                  # Tool executor (web/vector/file search, db explore)
└── utils/                  # Utilities (display, I/O)
```
