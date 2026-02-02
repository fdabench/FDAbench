# PUDDING

An agentic dataset construction framework that leverages LLM generation with iterative expert validation for reliable and scalable benchmark construction.

## Overview

PUDDING addresses the trade-off between manual annotation (high quality, low scalability) and fully automated LLM generation (scalable but prone to hallucination) by combining automation for efficiency with expert oversight for reliability.

## Three-Phase Workflow

### Phase 1: Initialization
Given a candidate query with database instance and gold SQL:
- Extract database schema and execute gold SQL for quantitative results
- Retrieve enterprise demonstrations from real application cases
- Collect external context via web search, vector retrieval, and file system search
- Build comprehensive context across heterogeneous data modalities

### Phase 2: Expert Verification and Refinement
Iterative agent-expert collaboration:
- Construction agent generates initial draft from comprehensive context
- Experts review following quality standards (integration necessity, answer correctness, realism)
- Decisions: **Accept**, **Revise** with feedback, or **Dispose**
- Achieves 75.9% first-iteration acceptance, averaging 1.9 iterations per task

### Phase 3: Finalization
Quality validation and difficulty classification:
- Single-source sufficiency tests (reject if solvable by any single source)
- Difficulty classification: Easy (20.68%), Medium (32.84%), Hard (46.49%)

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
│   ├── edges.py            # Conditional edges
│   ├── runner.py           # Workflow executor
│   └── state.py            # State definitions
├── generators/             # Content generation
├── models/                 # Data models (DAG, SubtaskResult, DatasetEntry)
├── tools/                  # External tools (Web/Vector/File search, DAG builder)
└── utils/                  # Utilities (display, I/O)
```
