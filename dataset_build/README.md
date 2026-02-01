# Dataset Build Module

This module provides an agent-expert collaboration framework for generating FDABench test cases using LangGraph-based stateful workflows.

## Quick Start

```bash
# Interactive mode (with human expert review)
python -m dataset_build.main

# Automatic mode (no human review)
python -m dataset_build.main --auto

# Limit number of queries
python -m dataset_build.main --limit 10

# Resume a previous session
python -m dataset_build.main --resume <thread_id>
```

## Module Structure

```
dataset_build/
├── main.py                 # Entry point
├── graph/                  # LangGraph workflow
│   ├── builder.py          # Graph construction
│   ├── nodes.py            # Processing nodes
│   ├── edges.py            # Conditional edges
│   ├── runner.py           # Workflow executor
│   └── state.py            # State definitions
├── generators/             # Content generation
│   └── question_generator.py
├── models/                 # Data models
│   ├── data_models.py      # SubtaskResult, DatasetEntry
│   └── dag_models.py       # DAG evaluation models
├── tools/                  # External tools
│   ├── external_tools.py   # Web/Vector/File search
│   ├── dag_builder.py      # DAG construction
│   └── multi_step_orchestrator.py
└── utils/                  # Utilities
    ├── display.py          # Console output
    └── io.py               # File I/O
```

## Generation Workflow

1. **Initialize**: Load queries from input JSONL
2. **Execute Subtasks**: Run SQL, web search, vector search
3. **Generate Content**: Create questions using LLM
4. **Human Review** (interactive mode):
   - `(a)` Accept - save to dataset
   - `(d)` Dispose - skip item
   - `(r)` Revise - regenerate with feedback
5. **Finalize**: Save results to output JSON

## Configuration

Set your API key in `.env`:
```
OPENROUTER_API_KEY=your-api-key
```

Input/output paths can be customized via CLI arguments:
```bash
python -m dataset_build.main --input path/to/queries.jsonl --output path/to/output.json
```
