# FDABench

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**FDABench** is the first data agent benchmark specifically designed for evaluating agents in multi-source data analytical scenarios. Our contributions include: (i) we construct a standard benchmark with 2,007 diverse tasks across different data sources, domains, difficulty levels, and task types to comprehensively evaluate data agent performance; (ii) we design an agent-expert collaboration dataset generation framework ensuring reliable and efficient heterogeneous data benchmark construction; (iii) we equip FDABench with strong generalization capabilities across diverse target systems and frameworks. We use FDABench to evaluate various data agent systems, revealing that each data agent system exhibits distinct advantages and limitations regarding response quality, accuracy, latency, and token cost.

## Overview

![FDABench Architecture](assets/overview.png)

*FDABench provides a comprehensive framework for evaluating data agents across multiple dimensions including accuracy, latency, cost, and tool usage efficiency. The benchmark supports diverse agent architectures and integrates with various database systems and semantic operators.*

## Key Features

- **Open-Source Data Agents Implementations**: Provides several ready-to-use data agent workflow implementations 
- **Agents Evaluation Framework**: Comprehensive support for evaluating diverse data agent architectures including Tool-Use Agents, Multi-Agent Systems, Planning Agents, and Reflection Agents
- **Universal Database Compatibility**: Seamlessly integrates with multiple database systems and real-world production environments
- **Flexible Task Architecture**: Supports three distinct workload types - single-choice questions, multiple-choice scenarios, and open-ended report generation tasks
- **Advanced Evaluation Metrics**: Built-in comprehensive evaluation system with detailed performance analytics and statistical insights
- **Rich Tool Ecosystem**: Extensive collection of integrated tools including database schema analysis, SQL query optimization, web search capabilities, and vector database operations
- **Extensible Agent Framework**: Modular base classes and interfaces that enable easy implementation and integration of custom agent architectures
- **Cost Monitoring**: Real-time token usage tracking and cost analysis for performance optimization and budget management

## Benchmark Workload

### Task Categories
- **Single Choice**: Multiple choice questions with one correct answer
- **Multiple Choice**: Questions allowing multiple correct answers
- **Free-form Report**: Open-ended analytical tasks requiring comprehensive database analysis

### Data Agent Interface
The benchmark uses a standardized agent interface that abstracts away implementation details, allowing fair comparison across different agent architectures and approaches.

## Environment Setup

### System Requirements

- **Python:** 3.10+
- **RAM:** 8GB+ recommended
- **Storage:** 5GB+ free space
- **OS:** Linux, macOS, Windows

### Option 1: One-Command Setup (Recommended)

Create the complete environment with all dependencies:

```bash
conda env create -f environment.yml
conda activate fdabench
```

This will:
- Create a new conda environment named `fdabench`
- Install Python 3.11 and all required dependencies
- Automatically install FDABench in development mode

### Option 2: Manual Setup

If you prefer manual installation:

```bash
# Create environment
conda create -n fdabench python=3.11
conda activate fdabench

# Install FDABench
pip install -e .
```

### API Configuration

Set up your API keys for LLM access:

```bash
# Option 1: Environment variables
export OPENROUTER_API_KEY="your-openrouter-api-key"

# Option 2: Create .env file in project root
echo "OPENROUTER_API_KEY=your-openrouter-api-key" >> .env
```

## Quick Start

After completing the environment setup above, you can immediately start using FDABench:

### Run Examples

**Built-in Sample Data**: The examples use sample data by default (`sample/sample_data.json` with `sample/regional_sales/regional_sales.sqlite` database) for immediate testing - no configuration needed!

```bash
# Activate your environment (if not already active)
conda activate fdabench

# Run your first example with built-in sample data
python examples/run_planning_agent.py
```

**Note**: If you want to run the full benchmark dataset with 2,007 tasks, follow the database configuration steps below:

### Database Configuration

FDABench supports multiple database types. You need to configure database paths and obtain required data:

#### 1. SQLite Databases

**BIRD Dataset**: Download from [BIRD repository](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird)

**Spider2-lite Dataset**: Download from [Spider2 spider-agent-lite](https://github.com/xlang-ai/Spider2/tree/main/methods/spider-agent-lite)

#### 2. Cloud Databases  

**BigQuery and Snowflake**: Follow registration and setup instructions from [Spider2 README](https://github.com/xlang-ai/Spider2/blob/main/README.md)

#### 3. Configure Database Paths

Edit `FDABench/utils/database_connection_manager.py` and update the configuration:

```python
default_config = {
    # SQLite database paths
    'bird_db_path': "/your/path/to/BIRD_train/train_databases",
    'local_db_path': "/your/path/to/local/databases", 
    'spider1_db_path': "/your/path/to/spider1/databases",
    
    # Cloud database credentials
    'bigquery_credentials_path': "/your/path/to/bigquery-service-account.json",
    'snowflake_config': {
        'account': 'your-snowflake-account',
        'user': 'your-username', 
        'password': 'your-password',
        'warehouse': 'your-warehouse',
        'database': 'your-database'
    }
}
```

#### 4. Directory Structure
```
your_databases/
├── BIRD_train/train_databases/
│   ├── california_schools/
│   │   └── california_schools.sqlite  
│   ├── card_games/
│   │   └── card_games.sqlite
│   └── ...
├── spider1_databases/
│   ├── concert_singer.sqlite
│   ├── pets_1.sqlite  
│   └── ...
├── local_databases/
│   └── merchant_data.db
└── credentials/
    └── bigquery-service-account.json
```

#### 5. Dataset Path Configuration

**Full Benchmark Dataset**: The complete FDABench dataset with 2,007 diverse tasks across multiple domains and difficulty levels is available in `dataset/` directory.

**Custom Datasets**: For your own datasets, you can:

```bash
# Option 1: Environment variable
export DATASET_PATH="/path/to/your/test/dataset"

# Option 2: Pass custom path to load_test_data() function
test_data = load_test_data("path/to/your/dataset.json")
```

### Available Examples

Test different agent implementations:

```bash
# Planning Agent - Uses step-by-step planning
python examples/run_planning_agent.py

# Multi-Agent - Coordinates multiple specialized agents
python examples/run_multi_agent.py

# Reflection Agent - Self-improving with reflection
python examples/run_reflection_agent.py

# Tool-Use Agent - Optimized for tool selection
python examples/run_tooluse_agent.py
```

#### Semantic Operator Agents

Specialized agents integrated with semantic data operators for advanced data processing:

```bash
# DocETL Semantic Operator Agent - Uses DocETL operators for document processing
python FDABench/examples/test_planning_agent_docetl_batch.py

# Lotus Semantic Operator Agent - Uses Lotus operators for natural language processing
python FDABench/examples/test_planning_agent_lotus_batch.py

# Palimpzest Semantic Operator Agent - Uses Palimpzest operators for data transformation
python FDABench/examples/test_planning_agent_pz_batch.py
```

**Note**: Semantic operator agents require additional environment setup. Check the respective environment files:
- `FDABench/examples/docetl_environment.yml`
- `FDABench/examples/lotus_environment.yml` 
- `FDABench/examples/palimpzest_environment.yml`

### Basic Usage Example

```python
from FDABench.agents.planning_agent import PlanningAgent
from FDABench.evaluation.evaluation_tools import ReportEvaluator

# Initialize agent with your preferred model
agent = PlanningAgent(
    model="openai/gpt-4",  # or "deepseek/deepseek-chat-v3"
    api_key="your-api-key"
)

# Process a single query
test_data = {
    "task_id": "example_task",
    "query": "What are the top 5 sales regions?",
    "question_type": "report",
    "db": "sales_database"
}

result = agent.process_query_from_json(test_data)
print(f"Generated report: {result['report'][:200]}...")
```

### Output and Results

All test results are automatically saved to:
- `results/` - DuckDB files with test results and metrics
- `FDABench/examples/data/` - Temporary processing files

## Agent Integration

### Adding a New Agent

To integrate a new DB agent, inherit from the `BaseAgent` class:

```python
from FDABench.core.base_agent import BaseAgent

class YourCustomAgent(BaseAgent):
    def __init__(self, model="openai/gpt-4", api_key=None, **kwargs):
        super().__init__(model=model, api_key=api_key, **kwargs)
        # Initialize your agent-specific components
    
    def process_query_from_json(self, query_data):
        """Main method to process queries from JSON format"""
        question_type = query_data.get("question_type", "report")
        
        if question_type == "single_choice":
            return self.process_single_choice(query_data)
        elif question_type == "multiple_choice":
            return self.process_multiple_choice(query_data)
        elif question_type == "report":
            return self.process_report(query_data)
    
    def process_single_choice(self, query_data):
        # Implement single choice question handling
        # Return {"selected_answer": "A", "metrics": {...}}
        pass
    
    def process_multiple_choice(self, query_data):
        # Implement multiple choice question handling  
        # Return {"selected_answers": ["A", "C"], "metrics": {...}}
        pass
    
    def process_report(self, query_data):
        # Implement report generation
        # Return {"report": "Generated report...", "metrics": {...}}
        pass
```

### Supported Integrations

- **OpenAI**: GPT-3.5, GPT-4, and other OpenAI models
- **LangChain**: Full LangChain ecosystem support
- **Private APIs**: Custom API integrations
- **Local Models**: Support for locally hosted LLMs

## Dataset Format

### Input Schema

The benchmark uses a structured JSON format for test cases:

```json
{
    "task_id": "FAD123",
    "instance_id": "bq001",
    "db": "ga360",
    "level": "hard",
    "database_type": "Spider2-lite",
    "question_type": "single_choice",
    "tools_available": ["get_schema_info", "generated_sql", "execute_sql"],
    "query": "Your database question here",
    "options": {
        "A": "Option A text",
        "B": "Option B text", 
        "C": "Option C text",
        "D": "Option D text"
    },
    "correct_answer": ["C"],
    "explanation": "Detailed explanation of the correct answer"
}
```

### Dataset Structure

```
dataset_path/
├── task_type_mapping.json          # Maps task IDs to agent types
├── test_singlechoice.json         # Single choice questions
├── test_multichoice.json          # Multiple choice questions  
└── test_report.json               # Report generation tasks
```

Datasets can be loaded using the built-in utilities or provided as custom paths to the examples.

## Evaluation Metrics

### Core Metrics

- **Accuracy**: Percentage of correctly answered questions
- **Execution Success**: Rate of successful SQL query execution
- **Latency**: Average response time per query
- **Token Efficiency**: Tokens used per successful query
- **Tool Usage Score**: Effectiveness of tool selection and usage

### Advanced Analytics

- **Error Analysis**: Categorization of failure modes
- **Complexity Scaling**: Performance across different difficulty levels
- **Database Type Performance**: Results segmented by database system
- **Agent Architecture Comparison**: Comparative analysis across agent types

## Directory Structure

```
FDABench/
├── FDABench/                     # Main package
│   ├── agents/                   # Pre-implemented agent types
│   │   ├── multi_agent.py       # Multi-agent coordination system
│   │   ├── planning_agent.py    # Step-by-step planning agent
│   │   ├── reflection_agent.py  # Self-reflective agent
│   │   └── tool_use_agent.py    # Tool-focused agent
│   ├── core/                    # Core framework components
│   │   ├── base_agent.py       # Base agent interface
│   │   ├── token_tracker.py    # Token usage monitoring
│   │   └── tool_registry.py    # Tool management system
│   ├── evaluation/              # Evaluation and scoring tools
│   │   └── evaluation_tools.py # Comprehensive evaluation suite
│   ├── tools/                   # Database and utility tools
│   │   ├── schema_tools.py     # Database schema analysis
│   │   ├── sql_tools.py        # SQL generation and optimization
│   │   └── search_tools.py     # Web and vector search tools
│   ├── utils/                   # Utility functions
│   │   └── database_connection_manager.py  # Database connectivity
│   └── prompts/                 # Prompt templates and management
│       └── prompts.py          # Standard prompts for agents
├── examples/                    # Usage examples and test scripts
│   ├── data/                   # Temporary processing files
│   ├── run_planning_agent.py   # Planning agent examples
│   ├── run_multi_agent.py      # Multi-agent examples
│   ├── run_reflection_agent.py # Reflection agent examples
│   └── run_tooluse_agent.py    # Tool-use agent examples
├── FDABench/examples/           # Semantic operator agents  
│   ├── test_planning_agent_docetl_batch.py   # DocETL semantic operators
│   ├── test_planning_agent_lotus_batch.py    # Lotus semantic operators
│   ├── test_planning_agent_pz_batch.py       # Palimpzest semantic operators
│   ├── docetl_environment.yml  # DocETL environment setup
│   ├── lotus_environment.yml   # Lotus environment setup
│   └── palimpzest_environment.yml # Palimpzest environment setup
├── sample/                      # Built-in sample data for testing
│   ├── sample_data.json        # Sample task configuration
│   └── regional_sales/         # Sample database directory
│       └── regional_sales.sqlite # Sample SQLite database
├── dataset/                     # Full benchmark dataset (2,007 tasks)
│   ├── test_singlechoice.json  # Single choice questions
│   ├── test_multichoice.json   # Multiple choice questions  
│   └── test_report.json        # Report generation tasks
├── results/                     # Test results and output files
├── environment.yml             # Conda environment specification
├── pyproject.toml              # Package configuration
├── INSTALL.md                  # Detailed installation guide
└── README.md                   # This file
```

## Contributing

We welcome contributions to FDABench! Here's how you can help:

### Adding New Benchmark Tasks

1. **Create task definitions** following the existing JSON schema
2. **Include gold standard answers** with detailed explanations
3. **Test with multiple agent types** to ensure fairness
4. **Document complexity levels** and expected difficulty

### Contributing New Agent Wrappers

1. **Inherit from BaseAgent** or CustomAgentBase
2. **Implement all required methods** (single_choice, multiple_choice, report)
3. **Add comprehensive error handling** and logging
4. **Include usage examples** and documentation
5. **Test with the evaluation suite** to ensure compatibility

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints for all public methods
- Include docstrings with parameter descriptions
- Write unit tests for new functionality
- Update documentation for API changes

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request with detailed description

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: FDABench is designed to provide fair and comprehensive evaluation of database agents. The benchmark continues to evolve with new task types, databases, and evaluation metrics to keep pace with advancing agent technologies.