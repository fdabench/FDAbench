"""
All prompt templates for FDABench.

Contains all prompts used across different agent patterns for database operations,
planning, evaluation, and tool selection extracted from original agents without modification.
"""

from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

# SQL Generation prompts
PROMPTS["SQL_GENERATION_FULL"] = """Given the following database schema for the SQLite database '{database_name}':
Schema:
{schema_info}

Based *only* on this schema and the user's request, generate a single, valid SQLite query to answer the following:
User Request: "{natural_language_query}"

Important Rules:
- Only use tables and columns listed in the schema
- Ensure correct SQLite syntax
- For JSON fields, use json_extract() function instead of -> or ->> operators
- Return ONLY the SQL query
- If the request cannot be answered with the given schema, return "QUERY_IMPOSSIBLE"

Example of JSON handling in SQLite:
- Instead of: column->>'$."en"'
- Use: json_extract(column, '$.en')
"""

PROMPTS["SQL_GENERATION_SIMPLE"] = """Generate {db_type_info} SQL for: "{natural_language_query}"
Schema: {schema_summary}
Rules: {syntax_notes}
Output only SQL or "QUERY_IMPOSSIBLE"."""

# SQL Optimization prompts
PROMPTS["SQL_OPTIMIZATION_FULL"] = """Given the following database schema and SQL query, optimize the query for better performance:

Database Schema:
{schema_info}

Original SQL Query:
{sql_query}

Please optimize this query considering:
1. Index usage
2. Join order
3. Subquery optimization
4. WHERE clause optimization
5. SELECT column optimization

Return ONLY the optimized SQL query. If no optimization is possible, return the original query.
"""

PROMPTS["SQL_OPTIMIZATION_SIMPLE"] = """Optimize SQL: {sql_query}
Schema: {schema_summary}
Return optimized SQL only."""

# SQL Debugging prompts
PROMPTS["SQL_DEBUG_FULL"] = """Given the following information:
1. Database Schema:
{schema_info}

2. Original Natural Language Query:
{natural_language_query}

3. Failed SQL Query:
{failed_sql}

4. Error Message:
{error}

Please analyze the error and generate a corrected SQL query that will work correctly.
Consider:
1. Syntax errors
2. Table/column name mismatches
3. Data type issues
4. Missing or incorrect joins
5. Incorrect function usage

Return ONLY the corrected SQL query. If you cannot fix the query, return "QUERY_UNFIXABLE".
"""

PROMPTS["SQL_DEBUG_SIMPLE"] = """Fix SQL error:
Query: {natural_language_query}
Failed: {failed_sql}
Error: {error}
Schema: {schema_summary}
Return corrected SQL only."""

# Planning prompts from planning_agent.py
PROMPTS["PLANNING_TASK_GENERATION"] = """Plan for: {query}...
DB: {db} ({database_type})
Tools: {tools_available}

Generate JSON array with tool, description, input fields.
Focus: schema_understanding → sql_generate → sql_execute
Output only JSON array."""

# Report generation prompts
PROMPTS["REPORT_GENERATION_FULL"] = """Based on the following complex query and the results from each subtask/tool, generate a structured and detailed analytical report.

Complex query: {advanced_query}

Tool results:
{tool_results}

Please output in markdown format, including sections such as Executive Summary, Analysis, and Conclusion.
"""

PROMPTS["REPORT_GENERATION_SIMPLE"] = """Query: {advanced_query}
Results: {results_summary}
Generate a concise analytical report with key findings."""

# Report reflection prompts
PROMPTS["REPORT_REFLECTION"] = """Please review the following report and provide suggestions for improvement (if any):
{final_report}"""

PROMPTS["REPORT_REVISION"] = """Please revise the report based on the following feedback:
Original report: {final_report}
Feedback: {reflection}"""

# Multiple choice answer generation prompts
PROMPTS["MULTIPLE_CHOICE_ANSWER"] = """Query: {advanced_query}
Options: {options}
Results: {tool_results}

Select ALL correct answers (e.g., A,B,C or B,D or A only):"""

# Single choice answer generation prompts  
PROMPTS["SINGLE_CHOICE_ANSWER"] = """Query: {advanced_query}
Options: {options}
Results: {tool_results}

Select ONE correct answer (A, B, C, or D):"""

# Tool selection prompts from tool_use_agent.py
PROMPTS["TOOL_SELECTION"] = """Query: {query}
Available: {valid_tools}
Used: {tools_executed}
Select next tool or "none" if sufficient."""

# Domain classification prompts from base_agent.py
PROMPTS["DOMAIN_CLASSIFICATION"] = """Given the following query, classify it into one or more domains from this list:
{domains_list}

Query: {query}

Return exactly three most relevant domain names separated by semicolons. Example: "Trading; Finance_Investment; E-commerce".
Do not include any other text, only return the three domain names separated by semicolons."""

# Vector search prompts from base_agent.py
PROMPTS["VECTOR_SEARCH"] = """Please search for and provide relevant information about: {query}. 
Focus on extracting key facts, statistics, definitions, examples, and relevant context that could help understand this topic. 
Include information on related concepts, historical background, common use cases, technical details, and potential applications if available. 
Gather as much information as possible, even if not perfectly precise - breadth is important. 
Don't worry about providing a complete or definitive answer, just collect all potentially useful information. 
If there are multiple interpretations or perspectives on the topic, include them all. 
Return all relevant information without attempting to synthesize or fully answer the query."""


def get_prompt(prompt_key: str, **kwargs) -> str:
    """
    Get a formatted prompt by key.
    
    Args:
        prompt_key: Key of the prompt in PROMPTS dictionary
        **kwargs: Variables to format into the prompt
        
    Returns:
        Formatted prompt string
    """
    if prompt_key not in PROMPTS:
        raise KeyError(f"Prompt '{prompt_key}' not found")
    
    return PROMPTS[prompt_key].format(**kwargs)


def list_prompts() -> list[str]:
    """List all available prompt keys."""
    return list(PROMPTS.keys())


def add_prompt(key: str, template: str) -> None:
    """Add a new prompt template."""
    PROMPTS[key] = template


def remove_prompt(key: str) -> bool:
    """Remove a prompt template."""
    if key in PROMPTS:
        del PROMPTS[key]
        return True
    return False