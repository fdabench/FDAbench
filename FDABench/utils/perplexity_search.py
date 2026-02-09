"""
Perplexity Search implementation using OpenRouter.

This module provides a perplexity_search function that uses OpenRouter
to call the perplexity/sonar model for web search functionality.
"""

import os
import logging
from typing import Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


def perplexity_search(query: str, use_web_search: bool = True, api_key: Optional[str] = None) -> str:
    """
    Perform web search using Perplexity Sonar model via OpenRouter.
    
    Args:
        query: Search query string
        use_web_search: Whether to enable web search (default: True)
        api_key: OpenRouter API key (will use environment variable if not provided)
        
    Returns:
        Search results as string
    """
    try:
        # Get API key from parameter or environment variable
        if not api_key:
            api_key = os.environ.get('OPENROUTER_API_KEY')
            if not api_key:
                logger.error("No OpenRouter API key provided")
                return "Error: No OpenRouter API key available for perplexity search"
        
        # Initialize OpenRouter client
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        # Prepare the search prompt
        if use_web_search:
            system_prompt = """You are a helpful web search assistant. Search the web for the given query and provide a comprehensive, accurate response based on current information. Include relevant facts, statistics, and context. Cite sources when possible."""
        else:
            system_prompt = """You are a helpful assistant. Answer the given query based on your knowledge. Provide accurate and helpful information."""
        
        # Create messages for the API call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        # Call the perplexity/sonar model via OpenRouter
        completion = client.chat.completions.create(
            model="perplexity/sonar",
            messages=messages,
            temperature=0.7,
            max_tokens=4000,
            extra_headers={
                "HTTP-Referer": "https://github.com/wa../FDABench",
                "X-Title": "FDABenchmark Perplexity Search"
            }
        )
        
        # Extract and return the response
        response = completion.choices[0].message.content
        logger.info(f"Perplexity search completed for query: {query[:50]}...")
        
        return response
        
    except Exception as e:
        logger.error(f"Perplexity search failed: {str(e)}")
        return f"Error performing perplexity search: {str(e)}"


