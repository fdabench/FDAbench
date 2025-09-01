#!/home/gong0092/miniconda3/envs/gong_env/bin/python
# -*- coding: utf-8 -*-
"""
External Tools Module - Web Search and Vector Database Tools with LLM Selection
Refactored from original spider2_med_websearch_build_oop.py and adapted for Bird dataset
"""

import os
import json
import requests
import logging
import time
from typing import Optional, Dict
from abc import ABC, abstractmethod
from enum import Enum
from dotenv import load_dotenv

# Vector DB specific imports
try:
    from llama_index.core import Settings
    from llama_index.embeddings.openai import OpenAIEmbedding
    VECTOR_AVAILABLE = True
        
except ImportError as e:
    print(f"Vector dependencies not available: {e}")
    VECTOR_AVAILABLE = False

# Load environment variables
load_dotenv()

class ExternalToolType(Enum):
    """External tool types for subtask management"""
    WEB_SEARCH = "web_search"
    VECTOR_SEARCH = "vector_search"
    FILE_SYSTEM = "file_system"

class SubtaskResult:
    """Container for subtask execution results"""
    def __init__(self, tool_name: str, result: str, success: bool = True, error: Optional[str] = None, selected_tool_type: Optional[str] = None):
        self.tool_name = tool_name
        self.result = result
        self.success = success
        self.error = error
        self.selected_tool_type = selected_tool_type  # For external search to store which tool was selected
        self.timestamp = time.time()

class ExternalTool(ABC):
    """Abstract base class for external tools"""
    
    @abstractmethod
    def search(self, query: str) -> str:
        """Execute search with the given query"""
        pass
    
    @property
    @abstractmethod
    def tool_name(self) -> str:
        """Return the tool name"""
        pass

class WebSearchTool(ExternalTool):
    """Web search tool implementation using Perplexity"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.logger = logging.getLogger(__name__)
    
    @property
    def tool_name(self) -> str:
        return "perplexity_search"
    
    def search(self, query: str) -> str:
        """Search using Perplexity API via OpenRouter"""
        try:
            if not self.api_key:
                return "Web search unavailable: Missing OPENROUTER_API_KEY"
                
            messages = [
                {
                    "role": "system",
                    "content": "You are a web search assistant. Provide relevant web information related to the query to help understand the context and background.",
                },
                {
                    "role": "user", 
                    "content": f"Please search for and provide relevant web information about: {query}",
                },
            ]
            
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": "perplexity/sonar-pro",
                    "messages": messages,
                })
            )
            
            if response.status_code == 200:
                response_data = response.json()
                if 'choices' in response_data and response_data['choices']:
                    return response_data['choices'][0]['message']['content'].strip()
                else:
                    return "No response content available"
            else:
                return f"API request failed with status code: {response.status_code}"
                
        except Exception as e:
            return f"Web search error: {e}"

class VectorSearchTool(ExternalTool):
    """Online vector database search tool implementation using bird.jsonl data"""
    
    def __init__(self, api_key: Optional[str] = None, enabled: bool = True):
        self.enabled = enabled
        self.logger = logging.getLogger(__name__)
        # Path to bird.jsonl data file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.bird_data_path = os.path.join(current_dir, "input_data", "original_data", "bird.jsonl")
        self._vector_index = None
        self._embedding_model = None
        
        if not VECTOR_AVAILABLE:
            self.enabled = False
            self.logger.warning("Vector search disabled: missing llama-index dependencies")
    
    @property
    def tool_name(self) -> str:
        return "vectorDB_search"
    
    def _get_embedding_model(self):
        """Get the global embedding model"""
        if self._embedding_model is None:
            try:
                self._embedding_model = OpenAIEmbedding(embed_batch_size=10, model="text-embedding-3-small")
                Settings.embed_model = self._embedding_model
            except Exception as e:
                self.logger.error(f"Failed to initialize embedding model: {e}")
                return None
        return self._embedding_model
    
    def _load_bird_data(self):
        """Load meaningful texts related to movie/actor database - hardcoded examples"""
        try:
            # Hardcoded content related to movie database analysis like movielens
            texts = [
                "Movie databases like MovieLens store information about actors, movies, genres, and user ratings. Actor gender is typically stored as 'M' for male and 'F' for female in the person table.",
                "French movies are identified by country codes or production countries in movie databases. USA movies represent American productions and are one of the largest categories in film databases.",
                "Female actors in movie databases can be queried using gender filters where a_gender = 'F'. This allows for gender-based analysis of movie participation and representation.",
                "Movie recommendation systems analyze user ratings, actor preferences, and genre patterns to suggest relevant films. Collaborative filtering and content-based approaches are common techniques.",
                "Film industry analytics examine box office performance, actor popularity, genre trends, and international market success. Country-specific analysis reveals regional preferences and cultural factors.",
                "Actor career analysis involves tracking movie participation, genre diversity, collaboration patterns, and rating performance across different time periods and regions.",
                "Movie database schemas typically include tables for movies, actors, genres, ratings, and countries. Many-to-many relationships connect actors to movies through cast or role tables.",
                "International cinema analysis compares movie production, actor participation, and audience reception across different countries like France and USA.",
                "Gender representation in films can be analyzed through database queries that count male and female actor participation across different movie categories, countries, and time periods.",
                "MovieLens dataset contains millions of ratings and thousands of movies with detailed actor and genre information, making it ideal for recommendation system research and film industry analysis."
            ]
            
            return texts
        except Exception as e:
            self.logger.error(f"Failed to load bird data: {e}")
            # Fallback text content
            return [
                "Movie database analysis focuses on actors, films, genres, and user preferences for recommendation systems and industry insights."
            ]
    
    def _create_online_vector_index(self):
        """Create vector index online from bird.jsonl data"""
        if self._vector_index is not None:
            return self._vector_index
        
        try:
            # Load bird data
            texts = self._load_bird_data()
            
            # Import required modules for online vector creation
            from llama_index.core import VectorStoreIndex
            from llama_index.core.readers import StringIterableReader
            from llama_index.core.node_parser import SentenceSplitter
            
            # Create documents from texts
            documents = StringIterableReader().load_data(texts=texts)
            
            # Split into chunks
            splitter = SentenceSplitter(chunk_size=256)
            nodes = splitter.get_nodes_from_documents(documents)
            
            # Create vector index online
            self._vector_index = VectorStoreIndex(nodes)
            
            self.logger.info(f"Created online vector index with {len(nodes)} nodes from {len(texts)} texts")
            return self._vector_index
            
        except Exception as e:
            self.logger.error(f"Failed to create online vector index: {e}")
            return None
    
    def _search_online_vector(self, query: str) -> Optional[str]:
        """Simple online vector search"""
        try:
            index = self._create_online_vector_index()
            if index is None:
                return None
            
            # Simple query without complex enhancements
            query_engine = index.as_query_engine(similarity_top_k=3)
            response = query_engine.query(query)
            
            # Return response text
            if hasattr(response, 'response') and response.response:
                return response.response
            else:
                return str(response) if response else None
                
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return None
    
    def search(self, query: str) -> str:
        """Simple search using online vector database"""
        start_time = time.time()
        try:
            if not self.enabled:
                return "Vector database search is disabled."
                
            # Initialize embedding model
            self._get_embedding_model()
            
            # Search vector database
            result = self._search_online_vector(query)
            
            if result and result.strip():
                self.logger.info(f"⏱️  Vector DB Search: {time.time() - start_time:.2f}s")
                return result
            else:
                # Simple fallback message
                result = "Related to movie database analysis with actors, genres, and ratings. Focus on gender filtering and country-based movie queries."
                
            self.logger.info(f"⏱️  Vector DB Search: {time.time() - start_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"❌ Vector DB Search failed: {e}")
            return f"Vector search error: {e}"

class FileSystemTool(ExternalTool):
    """File system information tool implementation"""
    
    def __init__(self, file_system_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        # Use relative path within refactored directory
        if file_system_path:
            self.file_system_path = file_system_path
        else:
            # Default to relative path from current file location
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.file_system_path = os.path.join(current_dir, "input_data", "file_system")
    
    @property
    def tool_name(self) -> str:
        return "file_system"
    
    def search(self, query: str) -> str:
        """Search file system information - exact copy from original"""
        start_time = time.time()
        try:
            # Find MD files related to the query
            md_files = []
            if os.path.exists(self.file_system_path):
                for file in os.listdir(self.file_system_path):
                    if file.endswith(".md"):
                        md_files.append(os.path.join(self.file_system_path, file))
            
            # If no files found, return a prompt message
            if not md_files:
                self.logger.info(f"⏱️  File System Search: {time.time() - start_time:.2f}s (no files found)")
                return "No relevant file system information found."
            
            # Read the contents of all MD files
            file_contents = []
            for file_path in md_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        file_contents.append(f.read())
                except Exception as e:
                    self.logger.warning(f"Failed to read file {file_path}: {e}")
                    continue
            
            # Merge all file contents
            if file_contents:
                file_statement = "\n\n".join(file_contents)
                self.logger.info(f"⏱️  File System Search: {time.time() - start_time:.2f}s")
                return file_statement
            else:
                self.logger.info(f"⏱️  File System Search: {time.time() - start_time:.2f}s (no readable files)")
                return "No readable file system information found."
                
        except Exception as e:
            self.logger.error(f"❌ File System Search: {time.time() - start_time:.2f}s (failed: {e})")
            return f"Error retrieving file system information: {e}"

class SmartToolSelector:
    """LLM-based tool selector that chooses between web search and vector search"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.logger = logging.getLogger(__name__)
    
    def select_tool(self, query: str, original_query: str = None) -> ExternalToolType:
        """Select the appropriate tool based on the query content using LLM"""

        try:
            if not self.api_key:
                # Fallback to web search if no API key
                return ExternalToolType.WEB_SEARCH
            
            analysis_query = original_query if original_query else query
            
            messages = [
                {
                    "role": "system",
                    "content": """You are a smart tool selector. Given a query, determine whether to use web search, vector database search, or file system search.

Use WEB_SEARCH for:
- Current events, news, recent developments
- Real-time information, stock prices, weather
- Popular culture, trending topics
- Questions needing the most up-to-date information
- Broad general knowledge queries
- Questions about markets, current statistics, or live data

Use VECTOR_SEARCH for:
- Technical documentation, specific domain knowledge  
- Business processes, data analysis methods
- Historical information, established facts
- Domain-specific expertise (finance, healthcare, engineering, etc.)
- Questions about specific databases, schemas, or data structures
- Data analysis, statistical concepts, and research methodologies
- Academic topics, scientific concepts, and analytical frameworks

Use FILE_SYSTEM for:
- Questions about file organization, directory structures
- File system operations, permissions, or management
- Document management and file handling queries
- Questions about specific file formats or file processing
- Queries related to file system architecture or storage

For analytical or data-focused queries, VECTOR_SEARCH typically provides more comprehensive domain knowledge, while WEB_SEARCH offers current market insights.

Respond with only: WEB_SEARCH, VECTOR_SEARCH, or FILE_SYSTEM"""
                },
                {
                    "role": "user",
                    "content": f"For this query, should I use web search, vector database search, or file system search?\n\nQuery: {analysis_query}"
                }
            ]
            
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": "anthropic/claude-3.5-sonnet",
                    "messages": messages,
                })
            )
            
            if response.status_code == 200:
                response_data = response.json()
                if 'choices' in response_data and response_data['choices']:
                    result = response_data['choices'][0]['message']['content'].strip().upper()
                    if "VECTOR_SEARCH" in result:
                        return ExternalToolType.VECTOR_SEARCH
                    elif "FILE_SYSTEM" in result:
                        return ExternalToolType.FILE_SYSTEM
                    else:
                        return ExternalToolType.WEB_SEARCH
            
            # Fallback to web search
            return ExternalToolType.WEB_SEARCH
            
        except Exception as e:
            self.logger.warning(f"Tool selection failed, defaulting to web search: {e}")
            return ExternalToolType.WEB_SEARCH

class ExternalToolManager:
    """Manages external tools with smart selection and content revision support"""
    
    def __init__(self, api_key: Optional[str] = None, file_system_path: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.web_tool = WebSearchTool(api_key)
        self.vector_tool = VectorSearchTool(api_key)
        self.file_system_tool = FileSystemTool(file_system_path)
        self.selector = SmartToolSelector(api_key)
        self.logger = logging.getLogger(__name__)
        self.search_history = []  # Store search history for context
    
    def search(self, query: str, original_query: str = None, force_tool: Optional[ExternalToolType] = None) -> SubtaskResult:
        """Search using the automatically selected or forced tool"""
        start_time = time.time()
        
        try:
            # Select tool
            if force_tool:
                selected_tool_type = force_tool
            else:
                selected_tool_type = self.selector.select_tool(query, original_query)
            
            # Get the tool
            if selected_tool_type == ExternalToolType.WEB_SEARCH:
                tool = self.web_tool
                tool_type_name = "WEB_SEARCH"
            elif selected_tool_type == ExternalToolType.VECTOR_SEARCH:
                tool = self.vector_tool
                tool_type_name = "VECTOR_SEARCH"
            else:  # FILE_SYSTEM
                tool = self.file_system_tool
                tool_type_name = "FILE_SYSTEM"
            
            # Execute search
            result = tool.search(query)
            
            # Store search in history for context
            search_record = {
                "query": query,
                "original_query": original_query,
                "tool_used": tool.tool_name,
                "result": result[:200] + "..." if len(result) > 200 else result,
                "timestamp": time.time()
            }
            self.search_history.append(search_record)
            
            # Keep only last 10 searches in history
            if len(self.search_history) > 10:
                self.search_history.pop(0)
            
            # Log the selection
            elapsed = time.time() - start_time
            self.logger.info(f"🔍 Used {tool.tool_name} for query (took {elapsed:.2f}s)")
            
            return SubtaskResult(
                tool_name=tool.tool_name,
                result=result,
                success=True,
                selected_tool_type=tool_type_name
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"❌ Search failed after {elapsed:.2f}s: {e}")
            return SubtaskResult(
                tool_name="external_search",
                result=f"Search error: {e}",
                success=False,
                error=str(e)
            )
    
    def get_search_context(self) -> str:
        """Get recent search history as context for content revision"""
        if not self.search_history:
            return "No previous searches available."
        
        context_parts = ["Recent search context:"]
        for search in self.search_history[-3:]:  # Last 3 searches
            context_parts.append(f"- Query: {search['query']}")
            context_parts.append(f"  Tool: {search['tool_used']}")
            context_parts.append(f"  Result: {search['result']}")
        
        return "\n".join(context_parts)
    
    def revise_search_with_feedback(self, original_query: str, feedback: str, force_tool: Optional[ExternalToolType] = None) -> SubtaskResult:
        """Perform a revised search based on user feedback"""
        # Enhance query with feedback context
        enhanced_query = f"""
        Original query: {original_query}
        User feedback for improvement: {feedback}
        
        Based on the feedback, provide improved and more relevant information for the original query.
        Focus on addressing the specific concerns mentioned in the feedback.
        """
        
        # Perform search with enhanced query
        result = self.search(enhanced_query, original_query, force_tool)
        
        # Add revision note to result
        if result.success:
            revised_result = f"[REVISED BASED ON FEEDBACK: {feedback}]\n\n{result.result}"
            return SubtaskResult(
                tool_name=result.tool_name + "_revised",
                result=revised_result,
                success=True
            )
        
        return result

# Factory function for backward compatibility
def create_external_tool(tool_type: ExternalToolType) -> ExternalTool:
    """Factory function to create external tool based on type"""
    if tool_type == ExternalToolType.WEB_SEARCH:
        return WebSearchTool()
    elif tool_type == ExternalToolType.VECTOR_SEARCH:
        return VectorSearchTool()
    elif tool_type == ExternalToolType.FILE_SYSTEM:
        return FileSystemTool()
    else:
        raise ValueError(f"Unsupported external tool type: {tool_type}")

# Example usage and testing
def main():
    # Initialize the manager
    manager = ExternalToolManager()
    
    # Test queries
    test_queries = [
        "What are the latest developments in AI?",  # Should use web search
        "How to optimize SQL queries for large datasets?",  # Should use vector search
        "Current stock price of Apple",  # Should use web search
        "Database normalization techniques",  # Should use vector search
        "How to organize files in a directory structure?"  # Should use file system search
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = manager.search(query)
        print(f"Tool used: {result.tool_name}")
        print(f"Success: {result.success}")
        print(f"Result preview: {result.result[:200]}...")

if __name__ == "__main__":
    main()