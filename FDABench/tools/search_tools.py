"""
Search-related tools for FDABench Package.

These tools provide web search and vector search capabilities.
They can integrate with external search services and vector databases.
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class WebSearchTool:
    """Tool for web search using various search APIs"""
    
    def __init__(self, api_key: str = None, search_engine: str = "perplexity", search_function=None):
        self.api_key = api_key
        self.search_engine = search_engine
        self.search_function = search_function  # Allow injecting custom search function
    
    def execute(self, query: str, expected_query: str = None, 
                max_results: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Perform web search.
        
        Args:
            query: Search query
            expected_query: Alternative query format
            max_results: Maximum number of results
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with status and results
        """
        try:
            search_query = query or expected_query
            if not search_query:
                return {"status": "error", "error": "No search query provided"}
            
            # Use custom search function if provided
            if self.search_function:
                try:
                    results = self.search_function(search_query, self.api_key, max_results)
                    return {
                        "status": "success",
                        "results": results,
                        "search_engine": self.search_engine,
                        "query": search_query
                    }
                except Exception as e:
                    logger.error(f"Custom search function failed: {e}")
                    return {"status": "error", "error": f"Custom search function failed: {str(e)}"}
            
            # Try to use real search API if available
            if self.api_key:
                try:
                    # Import and use the perplexity search function from utils
                    try:
                        from ..utils.perplexity_search import perplexity_search
                        results = perplexity_search(search_query, True, self.api_key)
                        if not results or results.strip() == "":
                            return {"status": "error", "error": "Search API returned empty results"}
                        if results.startswith("Error"):
                            return {"status": "error", "error": results}
                        return {
                            "status": "success",
                            "results": results,
                            "search_engine": self.search_engine,
                            "query": search_query
                        }
                    except ImportError:
                        logger.error("perplexity_search function not available")
                        return {"status": "error", "error": "Search function not available"}
                except Exception as e:
                    logger.error(f"Real search API failed: {e}")
                    return {"status": "error", "error": f"Search API failed: {str(e)}"}
            
            # If no API key provided, return error
            return {"status": "error", "error": "No API key provided for web search"}
            
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            return {"status": "error", "error": str(e)}


class VectorSearchTool:
    """Tool for vector database search"""
    
    def __init__(self, vector_db_config: Dict = None, vector_search_function=None):
        self.vector_db_config = vector_db_config or {}
        self.vector_search_function = vector_search_function  # Allow injecting custom vector search
        self.vector_manager = None
        
        # Try to initialize vector manager
        try:
            # This would be imported from utils if available
            # from ..utils import VectorSearchManager
            # self.vector_manager = VectorSearchManager()
            pass
        except ImportError:
            logger.warning("VectorSearchManager not available")
    
    def execute(self, query: str, expected_query: str = None,
                top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Perform vector database search.
        
        Args:
            query: Search query
            expected_query: Alternative query format
            top_k: Number of top results to return
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with status and results
        """
        try:
            search_query = query or expected_query
            if not search_query:
                return {"status": "error", "error": "No search query provided"}
            
            if self.vector_search_function:
                try:
                    results = self.vector_search_function(search_query, top_k)
                    return {
                        "status": "success", 
                        "results": results,
                        "query": search_query,
                        "top_k": top_k
                    }
                except Exception as e:
                    logger.error(f"Custom vector search function failed: {e}")
                    return {"status": "error", "error": f"Custom vector search function failed: {str(e)}"}
            
            if self.vector_manager:
                try:
                    results = self.vector_manager.search(search_query, top_k=top_k)
                    if not results:
                        return {"status": "error", "error": "Vector manager returned empty results"}
                    return {
                        "status": "success", 
                        "results": results,
                        "query": search_query,
                        "top_k": top_k
                    }
                except Exception as e:
                    logger.error(f"Vector manager search failed: {e}")
                    return {"status": "error", "error": f"Vector manager search failed: {str(e)}"}
            
            try:
                from llama_index import VectorStoreIndex, StorageContext, load_index_from_storage
                
                storage_context = StorageContext.from_defaults(persist_dir="./storage")
                index = load_index_from_storage(storage_context)
                
                query_engine = index.as_query_engine(similarity_top_k=top_k)
                response = query_engine.query(search_query)
                
                return {
                    "status": "success",
                    "results": str(response),
                    "query": search_query,
                    "top_k": top_k
                }
            except Exception as e:
                logger.error(f"LlamaIndex query failed: {e}")
                return {"status": "error", "error": f"Vector search failed: {str(e)}"}
            
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return {"status": "error", "error": str(e)}