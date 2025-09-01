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
                top_k: int = 5, domains: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform vector database search.
        
        Args:
            query: Search query
            expected_query: Alternative query format
            top_k: Number of top results to return
            domains: Specific domains to search in
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with status and results
        """
        try:
            search_query = query or expected_query
            if not search_query:
                return {"status": "error", "error": "No search query provided"}
            
            # Use custom vector search function if provided
            if self.vector_search_function:
                try:
                    results = self.vector_search_function(search_query, top_k, domains)
                    return {
                        "status": "success", 
                        "results": results,
                        "query": search_query,
                        "top_k": top_k,
                        "domains": domains
                    }
                except Exception as e:
                    logger.error(f"Custom vector search function failed: {e}")
                    return {"status": "error", "error": f"Custom vector search function failed: {str(e)}"}
            
            # Use vector manager if available
            if self.vector_manager:
                try:
                    results = self.vector_manager.search(search_query, top_k=top_k)
                    if not results:
                        return {"status": "error", "error": "Vector manager returned empty results"}
                    return {
                        "status": "success", 
                        "results": results,
                        "query": search_query,
                        "top_k": top_k,
                        "domains": domains
                    }
                except Exception as e:
                    logger.error(f"Vector manager search failed: {e}")
                    return {"status": "error", "error": f"Vector manager search failed: {str(e)}"}
            
            # Try to use base_agent vector search if available
            try:
                # Import vector search functions from base_agent
                from ..agents.base_agent import _search_single_domain, _get_domain_classification
                
                # Get domain classification
                if not domains:
                    # Create a simple domain classifier
                    if hasattr(self, '_get_domain_classification'):
                        domain_string = self._get_domain_classification(search_query)
                        domains = domain_string.split(";")[:3]  # Top 3 domains
                    else:
                        # Default domains
                        domains = ["E-commerce", "Business Management", "Database_Data"]
                
                # Search domains
                all_responses = []
                for domain in domains[:3]:  # Limit to 3 domains
                    domain = domain.strip()
                    if domain:
                        response = _search_single_domain(domain, search_query)
                        if response:
                            all_responses.append(response)
                
                if all_responses:
                    results = "\n\n".join(all_responses)
                    return {
                        "status": "success",
                        "results": results,
                        "query": search_query,
                        "top_k": top_k,
                        "domains": domains,
                        "searched_domains": len(all_responses)
                    }
                else:
                    return {"status": "error", "error": "No relevant information found in vector database"}
                
            except ImportError:
                logger.error("Base agent vector search functions not available")
                return {"status": "error", "error": "Vector search functions not available"}
            except Exception as e:
                logger.error(f"Base agent vector search failed: {e}")
                return {"status": "error", "error": f"Vector search failed: {str(e)}"}
            
            # If no vector search method available, return error
            return {"status": "error", "error": "No vector search functionality available"}
            
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return {"status": "error", "error": str(e)}