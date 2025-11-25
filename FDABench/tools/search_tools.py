"""
Search-related tools for FDABench Package.

These tools provide web search and vector search capabilities.
Vector search uses FAISS + OpenAI Embeddings (no LlamaIndex dependency).
"""

import os
import json
import logging
from typing import Dict, List, Any

import numpy as np

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
    """
    Tool for vector database search using FAISS + OpenAI Embeddings.

    This tool loads a pre-built FAISS index and performs semantic search
    using OpenAI's text-embedding-3-small model.
    """

    # Constants
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIM = 1536

    def __init__(
        self,
        vector_db_config: Dict = None,
        vector_search_function=None,
        storage_path: str = None,
        api_key: str = None
    ):
        """
        Initialize VectorSearchTool.

        Args:
            vector_db_config: Configuration dictionary for vector database
            vector_search_function: Custom search function to use
            storage_path: Path to the vector index storage directory.
                         If not specified, uses './storage' as default.
            api_key: OpenAI API key (optional, uses env var if not provided)
        """
        self.vector_db_config = vector_db_config or {}
        self.vector_search_function = vector_search_function
        self.storage_path = storage_path or self.vector_db_config.get('storage_path', './storage')
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        # Lazy-loaded components
        self._index = None
        self._chunks = None
        self._openai_client = None

    def _get_openai_client(self):
        """Get or create OpenAI client."""
        if self._openai_client is None:
            try:
                from openai import OpenAI
                if not self.api_key:
                    raise ValueError("OPENAI_API_KEY must be set")
                self._openai_client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("Please install openai: pip install openai")
        return self._openai_client

    def _load_index(self):
        """Load FAISS index and chunks from storage."""
        if self._index is not None:
            return

        try:
            import faiss
        except ImportError:
            raise ImportError("Please install faiss: pip install faiss-cpu")

        # Check if index exists
        index_path = os.path.join(self.storage_path, "faiss.index")
        chunks_path = os.path.join(self.storage_path, "chunks.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Chunks file not found at {chunks_path}")

        # Load FAISS index
        self._index = faiss.read_index(index_path)

        # Load chunks
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self._chunks = json.load(f)

        logger.info(f"Loaded FAISS index with {self._index.ntotal} vectors from {self.storage_path}")

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query using OpenAI API."""
        client = self._get_openai_client()
        response = client.embeddings.create(
            input=[query],
            model=self.EMBEDDING_MODEL
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def _search_index(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search the FAISS index."""
        try:
            import faiss
        except ImportError:
            raise ImportError("Please install faiss: pip install faiss-cpu")

        # Ensure query is 2D and normalized
        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)

        # Search
        distances, indices = self._index.search(query, min(top_k, self._index.ntotal))

        results = []
        for rank, (score, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0:  # FAISS returns -1 for not found
                continue

            chunk = self._chunks[idx]
            results.append({
                'text': chunk.get('text', ''),
                'metadata': chunk.get('metadata', {}),
                'score': float(score),
                'rank': rank + 1
            })

        return results

    def execute(
        self,
        query: str,
        expected_query: str = None,
        top_k: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
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

            # Use custom search function if provided
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

            # Load index if not loaded
            try:
                self._load_index()
            except FileNotFoundError as e:
                logger.error(f"Index not found: {e}")
                return {"status": "error", "error": str(e)}

            # Embed query
            query_embedding = self._embed_query(search_query)

            # Search
            results = self._search_index(query_embedding, top_k)

            # Format results as readable text
            formatted_results = self._format_results(results)

            return {
                "status": "success",
                "results": formatted_results,
                "raw_results": results,
                "query": search_query,
                "top_k": top_k,
                "num_results": len(results)
            }

        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    def _format_results(self, results: List[Dict]) -> str:
        """Format search results as readable text."""
        if not results:
            return "No results found."

        formatted_parts = []
        for r in results:
            metadata = r.get('metadata', {})
            category = metadata.get('category', 'Unknown')
            file_name = metadata.get('file_name', 'Unknown')
            score = r.get('score', 0)
            text = r.get('text', '')

            # Truncate text if too long
            max_text_len = 500
            if len(text) > max_text_len:
                text = text[:max_text_len] + "..."

            formatted_parts.append(
                f"[Rank {r['rank']}] (Score: {score:.4f})\n"
                f"Category: {category} | File: {file_name}\n"
                f"Content: {text}\n"
            )

        return "\n---\n".join(formatted_parts)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Direct search method returning raw results.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of result dictionaries
        """
        result = self.execute(query=query, top_k=top_k)
        if result["status"] == "success":
            return result.get("raw_results", [])
        else:
            raise RuntimeError(result.get("error", "Search failed"))


class FAISSVectorSearchTool(VectorSearchTool):
    """
    Alias for VectorSearchTool for backward compatibility.
    Uses FAISS + OpenAI Embeddings for vector search.
    """
    pass
