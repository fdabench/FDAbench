"""Unified tool executor with provenance tracking and frozen result building."""

import os
import re
import logging
import requests
from typing import Dict, Any, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# Import FAISS
try:
    from FDABench.utils.vector_index_builder import FAISSVectorIndex, OpenAIEmbedder
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available")

# Import tree models
from PUDDING.models.tree_models import ToolAction

# Default paths
DEFAULT_FAISS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "storage_faiss")


class WebSearchToolWithURLs:
    """Web search using Perplexity Sonar via OpenRouter with URL extraction."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')

    def search(self, query: str) -> Dict[str, Any]:
        if not self.api_key:
            return {"content": "Web search unavailable: Missing API key", "urls": [], "citations": []}

        try:
            prompt = f"""Search for information about: {query}

Please provide:
1. A comprehensive summary of the findings
2. Key statistics and data points
3. Recent trends and developments
4. Expert opinions and analysis

Format your response with clear sections and cite your sources."""

            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "perplexity/sonar",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 4000
                },
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                content = ""
                urls = []
                citations = []

                if 'choices' in data and data['choices']:
                    content = data['choices'][0]['message']['content'].strip()

                if 'citations' in data:
                    citations = data['citations']
                    urls = [c.get('url', c) if isinstance(c, dict) else c for c in citations]

                url_pattern = r'https?://[^\s\]\)"\']+'
                found_urls = re.findall(url_pattern, content)
                if found_urls:
                    urls.extend(found_urls)

                urls = list(dict.fromkeys(urls))

                return {
                    "content": content,
                    "urls": urls[:10],
                    "citations": citations
                }
            else:
                logger.warning(f"Web search API error: {response.status_code}")
                return {"content": f"API error: {response.status_code}", "urls": [], "citations": []}

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return {"content": f"Search error: {str(e)}", "urls": [], "citations": []}


class ToolExecutor:
    """Unified tool executor for tree-structured exploration."""

    def __init__(self, vector_index_path: str = None, api_key: str = None,
                 sql_dir: str = None, gold_result_dir: str = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.vector_index_path = vector_index_path or DEFAULT_FAISS_PATH
        self.sql_dir = sql_dir or ""
        self.gold_result_dir = gold_result_dir or ""
        self._web_tool = WebSearchToolWithURLs(self.api_key)
        self._vector_index = None
        self._embedder = None
        self._file_search_tool = None
        self._cache: Dict[str, ToolAction] = {}  # dedup cache: (tool, query) -> result

        self._load_vector_index()
        self._load_file_search_tool()

    def _load_vector_index(self):
        """Load FAISS index with 86K+ domain knowledge chunks."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available")
            return

        try:
            faiss_path = os.path.join(self.vector_index_path, "faiss.index")
            if os.path.exists(faiss_path):
                self._vector_index = FAISSVectorIndex.load(self.vector_index_path)
                self._embedder = OpenAIEmbedder(api_key=self.openai_key)
                logger.info(f"Loaded FAISS index with {self._vector_index.index.ntotal} vectors")
            else:
                logger.warning(f"FAISS index not found at {faiss_path}")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")

    def _load_file_search_tool(self):
        """Load the FileSystemTool for file_search."""
        try:
            from PUDDING.tools.external_tools import FileSystemTool
            file_system_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "input_data", "file_system"
            )
            self._file_search_tool = FileSystemTool(file_system_path)
            logger.info(f"Loaded FileSystemTool at {file_system_path}")
        except Exception as e:
            logger.warning(f"FileSystemTool not available: {e}")

    def execute(self, action: ToolAction) -> ToolAction:
        """Execute a tool action, filling output and provenance. Returns the same action mutated."""
        cache_key = f"{action.tool_name}::{action.input_params.get('query', '')}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            action.output = cached.output
            action.provenance = dict(cached.provenance)
            return action

        if action.tool_name == "web_search":
            self._execute_web_search(action)
        elif action.tool_name == "vector_search":
            self._execute_vector_search(action)
        elif action.tool_name == "file_search":
            self._execute_file_search(action)
        elif action.tool_name == "db_explore":
            self._execute_db_explore(action)
        else:
            action.output = f"Unknown tool: {action.tool_name}"
            action.provenance = {"error": True}

        self._cache[cache_key] = action
        return action

    def _execute_web_search(self, action: ToolAction):
        """Execute web search and populate output + provenance."""
        query = action.input_params.get("query", "")
        try:
            result = self._web_tool.search(query)
            action.output = result.get("content", "")
            action.provenance = {
                "urls": result.get("urls", []),
                "citations": result.get("citations", []),
                "success": bool(action.output and len(action.output) > 50),
            }
            logger.info(f"Web search: {len(result.get('urls', []))} URLs found")
        except Exception as e:
            action.output = f"Web search error: {str(e)}"
            action.provenance = {"error": str(e), "success": False}

    def _execute_vector_search(self, action: ToolAction, top_k: int = 5):
        """Execute vector search using FAISS index."""
        query = action.input_params.get("query", "")
        try:
            if not self._vector_index or not self._embedder:
                action.output = "Vector index not available"
                action.provenance = {"success": False}
                return

            query_emb = self._embedder.embed_query(query)
            results = self._vector_index.search(query_emb, top_k=top_k)

            chunks = []
            parts = []
            for r in results:
                chunk = r['chunk']
                chunks.append({
                    "chunk_id": chunk.get('id', ''),
                    "score": round(r['score'], 4),
                    "category": chunk.get('metadata', {}).get('category', ''),
                    "file_name": chunk.get('metadata', {}).get('file_name', ''),
                    "text_preview": chunk.get('text', '')[:300],
                })
                parts.append(f"[{chunk.get('metadata', {}).get('category', '')}] {chunk.get('text', '')[:400]}")

            action.output = "\n\n".join(parts) if parts else "No results found"
            action.provenance = {
                "chunks": chunks,
                "success": bool(chunks),
            }
            logger.info(f"Vector search: {len(chunks)} chunks found")
        except Exception as e:
            action.output = f"Vector search error: {str(e)}"
            action.provenance = {"error": str(e), "success": False}

    def _execute_file_search(self, action: ToolAction):
        """Execute file system search using FileSystemTool."""
        query = action.input_params.get("query", "")
        try:
            if not self._file_search_tool:
                action.output = "File search not available: FileSystemTool not loaded"
                action.provenance = {"success": False}
                return

            result = self._file_search_tool.search(query)
            if result and len(result) > 20 and "No relevant" not in result and "Error" not in result:
                action.output = result[:3000]
                action.provenance = {"success": True, "source": "file_system"}
                logger.info(f"File search: {len(result)} chars retrieved")
            else:
                action.output = result or "No relevant file system information found"
                action.provenance = {"success": False, "source": "file_system"}
        except Exception as e:
            action.output = f"File search error: {str(e)}"
            action.provenance = {"error": str(e), "success": False}

    def _execute_db_explore(self, action: ToolAction):
        """Explore database schema/data by reading SQL and CSV files for an instance."""
        query = action.input_params.get("query", "")
        instance_id = action.input_params.get("instance_id", "")
        try:
            parts = []

            # Try to load SQL file for schema hints
            if self.sql_dir and instance_id:
                sql_path = os.path.join(self.sql_dir, f"{instance_id}.sql")
                if os.path.exists(sql_path):
                    with open(sql_path, 'r') as f:
                        sql_content = f.read().strip()
                    parts.append(f"SQL Query:\n{sql_content[:1000]}")

            # Try to load CSV result for sample data
            if self.gold_result_dir and instance_id:
                csv_path = os.path.join(self.gold_result_dir, f"{instance_id}.csv")
                if os.path.exists(csv_path):
                    with open(csv_path, 'r') as f:
                        csv_content = f.read().strip()
                    # Show header + first rows for schema exploration
                    lines = csv_content.split('\n')
                    preview = '\n'.join(lines[:20])
                    parts.append(f"Data Preview ({len(lines)} rows):\n{preview}")

            if parts:
                action.output = "\n\n".join(parts)
                action.provenance = {"success": True, "source": "db_explore"}
                logger.info(f"DB explore: {len(action.output)} chars")
            else:
                action.output = f"No database files found for exploration (query: {query})"
                action.provenance = {"success": False, "source": "db_explore"}
        except Exception as e:
            action.output = f"DB explore error: {str(e)}"
            action.provenance = {"error": str(e), "success": False}

    def get_base_context(
        self,
        instance_id: str,
        gold_result_dir: str,
        sql_dir: str,
    ) -> Tuple[str, str]:
        """Load base context (sql_result, sql_statement) from files.

        Returns:
            (sql_result, sql_statement) tuple
        """
        sql_result = ""
        sql_statement = ""

        # Load SQL result from CSV
        csv_path = os.path.join(gold_result_dir, f"{instance_id}.csv")
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r') as f:
                    sql_result = f.read().strip()
            except Exception as e:
                logger.warning(f"Failed to load SQL result from {csv_path}: {e}")

        # Load SQL statement
        sql_path = os.path.join(sql_dir, f"{instance_id}.sql")
        if os.path.exists(sql_path):
            try:
                with open(sql_path, 'r') as f:
                    sql_statement = f.read().strip()
            except Exception as e:
                logger.warning(f"Failed to load SQL statement from {sql_path}: {e}")

        return sql_result, sql_statement

    @staticmethod
    def build_frozen_web_search(actions: List[ToolAction]) -> Dict:
        """Build frozen_web_search from accumulated web search ToolActions."""
        searches = []
        step_counter = 1
        for action in actions:
            if action.tool_name == "web_search":
                prov = action.provenance or {}
                searches.append({
                    "step": step_counter,
                    "query": action.input_params.get("query", ""),
                    "rationale": action.rationale,
                    "urls": prov.get("urls", []),
                    "citations": prov.get("citations", []),
                    "context_summary": action.output[:1500] if prov.get("success") else "",
                })
                step_counter += 1
        return {"searches": searches}

    @staticmethod
    def build_frozen_vector_search(actions: List[ToolAction]) -> Dict:
        """Build frozen_vector_search from accumulated vector search ToolActions."""
        searches = []
        step_counter = 1
        for action in actions:
            if action.tool_name == "vector_search":
                prov = action.provenance or {}
                searches.append({
                    "step": step_counter,
                    "query": action.input_params.get("query", ""),
                    "rationale": action.rationale,
                    "results": prov.get("chunks", []),
                    "context_summary": action.output[:800] if prov.get("success") else "",
                })
                step_counter += 1
        return {"searches": searches}
