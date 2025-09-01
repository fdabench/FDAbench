"""
Context management tools for FDABench Package.

These tools provide context and history management capabilities.
"""

import time
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class ContextHistoryTool:
    """Tool for managing context history"""
    
    def __init__(self, max_history_length: int = 100):
        self.max_history_length = max_history_length
        self.history = []
    
    def execute(self, action: str, data: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Manage context history records.
        
        Args:
            action: Operation type ("add", "get", "clear", "search")
            data: Operation related data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with status and results
        """
        try:
            if action == "add":
                return self._add_context(data or {})
            elif action == "get":
                return self._get_context(data or {})
            elif action == "clear":
                return self._clear_context()
            elif action == "search":
                return self._search_context(data or {})
            else:
                return {"status": "error", "error": f"Invalid action: {action}"}
                
        except Exception as e:
            logger.error(f"Context history operation failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def _add_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add new context entry"""
        if "content" not in data:
            return {"status": "error", "error": "Missing content in data"}
        
        context_entry = {
            "content": data["content"],
            "metadata": data.get("metadata", {}),
            "timestamp": time.time(),
            "id": len(self.history)
        }
        
        self.history.append(context_entry)
        
        # Maintain history size limit
        if len(self.history) > self.max_history_length:
            self.history.pop(0)
        
        return {
            "status": "success",
            "results": {
                "message": "Context added successfully",
                "current_length": len(self.history),
                "entry_id": context_entry["id"]
            }
        }
    
    def _get_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get context entries"""
        limit = data.get("limit", len(self.history))
        offset = data.get("offset", 0)
        
        # Get specified range
        history_slice = self.history[offset:offset + limit]
        
        return {
            "status": "success",
            "results": {
                "history": history_slice,
                "total": len(self.history),
                "returned": len(history_slice),
                "offset": offset,
                "limit": limit
            }
        }
    
    def _clear_context(self) -> Dict[str, Any]:
        """Clear all context history"""
        cleared_count = len(self.history)
        self.history = []
        
        return {
            "status": "success",
            "results": {
                "message": "Context history cleared",
                "cleared_entries": cleared_count
            }
        }
    
    def _search_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Search context history"""
        if "query" not in data:
            return {"status": "error", "error": "Missing query in data"}
        
        query = data["query"].lower()
        limit = data.get("limit", 10)
        
        # Search through history
        matches = []
        for entry in reversed(self.history):
            if query in entry["content"].lower():
                matches.append(entry)
                if len(matches) >= limit:
                    break
        
        return {
            "status": "success",
            "results": {
                "matches": matches,
                "total_matches": len(matches),
                "search_query": query,
                "limit": limit
            }
        }