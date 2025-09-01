"""
Tool registry system for FDABench.

This module provides a flexible tool registration and management system
that allows agents to dynamically discover and use available tools.
"""

from typing import Dict, List, Any, Optional, Callable, Union
import logging

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for managing tools that can be used by agents.
    
    This registry allows for dynamic tool registration, discovery, and execution.
    Tools can be functions, classes, or any callable objects.
    
    Example:
        registry = ToolRegistry()
        
        # Register a simple function
        @registry.register("my_tool")
        def my_function(input_data):
            return f"Processed: {input_data}"
        
        # Register a tool class
        registry.register("sql_tool", SQLGenerationTool())
        
        # Use tools
        result = registry.execute_tool("my_tool", "test data")
    """
    
    def __init__(self):
        self._tools: Dict[str, Any] = {}
        self._tool_metadata: Dict[str, Dict[str, Any]] = {}
        
    def register(self, name: str, tool: Union[Callable, Any], **metadata) -> None:
        """
        Register a tool with the registry.
        
        Args:
            name: Tool name for registration
            tool: Tool implementation (function, class instance, etc.)
            **metadata: Additional metadata about the tool
        """
        self._tools[name] = tool
        self._tool_metadata[name] = {
            "name": name,
            "type": type(tool).__name__,
            "description": metadata.get("description", ""),
            "category": metadata.get("category", "general"),
            **metadata
        }
        logger.info(f"Registered tool: {name}")
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a tool from the registry.
        
        Args:
            name: Tool name to unregister
            
        Returns:
            True if tool was found and removed, False otherwise
        """
        if name in self._tools:
            del self._tools[name]
            del self._tool_metadata[name]
            logger.info(f"Unregistered tool: {name}")
            return True
        return False
    
    def get_tool(self, name: str) -> Optional[Any]:
        """
        Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool implementation or None if not found
        """
        return self._tools.get(name)
    
    def get_tool_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a tool.
        
        Args:
            name: Tool name
            
        Returns:
            Tool metadata or None if not found
        """
        return self._tool_metadata.get(name)
    
    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """
        List all registered tool names.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of tool names
        """
        if category:
            return [
                name for name, metadata in self._tool_metadata.items()
                if metadata.get("category") == category
            ]
        return list(self._tools.keys())
    
    def get_tools_by_category(self) -> Dict[str, List[str]]:
        """
        Get tools organized by category.
        
        Returns:
            Dictionary mapping categories to tool names
        """
        categories = {}
        for name, metadata in self._tool_metadata.items():
            category = metadata.get("category", "general")
            if category not in categories:
                categories[category] = []
            categories[category].append(name)
        return categories
    
    def execute_tool(self, name: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool by name.
        
        Args:
            name: Tool name
            *args: Positional arguments for the tool
            **kwargs: Keyword arguments for the tool
            
        Returns:
            Dictionary with execution results
        """
        tool = self.get_tool(name)
        if not tool:
            return {
                "status": "error",
                "error": f"Tool not found: {name}"
            }
        
        try:
            # Handle different tool types
            if hasattr(tool, 'execute'):
                # Tool class with execute method
                result = tool.execute(*args, **kwargs)
            elif callable(tool):
                # Simple callable
                result = tool(*args, **kwargs)
            else:
                return {
                    "status": "error",
                    "error": f"Tool {name} is not callable"
                }
            
            # Ensure result is in standard format
            if isinstance(result, dict) and "status" in result:
                return result
            else:
                return {
                    "status": "success",
                    "results": result
                }
                
        except Exception as e:
            logger.error(f"Error executing tool {name}: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def validate_tool(self, name: str) -> Dict[str, Any]:
        """
        Validate that a tool is properly configured.
        
        Args:
            name: Tool name
            
        Returns:
            Validation results
        """
        tool = self.get_tool(name)
        if not tool:
            return {
                "valid": False,
                "error": f"Tool not found: {name}"
            }
        
        try:
            # Check if tool is callable or has execute method
            is_callable = callable(tool) or hasattr(tool, 'execute')
            
            # Check if tool has metadata
            metadata = self.get_tool_metadata(name)
            has_metadata = metadata is not None
            
            return {
                "valid": is_callable,
                "callable": is_callable,
                "has_metadata": has_metadata,
                "metadata": metadata
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }


# Global registry instance
_global_registry = ToolRegistry()


def register_tool(name: str, tool: Union[Callable, Any] = None, **metadata):
    """
    Register a tool with the global registry.
    
    Can be used as a decorator or function call.
    
    Args:
        name: Tool name
        tool: Tool implementation (optional if used as decorator)
        **metadata: Tool metadata
        
    Example:
        # As decorator
        @register_tool("my_tool", description="My custom tool")
        def my_function(data):
            return f"Processed: {data}"
        
        # As function call
        register_tool("sql_tool", SQLTool(), category="database")
    """
    if tool is None:
        # Used as decorator
        def decorator(func):
            _global_registry.register(name, func, **metadata)
            return func
        return decorator
    else:
        # Used as function call
        _global_registry.register(name, tool, **metadata)


def get_global_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    return _global_registry


def list_registered_tools(category: Optional[str] = None) -> List[str]:
    """List all tools in the global registry."""
    return _global_registry.list_tools(category)


def execute_registered_tool(name: str, *args, **kwargs) -> Dict[str, Any]:
    """Execute a tool from the global registry."""
    return _global_registry.execute_tool(name, *args, **kwargs)