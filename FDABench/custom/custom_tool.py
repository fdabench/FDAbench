"""
Custom tool interfaces for extending DB Agent Bench with new capabilities.

This module provides base classes for implementing custom tools that can be
registered and used by any agent in the framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import json


class CustomToolBase(ABC):
    """
    Abstract base class for custom tools.
    
    Inherit from this class to implement your own tools that can be registered
    with any agent in the framework.
    
    Example:
        class MyAPITool(CustomToolBase):
            def execute(self, input_data: Any) -> Dict[str, Any]:
                # Call your API
                response = my_api_call(input_data)
                return {"status": "success", "results": response}
                
            def get_tool_info(self) -> Dict[str, Any]:
                return {
                    "name": "my_api_tool",
                    "description": "Calls my custom API",
                    "input_type": "string",
                    "output_type": "dict"
                }
    """
    
    def __init__(self, name: str, description: str = "", **kwargs):
        """
        Initialize custom tool.
        
        Args:
            name: Tool name for registration
            description: Tool description
            **kwargs: Additional tool configuration
        """
        self.name = name
        self.description = description
        self.config = kwargs
        
    @abstractmethod
    def execute(self, input_data: Any) -> Dict[str, Any]:
        """
        Execute the tool with given input data.
        
        Args:
            input_data: Input data for the tool (can be string, dict, etc.)
            
        Returns:
            Dictionary with "status" and "results" or "error" keys
        """
        pass
    
    @abstractmethod
    def get_tool_info(self) -> Dict[str, Any]:
        """
        Get information about this tool.
        
        Returns:
            Dictionary with tool metadata (name, description, input/output types, etc.)
        """
        pass
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data for this tool.
        
        Override this method to implement custom input validation.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        return True


class CustomTool(CustomToolBase):
    """
    A concrete implementation of CustomToolBase with configurable behavior.
    
    This provides a ready-to-use custom tool that can be easily configured
    for specific use cases without requiring inheritance.
    
    Example:
        # Simple function tool
        def my_function(input_data):
            return f"Processed: {input_data}"
            
        tool = CustomTool(
            name="my_tool",
            description="My custom processing tool",
            execute_func=my_function
        )
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        execute_func: Optional[callable] = None,
        **kwargs
    ):
        """
        Initialize configurable custom tool.
        
        Args:
            name: Tool name
            description: Tool description
            execute_func: Custom function to execute
            **kwargs: Additional configuration
        """
        super().__init__(name, description, **kwargs)
        self.execute_func = execute_func
        
    def execute(self, input_data: Any) -> Dict[str, Any]:
        """
        Execute the tool based on configuration.
        """
        try:
            if not self.validate_input(input_data):
                return {"status": "error", "error": "Invalid input data"}
            
            if self.execute_func:
                result = self.execute_func(input_data)
                return {"status": "success", "results": result}
            else:
                return {
                    "status": "success",
                    "results": f"Default processing of: {input_data}"
                }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get information about this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "has_custom_function": self.execute_func is not None,
            "config": self.config
        }