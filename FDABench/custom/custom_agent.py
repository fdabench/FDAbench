"""
Custom agent interfaces for extending DB Agent Bench with new agent patterns.

This module provides base classes for implementing custom agent behaviors
while maintaining compatibility with the core framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from ..core.base_agent import BaseAgent


class CustomAgentBase(BaseAgent):
    """
    Abstract base class for custom agent patterns.
    
    Inherit from this class to implement your own agent design patterns
    while leveraging the core infrastructure (token tracking, tool registry, etc.).
    
    Example:
        class MyCustomAgent(CustomAgentBase):
            def __init__(self, model="moonshotai/kimi-k2", 
                        api_base="https://openrouter.ai/api/v1", api_key="your-key"):
                super().__init__(model=model, api_base=api_base, api_key=api_key)
                
            def process_query_logic(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
                # Implement your custom logic here
                return self._execute_custom_workflow(query_data)
    """
    
    def __init__(self, 
                 model: str = "moonshotai/kimi-k2",
                 api_base: str = "https://openrouter.ai/api/v1",
                 api_key: Optional[str] = None,
                 **kwargs):
        """
        Initialize custom agent with base functionality and API configuration.
        
        Args:
            model: LLM model to use (e.g., "moonshotai/kimi-k2", "gpt-4", etc.)
            api_base: API base URL (default: OpenRouter format)
            api_key: API key for the service
            **kwargs: Additional configuration parameters
        """
        super().__init__(model)
        self.api_base = api_base
        self.api_key = api_key
        self.custom_config = kwargs
        
        # Set up API client configuration
        self._setup_api_client()
        
    def _setup_api_client(self):
        """
        Set up API client based on configuration.
        Supports OpenRouter, OpenAI, and other OpenAI-compatible APIs.
        """
        import os
        from openai import OpenAI
        
        # Use provided API key or fallback to environment variables
        if not self.api_key:
            if "openrouter.ai" in self.api_base:
                self.api_key = os.environ.get("OPENROUTER_API_KEY")
            elif "api.openai.com" in self.api_base:
                self.api_key = os.environ.get("OPENAI_API_KEY")
            else:
                self.api_key = os.environ.get("CUSTOM_API_KEY")
        
        # Initialize OpenAI-compatible client
        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key
        )
        
    def call_llm(self, messages, model=None, category="general"):
        """
        Override LLM calling to use configured API client.
        """
        model = model or self.model
        
        try:
            # Estimate input tokens
            input_text = " ".join([msg.get("content", "") for msg in messages])
            input_tokens = self.estimate_tokens(input_text)
            
            # Make API call
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=20
            )
            
            response = completion.choices[0].message.content
            
            # Track token usage
            output_tokens = self.estimate_tokens(response)
            self.token_tracker.track_call(category, input_tokens, output_tokens, model)
            
            return response
            
        except Exception as e:
            # Fallback to parent implementation
            return super().call_llm(messages, model, category)
        
    @abstractmethod
    def process_query_logic(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement your custom query processing logic.
        
        This is the main method you need to implement for your custom agent pattern.
        
        Args:
            query_data: Query data dictionary containing instance_id, db, query, etc.
            
        Returns:
            Dictionary with processing results, metrics, and any custom fields
        """
        pass
    
    def process_query_from_json(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for processing queries. Calls your custom logic.
        """
        try:
            return self.process_query_logic(query_data)
        except Exception as e:
            return {
                "instance_id": query_data.get("instance_id", "unknown"),
                "error": str(e),
                "agent_type": "custom",
                "model": self.model,
                "api_base": self.api_base
            }
    
    def get_custom_tools(self) -> List[str]:
        """
        Override this method to specify which tools your custom agent supports.
        
        Returns:
            List of tool names that this agent can use
        """
        return []
    
    def validate_custom_config(self) -> bool:
        """
        Override this method to validate your custom configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        return True


class CustomAgent(CustomAgentBase):
    """
    A concrete implementation of CustomAgentBase with common patterns.
    
    This provides a ready-to-use custom agent that can be easily configured
    for specific use cases without requiring inheritance.
    
    Example:
        # Using OpenRouter (default)
        agent = CustomAgent(
            model="moonshotai/kimi-k2",
            api_key="your-openrouter-key",
            workflow_steps=["step1", "step2", "step3"]
        )
        
        # Using OpenAI
        agent = CustomAgent(
            model="gpt-4",
            api_base="https://api.openai.com/v1",
            api_key="your-openai-key"
        )
        
        # Using custom API endpoint
        agent = CustomAgent(
            model="custom-model",
            api_base="https://your-api.com/v1",
            api_key="your-custom-key"
        )
    """
    
    def __init__(self, 
                 model: str = "moonshotai/kimi-k2",
                 api_base: str = "https://openrouter.ai/api/v1",
                 api_key: Optional[str] = None,
                 workflow_steps: Optional[List[str]] = None,
                 custom_tools: Optional[List[str]] = None,
                 **kwargs):
        """
        Initialize configurable custom agent.
        
        Args:
            model: LLM model to use
            api_base: API base URL (supports OpenRouter, OpenAI, custom endpoints)
            api_key: API key for the service
            workflow_steps: List of workflow step names to execute
            custom_tools: List of custom tool names to register
            **kwargs: Additional configuration
        """
        super().__init__(model=model, api_base=api_base, api_key=api_key, **kwargs)
        self.workflow_steps = workflow_steps or ["default_step"]
        self.custom_tools = custom_tools or []
        
    def process_query_logic(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement a configurable workflow based on workflow_steps.
        """
        import time
        start_time = time.time()
        
        results = {}
        step_results = {}
        
        # Execute each workflow step
        for i, step in enumerate(self.workflow_steps):
            try:
                step_result = self._execute_workflow_step(step, query_data, results)
                step_results[f"step_{i+1}_{step}"] = step_result
                
                if step_result.get("status") == "success":
                    results[step] = step_result.get("results")
                    
            except Exception as e:
                step_results[f"step_{i+1}_{step}"] = {
                    "status": "error", 
                    "error": str(e)
                }
        
        # Generate final result
        end_time = time.time()
        token_summary = self.token_tracker.get_token_summary()
        
        return {
            "instance_id": query_data.get("instance_id", "unknown"),
            "agent_type": "custom",
            "model": self.model,
            "api_base": self.api_base,
            "workflow_steps": self.workflow_steps,
            "step_results": step_results,
            "final_results": results,
            "metrics": {
                "latency_seconds": round(end_time - start_time, 2),
                "token_usage": token_summary,
                "completed_steps": len([r for r in step_results.values() if r.get("status") == "success"]),
                "total_steps": len(self.workflow_steps)
            },
            "processing_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        }
    
    def _execute_workflow_step(
        self, 
        step_name: str, 
        query_data: Dict[str, Any], 
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single workflow step. Override this for custom step logic.
        
        Args:
            step_name: Name of the step to execute
            query_data: Original query data
            previous_results: Results from previous steps
            
        Returns:
            Step execution result
        """
        if step_name == "default_step":
            return {
                "status": "success",
                "results": f"Executed default step for query: {query_data.get('query', 'N/A')}"
            }
        
        # Try to execute as a registered tool
        if step_name in self.tools:
            tool_func = self.tools[step_name]
            try:
                result = tool_func(query_data.get("query", ""))
                return {"status": "success", "results": result}
            except Exception as e:
                return {"status": "error", "error": str(e)}
        
        return {
            "status": "error", 
            "error": f"Unknown workflow step: {step_name}"
        }
    
    def get_custom_tools(self) -> List[str]:
        """Return list of custom tools for this agent."""
        return self.custom_tools