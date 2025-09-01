"""
Optimization tools for FDABench Package.

These tools provide query optimization and performance analysis capabilities.
"""

import time
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class QueryOptimizationTool:
    """Tool for optimizing database queries"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    def execute(self, sql_query: str, database_type: str = "sqlite",
                schema_info: Dict = None, performance_metrics: Dict = None,
                **kwargs) -> Dict[str, Any]:
        """
        Optimize SQL queries for better performance.
        
        Args:
            sql_query: SQL query to optimize
            database_type: Type of database system
            schema_info: Database schema information
            performance_metrics: Current performance metrics
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with status and optimization results
        """
        try:
            if not sql_query:
                return {"status": "error", "error": "No SQL query provided"}
            
            # Mock optimization analysis - in real implementation would use LLM
            optimizations = []
            optimized_query = sql_query
            
            # Check for common optimization opportunities
            if "SELECT *" in sql_query.upper():
                optimizations.append({
                    "type": "column_selection",
                    "description": "Replace SELECT * with specific columns",
                    "impact": "Reduces data transfer and memory usage"
                })
                optimized_query = optimized_query.replace("SELECT *", "SELECT id, name, value")
            
            if "ORDER BY" in sql_query.upper() and "LIMIT" not in sql_query.upper():
                optimizations.append({
                    "type": "pagination",
                    "description": "Consider adding LIMIT clause with ORDER BY",
                    "impact": "Prevents sorting large result sets"
                })
            
            if "WHERE" not in sql_query.upper():
                optimizations.append({
                    "type": "filtering",
                    "description": "Add WHERE clause to filter results early",
                    "impact": "Reduces rows processed and improves performance"
                })
            
            # Add index suggestions based on schema
            index_suggestions = []
            if schema_info:
                for table_name, table_info in schema_info.get("tables", {}).items():
                    if table_name.lower() in sql_query.lower():
                        for column in table_info.get("columns", {}):
                            if column in sql_query and "WHERE" in sql_query.upper():
                                index_suggestions.append(f"CREATE INDEX idx_{table_name}_{column} ON {table_name}({column});")
            
            # Calculate estimated improvement
            estimated_improvement = len(optimizations) * 15  # Mock calculation
            
            return {
                "status": "success",
                "results": {
                    "original_query": sql_query,
                    "optimized_query": optimized_query,
                    "optimizations": optimizations,
                    "index_suggestions": index_suggestions,
                    "estimated_improvement_percent": min(estimated_improvement, 80),
                    "database_type": database_type,
                    "optimization_timestamp": time.time()
                }
            }
            
        except Exception as e:
            logger.error(f"Query optimization failed: {str(e)}")
            return {"status": "error", "error": str(e)}


class PerformanceTool:
    """Tool for analyzing and monitoring performance"""
    
    def __init__(self):
        self.metrics_history = []
    
    def execute(self, operation: str, execution_time: float = None,
                query: str = None, result_count: int = None,
                memory_usage: float = None, **kwargs) -> Dict[str, Any]:
        """
        Track and analyze performance metrics.
        
        Args:
            operation: Type of operation being measured
            execution_time: Time taken for execution
            query: SQL query or operation description
            result_count: Number of results returned
            memory_usage: Memory usage in MB
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with status and performance analysis
        """
        try:
            # Record performance metrics
            metrics = {
                "operation": operation,
                "execution_time": execution_time,
                "query": query,
                "result_count": result_count,
                "memory_usage": memory_usage,
                "timestamp": time.time()
            }
            
            self.metrics_history.append(metrics)
            
            # Keep only recent metrics (last 100)
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]
            
            # Analyze performance
            analysis = self._analyze_performance(metrics)
            
            return {
                "status": "success",
                "results": {
                    "current_metrics": metrics,
                    "performance_analysis": analysis,
                    "recommendations": self._get_recommendations(analysis),
                    "metrics_count": len(self.metrics_history)
                }
            }
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def _analyze_performance(self, current_metrics: Dict) -> Dict[str, Any]:
        """Analyze current performance against historical data"""
        if not self.metrics_history or len(self.metrics_history) < 2:
            return {"status": "insufficient_data"}
        
        # Calculate averages for comparison
        recent_metrics = self.metrics_history[-10:]  # Last 10 operations
        
        avg_execution_time = sum(m.get("execution_time", 0) for m in recent_metrics) / len(recent_metrics)
        avg_result_count = sum(m.get("result_count", 0) for m in recent_metrics) / len(recent_metrics)
        
        current_time = current_metrics.get("execution_time", 0)
        current_results = current_metrics.get("result_count", 0)
        
        return {
            "execution_time_vs_average": (current_time / avg_execution_time) if avg_execution_time > 0 else 1,
            "result_count_vs_average": (current_results / avg_result_count) if avg_result_count > 0 else 1,
            "performance_trend": "improving" if current_time < avg_execution_time else "degrading",
            "average_execution_time": avg_execution_time,
            "average_result_count": avg_result_count
        }
    
    def _get_recommendations(self, analysis: Dict) -> List[str]:
        """Get performance recommendations based on analysis"""
        recommendations = []
        
        if analysis.get("status") == "insufficient_data":
            return ["Collect more performance data for meaningful analysis"]
        
        exec_ratio = analysis.get("execution_time_vs_average", 1)
        if exec_ratio > 1.5:
            recommendations.append("Current operation is significantly slower than average - consider optimization")
        elif exec_ratio > 1.2:
            recommendations.append("Operation is slower than usual - monitor for patterns")
        
        result_ratio = analysis.get("result_count_vs_average", 1)
        if result_ratio > 2:
            recommendations.append("Large result set detected - consider pagination or filtering")
        
        if analysis.get("performance_trend") == "degrading":
            recommendations.append("Performance trend is degrading - review recent changes")
        
        if not recommendations:
            recommendations.append("Performance is within normal ranges")
        
        return recommendations