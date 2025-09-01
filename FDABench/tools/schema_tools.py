"""
Schema-related tools for FDABench Package.

These tools provide database schema inspection and analysis capabilities.
They integrate with the DatabaseConnectionManager for flexible database access.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class SchemaInfoTool:
    """Tool for getting database schema information"""
    
    def __init__(self, db_manager=None):
        """
        Initialize the schema info tool
        
        Args:
            db_manager: DatabaseConnectionManager instance for database operations
        """
        self.db_manager = db_manager
        
        # If no database manager provided, try to create one
        if self.db_manager is None:
            try:
                from ..utils.database_connection_manager import DatabaseConnectionManager
                self.db_manager = DatabaseConnectionManager()
                logger.info("Created default DatabaseConnectionManager")
            except ImportError as e:
                logger.warning(f"Could not import DatabaseConnectionManager: {e}")
                self.db_manager = None
    
    def execute(self, database_name: str = None, database_type: str = None,
                instance_id: str = None, **kwargs) -> Dict[str, Any]:
        """
        Get database schema information.
        """
        try:
            if not database_name:
                return {"status": "error", "error": "No database name provided"}
            if not instance_id:
                return {
                    "status": "error",
                    "error": "No instance_id provided for database connection. Please specify a valid instance_id."
                }
            if not self.db_manager or not hasattr(self.db_manager, 'get_database_config'):
                return {
                    "status": "error",
                    "error": "No valid database manager with get_database_config available for schema retrieval."
                }
            config = self.db_manager.get_database_config(instance_id, database_name, database_type)
            if not config.connection_params:
                return {
                    "status": "error",
                    "error": "DatabaseConfig.connection_params is None. Please check database configuration."
                }
            schema_result = self.db_manager.get_schema_info(config)
            if not isinstance(schema_result, dict):
                return {
                    "status": "error",
                    "error": "Database manager get_schema_info returned None or invalid type"
                }
            if "error" not in schema_result:
                return {
                    "status": "success",
                    "results": {
                        **schema_result,
                        "database_name": database_name,
                        "database_type": database_type,
                        "instance_id": instance_id,
                        "source": "database_manager"
                    }
                }
            else:
                return {
                    "status": "error",
                    "error": f"Database schema retrieval failed: {schema_result['error']}"
                }
        except Exception as e:
            logger.error(f"Database manager schema retrieval failed: {e}")
            return {
                "status": "error",
                "error": f"Database connection error: {str(e)}"
            }


class SchemaInspectionTool:
    """Tool for inspecting database schemas"""
    
    def __init__(self, db_manager=None):
        """
        Initialize the schema inspection tool
        
        Args:
            db_manager: DatabaseConnectionManager instance for database operations
        """
        self.db_manager = db_manager
    
    def execute(self, database_name: str, database_type: str = None,
                instance_id: str = None, **kwargs) -> Dict[str, Any]:
        """
        Get database schema information.
        
        Args:
            database_name: Name of the database
            database_type: Type of database (bird, spider2-lite, spider1, dabstep, etc.)
            instance_id: Instance identifier for database connection
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with status and schema results
        """
        try:
            if not database_name:
                return {"status": "error", "error": "No database name provided"}
            
            # Use SchemaInfoTool for consistency
            schema_tool = SchemaInfoTool(self.db_manager)
            return schema_tool.execute(
                database_name=database_name,
                database_type=database_type,
                instance_id=instance_id,
                **kwargs
            )
            
        except Exception as e:
            logger.error(f"Schema inspection failed: {str(e)}")
            return {"status": "error", "error": str(e)}


class SchemaAnalysisTool:
    """Tool for analyzing database schemas and providing insights"""
    
    def __init__(self, db_manager=None):
        """
        Initialize the schema analysis tool
        
        Args:
            db_manager: DatabaseConnectionManager instance for database operations
        """
        self.db_manager = db_manager
    
    def execute(self, database_name: str, database_type: str = None,
                instance_id: str = None, analysis_type: str = "basic", **kwargs) -> Dict[str, Any]:
        """
        Analyze database schema and provide insights.
        
        Args:
            database_name: Name of the database
            database_type: Type of database (bird, spider2-lite, spider1, dabstep, etc.)
            instance_id: Instance identifier for database connection
            analysis_type: Type of analysis ("basic", "relationships", "complexity")
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with status and analysis results
        """
        try:
            # First get the schema
            schema_tool = SchemaInfoTool(self.db_manager)
            schema_result = schema_tool.execute(
                database_name=database_name,
                database_type=database_type,
                instance_id=instance_id,
                **kwargs
            )
            
            if schema_result["status"] != "success":
                return schema_result
            
            schema_data = schema_result["results"]
            tables = schema_data.get("tables", {})
            
            # Perform analysis based on type
            analysis = {
                "database_name": database_name,
                "database_type": database_type,
                "analysis_type": analysis_type,
                "table_count": len(tables),
                "tables": {}
            }
            
            for table_name, table_info in tables.items():
                columns = table_info.get("columns", {})
                foreign_keys = table_info.get("foreign_keys", [])
                
                table_analysis = {
                    "column_count": len(columns),
                    "column_types": {},
                    "has_primary_key": False,
                    "foreign_key_count": len(foreign_keys),
                    "foreign_keys": foreign_keys
                }
                
                # Analyze column types
                for col_name, col_type in columns.items():
                    if isinstance(col_type, dict):
                        col_type_name = col_type.get("type", "UNKNOWN")
                        if col_type.get("primary_key"):
                            table_analysis["has_primary_key"] = True
                    else:
                        col_type_name = str(col_type)
                    
                    if col_type_name not in table_analysis["column_types"]:
                        table_analysis["column_types"][col_type_name] = 0
                    table_analysis["column_types"][col_type_name] += 1
                
                analysis["tables"][table_name] = table_analysis
            
            # Add summary statistics
            if analysis_type == "basic":
                total_columns = sum(t["column_count"] for t in analysis["tables"].values())
                total_foreign_keys = sum(t["foreign_key_count"] for t in analysis["tables"].values())
                tables_with_pk = sum(1 for t in analysis["tables"].values() if t["has_primary_key"])
                
                analysis["summary"] = {
                    "total_columns": total_columns,
                    "total_foreign_keys": total_foreign_keys,
                    "tables_with_primary_key": tables_with_pk,
                    "average_columns_per_table": round(total_columns / len(tables), 2) if tables else 0
                }
            
            return {
                "status": "success",
                "results": analysis
            }
            
        except Exception as e:
            logger.error(f"Schema analysis failed: {str(e)}")
            return {"status": "error", "error": str(e)}