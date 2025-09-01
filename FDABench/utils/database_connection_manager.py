"""
Database Connection Manager for FDABench Package.

This module provides database connection management for different database types
including SQLite, BigQuery, and Snowflake.
"""

import os
import sqlite3
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Configuration for database connections"""
    database_type: str
    instance_id: str
    db_name: str
    connection_params: Dict[str, Any] = None

class DatabaseConnectionManager:
    """Manages connections to different types of databases"""
    
    def __init__(self, config_overrides: Dict[str, Any] = None):
        """
        Initialize the database connection manager
        
        Args:
            config_overrides: Optional configuration overrides for paths and credentials
        """
        # Default configuration - Update these paths to match your environment
        default_config = {
            'bird_db_path': "/path/to/your/BIRD_train/train_databases",
            'local_db_path': "/path/to/your/local/databases",
            'sample_db_path': "sample/regional_sales",  # Built-in sample database for testing
            'bigquery_credentials_path': "/path/to/your/bigquery-service-account.json",
            'snowflake_config': {
                'account': 'your-snowflake-account',
                'user': 'your-username',
                'password': 'your-password',
                'role': 'your-role',
                'database': 'your-database',
                'schema': 'your-schema',
                'warehouse': 'your-warehouse'
            }
        }
        
        # Apply overrides if provided
        if config_overrides:
            default_config.update(config_overrides)
        
        # Set configuration
        self.bird_db_path = default_config['bird_db_path']
        self.local_db_path = default_config['local_db_path']
        self.sample_db_path = default_config['sample_db_path']
        self.bigquery_credentials_path = default_config['bigquery_credentials_path']
        self.snowflake_config = default_config['snowflake_config']
        
        # Initialize clients
        self._bigquery_client = None
        
        logger.info("DatabaseConnectionManager initialized")

    def _find_dataset_by_fuzzy_match(self, dataset_name: str) -> str:
        """
        Find actual dataset name using prefix matching and dynamic discovery
        
        Args:
            dataset_name: The dataset name to search for
            
        Returns:
            The actual dataset name if found, otherwise the original name
        """
        try:
            client = self._get_bigquery_client()
            
            # First try to access the dataset directly
            try:
                dataset_ref = client.dataset(dataset_name)
                client.get_dataset(dataset_ref)
                logger.info(f"Found exact dataset: {dataset_name}")
                return dataset_name
            except Exception:
                logger.info(f"Dataset {dataset_name} not found, searching for prefix matches...")
            
            # Search for prefix matches in public datasets
            prefix_matches = self._discover_public_datasets(dataset_name)
            
            if prefix_matches:
                # Return the first matching dataset
                best_match = prefix_matches[0]
                actual_name = best_match["dataset_id"]
                logger.info(f"Found prefix match: {dataset_name} -> {actual_name}")
                return actual_name
            
            # If no prefix matches found, return original name
            logger.info(f"No prefix matches found for {dataset_name}, using original name")
            return dataset_name
            
        except Exception as e:
            logger.warning(f"Error in fuzzy matching for {dataset_name}: {e}")
            return dataset_name

    def _discover_public_datasets(self, search_pattern: str = None) -> List[Dict[str, Any]]:
        """
        Discover available public datasets that might match the search pattern
        
        Args:
            search_pattern: Optional pattern to search for in dataset names
            
        Returns:
            List of matching datasets with their information, sorted by relevance
        """
        try:
            client = self._get_bigquery_client()
            
            # List of common public datasets to check
            public_datasets = [
                ("bigquery-public-data", "world_bank_wdi"),
                ("bigquery-public-data", "google_analytics_sample"),
                ("bigquery-public-data", "census_bureau_international"),
                ("bigquery-public-data", "covid19_jhu_csse"),
                ("bigquery-public-data", "new_york_citibike"),
                ("bigquery-public-data", "samples"),
                ("bigquery-public-data", "github_repos"),
                ("bigquery-public-data", "stackoverflow"),
                ("bigquery-public-data", "baseball_games"),
                ("bigquery-public-data", "austin_bikeshare"),
                ("bigquery-public-data", "chicago_taxi_trips"),
                ("bigquery-public-data", "san_francisco_bikeshare"),
                ("bigquery-public-data", "london_bicycles"),
                ("bigquery-public-data", "nasa_asteroids"),
                ("bigquery-public-data", "ml_datasets"),
                ("bigquery-public-data", "weather_gsod"),
                ("bigquery-public-data", "reddit_comments"),
                ("bigquery-public-data", "wikipedia_pageviews"),
                ("bigquery-public-data", "patents_public_data"),
                ("bigquery-public-data", "open_images"),
                ("bigquery-public-data", "bigquery_geolocation"),
                ("bigquery-public-data", "utility_us"),
                ("bigquery-public-data", "utility_eu"),
                ("bigquery-public-data", "utility_au")
            ]
            
            matching_datasets = []
            
            for project_id, dataset_name in public_datasets:
                try:
                    # Check if dataset name matches search pattern
                    if search_pattern:
                        # Convert both to lowercase for case-insensitive matching
                        pattern_lower = search_pattern.lower()
                        dataset_lower = dataset_name.lower()
                        
                        # Check different types of matches
                        is_exact_match = pattern_lower == dataset_lower
                        is_prefix_match = dataset_lower.startswith(pattern_lower)
                        is_contains_match = pattern_lower in dataset_lower
                        is_word_match = any(word in dataset_lower for word in pattern_lower.split('_'))
                        
                        # Calculate relevance score
                        relevance_score = 0
                        if is_exact_match:
                            relevance_score = 100
                        elif is_prefix_match:
                            relevance_score = 80
                        elif is_contains_match:
                            relevance_score = 60
                        elif is_word_match:
                            relevance_score = 40
                        else:
                            continue  # Skip if no match
                        
                        # Get dataset info
                        dataset_ref = client.dataset(dataset_name, project=project_id)
                        dataset = client.get_dataset(dataset_name, project=project_id)
                        
                        # Get basic table count
                        tables = list(client.list_tables(dataset_ref))
                        
                        matching_datasets.append({
                            "project_id": project_id,
                            "dataset_id": dataset_name,
                            "description": dataset.description or "No description",
                            "table_count": len(tables),
                            "created": dataset.created.isoformat() if dataset.created else None,
                            "relevance_score": relevance_score,
                            "match_type": "exact" if is_exact_match else "prefix" if is_prefix_match else "contains" if is_contains_match else "word"
                        })
                    else:
                        # If no search pattern, return all datasets
                        dataset_ref = client.dataset(dataset_name, project=project_id)
                        dataset = client.get_dataset(dataset_name, project=project_id)
                        
                        tables = list(client.list_tables(dataset_ref))
                        
                        matching_datasets.append({
                            "project_id": project_id,
                            "dataset_id": dataset_name,
                            "description": dataset.description or "No description",
                            "table_count": len(tables),
                            "created": dataset.created.isoformat() if dataset.created else None,
                            "relevance_score": 0,
                            "match_type": "all"
                        })
                    
                except Exception as e:
                    logger.debug(f"Could not access dataset {project_id}.{dataset_name}: {e}")
                    continue
            
            # Sort by relevance score (highest first)
            matching_datasets.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return matching_datasets
            
        except Exception as e:
            logger.error(f"Error discovering public datasets: {e}")
            return []

    def discover_datasets(self, search_pattern: str = None) -> Dict[str, Any]:
        """
        Discover available datasets that match the search pattern
        
        Args:
            search_pattern: Optional pattern to search for in dataset names
            
        Returns:
            Dictionary with discovery results
        """
        try:
            # Get public datasets
            public_datasets = self._discover_public_datasets(search_pattern)
            
            # Get current project datasets
            current_project_datasets = []
            try:
                client = self._get_bigquery_client()
                datasets = list(client.list_datasets())
                for dataset in datasets:
                    if search_pattern is None or search_pattern.lower() in dataset.dataset_id.lower():
                        current_project_datasets.append({
                            "project_id": client.project,
                            "dataset_id": dataset.dataset_id,
                            "description": "Current project dataset",
                            "table_count": "Unknown",
                            "created": None
                        })
            except Exception as e:
                logger.warning(f"Could not list current project datasets: {e}")
            
            return {
                "status": "success",
                "public_datasets": public_datasets,
                "current_project_datasets": current_project_datasets,
                "search_pattern": search_pattern,
                "total_found": len(public_datasets) + len(current_project_datasets)
            }
            
        except Exception as e:
            logger.error(f"Error discovering datasets: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def get_dataset_alias_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get information about dataset discovery and matching
        
        Args:
            dataset_name: The dataset name to check
            
        Returns:
            Dictionary with discovery information
        """
        # Try to find the dataset using fuzzy matching
        actual_name = self._find_dataset_by_fuzzy_match(dataset_name)
        
        # Discover all matching datasets
        matching_datasets = self._discover_public_datasets(dataset_name)
        
        # Check if it's an exact match
        is_exact_match = actual_name == dataset_name
        
        # Check if it's a prefix match
        is_prefix_match = False
        prefix_matches = []
        for dataset in matching_datasets:
            if dataset["match_type"] in ["exact", "prefix"]:
                is_prefix_match = True
                prefix_matches.append({
                    "dataset_id": dataset["dataset_id"],
                    "project_id": dataset["project_id"],
                    "table_count": dataset["table_count"],
                    "match_type": dataset["match_type"],
                    "relevance_score": dataset["relevance_score"]
                })
        
        return {
            "original_name": dataset_name,
            "actual_name": actual_name,
            "is_exact_match": is_exact_match,
            "is_prefix_match": is_prefix_match,
            "prefix_matches": prefix_matches,
            "all_matches": matching_datasets,
            "total_matches": len(matching_datasets)
        }

    def get_database_config(self, instance_id: str, db_name: str, database_type: str) -> DatabaseConfig:
        """Get database configuration based on instance_id and database_type"""
        config = DatabaseConfig(
            database_type=database_type,
            instance_id=instance_id,
            db_name=db_name
        )
        
        if database_type.lower() == "bird":
            config.connection_params = {
                "db_path": os.path.join(self.bird_db_path, db_name, f"{db_name}.sqlite")
            }
        elif database_type.lower() == "spider2-lite":
            if instance_id.startswith("bq"):
                config.connection_params = {
                    "connection_type": "bigquery",
                    "credentials_path": self.bigquery_credentials_path
                }
            elif instance_id.startswith("sf"):
                config.connection_params = {
                    "connection_type": "snowflake",
                    "config": self.snowflake_config.copy()
                }
            elif instance_id.startswith("local"):
                config.connection_params = {
                    "connection_type": "local_sqlite",
                    "db_path": os.path.join(self.local_db_path, f"{db_name}.sqlite")
                }
            else:
                raise ValueError(f"Unknown Spider2-lite instance type: {instance_id}")
        elif database_type.lower() == "spider1":
            # Spider1 databases are in spider_data/test_database/db_name directory
            spider_path = os.path.join(os.path.dirname(os.path.dirname(self.bird_db_path)), "spider_data", "test_database")
            config.connection_params = {
                "connection_type": "local_sqlite",
                "db_path": os.path.join(spider_path, db_name, f"{db_name}.sqlite")
            }
        elif database_type.lower() == "dabstep":
            # Dabstep uses specific merchant_data.db file
            dabstep_path = os.path.join(os.path.dirname(os.path.dirname(self.bird_db_path)), "dabstep", "merchant_data.db")
            config.connection_params = {
                "connection_type": "local_sqlite", 
                "db_path": dabstep_path
            }
        elif database_type.lower() == "local":
            # Sample/local databases for testing and examples
            config.connection_params = {
                "connection_type": "local_sqlite",
                "db_path": os.path.join(self.sample_db_path, f"{db_name}.sqlite")
            }
        else:
            raise ValueError(f"Unsupported database type: {database_type}")
            
        return config

    def _get_bigquery_client(self):
        """Get or create BigQuery client"""
        if self._bigquery_client is None:
            try:
                from google.cloud import bigquery
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.bigquery_credentials_path
                self._bigquery_client = bigquery.Client()
            except ImportError:
                logger.error("google-cloud-bigquery not installed. BigQuery functionality disabled.")
                raise
        return self._bigquery_client

    def execute_sql_sqlite(self, db_path: str, sql_query: str) -> Tuple[List[List], List[str]]:
        """Execute SQL query on SQLite database"""
        try:
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"Database file not found: {db_path}")
                
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            
            rows = cursor.fetchall()
            column_names = [description[0] for description in cursor.description] if cursor.description else []
            
            cursor.close()
            conn.close()
            
            logger.info(f"SQLite query executed successfully: {len(rows)} rows returned")
            return rows, column_names
            
        except Exception as e:
            logger.error(f"Error executing SQLite query: {str(e)}")
            raise

    def execute_sql_bigquery(self, sql_query: str) -> Tuple[List[List], List[str]]:
        """Execute SQL query on BigQuery"""
        try:
            client = self._get_bigquery_client()
            query_job = client.query(sql_query)
            results = query_job.result().to_dataframe()
            
            if results.empty:
                logger.warning("No data found for BigQuery query")
                return [], []
            
            # Convert DataFrame to list of lists and get column names
            rows = results.values.tolist()
            column_names = results.columns.tolist()
            
            logger.info(f"BigQuery query executed successfully: {len(rows)} rows returned")
            return rows, column_names
            
        except Exception as e:
            logger.error(f"Error executing BigQuery query: {str(e)}")
            raise

    def execute_sql_snowflake(self, sql_query: str) -> Tuple[List[List], List[str]]:
        """Execute SQL query on Snowflake"""
        try:
            import snowflake.connector
            conn = snowflake.connector.connect(**self.snowflake_config)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            
            # Fetch results and convert to DataFrame
            results = cursor.fetch_pandas_all()
            
            cursor.close()
            conn.close()
            
            if results.empty:
                logger.warning("No data found for Snowflake query")
                return [], []
            
            # Convert DataFrame to list of lists and get column names
            rows = results.values.tolist()
            column_names = results.columns.tolist()
            
            logger.info(f"Snowflake query executed successfully: {len(rows)} rows returned")
            return rows, column_names
            
        except Exception as e:
            logger.error(f"Error executing Snowflake query: {str(e)}")
            raise

    def get_schema_info(self, config: DatabaseConfig) -> Dict[str, Any]:
        """Get schema information for the database"""
        try:
            if config.database_type.lower() == "bird":
                return self._get_sqlite_schema_info(config.connection_params["db_path"])
            elif config.database_type.lower() == "spider2-lite":
                if config.connection_params["connection_type"] == "bigquery":
                    return self._get_bigquery_schema_info(config.db_name)
                elif config.connection_params["connection_type"] == "snowflake":
                    return self._get_snowflake_schema_info(config.db_name)
                elif config.connection_params["connection_type"] == "local_sqlite":
                    return self._get_sqlite_schema_info(config.connection_params["db_path"])
            elif config.database_type.lower() == "spider1":
                return self._get_sqlite_schema_info(config.connection_params["db_path"])
            elif config.database_type.lower() == "dabstep":
                return self._get_sqlite_schema_info(config.connection_params["db_path"])
            
            raise ValueError(f"Unsupported database configuration: {config}")
            
        except Exception as e:
            logger.error(f"Error getting schema info: {str(e)}")
            return {"error": str(e)}

    def _get_sqlite_schema_info(self, db_path: str) -> Dict[str, Any]:
        """Get schema information for SQLite database"""
        try:
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"Database file not found: {db_path}")
                
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            schema_info = {"tables": {}}
            
            # Get schema for each table
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table});")
                columns = cursor.fetchall()
                
                schema_info["tables"][table] = {
                    "columns": {col[1]: col[2] for col in columns}
                }
                
                # Get foreign key information
                cursor.execute(f"PRAGMA foreign_key_list({table});")
                foreign_keys = cursor.fetchall()
                if foreign_keys:
                    schema_info["tables"][table]["foreign_keys"] = [
                        {
                            "from_column": fk[3],
                            "to_table": fk[2],
                            "to_column": fk[4]
                        }
                        for fk in foreign_keys
                    ]
            
            cursor.close()
            conn.close()
            
            return schema_info
            
        except Exception as e:
            logger.error(f"Error getting SQLite schema: {str(e)}")
            return {"error": str(e)}

    def _get_bigquery_schema_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get schema information for BigQuery dataset"""
        try:
            client = self._get_bigquery_client()
            
            # Use fuzzy matching to find the actual dataset name
            actual_dataset_name = self._find_dataset_by_fuzzy_match(dataset_name)
            logger.info(f"Looking for dataset: {dataset_name} -> {actual_dataset_name}")
            
            # Try to access the specified dataset first
            try:
                dataset_ref = client.dataset(actual_dataset_name)
                tables = list(client.list_tables(dataset_ref))
                
                schema_info = {
                    "tables": {},
                    "note": f"BigQuery dataset: {actual_dataset_name} (matched from: {dataset_name})",
                    "connection_type": "bigquery",
                    "dataset_name": actual_dataset_name,
                    "original_search": dataset_name,
                    "match_type": "exact" if actual_dataset_name == dataset_name else "fuzzy"
                }
                
                # Get schema for each table (limit to avoid timeout)
                for table in tables[:10]:
                    table_ref = dataset_ref.table(table.table_id)
                    table_obj = client.get_table(table_ref)
                    
                    columns = {}
                    for field in table_obj.schema:
                        columns[field.name] = field.field_type
                    
                    schema_info["tables"][table.table_id] = {
                        "columns": columns
                    }
                
                return schema_info
                
            except Exception as dataset_error:
                logger.warning(f"Failed to access dataset {actual_dataset_name}: {dataset_error}")
                
                # Try to discover similar public datasets
                logger.info(f"Attempting to discover similar datasets for: {dataset_name}")
                similar_datasets = self._discover_public_datasets(dataset_name)
                
                if similar_datasets:
                    # Use the first similar dataset as fallback
                    fallback_dataset = similar_datasets[0]
                    fallback_project = fallback_dataset["project_id"]
                    fallback_name = fallback_dataset["dataset_id"]
                    
                    try:
                        fallback_dataset_ref = client.dataset(fallback_name, project=fallback_project)
                        tables = list(client.list_tables(fallback_dataset_ref))
                        
                        schema_info = {
                            "tables": {},
                            "note": f"BigQuery public dataset: {fallback_project}.{fallback_name} (discovered fallback for: {dataset_name})",
                            "connection_type": "bigquery",
                            "is_public_dataset": True,
                            "original_dataset": dataset_name,
                            "fallback_dataset": f"{fallback_project}.{fallback_name}",
                            "discovered_options": similar_datasets
                        }
                        
                        # Get schema for each table (limit to first 5 to avoid timeout)
                        for table in tables[:5]:
                            table_ref = fallback_dataset_ref.table(table.table_id)
                            table_obj = client.get_table(table_ref)
                            
                            columns = {}
                            for field in table_obj.schema:
                                columns[field.name] = field.field_type
                            
                            schema_info["tables"][table.table_id] = {
                                "columns": columns
                            }
                        
                        return schema_info
                        
                    except Exception as fallback_error:
                        logger.error(f"Failed to access fallback dataset: {fallback_error}")
                
                # Final fallback: use census dataset
                try:
                    public_dataset_ref = client.dataset('census_bureau_international', project='bigquery-public-data')
                    tables = list(client.list_tables(public_dataset_ref))
                    
                    schema_info = {
                        "tables": {},
                        "note": f"BigQuery public dataset: bigquery-public-data.census_bureau_international (final fallback from: {dataset_name})",
                        "connection_type": "bigquery",
                        "is_public_dataset": True,
                        "original_dataset": dataset_name,
                        "fallback_type": "final_census"
                    }
                    
                    # Get schema for each table (limit to first 5 to avoid timeout)
                    for table in tables[:5]:
                        table_ref = public_dataset_ref.table(table.table_id)
                        table_obj = client.get_table(table_ref)
                        
                        columns = {}
                        for field in table_obj.schema:
                            columns[field.name] = field.field_type
                        
                        schema_info["tables"][table.table_id] = {
                            "columns": columns
                        }
                    
                    return schema_info
                    
                except Exception as public_error:
                    logger.error(f"Failed to access public dataset: {public_error}")
                    return {
                        "tables": {},
                        "note": f"BigQuery schema for dataset: {dataset_name}",
                        "connection_type": "bigquery",
                        "error": f"Dataset access failed: {dataset_error}, Discovery failed, Public dataset fallback failed: {public_error}",
                        "discovery_attempted": True,
                        "fuzzy_match_attempted": True
                    }
                
        except Exception as e:
            logger.error(f"Error getting BigQuery schema: {e}")
            return {
                "tables": {},
                "note": f"BigQuery schema for dataset: {dataset_name}",
                "connection_type": "bigquery",
                "error": str(e)
            }

    def _get_snowflake_schema_info(self, database_name: str) -> Dict[str, Any]:
        """Get schema information for Snowflake database"""
        try:
            import snowflake.connector
            conn = snowflake.connector.connect(**self.snowflake_config)
            cursor = conn.cursor()
            
            # Get tables information
            cursor.execute("SHOW TABLES")
            tables_result = cursor.fetchall()
            
            schema_info = {
                "tables": {},
                "note": f"Snowflake schema for database: {database_name}",
                "connection_type": "snowflake"
            }
            
            # Process table information
            for table_row in tables_result:
                table_name = table_row[1]  # Table name is usually in the second column
                
                # Get column information for each table
                cursor.execute(f"DESCRIBE TABLE {table_name}")
                columns_result = cursor.fetchall()
                
                columns = {}
                for col_row in columns_result:
                    columns[col_row[0]] = col_row[1]  # Column name and type
                
                schema_info["tables"][table_name] = {"columns": columns}
            
            cursor.close()
            conn.close()
            
            return schema_info
            
        except Exception as e:
            logger.error(f"Error getting Snowflake schema: {str(e)}")
            return {"error": str(e)}

    def execute_sql(self, config: DatabaseConfig, sql_query: str) -> Dict[str, Any]:
        """Execute SQL query based on database configuration"""
        try:
            if config.database_type.lower() == "bird":
                rows, columns = self.execute_sql_sqlite(
                    config.connection_params["db_path"], 
                    sql_query
                )
            elif config.database_type.lower() == "spider2-lite":
                if config.connection_params["connection_type"] == "bigquery":
                    rows, columns = self.execute_sql_bigquery(sql_query)
                elif config.connection_params["connection_type"] == "snowflake":
                    rows, columns = self.execute_sql_snowflake(sql_query)
                elif config.connection_params["connection_type"] == "local_sqlite":
                    rows, columns = self.execute_sql_sqlite(
                        config.connection_params["db_path"], 
                        sql_query
                    )
                else:
                    raise ValueError(f"Unknown Spider2-lite connection type: {config.connection_params['connection_type']}")
            elif config.database_type.lower() in ["spider1", "dabstep"]:
                rows, columns = self.execute_sql_sqlite(
                    config.connection_params["db_path"], 
                    sql_query
                )
            else:
                raise ValueError(f"Unsupported database type: {config.database_type}")
            
            # Convert results to a standard format
            results = [dict(zip(columns, row)) for row in rows]
            
            return {
                "status": "success",
                "results": {
                    "query_results": results[:20],  # Limit to first 20 rows
                    "total_results_count": len(results),
                    "columns": columns
                }
            }
            
        except Exception as e:
            logger.error(f"Error executing SQL: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def test_connection(self, config: DatabaseConfig) -> Dict[str, Any]:
        """Test database connection"""
        try:
            if config.database_type.lower() == "bird":
                db_path = config.connection_params["db_path"]
                if not os.path.exists(db_path):
                    return {"status": "error", "error": f"Database file not found: {db_path}"}
                
                # Test with a simple query
                rows, columns = self.execute_sql_sqlite(db_path, "SELECT 1 as test")
                return {"status": "success", "message": "BIRD SQLite connection successful"}
                
            elif config.database_type.lower() == "spider2-lite":
                if config.connection_params["connection_type"] == "bigquery":
                    # Test BigQuery connection
                    client = self._get_bigquery_client()
                    test_query = "SELECT 1 as test"
                    query_job = client.query(test_query)
                    query_job.result()
                    return {"status": "success", "message": "BigQuery connection successful"}
                    
                elif config.connection_params["connection_type"] == "snowflake":
                    # Test Snowflake connection
                    import snowflake.connector
                    conn = snowflake.connector.connect(**self.snowflake_config)
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1 as test")
                    cursor.close()
                    conn.close()
                    return {"status": "success", "message": "Snowflake connection successful"}
                    
                elif config.connection_params["connection_type"] == "local_sqlite":
                    db_path = config.connection_params["db_path"]
                    if not os.path.exists(db_path):
                        return {"status": "error", "error": f"Database file not found: {db_path}"}
                    
                    # Test with a simple query
                    rows, columns = self.execute_sql_sqlite(db_path, "SELECT 1 as test")
                    return {"status": "success", "message": "Local SQLite connection successful"}
                    
            elif config.database_type.lower() in ["spider1", "dabstep"]:
                db_path = config.connection_params["db_path"]
                if not os.path.exists(db_path):
                    return {"status": "error", "error": f"Database file not found: {db_path}"}
                
                # Test with a simple query
                rows, columns = self.execute_sql_sqlite(db_path, "SELECT 1 as test")
                return {"status": "success", "message": f"{config.database_type} SQLite connection successful"}
                    
            return {"status": "error", "error": "Unknown database configuration"}
            
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return {"status": "error", "error": str(e)}

# Global instance for easy access
db_connection_manager = DatabaseConnectionManager() 