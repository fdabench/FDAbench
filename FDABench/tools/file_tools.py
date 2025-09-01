"""
File system tools for FDABench Package.

These tools provide file system operations and file search capabilities.
The base path can be configured to search in different directories.
"""

import os
import fnmatch
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class FileSystemSearchTool:
    """Tool for searching files in the file system"""
    
    def __init__(self, base_path: str = None):
        # Allow configuration of base path, with fallback to current directory
        self.base_path = base_path or os.environ.get('DB_AGENT_BENCH_FILE_PATH', '.')
    
    def execute(self, pattern: str = "*", search_params: Dict = None,
                content_pattern: str = None, file_types: List[str] = None,
                max_results: int = 100, **kwargs) -> Dict[str, Any]:
        """
        Search files in the file system.
        
        Args:
            pattern: File name pattern (supports wildcards)
            search_params: Additional search parameters
            content_pattern: Pattern to search within file contents
            file_types: List of file extensions to include
            max_results: Maximum number of results
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with status and search results
        """
        try:
            # Use search_params if provided
            if search_params:
                pattern = search_params.get("pattern", pattern)
                content_pattern = search_params.get("content_pattern", content_pattern)
                file_types = search_params.get("file_types", file_types)
                # Allow overriding base path in search_params
                custom_path = search_params.get("base_path")
                if custom_path:
                    search_path = custom_path
                else:
                    search_path = self.base_path
            else:
                search_path = kwargs.get("base_path", self.base_path)
            
            results = []
            
            # Ensure search path exists
            if not os.path.exists(search_path):
                return {
                    "status": "error",
                    "error": f"Search path does not exist: {search_path}"
                }
            
            # Search files
            for root, dirs, files in os.walk(search_path):
                if len(results) >= max_results:
                    break
                    
                for file in files:
                    if len(results) >= max_results:
                        break
                    
                    # Check file name pattern
                    if not fnmatch.fnmatch(file, pattern):
                        continue
                    
                    # Check file type
                    if file_types:
                        if not any(file.endswith(ft) for ft in file_types):
                            continue
                    
                    full_path = os.path.join(root, file)
                    
                    # Check content pattern if specified
                    if content_pattern:
                        try:
                            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                if content_pattern.lower() not in content.lower():
                                    continue
                        except Exception:
                            continue
                    
                    # Add file info
                    try:
                        stat = os.stat(full_path)
                        results.append({
                            "path": full_path,
                            "name": file,
                            "size": stat.st_size,
                            "modified": stat.st_mtime,
                            "type": os.path.splitext(file)[1],
                            "directory": root
                        })
                    except Exception as e:
                        logger.warning(f"Could not stat file {full_path}: {e}")
            
            return {
                "status": "success",
                "results": {
                    "files": results,
                    "total_count": len(results),
                    "search_path": search_path,
                    "pattern": pattern,
                    "content_pattern": content_pattern,
                    "file_types": file_types
                }
            }
            
        except Exception as e:
            logger.error(f"File system search failed: {str(e)}")
            return {"status": "error", "error": str(e)}


class FileReaderTool:
    """Tool for reading file contents"""
    
    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding
    
    def execute(self, file_path: str, max_lines: int = None,
                offset: int = 0, **kwargs) -> Dict[str, Any]:
        """
        Read file contents.
        
        Args:
            file_path: Path to the file
            max_lines: Maximum number of lines to read
            offset: Starting line offset
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with status and file contents
        """
        try:
            if not file_path:
                return {"status": "error", "error": "No file path provided"}
            
            if not os.path.exists(file_path):
                return {"status": "error", "error": f"File not found: {file_path}"}
            
            with open(file_path, 'r', encoding=self.encoding, errors='ignore') as f:
                lines = f.readlines()
            
            # Apply offset and limit
            if offset > 0:
                lines = lines[offset:]
            if max_lines:
                lines = lines[:max_lines]
            
            content = ''.join(lines)
            
            return {
                "status": "success",
                "results": {
                    "content": content,
                    "lines": lines,
                    "total_lines": len(lines),
                    "file_path": file_path,
                    "encoding": self.encoding
                }
            }
            
        except Exception as e:
            logger.error(f"File read failed: {str(e)}")
            return {"status": "error", "error": str(e)}


class FileWriterTool:
    """Tool for writing content to files"""
    
    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding
    
    def execute(self, file_path: str, content: str, 
                mode: str = "w", create_dirs: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Write content to file.
        
        Args:
            file_path: Path to the file
            content: Content to write
            mode: Write mode ('w' for overwrite, 'a' for append)
            create_dirs: Whether to create parent directories
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with status and write results
        """
        try:
            if not file_path:
                return {"status": "error", "error": "No file path provided"}
            
            if content is None:
                return {"status": "error", "error": "No content provided"}
            
            # Create parent directories if needed
            if create_dirs:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, mode, encoding=self.encoding) as f:
                f.write(content)
            
            # Get file info
            stat = os.stat(file_path)
            
            return {
                "status": "success",
                "results": {
                    "file_path": file_path,
                    "bytes_written": len(content.encode(self.encoding)),
                    "mode": mode,
                    "encoding": self.encoding,
                    "file_size": stat.st_size
                }
            }
            
        except Exception as e:
            logger.error(f"File write failed: {str(e)}")
            return {"status": "error", "error": str(e)}


# Alias for backward compatibility
FileSystemTool = FileSystemSearchTool