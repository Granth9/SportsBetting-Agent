"""Data storage and retrieval manager."""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class DataManager:
    """Manage data storage, caching, and retrieval."""
    
    def __init__(self, raw_path: str = "data/raw", processed_path: str = "data/processed"):
        """Initialize the data manager.
        
        Args:
            raw_path: Path to raw data directory
            processed_path: Path to processed data directory
        """
        self.raw_path = Path(raw_path)
        self.processed_path = Path(processed_path)
        
        # Create directories if they don't exist
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
    
    def save_raw_data(self, data: pd.DataFrame, name: str, format: str = "parquet") -> Path:
        """Save raw data to disk.
        
        Args:
            data: DataFrame to save
            name: Name for the file (without extension)
            format: File format ('parquet', 'csv')
            
        Returns:
            Path to saved file
        """
        if format == "parquet":
            path = self.raw_path / f"{name}.parquet"
            data.to_parquet(path)
        elif format == "csv":
            path = self.raw_path / f"{name}.csv"
            data.to_csv(path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved raw data to {path}")
        return path
    
    def save_processed_data(self, data: pd.DataFrame, name: str, format: str = "parquet") -> Path:
        """Save processed data to disk.
        
        Args:
            data: DataFrame to save
            name: Name for the file (without extension)
            format: File format ('parquet', 'csv')
            
        Returns:
            Path to saved file
        """
        if format == "parquet":
            path = self.processed_path / f"{name}.parquet"
            data.to_parquet(path)
        elif format == "csv":
            path = self.processed_path / f"{name}.csv"
            data.to_csv(path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved processed data to {path}")
        return path
    
    def load_raw_data(self, name: str, format: str = "parquet") -> Optional[pd.DataFrame]:
        """Load raw data from disk.
        
        Args:
            name: Name of the file (without extension)
            format: File format ('parquet', 'csv')
            
        Returns:
            DataFrame or None if file doesn't exist
        """
        if format == "parquet":
            path = self.raw_path / f"{name}.parquet"
            if path.exists():
                return pd.read_parquet(path)
        elif format == "csv":
            path = self.raw_path / f"{name}.csv"
            if path.exists():
                return pd.read_csv(path)
        
        logger.warning(f"Raw data file not found: {name}.{format}")
        return None
    
    def load_processed_data(self, name: str, format: str = "parquet") -> Optional[pd.DataFrame]:
        """Load processed data from disk.
        
        Args:
            name: Name of the file (without extension)
            format: File format ('parquet', 'csv')
            
        Returns:
            DataFrame or None if file doesn't exist
        """
        if format == "parquet":
            path = self.processed_path / f"{name}.parquet"
            if path.exists():
                return pd.read_parquet(path)
        elif format == "csv":
            path = self.processed_path / f"{name}.csv"
            if path.exists():
                return pd.read_csv(path)
        
        logger.warning(f"Processed data file not found: {name}.{format}")
        return None
    
    def list_files(self, data_type: str = "raw") -> List[str]:
        """List all data files of a given type.
        
        Args:
            data_type: 'raw' or 'processed'
            
        Returns:
            List of file names (without extensions)
        """
        path = self.raw_path if data_type == "raw" else self.processed_path
        
        files = []
        for file_path in path.glob("*"):
            if file_path.is_file() and file_path.suffix in ['.parquet', '.csv']:
                files.append(file_path.stem)
        
        return sorted(files)
    
    def delete_file(self, name: str, data_type: str = "raw") -> bool:
        """Delete a data file.
        
        Args:
            name: Name of the file (without extension)
            data_type: 'raw' or 'processed'
            
        Returns:
            True if deleted, False if not found
        """
        path = self.raw_path if data_type == "raw" else self.processed_path
        
        for ext in ['.parquet', '.csv']:
            file_path = path / f"{name}{ext}"
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted {file_path}")
                return True
        
        logger.warning(f"File not found: {name}")
        return False
    
    def get_file_info(self, name: str, data_type: str = "raw") -> Optional[Dict[str, Any]]:
        """Get information about a data file.
        
        Args:
            name: Name of the file (without extension)
            data_type: 'raw' or 'processed'
            
        Returns:
            Dictionary with file info or None
        """
        path = self.raw_path if data_type == "raw" else self.processed_path
        
        for ext in ['.parquet', '.csv']:
            file_path = path / f"{name}{ext}"
            if file_path.exists():
                stat = file_path.stat()
                return {
                    'name': name,
                    'format': ext.lstrip('.'),
                    'size_mb': stat.st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(stat.st_mtime),
                    'path': str(file_path)
                }
        
        return None

