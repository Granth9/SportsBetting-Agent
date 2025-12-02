"""Configuration loader utility."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv


class ConfigLoader:
    """Load and manage configuration from YAML and environment variables."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the config loader.
        
        Args:
            config_path: Path to the YAML config file
        """
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_env()
        self._load_yaml()
    
    def _load_env(self):
        """Load environment variables from .env file."""
        load_dotenv()
    
    def _load_yaml(self):
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key (supports nested keys with dots).
        
        Args:
            key: Configuration key (e.g., 'anthropic.model')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_env(self, key: str, default: Any = None) -> Any:
        """Get an environment variable.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            
        Returns:
            Environment variable value
        """
        return os.getenv(key, default)
    
    @property
    def anthropic_api_key(self) -> str:
        """Get Anthropic API key from environment."""
        key = self.get_env('ANTHROPIC_API_KEY')
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        return key
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary."""
        return self._config


# Global config instance
_config_instance = None


def get_config() -> ConfigLoader:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader()
    return _config_instance

