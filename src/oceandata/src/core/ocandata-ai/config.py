"""
Configuration handler for OceanData.

This module provides utilities for loading, validating, and accessing configuration.
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

# Default configuration
DEFAULT_CONFIG = {
    # Anomaly detection settings
    "anomaly_detection": {
        "method": "isolation_forest",
        "contamination": 0.05,
    },
    
    # Semantic analysis settings
    "semantic_analysis": {
        "model_type": "bert",
        "model_name": "bert-base-uncased",
        "cache_embeddings": True,
    },
    
    # Predictive modeling settings
    "predictive_modeling": {
        "model_type": "lstm",
        "forecast_horizon": 7,
        "lookback": 10,
    },
    
    # Data synthesis settings
    "data_synthesis": {
        "categorical_threshold": 10,
        "noise_dim": 100,
        "enabled": True,
    },
    
    # Privacy and compute-to-data settings
    "privacy": {
        "min_group_size": 5,
        "noise_level": 0.01,
        "outlier_removal": True,
    },
    
    # Data monetization settings
    "monetization": {
        "base_token_value": 5.0,
        "min_token_value": 1.0,
        "max_token_value": 10.0,
    },
    
    # System settings
    "system": {
        "log_level": "INFO",
        "log_file": None,
        "use_gpu": True,
    }
}


class ConfigManager:
    """Manager for OceanData configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to a JSON configuration file.
        """
        self.config = DEFAULT_CONFIG.copy()
        
        if config_path:
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str) -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to a JSON configuration file.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        
        # Update the default configuration with user settings
        self._update_nested_dict(self.config, user_config)
    
    def load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Load configuration from a dictionary.
        
        Args:
            config_dict: Configuration dictionary.
        """
        self._update_nested_dict(self.config, config_dict)
    
    def _update_nested_dict(self, d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a nested dictionary with values from another dictionary.
        
        Args:
            d: Target dictionary to update.
            u: Source dictionary with new values.
            
        Returns:
            Updated dictionary.
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key using dot notation (e.g., 'privacy.noise_level').
            default: Default value to return if the key is not found.
            
        Returns:
            Configuration value or default.
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def save_to_file(self, config_path: str) -> None:
        """
        Save the current configuration to a JSON file.
        
        Args:
            config_path: Path to save the configuration.
        """
        # Create directory if it doesn't exist
        directory = os.path.dirname(config_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def __getitem__(self, key: str) -> Any:
        """
        Get a configuration section or value.
        
        Args:
            key: Configuration key.
            
        Returns:
            Configuration section or value.
        """
        return self.config[key]
