"""Configuration loader and validator for the damage detection system."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class Config:
    """Configuration manager for the vehicle damage detection system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to config.yaml file. If None, uses default location.
        """
        if config_path is None:
            # Default to project root
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        self._validate()
    
    def _validate(self):
        """Validate configuration structure and values."""
        required_sections = ['system', 'paths', 'dataset', 'detection', 'segmentation', 
                           'classification', 'severity', 'api']
        
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        logger.info("Configuration validated successfully")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'system.device')
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
    
    @property
    def device(self) -> str:
        """Get the computation device (mps, cuda, or cpu)."""
        import torch
        
        device = self.get('system.device', 'cpu')
        
        # Verify MPS availability for M4 Pro
        if device == 'mps':
            if not torch.backends.mps.is_available():
                logger.warning("MPS not available, falling back to CPU")
                device = self.get('system.device_fallback', 'cpu')
            else:
                logger.info("Using MPS (Metal Performance Shaders) for acceleration")
        
        return device
    
    @property
    def damage_classes(self) -> list:
        """Get list of damage classes."""
        return self.get('dataset.damage_classes', [])
    
    @property
    def num_damage_classes(self) -> int:
        """Get number of damage classes."""
        return len(self.damage_classes)
    
    @property
    def severity_levels(self) -> list:
        """Get list of severity levels."""
        return self.get('dataset.severity_levels', ['low', 'medium', 'high'])
    
    @property
    def vehicle_parts(self) -> list:
        """Get list of vehicle parts."""
        return self.get('dataset.vehicle_parts', [])
    
    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        return Path(self.get('paths.data_dir', 'data'))
    
    @property
    def models_dir(self) -> Path:
        """Get models directory path."""
        return Path(self.get('paths.models_dir', 'models'))
    
    @property
    def outputs_dir(self) -> Path:
        """Get outputs directory path."""
        return Path(self.get('paths.outputs_dir', 'outputs'))
    
    @property
    def logs_dir(self) -> Path:
        """Get logs directory path."""
        return Path(self.get('paths.logs_dir', 'logs'))
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return the entire configuration as a dictionary."""
        return self._config.copy()


# Global config instance
_global_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get or create the global configuration instance.
    
    Args:
        config_path: Path to config file (only used on first call)
        
    Returns:
        Config instance
    """
    global _global_config
    
    if _global_config is None:
        _global_config = Config(config_path)
    
    return _global_config


def reset_config():
    """Reset the global configuration instance."""
    global _global_config
    _global_config = None
