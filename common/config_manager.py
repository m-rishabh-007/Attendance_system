"""
Configuration Manager - Singleton Pattern

Provides centralized configuration management ensuring a single source of truth
throughout the application lifecycle. Prevents redundant file I/O and maintains
consistency across all components.

Design Pattern: Singleton
Thread-Safe: No (single-threaded application assumed)
"""

from typing import Any, Optional, Dict
from pathlib import Path
import yaml
import logging


class ConfigManager:
    """
    Singleton configuration manager for the Face Attendance System.
    
    This class ensures only one configuration instance exists throughout
    the application, preventing redundant file reads and maintaining
    consistency across all components.
    
    Thread Safety:
        Not thread-safe. Designed for single-threaded applications.
        For multi-threaded use, add threading.Lock() protection.
    
    Example:
        >>> config = ConfigManager()
        >>> config.load('config.yaml')
        >>> camera_id = config.get('deployment_config.camera_id', default=0)
        >>> 
        >>> # Same instance anywhere in code
        >>> config2 = ConfigManager()
        >>> assert config is config2  # True
    """
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[Dict[str, Any]] = None
    _config_path: Optional[Path] = None
    _logger = logging.getLogger('ConfigManager')
    
    def __new__(cls) -> 'ConfigManager':
        """
        Create or return existing singleton instance.
        
        Returns:
            ConfigManager: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._logger.debug("ConfigManager singleton instance created")
        return cls._instance
    
    def load(self, config_path: Path | str) -> None:
        """
        Load configuration from YAML file.
        
        Configuration is loaded only once. Subsequent calls to load()
        will be ignored unless force_reload=True is added in future.
        
        Args:
            config_path: Path to YAML configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is malformed
        """
        config_path = Path(config_path)
        
        if self._config is not None:
            self._logger.warning(
                f"Configuration already loaded from {self._config_path}. "
                "Ignoring new load request."
            )
            return
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            self._config_path = config_path
            self._logger.info(f"Configuration loaded from {config_path}")
        except yaml.YAMLError as e:
            self._logger.error(f"Failed to parse YAML config: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key path.
        
        Supports nested dictionary access using dot notation.
        Example: 'inference_settings.confidence_threshold'
        
        Args:
            key: Dot-separated key path (e.g., 'model.type')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            >>> config.get('inference_settings.confidence_threshold', 0.5)
            0.6
            >>> config.get('nonexistent.key', 'fallback')
            'fallback'
        """
        if self._config is None:
            self._logger.warning("Configuration not loaded. Returning default value.")
            return default
        
        # Navigate nested dictionary
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Top-level section name
            
        Returns:
            Dictionary containing section configuration
            
        Raises:
            KeyError: If section doesn't exist
        """
        if self._config is None:
            raise ValueError("Configuration not loaded")
        
        if section not in self._config:
            raise KeyError(f"Configuration section '{section}' not found")
        
        return self._config[section]
    
    def is_loaded(self) -> bool:
        """Check if configuration has been loaded."""
        return self._config is not None
    
    def reload(self) -> None:
        """
        Reload configuration from last loaded path.
        
        Useful for development when config changes without restart.
        """
        if self._config_path is None:
            raise ValueError("No configuration path set. Call load() first.")
        
        self._config = None  # Clear existing config
        self.load(self._config_path)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        loaded = "loaded" if self._config is not None else "not loaded"
        path = self._config_path if self._config_path else "none"
        return f"ConfigManager(status={loaded}, path={path})"
