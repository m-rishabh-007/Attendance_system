"""
Recognizer Factory - Config-driven recognizer creation

This module implements the Factory pattern for creating face recognizers
from configuration. It abstracts the instantiation logic and allows
different recognizer types to be swapped by changing config.yaml.

Design Pattern: Factory Pattern
- Centralizes object creation logic
- Allows adding new recognizers without changing client code
- Config-driven: type specified in config.yaml

Supported Recognizers:
- 'auraface': AuraFace ResNet100 (Apache 2.0 license)
- (Future) 'arcface': ArcFace models
- (Future) 'facenet': FaceNet models

Usage:
    >>> from recognizers.factory import RecognizerFactory
    >>> config = {
    ...     'model_type': 'auraface',
    ...     'model_path': 'models/recognition/auraface_resnet100_fp32.onnx',
    ...     'embedding_size': 512,
    ...     'input_size': [112, 112],
    ...     'num_threads': 4
    ... }
    >>> recognizer = RecognizerFactory.create(config)
    >>> recognizer.load_model()

Author: Attendance System Team
Created: November 2025
Phase: 3A - Core Recognition (Week 1)
"""

from typing import Dict, Any
import logging
from pathlib import Path

from recognizers.base_recognizer import BaseRecognizer
from recognizers.auraface_recognizer import AuraFaceRecognizer


logger = logging.getLogger(__name__)


class RecognizerFactory:
    """
    Factory class for creating face recognizers from configuration.
    
    This class implements the Factory pattern to create recognizer instances
    based on configuration. It validates config parameters and provides
    helpful error messages for common issues.
    
    The factory supports multiple recognizer types and can be easily extended
    by adding new types to the _RECOGNIZER_TYPES registry.
    
    Class Methods:
        create: Create recognizer from config dictionary
        get_supported_types: Get list of supported recognizer types
    
    Example:
        >>> # From config.yaml
        >>> config = load_config('config.yaml')
        >>> recognizer = RecognizerFactory.create(config['recognition'])
        >>> recognizer.load_model()
        >>> 
        >>> # Direct config dict
        >>> config = {
        ...     'model_type': 'auraface',
        ...     'model_path': 'models/recognition/auraface_resnet100_fp32.onnx',
        ...     'embedding_size': 512,
        ...     'num_threads': 4
        ... }
        >>> recognizer = RecognizerFactory.create(config)
    """
    
    # Registry of supported recognizer types
    _RECOGNIZER_TYPES: Dict[str, type] = {
        'auraface': AuraFaceRecognizer,
        # Future recognizers:
        # 'arcface': ArcFaceRecognizer,
        # 'facenet': FaceNetRecognizer,
    }
    
    @classmethod
    def create(cls, config: Dict[str, Any]) -> BaseRecognizer:
        """
        Create recognizer from configuration dictionary.
        
        This method:
        1. Validates required config parameters
        2. Extracts recognizer type
        3. Maps type to recognizer class
        4. Instantiates recognizer with config
        5. Returns recognizer instance (not loaded yet)
        
        The returned recognizer needs load_model() called before use.
        
        Args:
            config: Configuration dictionary with keys:
                - model_type (str): Recognizer type ('auraface', etc.)
                - model_path (str): Path to model file
                - embedding_size (int): Embedding dimension (default: 512)
                - input_size (list): Input size [H, W] (default: [112, 112])
                - num_threads (int): CPU threads (default: 4)
                - (optional) quantized (bool): INT8 quantized (default: False)
                - (optional) backend (str): Inference backend
                - (optional) device (str): Device type ('cpu', 'gpu')
        
        Returns:
            Recognizer instance (not loaded - call load_model() before use)
        
        Raises:
            ValueError: If config is invalid or missing required parameters
            KeyError: If model_type is not supported
            FileNotFoundError: If model file doesn't exist
        
        Example:
            >>> config = {
            ...     'model_type': 'auraface',
            ...     'model_path': 'models/recognition/auraface_resnet100_fp32.onnx',
            ...     'embedding_size': 512,
            ...     'input_size': [112, 112],
            ...     'num_threads': 4
            ... }
            >>> recognizer = RecognizerFactory.create(config)
            >>> print(type(recognizer).__name__)  # AuraFaceRecognizer
            >>> recognizer.load_model()
            >>> print(recognizer.is_loaded)  # True
        """
        # Validate config
        if not config:
            raise ValueError("Config dictionary is empty")
        
        if not isinstance(config, dict):
            raise ValueError(f"Config must be dict, got {type(config)}")
        
        # Extract model type
        model_type = config.get('model_type', '').lower()
        if not model_type:
            raise ValueError(
                "Missing 'model_type' in config. "
                f"Supported types: {list(cls._RECOGNIZER_TYPES.keys())}"
            )
        
        # Check if model type is supported
        if model_type not in cls._RECOGNIZER_TYPES:
            supported = ', '.join(cls._RECOGNIZER_TYPES.keys())
            raise KeyError(
                f"Unsupported recognizer type: '{model_type}'. "
                f"Supported types: {supported}"
            )
        
        # Validate model path
        model_path = config.get('model_path')
        if not model_path:
            raise ValueError("Missing 'model_path' in config")
        
        # Check if model file exists
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please download the model first. See: recognizers/README.md"
            )
        
        # Extract common parameters with defaults
        params = {
            'model_path': str(model_path),
            'embedding_size': config.get('embedding_size', 512),
            'input_size': tuple(config.get('input_size', [112, 112])),
            'quantized': config.get('quantized', False),
        }
        
        # Extract recognizer-specific parameters
        # AuraFace-specific
        if model_type == 'auraface':
            params['num_threads'] = config.get('num_threads', 4)
        
        # Get recognizer class
        recognizer_class = cls._RECOGNIZER_TYPES[model_type]
        
        # Instantiate recognizer
        try:
            recognizer = recognizer_class(**params)
            logger.info(
                f"Created {recognizer_class.__name__}: "
                f"model={model_path.name}, "
                f"quantized={params['quantized']}"
            )
            return recognizer
            
        except Exception as e:
            logger.error(f"Failed to create recognizer: {e}", exc_info=True)
            raise RuntimeError(
                f"Failed to instantiate {recognizer_class.__name__}: {e}"
            ) from e
    
    @classmethod
    def get_supported_types(cls) -> list:
        """
        Get list of supported recognizer types.
        
        Returns:
            List of supported type strings (e.g., ['auraface'])
        
        Example:
            >>> types = RecognizerFactory.get_supported_types()
            >>> print(types)  # ['auraface']
            >>> 'auraface' in types  # True
        """
        return list(cls._RECOGNIZER_TYPES.keys())
    
    @classmethod
    def register_recognizer(cls, name: str, recognizer_class: type):
        """
        Register a new recognizer type (for future extensibility).
        
        This allows third-party recognizers to be added without
        modifying the factory code.
        
        Args:
            name: Type name (e.g., 'custom_recognizer')
            recognizer_class: Recognizer class (must inherit BaseRecognizer)
        
        Raises:
            ValueError: If class doesn't inherit BaseRecognizer
            KeyError: If name already registered
        
        Example:
            >>> class MyRecognizer(BaseRecognizer):
            ...     # Implementation
            ...     pass
            >>> 
            >>> RecognizerFactory.register_recognizer('my_recognizer', MyRecognizer)
            >>> config = {'model_type': 'my_recognizer', ...}
            >>> recognizer = RecognizerFactory.create(config)
        """
        # Validate recognizer class
        if not issubclass(recognizer_class, BaseRecognizer):
            raise ValueError(
                f"{recognizer_class.__name__} must inherit from BaseRecognizer"
            )
        
        # Check if name already registered
        if name.lower() in cls._RECOGNIZER_TYPES:
            raise KeyError(
                f"Recognizer type '{name}' already registered. "
                f"Use a different name or unregister first."
            )
        
        # Register
        cls._RECOGNIZER_TYPES[name.lower()] = recognizer_class
        logger.info(f"Registered new recognizer type: '{name}'")
