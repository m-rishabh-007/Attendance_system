"""
Detector Factory - Factory Pattern Implementation

Creates appropriate face detector based on configuration, hiding
implementation details and allowing easy model switching.

Design Pattern: Factory
Benefits:
    - Centralized model creation logic
    - Easy to add new detector types
    - Client code doesn't know about concrete classes
"""

from typing import Dict, Any
from pathlib import Path
import logging

from common.base_classes import BaseFaceDetector


class DetectorFactory:
    """
    Factory for creating face detector instances.
    
    Supports multiple detector types:
    - 'yolo': Ultralytics YOLO with TFLite model (recommended)
    - Custom detectors can be registered via register_detector()
    
    Example:
        >>> config = {
        ...     'type': 'yolo',
        ...     'model_path': 'models/yolov8n_face_int8.tflite',
        ...     'confidence_threshold': 0.5
        ... }
        >>> detector = DetectorFactory.create(config)
        >>> detector.initialize()
        >>> detections = detector.detect(frame)
    """
    
    _logger = logging.getLogger('DetectorFactory')
    
    # Registry of available detectors
    _detector_registry = {}
    
    @classmethod
    def register_detector(cls, detector_type: str, detector_class: type):
        """
        Register a new detector type.
        
        Allows plugins/extensions to add new detector types dynamically.
        
        Args:
            detector_type: String identifier for detector
            detector_class: Class that inherits from BaseFaceDetector
        """
        cls._detector_registry[detector_type] = detector_class
        cls._logger.info(f"Registered detector type: {detector_type}")
    
    @classmethod
    def create(cls, config: Dict[str, Any]) -> BaseFaceDetector:
        """
        Create face detector based on configuration.
        
        Args:
            config: Detector configuration dictionary containing:
                - type: Detector type ('yolo', 'tflite', etc.)
                - model_path: Path to model file
                - Other detector-specific settings
                
        Returns:
            Initialized BaseFaceDetector instance
            
        Raises:
            ValueError: If detector type is unsupported
            KeyError: If required config keys are missing
        
        Example:
            >>> config = {
            ...     'type': 'yolo',
            ...     'model_path': 'models/yolov8n_face_int8.tflite',
            ...     'input_size': 256,
            ...     'confidence_threshold': 0.5
            ... }
            >>> detector = DetectorFactory.create(config)
        """
        detector_type = config.get('type')
        
        if not detector_type:
            raise KeyError("'type' field required in detector config")
        
        # Lazy import to avoid circular dependencies
        # Only import the detector that's actually needed
        if detector_type == 'yolo':
            from detectors.yolo_detector import YOLODetector
            cls._logger.info("Creating YOLO detector")
            return YOLODetector(config)
        
        # Check custom registered detectors
        elif detector_type in cls._detector_registry:
            detector_class = cls._detector_registry[detector_type]
            cls._logger.info(f"Creating custom detector: {detector_type}")
            return detector_class(config)
        
        else:
            available = ['yolo'] + list(cls._detector_registry.keys())
            raise ValueError(
                f"Unsupported detector type: '{detector_type}'. "
                f"Available types: {available}"
            )
    
    @classmethod
    def get_available_types(cls) -> list:
        """Get list of available detector types."""
        return ['yolo'] + list(cls._detector_registry.keys())
