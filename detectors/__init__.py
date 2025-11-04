"""
Face Detector Module - Factory Pattern

Provides unified interface for different face detection models.
Supports easy swapping between YOLO, MTCNN, RetinaFace, etc.

Design Pattern: Factory
Use Cases:
    - Switch detection models via configuration
    - A/B testing different detectors
    - Platform-specific model selection (Pi vs Laptop)
"""

__version__ = "2.0.0"
