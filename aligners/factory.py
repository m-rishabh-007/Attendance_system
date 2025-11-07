"""
Factory for creating face alignment instances.

Implements Factory Pattern to decouple alignment algorithm selection
from client code. Allows easy swapping between MediaPipe, dlib, MTCNN, etc.
"""

from typing import Dict, Any
import logging

from aligners.base_aligner import BaseAligner
from aligners.mediapipe_aligner import MediaPipeAligner


logger = logging.getLogger(__name__)


class AlignerFactory:
    """
    Factory class for creating face alignment instances.
    
    Design Pattern: Factory Pattern
    - Centralizes object creation logic
    - Allows adding new aligners without modifying client code
    - Configuration-driven aligner selection
    
    Usage:
        config = {'type': 'mediapipe', 'output_size': (112, 112)}
        aligner = AlignerFactory.create_aligner(config)
    """
    
    @staticmethod
    def create_aligner(config: Dict[str, Any]) -> BaseAligner:
        """
        Create face aligner based on configuration.
        
        Args:
            config: Aligner configuration dictionary with 'type' key:
                - 'mediapipe': MediaPipe Face Mesh alignment
                - 'dlib': dlib 68-point alignment (future)
                - 'mtcnn': MTCNN alignment (future)
                
        Returns:
            Initialized BaseAligner instance
            
        Raises:
            ValueError: If aligner type is unknown
        """
        aligner_type = config.get('type', 'mediapipe').lower()
        
        if aligner_type == 'mediapipe':
            logger.info("Creating MediaPipe aligner")
            return MediaPipeAligner(config)
        
        # Future aligners
        # elif aligner_type == 'dlib':
        #     return DlibAligner(config)
        # elif aligner_type == 'mtcnn':
        #     return MTCNNAligner(config)
        
        else:
            raise ValueError(
                f"Unknown aligner type: {aligner_type}. "
                f"Supported types: ['mediapipe']"
            )
