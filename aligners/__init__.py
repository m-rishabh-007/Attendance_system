"""
Face alignment module for normalizing face crops to canonical poses.

This module provides face alignment implementations for preparing faces
for recognition. Alignment transforms arbitrary face crops (varying pose,
scale, rotation) into normalized 112x112 crops suitable for ArcFace models.

Available Aligners:
- MediaPipeAligner: 468-landmark alignment using MediaPipe Face Mesh

Usage:
    from aligners import AlignerFactory
    
    config = {
        'type': 'mediapipe',
        'output_size': (112, 112),
        'min_detection_confidence': 0.5
    }
    
    aligner = AlignerFactory.create_aligner(config)
    aligned_face = aligner.align(frame, bbox)
"""

from aligners.base_aligner import BaseAligner
from aligners.mediapipe_aligner import MediaPipeAligner
from aligners.factory import AlignerFactory

__version__ = '3.0.0'
__all__ = [
    'BaseAligner',
    'MediaPipeAligner',
    'AlignerFactory'
]
