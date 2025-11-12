"""
Recognizers Module - Face Recognition Implementations

This module provides face recognition functionality for the Attendance System.
It follows the Strategy pattern, allowing different recognition algorithms
to be swapped without modifying the pipeline code.

Phase 3A (Week 1): Core Recognition with FP32 models
Phase 3B (Week 2): INT8 quantization + quality-aware caching
Phase 3C (Week 3): Database integration + similarity matching

Current Status: Week 1 - BaseRecognizer interface complete

Available Classes:
    - BaseRecognizer: Abstract interface for all recognizers
    - (Future) AuraFaceRecognizer: AuraFace ResNet100 implementation
    - (Future) RecognizerFactory: Config-driven recognizer creation
    - (Future) QualityScorer: Face quality assessment for caching

Author: Attendance System Team
Created: November 2025
Phase: 3A - Core Recognition
"""

from recognizers.base_recognizer import BaseRecognizer
from recognizers.auraface_recognizer import AuraFaceRecognizer
from recognizers.factory import RecognizerFactory
from recognizers.quality_scorer import QualityScorer

__all__ = [
    'BaseRecognizer',
    'AuraFaceRecognizer',       # ✅ Day 3-4
    'RecognizerFactory',         # ✅ Day 3-4
    'QualityScorer',             # ✅ Day 5
]

__version__ = '0.1.0'
__author__ = 'Attendance System Team'
__phase__ = '3A - Core Recognition (Week 1)'
