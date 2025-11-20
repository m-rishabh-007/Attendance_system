"""
Detection Stage - Face detection orchestration.

This stage coordinates face detection using the DetectorFactory.
It doesn't care about implementation details - just calls detect().

Author: Rishabh Mishra
Date: November 4, 2025
Version: 3.0.0
"""

import logging
from typing import List, Dict, Any

from detectors.factory import DetectorFactory


class DetectionStage:
    """
    Face detection pipeline stage.
    
    Responsibilities:
    1. Get detector from factory (based on config)
    2. Call detect() on each frame
    3. Return raw detections (no tracking yet)
    
    Does NOT care about:
    - Which detector implementation is used (YOLO, RetinaFace, MTCNN)
    - How detection works internally
    - What happens to detections afterwards
    
    Example:
        >>> config = {'type': 'yolo', 'model_path': 'models/detection/yolov8n.tflite'}
        >>> stage = DetectionStage(config)
        >>> detections = stage.process(frame)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize detection stage.
        
        Args:
            config: Detection configuration dict
        """
        self.logger = logging.getLogger('DetectionStage')
        
        # Create detector using Factory pattern
        self.logger.info("Creating detector from factory...")
        self.detector = DetectorFactory.create(config)
        self.detector.initialize()
        
        self.logger.info(f"✅ DetectionStage initialized with {self.detector}")
    
    def process(self, frame) -> List:
        """
        Detect faces in frame.
        
        ⚠️ IMPORTANT: If using YOLO detector, this calls detect() which does NOT
        include tracking persistence. For stable track IDs, the pipeline should
        use detect_and_track() instead (handled by orchestrator).
        
        Args:
            frame: Input frame (BGR format, numpy array)
        
        Returns:
            List of Detection objects from the detector
        """
        return self.detector.detect(frame)
    
    def process_with_tracking(self, frame, tracker_config='botsort.yaml'):
        """
        Detect AND track faces using built-in YOLO tracking (RECOMMENDED).
        
        This bypasses the separate tracking stage and uses YOLO's built-in
        tracker with persist=True for stable track IDs.
        
        Args:
            frame: Input frame (BGR format, numpy array)
            tracker_config: Tracker config ('botsort.yaml' or 'bytetrack.yaml')
        
        Returns:
            Tuple of (detections, tracks) with persistent track IDs
        """
        # Type guard: Check if detector supports integrated tracking
        if hasattr(self.detector, 'detect_and_track') and callable(getattr(self.detector, 'detect_and_track', None)):
            return self.detector.detect_and_track(frame, tracker_config)  # type: ignore[attr-defined]
        else:
            # Fallback: detector doesn't support integrated tracking
            self.logger.warning(
                "Detector doesn't support integrated tracking. "
                "Using detect() only - track IDs may fluctuate!"
            )
            detections = self.detector.detect(frame)
            return detections, []
    
    def __repr__(self) -> str:
        return f"DetectionStage(detector={self.detector})"
