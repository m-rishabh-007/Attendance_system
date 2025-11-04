"""
Tracking Stage - Face tracking orchestration.

This stage coordinates face tracking using the TrackerFactory.
It receives detections and maintains persistent track IDs.

Author: Rishabh Mishra
Date: November 4, 2025
Version: 3.0.0
"""

import logging
from typing import List, Any

from tracking.factory import TrackerFactory


class TrackingStage:
    """
    Face tracking pipeline stage.
    
    Responsibilities:
    1. Get tracker from factory (based on config)
    2. Update tracker with new detections
    3. Return tracks with persistent IDs
    
    Does NOT care about:
    - Which tracker implementation is used (BoT-SORT, ByteTrack, DeepSORT)
    - How tracking algorithm works internally
    - What happens to tracks afterwards
    
    Example:
        >>> config = {'type': 'botsort', 'track_thresh': 0.5}
        >>> stage = TrackingStage(config)
        >>> tracks = stage.process(detections)
    """
    
    def __init__(self, config: dict):
        """
        Initialize tracking stage.
        
        Args:
            config: Tracking configuration dict
        """
        self.logger = logging.getLogger('TrackingStage')
        
        # Create tracker using Factory pattern
        self.logger.info("Creating tracker from factory...")
        self.tracker = TrackerFactory.create(config)
        
        self.logger.info(f"✅ TrackingStage initialized with {self.tracker}")
    
    def process(self, detections: List[Any]) -> List[Any]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dicts from DetectionStage
        
        Returns:
            List of Track objects with persistent IDs
        """
        return self.tracker.update(detections)
    
    def __repr__(self) -> str:
        return f"TrackingStage(tracker={self.tracker})"
