"""
No Tracker Implementation

Passes through detections without tracking. Useful for:
- Performance testing (measure detection-only speed)
- Debugging (isolate tracking issues)
- Simple applications that don't need persistent IDs
"""

from typing import List, Dict, Any
import logging

from common.base_classes import BaseTracker, Detection, Track


class NoTracker(BaseTracker):
    """
    Dummy tracker that assigns sequential IDs without actual tracking.
    
    Use cases:
    - Performance testing (measure detection overhead only)
    - Debugging (isolate tracking issues)
    - Simple applications where persistent IDs aren't needed
    - Baseline comparison for tracking algorithms
    
    Note: IDs will change every frame since there's no temporal association.
    
    Example:
        >>> tracker = NoTracker({})
        >>> tracks = tracker.update(detections)
        >>> # tracks[0].track_id will be different each frame
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize NoTracker.
        
        Args:
            config: Configuration dictionary (not used, for interface compatibility)
        """
        super().__init__(config)
        self._logger = logging.getLogger('NoTracker')
        self._logger.info("NoTracker created (no tracking will be performed)")
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Convert detections to tracks with sequential IDs.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            List of Track objects with sequential IDs (1, 2, 3, ...)
        """
        tracks = []
        for idx, det in enumerate(detections):
            track = Track(
                track_id=idx + 1,  # Sequential IDs starting from 1
                bbox=det.bbox,
                confidence=det.confidence,
                state='tracked'
            )
            tracks.append(track)
        
        self.increment_frame()
        self._logger.debug(f"Frame {self.frame_count}: {len(tracks)} detections (no tracking)")
        return tracks
    
    def reset(self) -> None:
        """Reset frame counter."""
        self.frame_count = 0
        self._logger.info("NoTracker reset")
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"NoTracker(frame_count={self.frame_count})"
