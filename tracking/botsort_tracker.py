"""
BoT-SORT Tracker Implementation - FALLBACK WRAPPER ONLY

⚠️ CRITICAL UNDERSTANDING: In V3 HYBRID, this is a FALLBACK wrapper!

NORMAL OPERATION (V3 HYBRID):
    BoT-SORT runs INSIDE Ultralytics when you call:
        model.track(persist=True, tracker='botsort.yaml')
    
    This provides:
    ✅ Persistent Track IDs (same ID across frames)
    ✅ Kalman filter state maintenance
    ✅ ReID features (appearance similarity)
    ✅ Excellent tracking quality

FALLBACK OPERATION (This Class):
    This wrapper is used ONLY when:
    1. Using TFLiteDetector (no integrated tracking)
    2. Someone incorrectly uses: detect() + update()
    3. Unit testing without full pipeline
    
    This provides:
    ❌ NO persistent Track IDs (just sequential IDs)
    ❌ NO Kalman filter state
    ❌ NO ReID features
    ❌ Poor tracking quality (IDs fluctuate)

EXECUTION FLOW COMPARISON:

Normal (Persistent IDs):
    orchestrator.process_frame(frame)
        ↓
    detection_stage.process_with_tracking(frame, 'botsort.yaml')
        ↓
    yolo_detector.detect_and_track(frame, 'botsort.yaml')
        ↓
    model.track(persist=True, tracker='botsort.yaml')
        ↓ BoT-SORT runs HERE inside Ultralytics!
    Returns: (detections, tracks) with stable IDs ✅

Fallback (Fluctuating IDs):
    orchestrator.process_frame(frame)
        ↓
    detection_stage.process(frame)
        ↓
    tflite_detector.detect(frame)
        ↓
    tracking_stage.process(detections)
        ↓
    botsort_tracker.update(detections)  ← THIS CLASS runs here
        ↓
    Returns: tracks with unstable IDs ❌

For more details, see:
- tracking/README.md - Complete execution flow documentation
- detectors/README.md - Integrated tracking explanation
- docs/DEVELOPER_GUIDE.md - Architecture overview

Advantages of Real BoT-SORT (Ultralytics):
- Superior track persistence (maintains IDs through occlusions)
- Handles motion blur and camera movement
- Battle-tested by millions of users
- Active development and updates

Performance: Excellent ID persistence, handles head shaking well
"""

from typing import List, Dict, Any
import numpy as np
import logging

from common.base_classes import BaseTracker, Detection, Track

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logging.warning("Ultralytics not available. BotSORTTracker will not work.")


class BotSORTTracker(BaseTracker):
    """
    BoT-SORT tracker using Ultralytics implementation.
    
    BoT-SORT (Bag of Tricks + SORT) combines:
    - Motion prediction (Kalman filter)
    - Appearance features (Re-identification model)
    - Camera motion compensation
    
    Best for:
    - Production deployments requiring reliability
    - Scenarios with occlusions and motion blur
    - When track persistence is critical
    
    Note: This is a simplified wrapper. Full BoT-SORT requires
    the model to be loaded first (handled by detector).
    
    Example:
        >>> config = {
        ...     'track_thresh': 0.4,
        ...     'track_buffer': 90,
        ...     'match_thresh': 0.4
        ... }
        >>> tracker = BotSORTTracker(config)
        >>> tracks = tracker.update(detections)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize BoT-SORT tracker.
        
        Args:
            config: Configuration dictionary containing:
                - track_thresh: Minimum confidence to start tracking (default: 0.4)
                - track_buffer: Frames to keep lost tracks (default: 90)
                - match_thresh: IoU threshold for matching (default: 0.4)
        """
        super().__init__(config)
        
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "Ultralytics not installed. Install with: pip install ultralytics"
            )
        
        self._logger = logging.getLogger('BotSORTTracker')
        
        # Configuration
        self.track_thresh = config.get('track_thresh', 0.4)
        self.track_buffer = config.get('track_buffer', 90)
        self.match_thresh = config.get('match_thresh', 0.4)
        
        # Note: Actual BoT-SORT tracker is created and managed by Ultralytics YOLO
        # when track() is called. This class provides the unified interface.
        
        self._logger.info(
            f"BotSORTTracker created (track_thresh={self.track_thresh}, "
            f"buffer={self.track_buffer}, match_thresh={self.match_thresh})"
        )
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracker with new detections.
        
        ⚠️ WARNING: This is a FALLBACK implementation with NO tracking persistence!
        Track IDs will fluctuate because this doesn't maintain state between frames.
        
        For stable track IDs, use YOLO's built-in tracking:
            results = model.track(frame, persist=True, tracker='botsort.yaml')
        
        This is handled automatically when using:
            detector.detect_and_track(frame)  # ✅ Correct way
        
        Instead of:
            detections = detector.detect(frame)  # ❌ No tracking state
            tracks = tracker.update(detections)  # ❌ Sequential IDs only
        
        Args:
            detections: List of Detection objects
            
        Returns:
            List of Track objects with sequential IDs (NOT persistent!)
        """
        self._logger.warning(
            "⚠️ BotSORTTracker.update() called directly - Track IDs will fluctuate! "
            "For stable IDs, use YOLO.track() with persist=True via detect_and_track()."
        )
        
        # Convert detections to tracks with sequential IDs
        # (This is a fallback; real tracking happens in YOLO.track())
        tracks = []
        for idx, det in enumerate(detections):
            track = Track(
                track_id=idx + 1,
                bbox=det.bbox,
                confidence=det.confidence,
                state='tracked'
            )
            tracks.append(track)
        
        self.increment_frame()
        return tracks
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.frame_count = 0
        self._logger.info("BotSORTTracker reset")
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"BotSORTTracker(track_thresh={self.track_thresh}, "
            f"buffer={self.track_buffer}, frame_count={self.frame_count})"
        )
