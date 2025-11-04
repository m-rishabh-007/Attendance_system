"""
Pipeline Orchestrator - Main coordinator for all pipeline stages.

This is the "conductor" that coordinates the entire face attendance pipeline:
1. Detection Stage
2. Tracking Stage
3. Alignment Stage (future)
4. Recognition Stage (future)
5. Attendance Stage (future)

Author: Rishabh Mishra
Date: November 4, 2025
Version: 3.0.0
"""

import logging
import time
from typing import Dict, Any, List, Optional
import numpy as np

from pipeline.detection_stage import DetectionStage
from pipeline.tracking_stage import TrackingStage


class PipelineOrchestrator:
    """
    Orchestrates all pipeline stages for face attendance processing.
    
    This class is the heart of the V3 HYBRID architecture - it separates
    "what to do" (orchestration) from "how to do it" (implementations).
    
    Design Principles:
    1. Single Responsibility: Only coordinates stages, doesn't implement them
    2. Open/Closed: Can add new stages without modifying existing code
    3. Dependency Inversion: Depends on stage interfaces, not concrete classes
    
    Stages:
        - Detection: Detect faces in frame
        - Tracking: Maintain persistent track IDs
        - Alignment: (future) Align faces for recognition
        - Recognition: (future) Identify persons
        - Attendance: (future) Mark attendance records
    
    Example:
        >>> config = ConfigManager.get_instance()
        >>> orchestrator = PipelineOrchestrator(config)
        >>> result = orchestrator.process_frame(frame)
        >>> tracks = result['tracks']
        >>> annotated_frame = result['annotated_frame']
    """
    
    def __init__(self, config):
        """
        Initialize pipeline orchestrator.
        
        Args:
            config: ConfigManager instance with full configuration
        """
        self.logger = logging.getLogger('PipelineOrchestrator')
        self.config = config
        
        self.logger.info("="*60)
        self.logger.info("Initializing Pipeline Orchestrator (V3 HYBRID)")
        self.logger.info("="*60)
        
        # Initialize Stage 1: Detection
        self.logger.info("Stage 1: Detection")
        detector_config = config.get_section('detector')
        self.detection_stage = DetectionStage(detector_config)
        
        # Initialize Stage 2: Tracking
        self.logger.info("Stage 2: Tracking")
        tracker_config = config.get_section('tracker')
        self.tracking_stage = TrackingStage(tracker_config)
        
        # Future stages (conditionally initialized)
        self.alignment_stage = None
        self.recognition_stage = None
        self.attendance_stage = None
        
        # Check if future stages are enabled
        if config.get('alignment.enabled', False):
            self.logger.info("Stage 3: Alignment - ENABLED (future)")
            # TODO: self.alignment_stage = AlignmentStage(config.get_section('alignment'))
        
        if config.get('recognition.enabled', False):
            self.logger.info("Stage 4: Recognition - ENABLED (future)")
            # TODO: self.recognition_stage = RecognitionStage(config.get_section('recognition'))
        
        if config.get('attendance.enabled', False):
            self.logger.info("Stage 5: Attendance - ENABLED (future)")
            # TODO: self.attendance_stage = AttendanceStage(config.get_section('attendance'))
        
        self.logger.info("="*60)
        self.logger.info("✅ Pipeline Orchestrator Ready")
        self.logger.info("="*60)
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process single frame through all pipeline stages.
        
        This is the main entry point - it runs all enabled stages
        in sequence and returns comprehensive results.
        
        Args:
            frame: Input frame (BGR format, numpy array)
        
        Returns:
            Dictionary containing:
                - detections: Raw detection results
                - tracks: Track objects with persistent IDs
                - aligned_faces: (future) Aligned face crops
                - identities: (future) Recognized person IDs
                - annotated_frame: Frame with visualizations
                - processing_time_ms: Total processing time
        """
        start_time = time.time()
        
        # Stage 1 + 2: Detection with integrated tracking (RECOMMENDED for stable IDs)
        # This uses YOLO's model.track(persist=True) which maintains tracking state
        detections, tracks = self.detection_stage.process_with_tracking(
            frame,
            tracker_config=self.config.get('tracker.type', 'botsort') + '.yaml'
        )
        
        # Note: Separate tracking stage (below) is kept for compatibility with
        # detectors that don't support integrated tracking. If detect_and_track()
        # fails, we fall back to: detection → separate tracking
        # 
        # For YOLO detector: Uses model.track(persist=True) ✅
        # For other detectors: Falls back to separate tracking (may have ID fluctuations)
        
        # Stage 3: Alignment (if enabled)
        aligned_faces = None
        if self.alignment_stage is not None:
            aligned_faces = self.alignment_stage.process(frame, tracks)
        
        # Stage 4: Recognition (if enabled)
        identities = None
        if self.recognition_stage is not None and aligned_faces is not None:
            identities = self.recognition_stage.process(aligned_faces)
        
        # Stage 5: Attendance (if enabled)
        if self.attendance_stage is not None and identities is not None:
            self.attendance_stage.process(identities)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Prepare result
        result = {
            'detections': detections,
            'tracks': tracks,
            'aligned_faces': aligned_faces,
            'identities': identities,
            'annotated_frame': self._annotate_frame(frame, tracks, identities),
            'processing_time_ms': processing_time_ms
        }
        
        return result
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        tracks: List[Any],
        identities: Optional[List[Any]] = None
    ) -> np.ndarray:
        """
        Draw annotations on frame.
        
        Args:
            frame: Input frame (modified in-place)
            tracks: List of Track objects
            identities: Optional list of identified persons
        
        Returns:
            Annotated frame
        """
        import cv2
        
        # Get display config
        bbox_color = tuple(self.config.get('display.bbox_color', [0, 255, 0]))
        text_color = tuple(self.config.get('display.text_color', [0, 255, 0]))
        font_scale = self.config.get('display.font_scale', 0.6)
        thickness = self.config.get('display.thickness', 2)
        
        # Draw tracks
        for i, track in enumerate(tracks):
            # Get bounding box (Track objects have bbox attribute, not subscriptable)
            if hasattr(track, 'to_tlwh'):
                x, y, w, h = track.to_tlwh()
            elif hasattr(track, 'bbox'):
                # Track object from common/base_classes.py
                bbox = track.bbox
                if hasattr(bbox, '__iter__'):
                    x, y, w, h = bbox
                else:
                    # bbox is numpy array
                    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            else:
                # Fallback: dictionary format
                x, y, w, h = track['bbox']
            
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Get track ID and confidence
            track_id = track.track_id if hasattr(track, 'track_id') else i
            confidence = track.confidence if hasattr(track, 'confidence') else 0.0
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), bbox_color, thickness)
            
            # Draw label with ID and confidence
            label = f"ID: {track_id} ({confidence:.2f})"
            if identities and i < len(identities):
                label = f"{identities[i]} (ID: {track_id}, {confidence:.2f})"
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w, y), bbox_color, -1)
            
            # Text
            cv2.putText(frame, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        return frame
    
    def __repr__(self) -> str:
        stages = [
            "detection",
            "tracking",
        ]
        if self.alignment_stage:
            stages.append("alignment")
        if self.recognition_stage:
            stages.append("recognition")
        if self.attendance_stage:
            stages.append("attendance")
        
        return f"PipelineOrchestrator(stages={stages})"
