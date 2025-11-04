"""
YOLO Detector Implementation

Uses Ultralytics YOLO API with TFLite model for face detection.
Provides high-level interface with built-in preprocessing and NMS.

Model: YOLOv8n INT8 TFLite (1.5MB)
Performance: ~40-50ms inference on Raspberry Pi, ~20-30ms on laptop
"""

from typing import List, Dict, Any
from pathlib import Path
import numpy as np
import logging

from common.base_classes import BaseFaceDetector, Detection

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logging.warning("Ultralytics not available. YOLODetector will not work.")


class YOLODetector(BaseFaceDetector):
    """
    YOLO-based face detector using Ultralytics API.
    
    Advantages:
    - Battle-tested library (millions of users)
    - Built-in preprocessing and NMS
    - Easy to use high-level API
    - Active development and bug fixes
    
    Disadvantages:
    - Heavier dependencies than raw TFLite
    - Slightly slower than direct TFLite inference
    
    Use When:
    - Need reliability over raw speed
    - Want easy integration
    - Acceptable to have larger dependency footprint
    
    Example:
        >>> config = {
        ...     'model_path': 'models/yolov8n_face_int8.tflite',
        ...     'confidence_threshold': 0.5,
        ...     'iou_threshold': 0.3,
        ...     'input_size': 256
        ... }
        >>> detector = YOLODetector(config)
        >>> detector.initialize()
        >>> detections = detector.detect(frame)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize YOLO detector.
        
        Args:
            config: Configuration dictionary containing:
                - model_path: Path to YOLO TFLite model
                - confidence_threshold: Minimum detection confidence (default: 0.5)
                - iou_threshold: NMS IoU threshold (default: 0.3)
                - input_size: Model input size (default: 256)
        """
        super().__init__(config)
        
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "Ultralytics not installed. Install with: "
                "pip install ultralytics"
            )
        
        self._logger = logging.getLogger('YOLODetector')
        self.model = None
        
        # Configuration
        self.model_path = Path(config['model_path'])
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.iou_threshold = config.get('iou_threshold', 0.3)
        self.input_size = config.get('input_size', 256)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self._logger.info(
            f"YOLODetector created (conf={self.confidence_threshold}, "
            f"iou={self.iou_threshold}, size={self.input_size})"
        )
    
    def initialize(self) -> None:
        """
        Load YOLO model.
        
        Called once before first detection to avoid redundant loading.
        """
        if self._is_initialized:
            self._logger.warning("Already initialized. Skipping.")
            return
        
        self._logger.info(f"Loading YOLO model from {self.model_path}")
        
        try:
            self.model = YOLO(self.model_path, task='detect')
            self._is_initialized = True
            self._logger.info("YOLO model loaded successfully")
        except Exception as e:
            self._logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect faces in frame using YOLO.
        
        ⚠️ NOTE: This method does NOT include tracking.
        For stable track IDs, use the integrated tracker via model.track()
        (see YOLODetector.detect_and_track() or use proper tracker integration).
        
        Args:
            frame: Input image as numpy array (H, W, 3) in BGR format
            
        Returns:
            List of Detection objects
            
        Raises:
            RuntimeError: If detector not initialized
        """
        if not self._is_initialized:
            raise RuntimeError("Detector not initialized. Call initialize() first.")
        
        # Run YOLO inference (detection only, no tracking persistence)
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            verbose=False  # Suppress per-frame output
        )
        
        # Convert YOLO results to Detection objects
        detections = []
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # YOLO returns xyxy format, convert to tlwh
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    
                    # Convert to tlwh (top, left, width, height)
                    bbox = np.array([x1, y1, x2 - x1, y2 - y1])
                    confidence = float(box.conf[0])
                    
                    detection = Detection(
                        bbox=bbox,
                        confidence=confidence,
                        class_id=0  # Single class: face
                    )
                    detections.append(detection)
        
        self._logger.debug(f"Detected {len(detections)} faces")
        return detections
    
    def detect_and_track(self, frame: np.ndarray, tracker_config: str = 'botsort.yaml') -> tuple:
        """
        Detect AND track faces using YOLO's built-in tracker (PROPER WAY).
        
        This is the correct method for stable track IDs! Uses model.track()
        with persist=True to maintain tracking state between frames.
        
        Args:
            frame: Input image as numpy array (H, W, 3) in BGR format
            tracker_config: Tracker config file ('botsort.yaml' or 'bytetrack.yaml')
            
        Returns:
            Tuple of (detections, tracks) where tracks include track_ids
            
        Raises:
            RuntimeError: If detector not initialized
        """
        if not self._is_initialized:
            raise RuntimeError("Detector not initialized. Call initialize() first.")
        
        # Run YOLO with tracking (THIS IS THE KEY FIX!)
        results = self.model.track(
            frame,
            persist=True,  # ✅ CRITICAL: Maintains track IDs across frames!
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            verbose=False,
            tracker=tracker_config  # BoT-SORT or ByteTrack
        )
        
        # Convert results to Detection objects (with track IDs if available)
        detections = []
        tracks = []
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            
            if boxes is not None and len(boxes) > 0:
                # Check if we have track IDs
                has_track_ids = boxes.id is not None
                
                for i, box in enumerate(boxes):
                    # YOLO returns xyxy format, convert to tlwh
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    bbox = np.array([x1, y1, x2 - x1, y2 - y1])
                    confidence = float(box.conf[0])
                    
                    detection = Detection(
                        bbox=bbox,
                        confidence=confidence,
                        class_id=0
                    )
                    detections.append(detection)
                    
                    # Add track info if available
                    if has_track_ids:
                        from common.base_classes import Track
                        track_id = int(boxes.id[i].item())
                        track = Track(
                            track_id=track_id,
                            bbox=bbox,
                            confidence=confidence,
                            state='tracked',
                            frames_since_update=0
                        )
                        tracks.append(track)
        
        self._logger.debug(f"Detected {len(detections)} faces, tracked {len(tracks)} faces")
        return detections, tracks
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"YOLODetector(model={self.model_path.name}, "
            f"conf={self.confidence_threshold}, initialized={self._is_initialized})"
        )
