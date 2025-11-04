"""
Base classes and interfaces for Face Attendance System components.

Defines abstract base classes (interfaces) that all implementations must follow.
This ensures consistency and allows easy component swapping.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class Detection:
    """
    Standard detection result format.
    
    All detectors must return detections in this format for consistency.
    
    Attributes:
        bbox: Bounding box in tlwh format [top, left, width, height]
        confidence: Detection confidence score [0.0-1.0]
        class_id: Class ID (0 for face in our case)
        landmarks: Optional facial landmarks (future use)
    """
    bbox: np.ndarray  # Shape: (4,) - [x, y, w, h] in tlwh format
    confidence: float
    class_id: int = 0
    landmarks: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary."""
        return {
            'bbox': self.bbox.tolist(),
            'confidence': float(self.confidence),
            'class_id': int(self.class_id),
            'landmarks': self.landmarks.tolist() if self.landmarks is not None else None
        }


@dataclass
class Track:
    """
    Standard tracked object format.
    
    Represents a tracked face across multiple frames.
    
    Attributes:
        track_id: Unique persistent ID for this track
        bbox: Current bounding box in tlwh format
        confidence: Current detection confidence
        state: Track state ('tracked', 'lost', 'deleted')
        frames_since_update: Frames since last detection (for lost tracks)
    """
    track_id: int
    bbox: np.ndarray  # Shape: (4,) - [x, y, w, h] in tlwh format
    confidence: float
    state: str = 'tracked'  # 'tracked', 'lost', 'deleted'
    frames_since_update: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert track to dictionary."""
        return {
            'track_id': int(self.track_id),
            'bbox': self.bbox.tolist(),
            'confidence': float(self.confidence),
            'state': self.state,
            'frames_since_update': self.frames_since_update
        }


class BaseFaceDetector(ABC):
    """
    Abstract base class for all face detectors.
    
    All detector implementations (YOLO, MTCNN, RetinaFace, etc.) must
    inherit from this class and implement the detect() method.
    
    This ensures:
    - Consistent interface across different detectors
    - Easy swapping of detection models
    - Type safety and clear contracts
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize detector with configuration.
        
        Args:
            config: Detector-specific configuration dictionary
        """
        self.config = config
        self._is_initialized = False
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect faces in a frame.
        
        Args:
            frame: Input image as numpy array (H, W, 3) in BGR format
            
        Returns:
            List of Detection objects
            
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize detector (load model, allocate resources).
        
        Called once before first detection. Allows lazy initialization.
        """
        pass
    
    def is_initialized(self) -> bool:
        """Check if detector has been initialized."""
        return self._is_initialized
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.__class__.__name__}(initialized={self._is_initialized})"


class BaseTracker(ABC):
    """
    Abstract base class for all object trackers.
    
    All tracker implementations (ByteTrack, BoT-SORT, DeepSORT, etc.)
    must inherit from this class and implement the update() method.
    
    This allows easy swapping of tracking algorithms via Strategy pattern.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize tracker with configuration.
        
        Args:
            config: Tracker-specific configuration dictionary
        """
        self.config = config
        self.frame_count = 0
    
    @abstractmethod
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of Detection objects from current frame
            
        Returns:
            List of Track objects (tracked faces with persistent IDs)
            
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset tracker state (clear all tracks).
        
        Useful when starting new video stream or after long pause.
        """
        pass
    
    def increment_frame(self) -> None:
        """Increment internal frame counter."""
        self.frame_count += 1
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.__class__.__name__}(frame_count={self.frame_count})"
