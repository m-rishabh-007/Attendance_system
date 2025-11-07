"""
Base aligner interface for face alignment implementations.

All alignment methods must inherit from this abstract base class to ensure
consistent API across different alignment strategies (MediaPipe, dlib, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np


class BaseAligner(ABC):
    """
    Abstract base class for face alignment implementations.
    
    Alignment converts arbitrary face crops to normalized, canonical poses
    suitable for face recognition. This typically involves:
    1. Detecting facial landmarks (eyes, nose, mouth)
    2. Computing affine transformation to canonical position
    3. Warping face to fixed size (e.g., 112x112)
    
    Design Pattern: Strategy Pattern
    - Allows swapping alignment algorithms (MediaPipe, dlib, MTCNN)
    - Client code works with BaseAligner interface
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize aligner with configuration.
        
        Args:
            config: Aligner configuration dictionary containing:
                - output_size: (width, height) of aligned face
                - min_detection_confidence: Minimum confidence for landmarks
                - model_path: Path to alignment model (if applicable)
        """
        self.config = config
        self.output_size = config.get('output_size', (112, 112))
        self.min_confidence = config.get('min_detection_confidence', 0.5)
        
    @abstractmethod
    def align(
        self, 
        frame: np.ndarray, 
        bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Align a face crop to canonical position.
        
        Args:
            frame: Full input frame (BGR format)
            bbox: Face bounding box (x1, y1, x2, y2)
            
        Returns:
            Aligned face crop (RGB, normalized to output_size) or None if alignment fails
            
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass
    
    @abstractmethod
    def align_batch(
        self, 
        frame: np.ndarray, 
        bboxes: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Align multiple faces in a single frame (batch processing).
        
        Args:
            frame: Full input frame (BGR format)
            bboxes: Array of bounding boxes (N, 4) where each row is (x1, y1, x2, y2)
            
        Returns:
            Dictionary mapping bbox_index → aligned_face_crop
            Faces that fail alignment are excluded from output
            
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass
    
    @abstractmethod
    def get_landmarks(
        self, 
        frame: np.ndarray, 
        bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Extract facial landmarks without performing alignment.
        
        Useful for visualization or custom alignment strategies.
        
        Args:
            frame: Full input frame (BGR format)
            bbox: Face bounding box (x1, y1, x2, y2)
            
        Returns:
            Facial landmarks array (N, 2) or None if detection fails
            Shape depends on model: MediaPipe=468 points, dlib=68 points
            
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass
    
    def validate_bbox(
        self, 
        bbox: Tuple[int, int, int, int], 
        frame_shape: Tuple[int, int, int]
    ) -> bool:
        """
        Validate bounding box is within frame bounds and has positive area.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            frame_shape: Frame dimensions (height, width, channels)
            
        Returns:
            True if bbox is valid, False otherwise
        """
        x1, y1, x2, y2 = bbox
        h, w = frame_shape[:2]
        
        # Check bounds
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return False
        
        # Check positive area
        if x2 <= x1 or y2 <= y1:
            return False
        
        return True
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"{self.__class__.__name__}("
                f"output_size={self.output_size}, "
                f"min_confidence={self.min_confidence})")
