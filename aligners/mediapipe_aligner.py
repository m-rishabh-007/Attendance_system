"""
MediaPipe Face Mesh-based face alignment implementation.

Uses MediaPipe's 468-landmark face mesh model to perform robust face alignment.
Optimized for CPU inference (Raspberry Pi compatible).

Key Features:
- 468 facial landmarks (vs 68 for dlib)
- Real-time CPU performance (~40ms per face)
- Handles partial occlusions and head rotation
- No GPU required

Reference: https://google.github.io/mediapipe/solutions/face_mesh.html
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Optional, Tuple, List
import logging

from aligners.base_aligner import BaseAligner


logger = logging.getLogger(__name__)


class MediaPipeAligner(BaseAligner):
    """
    Face alignment using MediaPipe Face Mesh.
    
    Alignment Strategy:
    1. Extract 468 facial landmarks using MediaPipe Face Mesh
    2. Select 5 key points: left_eye, right_eye, nose, left_mouth, right_mouth
    3. Compute similarity transform to canonical positions
    4. Warp face to 112x112 output size
    
    MediaPipe Landmark Indices (for 5-point alignment):
    - Left Eye:   33  (left eye outer corner)
    - Right Eye:  263 (right eye outer corner)
    - Nose Tip:   1   (nose tip)
    - Left Mouth: 61  (left mouth corner)
    - Right Mouth: 291 (right mouth corner)
    """
    
    # Canonical 5-point positions for 112x112 face (ArcFace standard)
    CANONICAL_POINTS = np.array([
        [38.2946, 51.6963],  # Left eye
        [73.5318, 51.5014],  # Right eye
        [56.0252, 71.7366],  # Nose tip
        [41.5493, 92.3655],  # Left mouth
        [70.7299, 92.2041]   # Right mouth
    ], dtype=np.float32)
    
    # MediaPipe landmark indices for 5 key points
    LANDMARK_INDICES = {
        'left_eye': 33,
        'right_eye': 263,
        'nose': 1,
        'left_mouth': 61,
        'right_mouth': 291
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MediaPipe Face Mesh aligner.
        
        Args:
            config: Configuration dictionary with keys:
                - output_size: (width, height) of aligned face (default: 112x112)
                - min_detection_confidence: Minimum face detection confidence (default: 0.5)
                - min_tracking_confidence: Minimum landmark tracking confidence (default: 0.5)
                - refine_landmarks: Use attention mesh model for iris landmarks (default: False)
        """
        super().__init__(config)
        
        # MediaPipe Face Mesh configuration
        self.refine_landmarks = config.get('refine_landmarks', False)
        self.tracking_confidence = config.get('min_tracking_confidence', 0.5)
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh  # type: ignore
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,           # Video mode (faster)
            max_num_faces=10,                  # Support multiple faces
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        
        # Store last computed angle for debugging/visualization
        self.last_angle = None
        
        logger.info(f"Initialized {self}")
    
    def align(
        self, 
        frame: np.ndarray, 
        bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Align a single face using MediaPipe landmarks.
        
        Args:
            frame: Input frame (BGR format)
            bbox: Face bounding box (x1, y1, x2, y2)
            
        Returns:
            Aligned face crop (RGB, 112x112) or None if alignment fails
        """
        # Validate bbox
        if not self.validate_bbox(bbox, frame.shape[:3]):  # type: ignore
            logger.warning(f"Invalid bbox: {bbox} for frame shape {frame.shape}")
            return None
        
        # Extract face crop with padding
        x1, y1, x2, y2 = bbox
        crop, crop_bbox = self._extract_crop_with_padding(frame, bbox, padding=0.2)
        
        # Convert BGR → RGB for MediaPipe
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Run MediaPipe Face Mesh
        results = self.face_mesh.process(crop_rgb)
        
        if not results.multi_face_landmarks:
            logger.debug(f"No landmarks detected for bbox {bbox}")
            return None
        
        # Get first face (closest to bbox)
        landmarks = results.multi_face_landmarks[0]
        
        # Estimate face angle (for debugging and adaptive alignment)
        face_angle = self._estimate_yaw_angle(landmarks)
        self.last_angle = face_angle  # Store for external access (visualization)
        logger.debug(f"Face angle: {face_angle:.1f}° (bbox: {bbox})")
        
        # Extract 5 key points
        key_points = self._extract_5_key_points(landmarks, crop.shape[:3])  # type: ignore
        
        if key_points is None:
            logger.debug(f"Failed to extract 5 key points for bbox {bbox}")
            return None
        
        # Adjust key points to original frame coordinates
        cx1, cy1, _, _ = crop_bbox
        key_points[:, 0] += cx1
        key_points[:, 1] += cy1
        
        # Compute similarity transform
        transform_matrix = self._compute_similarity_transform(
            key_points, 
            self.CANONICAL_POINTS
        )
        
        # Warp face to canonical position
        aligned_face = cv2.warpAffine(
            frame, 
            transform_matrix, 
            self.output_size, 
            flags=cv2.INTER_LINEAR
        )
        
        # Convert BGR → RGB for recognition models (ArcFace expects RGB)
        aligned_face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        
        return aligned_face_rgb
    
    def align_batch(
        self, 
        frame: np.ndarray, 
        bboxes: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Align multiple faces in a single frame.
        
        Args:
            frame: Input frame (BGR format)
            bboxes: Array of bounding boxes (N, 4) where each row is (x1, y1, x2, y2)
            
        Returns:
            Dictionary mapping bbox_index → aligned_face_crop
        """
        aligned_faces = {}
        
        for idx, bbox in enumerate(bboxes):
            aligned_face = self.align(frame, tuple(bbox))
            if aligned_face is not None:
                aligned_faces[idx] = aligned_face
        
        return aligned_faces
    
    def get_landmarks(
        self, 
        frame: np.ndarray, 
        bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Extract all 468 MediaPipe landmarks for visualization.
        
        Args:
            frame: Input frame (BGR format)
            bbox: Face bounding box (x1, y1, x2, y2)
            
        Returns:
            Landmarks array (468, 2) or None if detection fails
        """
        # Validate bbox
        if not self.validate_bbox(bbox, frame.shape[:3]):  # type: ignore
            return None
        
        # Extract face crop
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        
        # Convert BGR → RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Run MediaPipe
        results = self.face_mesh.process(crop_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        # Convert normalized landmarks to pixel coordinates
        landmarks = results.multi_face_landmarks[0]
        h, w = crop.shape[:2]
        
        landmark_points = np.array([
            [lm.x * w + x1, lm.y * h + y1] 
            for lm in landmarks.landmark
        ], dtype=np.float32)
        
        return landmark_points
    
    def _extract_crop_with_padding(
        self, 
        frame: np.ndarray, 
        bbox: Tuple[int, int, int, int], 
        padding: float = 0.2
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extract face crop with padding to include more context.
        
        Args:
            frame: Input frame
            bbox: Original bounding box (x1, y1, x2, y2)
            padding: Padding ratio (e.g., 0.2 = 20% padding on each side)
            
        Returns:
            (cropped_image, padded_bbox)
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # Compute padding
        box_w = x2 - x1
        box_h = y2 - y1
        pad_w = int(box_w * padding)
        pad_h = int(box_h * padding)
        
        # Apply padding with bounds checking
        cx1 = max(0, x1 - pad_w)
        cy1 = max(0, y1 - pad_h)
        cx2 = min(w, x2 + pad_w)
        cy2 = min(h, y2 + pad_h)
        
        crop = frame[cy1:cy2, cx1:cx2]
        
        return crop, (cx1, cy1, cx2, cy2)
    
    def _extract_5_key_points(
        self, 
        landmarks, 
        crop_shape: Tuple[int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Extract 5 key facial points from 468 MediaPipe landmarks.
        
        Args:
            landmarks: MediaPipe face_landmarks object
            crop_shape: Shape of cropped face (height, width, channels)
            
        Returns:
            Key points array (5, 2) or None if extraction fails
        """
        h, w = crop_shape[:2]
        
        try:
            key_points = np.array([
                [landmarks.landmark[self.LANDMARK_INDICES['left_eye']].x * w,
                 landmarks.landmark[self.LANDMARK_INDICES['left_eye']].y * h],
                
                [landmarks.landmark[self.LANDMARK_INDICES['right_eye']].x * w,
                 landmarks.landmark[self.LANDMARK_INDICES['right_eye']].y * h],
                
                [landmarks.landmark[self.LANDMARK_INDICES['nose']].x * w,
                 landmarks.landmark[self.LANDMARK_INDICES['nose']].y * h],
                
                [landmarks.landmark[self.LANDMARK_INDICES['left_mouth']].x * w,
                 landmarks.landmark[self.LANDMARK_INDICES['left_mouth']].y * h],
                
                [landmarks.landmark[self.LANDMARK_INDICES['right_mouth']].x * w,
                 landmarks.landmark[self.LANDMARK_INDICES['right_mouth']].y * h]
            ], dtype=np.float32)
            
            return key_points
            
        except (IndexError, AttributeError) as e:
            logger.error(f"Failed to extract 5 key points: {e}")
            return None
    
    def _compute_similarity_transform(
        self, 
        src_points: np.ndarray, 
        dst_points: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity transformation matrix (rotation + scale + translation).
        
        Args:
            src_points: Source points (5, 2) - detected landmarks
            dst_points: Destination points (5, 2) - canonical positions
            
        Returns:
            Affine transformation matrix (2, 3)
        """
        # Use OpenCV's estimateAffinePartial2D for robust similarity transform
        # This handles rotation, scale, and translation (but not shear)
        transform_matrix, _ = cv2.estimateAffinePartial2D(
            src_points, 
            dst_points, 
            method=cv2.LMEDS  # Least Median of Squares (robust to outliers)
        )
        
        return transform_matrix
    
    def _estimate_yaw_angle(self, landmarks) -> float:
        """
        Estimate face yaw angle (left-right rotation) from landmarks.
        
        Uses nose-to-eye distance ratio to compute yaw:
        - Frontal face: left_eye-nose ≈ right_eye-nose (ratio ≈ 1.0)
        - Right profile: right_eye closer to nose (ratio > 1.3)
        - Left profile: left_eye closer to nose (ratio < 0.7)
        
        Args:
            landmarks: MediaPipe face_landmarks object
            
        Returns:
            Yaw angle in degrees:
            - 0°: Frontal face
            - +60° to +90°: Right profile
            - -60° to -90°: Left profile
            
        Example:
            >>> landmarks = face_mesh.process(image).multi_face_landmarks[0]
            >>> angle = self._estimate_yaw_angle(landmarks)
            >>> print(f"Face angle: {angle:.1f}°")
            Face angle: 5.2°
        """
        # Extract key landmarks for angle estimation
        nose = landmarks.landmark[self.LANDMARK_INDICES['nose']]
        left_eye = landmarks.landmark[self.LANDMARK_INDICES['left_eye']]
        right_eye = landmarks.landmark[self.LANDMARK_INDICES['right_eye']]
        
        # Compute Euclidean distances
        left_dist = np.sqrt(
            (nose.x - left_eye.x)**2 + 
            (nose.y - left_eye.y)**2
        )
        right_dist = np.sqrt(
            (nose.x - right_eye.x)**2 + 
            (nose.y - right_eye.y)**2
        )
        
        # Avoid division by zero
        if right_dist < 1e-6:
            logger.warning("Right eye-nose distance too small, assuming right profile")
            return 90.0
        
        # Compute ratio (indicator of face rotation)
        ratio = left_dist / right_dist
        
        # Empirical mapping: ratio → yaw angle
        # Calibrated based on MediaPipe landmark behavior
        if ratio < 0.7:
            # Left profile (left eye much closer)
            yaw = -60 - (0.7 - ratio) * 100  # Extrapolate beyond -60°
            yaw = max(yaw, -90)  # Clamp to -90°
        elif ratio > 1.3:
            # Right profile (right eye much closer)
            yaw = 60 + (ratio - 1.3) * 100  # Extrapolate beyond 60°
            yaw = min(yaw, 90)  # Clamp to 90°
        else:
            # Frontal to semi-profile (linear interpolation)
            yaw = (ratio - 1.0) * 60
        
        return float(yaw)
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
