"""
Base Recognizer Module - Abstract interface for face recognition models.

This module defines the abstract base class that all face recognition
implementations must inherit from. It follows the Strategy pattern,
allowing different recognition models (ArcFace, FaceNet, etc.) to be
swapped without changing the pipeline code.

IMPORTANT: This implementation supports BOTH FP32 and INT8 quantized models.
Week 1: FP32 baseline
Week 2: INT8 quantized (custom inference with onnxruntime)

Architecture:
    - Strategy Pattern: Allows swapping recognition algorithms
    - Type hints: Full Python 3.11+ type annotation support
    - Abstract methods: Enforces consistent interface across implementations

Usage:
    This is an abstract class - do not instantiate directly.
    Use RecognizerFactory to create concrete implementations.

Example:
    >>> from recognizers.factory import RecognizerFactory
    >>> recognizer = RecognizerFactory.create(config)
    >>> embedding = recognizer.get_embedding(aligned_face)

Author: Attendance System Team
Created: November 2025
Phase: 3A - Core Recognition
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import numpy as np
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


class BaseRecognizer(ABC):
    """
    Abstract base class for face recognition models.
    
    All face recognition implementations (ArcFace, FaceNet, etc.) must
    inherit from this class and implement the required methods.
    
    The recognizer takes an aligned face image (typically 112x112 from
    Phase 2 alignment) and produces a normalized embedding vector
    (typically 512 dimensions) that represents the face in feature space.
    
    Supports both FP32 (Week 1) and INT8 quantized models (Week 2).
    
    Attributes:
        model_path (Path): Path to the recognition model file
        embedding_size (int): Dimension of output embedding (e.g., 512)
        input_size (Tuple[int, int]): Expected input image size (H, W)
        is_loaded (bool): Whether model is successfully loaded
        quantized (bool): Whether model is INT8 quantized
    
    Design Patterns:
        - Strategy: Allows swapping recognition algorithms
        - Template Method: Defines common workflow (preprocess → infer → postprocess)
    """
    
    def __init__(
        self,
        model_path: str,
        embedding_size: int = 512,
        input_size: Tuple[int, int] = (112, 112),
        quantized: bool = False,
        **kwargs
    ):
        """
        Initialize base recognizer with common attributes.
        
        Args:
            model_path: Path to model file (ONNX, TFLite, etc.)
            embedding_size: Dimension of output embedding vector
            input_size: Expected input image dimensions (height, width)
            quantized: Whether model is INT8 quantized (Week 2)
            **kwargs: Additional model-specific parameters
        
        Raises:
            FileNotFoundError: If model_path does not exist
            ValueError: If embedding_size or input_size are invalid
        """
        self.model_path = Path(model_path)
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.quantized = quantized
        self.is_loaded = False
        self.config = kwargs
        
        # Validate inputs
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Please download the model to this location."
            )
        
        if embedding_size <= 0:
            raise ValueError(f"embedding_size must be positive, got {embedding_size}")
        
        if len(input_size) != 2 or any(s <= 0 for s in input_size):
            raise ValueError(f"input_size must be (H, W) with positive values, got {input_size}")
        
        logger.info(f"Initialized {self.__class__.__name__} (quantized={quantized})")
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Load the recognition model into memory.
        
        This method must:
        1. Load model file from self.model_path
        2. Initialize inference session/runtime
        3. Verify input/output shapes match expectations
        4. Set self.is_loaded = True on success
        
        For INT8 models (Week 2):
        - Use onnxruntime.InferenceSession directly
        - Configure CPUExecutionProvider
        - Verify quantized graph structure
        
        Raises:
            RuntimeError: If model loading fails
            ValueError: If model structure doesn't match expectations
        
        Implementation Example (FP32):
            >>> def load_model(self):
            >>>     import onnxruntime as ort
            >>>     self.session = ort.InferenceSession(
            >>>         str(self.model_path),
            >>>         providers=['CPUExecutionProvider']
            >>>     )
            >>>     # Verify shapes...
            >>>     self.is_loaded = True
        
        Implementation Example (INT8 - Week 2):
            >>> def load_model(self):
            >>>     import onnxruntime as ort
            >>>     # Load custom INT8 ONNX model
            >>>     self.session = ort.InferenceSession(
            >>>         str(self.model_path),  # e.g., auraface_int8.onnx
            >>>         providers=['CPUExecutionProvider']
            >>>     )
            >>>     # Verify quantized graph
            >>>     self.is_loaded = True
        """
        pass
    
    @abstractmethod
    def preprocess(self, face: np.ndarray) -> np.ndarray:
        """
        Preprocess aligned face for model inference.
        
        CRITICAL: Preprocessing must EXACTLY match the model's training.
        This is especially important for quantization (Week 2).
        
        Common preprocessing steps:
        1. Resize to model input size (if needed)
        2. Normalize pixel values (e.g., [0, 255] → [0, 1])
        3. Apply mean/std normalization (e.g., (x - 0.5) / 0.5)
        4. Transpose dimensions (HWC → CHW for ONNX)
        5. Add batch dimension (H, W, C) → (1, C, H, W)
        6. Convert data type (uint8 → float32)
        
        For INT8 models (Week 2):
        - Preprocessing must match calibration data
        - May need to convert to int8 input/output
        
        Args:
            face: Aligned face image from Phase 2 aligner
                  Shape: (H, W, 3) in RGB format
                  Dtype: uint8, values [0, 255]
        
        Returns:
            Preprocessed array ready for model inference
            Shape: Model-specific (e.g., (1, 3, 112, 112))
            Dtype: float32 (FP32) or int8 (INT8)
        
        Raises:
            ValueError: If face is invalid (wrong shape, type, or values)
        
        Implementation Example (ArcFace standard):
            >>> def preprocess(self, face):
            >>>     # Resize if needed
            >>>     if face.shape[:2] != self.input_size:
            >>>         face = cv2.resize(face, self.input_size[::-1])
            >>>     
            >>>     # Normalize to [0, 1]
            >>>     face = face.astype(np.float32) / 255.0
            >>>     
            >>>     # Apply mean/std normalization
            >>>     face = (face - 0.5) / 0.5  # Range: [-1, 1]
            >>>     
            >>>     # Transpose to CHW (ONNX format)
            >>>     face = np.transpose(face, (2, 0, 1))
            >>>     
            >>>     # Add batch dimension
            >>>     return np.expand_dims(face, axis=0)
        """
        pass
    
    @abstractmethod
    def get_embedding(self, face: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract normalized embedding vector from aligned face.
        
        This is the main public method that:
        1. Validates input
        2. Preprocesses face
        3. Runs model inference
        4. Normalizes embedding (L2 normalization)
        5. Returns embedding vector
        
        The embedding is L2-normalized so that cosine similarity
        can be computed efficiently as a dot product.
        
        Supports both FP32 (Week 1) and INT8 (Week 2) models.
        
        Args:
            face: Aligned face image from Phase 2 aligner
                  Shape: (H, W, 3) in RGB format
                  Dtype: uint8, values [0, 255]
        
        Returns:
            Normalized embedding vector if successful, None if failed
            Shape: (embedding_size,) e.g., (512,)
            Dtype: float32
            Properties: L2 norm = 1.0 (normalized)
        
        Raises:
            RuntimeError: If model not loaded (call load_model() first)
            ValueError: If face is invalid
        
        Example:
            >>> recognizer = AuraFaceRecognizer(model_path)
            >>> recognizer.load_model()
            >>> aligned_face = aligner.align(frame, bbox)  # From Phase 2
            >>> embedding = recognizer.get_embedding(aligned_face)
            >>> print(embedding.shape)  # (512,)
            >>> print(np.linalg.norm(embedding))  # 1.0 (normalized)
        
        Implementation Template:
            >>> def get_embedding(self, face):
            >>>     if not self.is_loaded:
            >>>         raise RuntimeError("Model not loaded")
            >>>     
            >>>     # Validate input
            >>>     if not self._validate_face(face):
            >>>         return None
            >>>     
            >>>     # Preprocess
            >>>     input_tensor = self.preprocess(face)
            >>>     
            >>>     # Inference
            >>>     embedding = self._run_inference(input_tensor)
            >>>     
            >>>     # L2 normalize
            >>>     embedding = self._normalize_embedding(embedding)
            >>>     
            >>>     return embedding
        """
        pass
    
    def _validate_face(self, face: np.ndarray) -> bool:
        """
        Validate input face array.
        
        Checks:
        1. Not None
        2. Non-empty
        3. 3D array (H, W, C)
        4. 3 channels (RGB)
        5. Valid data type (uint8)
        6. Valid value range [0, 255]
        
        Args:
            face: Face image to validate
        
        Returns:
            True if valid, False otherwise
        
        Note:
            This is a helper method - implementations can use it
            or implement custom validation.
        """
        if face is None or face.size == 0:
            logger.warning("Face is None or empty")
            return False
        
        if face.ndim != 3:
            logger.warning(f"Face must be 3D array, got {face.ndim}D")
            return False
        
        if face.shape[2] != 3:
            logger.warning(f"Face must have 3 channels (RGB), got {face.shape[2]}")
            return False
        
        if face.dtype != np.uint8:
            logger.warning(f"Face must be uint8, got {face.dtype}")
            return False
        
        if face.min() < 0 or face.max() > 255:
            logger.warning(f"Face values must be [0, 255], got [{face.min()}, {face.max()}]")
            return False
        
        return True
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        L2-normalize embedding vector.
        
        After normalization, the embedding has unit length (L2 norm = 1.0).
        This allows cosine similarity to be computed as a simple dot product:
        
            similarity = dot(emb1, emb2)  # Range: [-1, 1]
        
        Args:
            embedding: Raw embedding vector from model
        
        Returns:
            L2-normalized embedding vector
        
        Example:
            >>> raw_emb = np.array([3.0, 4.0])
            >>> norm_emb = self._normalize_embedding(raw_emb)
            >>> print(norm_emb)  # [0.6, 0.8]
            >>> print(np.linalg.norm(norm_emb))  # 1.0
        """
        norm = np.linalg.norm(embedding)
        if norm == 0:
            logger.warning("Embedding norm is zero, returning unnormalized")
            return embedding  # Avoid division by zero
        return embedding / norm
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model configuration and runtime information.
        
        Returns:
            Dictionary with model metadata:
            - model_path: Path to model file
            - embedding_size: Embedding dimension
            - input_size: Input image size
            - quantized: Whether model is INT8
            - is_loaded: Whether model is loaded
            - backend: Inference backend (e.g., 'onnxruntime')
        """
        return {
            'model_path': str(self.model_path),
            'embedding_size': self.embedding_size,
            'input_size': self.input_size,
            'quantized': self.quantized,
            'is_loaded': self.is_loaded,
            'backend': self.__class__.__name__
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"model_path='{self.model_path.name}', "
            f"embedding_size={self.embedding_size}, "
            f"input_size={self.input_size}, "
            f"quantized={self.quantized}, "
            f"is_loaded={self.is_loaded})"
        )
