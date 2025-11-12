"""
AuraFace Recognizer - ResNet100 Face Recognition Implementation

This module implements face recognition using the AuraFace ResNet100 model.
AuraFace is trained on Asian-Celeb dataset and achieves 99.83% accuracy on LFW.

Key Features:
- Apache 2.0 license (commercial use allowed)
- 512-dimensional L2-normalized embeddings
- Supports both FP32 and INT8 quantized models
- ONNX Runtime backend for cross-platform compatibility

Architecture:
- Inherits from BaseRecognizer (Strategy pattern)
- Uses ONNX Runtime for inference
- Preprocesses faces according to ArcFace standard
- Returns L2-normalized embeddings for cosine similarity

Usage:
    >>> from recognizers import AuraFaceRecognizer
    >>> recognizer = AuraFaceRecognizer(
    ...     model_path='models/recognition/auraface_resnet100_fp32.onnx',
    ...     embedding_size=512,
    ...     input_size=(112, 112),
    ...     quantized=False
    ... )
    >>> recognizer.load_model()
    >>> embedding = recognizer.get_embedding(aligned_face)
    >>> print(embedding.shape)  # (512,)

Author: Attendance System Team
Created: November 2025
Phase: 3A - Core Recognition (Week 1)
"""

from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING
import numpy as np
import cv2
import logging
from pathlib import Path

if TYPE_CHECKING:
    import onnxruntime as ort

try:
    import onnxruntime as ort  # type: ignore
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("onnxruntime not installed. Run: pip install onnxruntime")

from recognizers.base_recognizer import BaseRecognizer


logger = logging.getLogger(__name__)


class AuraFaceRecognizer(BaseRecognizer):
    """
    AuraFace ResNet100 face recognition implementation.
    
    This class implements the AuraFace face recognition model using ONNX Runtime.
    It supports both FP32 (Week 1) and INT8 quantized models (Week 2).
    
    The model takes a 112x112 aligned face image and produces a 512-dimensional
    L2-normalized embedding vector suitable for cosine similarity matching.
    
    Model Details:
        - Architecture: ResNet100 backbone
        - Training Data: Asian-Celeb dataset (includes South Asian)
        - Accuracy: 99.83% on LFW benchmark
        - License: Apache 2.0 (commercial use allowed)
        - Input: (1, 3, 112, 112) NCHW format, RGB, [-1, 1] range
        - Output: (1, 512) embedding vector, L2-normalized
    
    Performance (Raspberry Pi 4):
        - FP32: 80-100ms per face
        - INT8: 40-50ms per face (Week 2)
    
    Attributes:
        session (ort.InferenceSession): ONNX Runtime inference session
        input_name (str): Model input tensor name
        output_name (str): Model output tensor name
        num_threads (int): Number of CPU threads for inference
    
    Example:
        >>> recognizer = AuraFaceRecognizer(
        ...     model_path='models/recognition/auraface_resnet100_fp32.onnx',
        ...     num_threads=4
        ... )
        >>> recognizer.load_model()
        >>> 
        >>> # Get embedding from aligned face
        >>> aligned_face = aligner.align(frame, bbox)  # From Phase 2
        >>> embedding = recognizer.get_embedding(aligned_face)
        >>> 
        >>> # Compare two faces
        >>> emb1 = recognizer.get_embedding(face1)
        >>> emb2 = recognizer.get_embedding(face2)
        >>> similarity = np.dot(emb1, emb2)  # Cosine similarity
        >>> is_same_person = similarity > 0.6
    """
    
    def __init__(
        self,
        model_path: str,
        embedding_size: int = 512,
        input_size: Tuple[int, int] = (112, 112),
        quantized: bool = False,
        num_threads: int = 4,
        **kwargs
    ):
        """
        Initialize AuraFace recognizer.
        
        Args:
            model_path: Path to ONNX model file
            embedding_size: Embedding dimension (default: 512)
            input_size: Input image size (height, width) (default: 112x112)
            quantized: Whether model is INT8 quantized (default: False)
            num_threads: CPU threads for inference (default: 4)
            **kwargs: Additional arguments passed to BaseRecognizer
        
        Raises:
            ImportError: If onnxruntime is not installed
            FileNotFoundError: If model file does not exist
        """
        if not ONNX_AVAILABLE:
            raise ImportError(
                "onnxruntime is required for AuraFaceRecognizer.\n"
                "Install with: pip install onnxruntime"
            )
        
        super().__init__(
            model_path=model_path,
            embedding_size=embedding_size,
            input_size=input_size,
            quantized=quantized,
            **kwargs
        )
        
        self.num_threads = num_threads
        self.session: Optional['ort.InferenceSession'] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        
        logger.info(
            f"Initialized AuraFaceRecognizer: "
            f"model={self.model_path.name}, "
            f"quantized={quantized}, "
            f"threads={num_threads}"
        )
    
    def load_model(self) -> None:
        """
        Load AuraFace ONNX model into memory.
        
        This method:
        1. Creates ONNX Runtime session with CPU provider
        2. Configures number of threads
        3. Verifies input/output shapes match expectations
        4. Sets self.is_loaded = True on success
        
        Supports both FP32 and INT8 quantized models automatically.
        ONNX Runtime handles quantized inference transparently.
        
        Raises:
            RuntimeError: If model loading fails
            ValueError: If model structure doesn't match expectations
        
        Example:
            >>> recognizer = AuraFaceRecognizer(model_path)
            >>> recognizer.load_model()
            >>> print(recognizer.is_loaded)  # True
        """
        try:
            # Configure ONNX Runtime session options
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self.num_threads
            sess_options.inter_op_num_threads = self.num_threads
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Create inference session
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )
            
            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            # Verify input shape
            input_shape = self.session.get_inputs()[0].shape
            expected_shape = [1, 3, self.input_size[0], self.input_size[1]]
            
            # Handle dynamic batch dimension (-1)
            if input_shape[0] == -1 or input_shape[0] == 'batch':
                input_shape[0] = 1
            
            if list(input_shape) != expected_shape:
                logger.warning(
                    f"Input shape mismatch: expected {expected_shape}, "
                    f"got {input_shape}. Model may still work."
                )
            
            # Verify output shape
            output_shape = self.session.get_outputs()[0].shape
            expected_output = [1, self.embedding_size]
            
            if output_shape[0] == -1 or output_shape[0] == 'batch':
                output_shape[0] = 1
            
            if list(output_shape) != expected_output:
                logger.warning(
                    f"Output shape mismatch: expected {expected_output}, "
                    f"got {output_shape}. Embedding size may differ."
                )
            
            self.is_loaded = True
            
            logger.info(
                f"✅ Model loaded successfully: "
                f"input={input_shape}, output={output_shape}, "
                f"quantized={self.quantized}"
            )
            
        except Exception as e:
            self.is_loaded = False
            raise RuntimeError(f"Failed to load ONNX model: {e}") from e
    
    def preprocess(self, face: np.ndarray) -> np.ndarray:
        """
        Preprocess aligned face for AuraFace model inference.
        
        Preprocessing steps (ArcFace standard):
        1. Resize to 112x112 (if needed)
        2. Convert to float32 and normalize to [0, 1]
        3. Apply mean/std normalization: (x - 0.5) / 0.5 → [-1, 1]
        4. Transpose from HWC to CHW format (ONNX/PyTorch convention)
        5. Add batch dimension: (3, 112, 112) → (1, 3, 112, 112)
        
        CRITICAL: This preprocessing MUST match the model's training.
        Any deviation will cause significant accuracy drop!
        
        Args:
            face: Aligned face image from Phase 2 aligner
                  Shape: (H, W, 3) in RGB format
                  Dtype: uint8, values [0, 255]
        
        Returns:
            Preprocessed tensor ready for ONNX inference
            Shape: (1, 3, 112, 112)
            Dtype: float32
            Range: [-1, 1]
        
        Raises:
            ValueError: If face has invalid shape or dtype
        
        Example:
            >>> face = cv2.imread('face.jpg')  # (112, 112, 3) uint8
            >>> input_tensor = recognizer.preprocess(face)
            >>> print(input_tensor.shape)  # (1, 3, 112, 112)
            >>> print(input_tensor.dtype)  # float32
            >>> print(input_tensor.min(), input_tensor.max())  # -1.0, 1.0
        """
        # Validate input (inherited from BaseRecognizer)
        if not self._validate_face(face):
            raise ValueError("Invalid face image for preprocessing")
        
        # Step 1: Resize to model input size if needed
        if face.shape[:2] != self.input_size:
            face = cv2.resize(
                face,
                (self.input_size[1], self.input_size[0]),  # (width, height)
                interpolation=cv2.INTER_LINEAR
            )
        
        # Step 2: Convert to float32 and normalize to [0, 1]
        face = face.astype(np.float32) / 255.0
        
        # Step 3: Apply mean/std normalization to [-1, 1]
        # This is the standard ArcFace preprocessing
        face = (face - 0.5) / 0.5
        
        # Step 4: Transpose from HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
        # ONNX/PyTorch convention: (H, W, C) → (C, H, W)
        face = np.transpose(face, (2, 0, 1))
        
        # Step 5: Add batch dimension
        # (C, H, W) → (1, C, H, W)
        face = np.expand_dims(face, axis=0)
        
        # Ensure correct dtype for ONNX Runtime
        face = face.astype(np.float32)
        
        return face
    
    def get_embedding(self, face: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract L2-normalized embedding from aligned face.
        
        This is the main method for face recognition:
        1. Validates input face
        2. Preprocesses face (resize, normalize, transpose)
        3. Runs ONNX inference
        4. Extracts embedding vector
        5. L2-normalizes embedding (for cosine similarity)
        
        The returned embedding can be compared with other embeddings
        using dot product (cosine similarity):
            similarity = np.dot(emb1, emb2)
        
        Args:
            face: Aligned face image from Phase 2 aligner
                  Shape: (H, W, 3) in RGB format
                  Dtype: uint8, values [0, 255]
        
        Returns:
            L2-normalized embedding vector if successful, None if failed
            Shape: (512,)
            Dtype: float32
            Properties: L2 norm = 1.0 (unit vector)
        
        Raises:
            RuntimeError: If model not loaded (call load_model() first)
        
        Example:
            >>> # Extract embeddings
            >>> emb1 = recognizer.get_embedding(aligned_face1)
            >>> emb2 = recognizer.get_embedding(aligned_face2)
            >>> 
            >>> # Compute similarity
            >>> similarity = np.dot(emb1, emb2)
            >>> print(f"Similarity: {similarity:.4f}")
            >>> 
            >>> # Check if same person (threshold from config)
            >>> is_same = similarity > 0.6
            >>> print(f"Same person: {is_same}")
        """
        # Check if model is loaded
        if not self.is_loaded or self.session is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() before get_embedding()."
            )
        
        # Validate input face
        if not self._validate_face(face):
            logger.warning("Invalid face provided to get_embedding()")
            return None
        
        try:
            # Preprocess face
            input_tensor = self.preprocess(face)
            
            # Run ONNX inference
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: input_tensor}
            )
            
            # Extract embedding (first output, first batch)
            # Type: outputs is List[np.ndarray]
            embedding: np.ndarray = outputs[0][0]  # type: ignore  # (1, 512) → (512,)
            
            # L2-normalize embedding (inherited from BaseRecognizer)
            embedding = self._normalize_embedding(embedding)
            
            # Verify embedding properties
            if embedding.shape != (self.embedding_size,):
                logger.error(
                    f"Embedding shape mismatch: expected ({self.embedding_size},), "
                    f"got {embedding.shape}"
                )
                return None
            
            # Verify L2 norm is approximately 1.0
            norm = np.linalg.norm(embedding)
            if not np.isclose(norm, 1.0, atol=1e-5):
                logger.warning(
                    f"Embedding L2 norm is {norm:.6f}, expected 1.0. "
                    f"Re-normalizing..."
                )
                embedding = self._normalize_embedding(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}", exc_info=True)
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed model information.
        
        Returns:
            Dictionary with model metadata including:
            - model_path, embedding_size, input_size (from BaseRecognizer)
            - num_threads, input_name, output_name (AuraFace-specific)
            - quantized status and backend info
        
        Example:
            >>> info = recognizer.get_model_info()
            >>> print(info['backend'])  # 'AuraFaceRecognizer'
            >>> print(info['num_threads'])  # 4
            >>> print(info['quantized'])  # False
        """
        base_info = super().get_model_info()
        base_info.update({
            'num_threads': self.num_threads,
            'input_name': self.input_name,
            'output_name': self.output_name,
            'backend': 'ONNX Runtime',
            'model_type': 'AuraFace ResNet100'
        })
        return base_info
    
    def __del__(self):
        """Cleanup ONNX session on deletion."""
        if self.session is not None:
            del self.session
            self.session = None
