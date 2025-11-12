"""
Recognition Stage - Integrates alignment, recognition, and quality scoring.

This stage coordinates:
1. Face alignment (Phase 2 - MediaPipe)
2. Quality assessment (Phase 3A Day 5 - QualityScorer)
3. Face recognition (Phase 3A Day 3-4 - AuraFace)

Author: AI Agent + Rishabh
Date: November 8, 2025
Phase: 3A Week 1 Day 6
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from aligners import AlignerFactory
from recognizers import RecognizerFactory, QualityScorer


class RecognitionStage:
    """
    Recognition stage for face attendance pipeline.
    
    This stage processes tracked faces through:
    1. Alignment (crop + warp to 112x112)
    2. Quality scoring (assess face quality)
    3. Recognition (extract 512-dim embedding)
    
    The stage operates in two modes:
    - **Week 1 (Current)**: Quality-aware sampling (no caching)
      - Process ALL frames
      - Score quality for each frame
      - Useful for benchmarking and quality distribution analysis
    
    - **Week 2 (Future)**: Quality-aware caching
      - Sample first 10 frames per track_id
      - Cache best quality embedding
      - Skip recognition for subsequent frames
      - 97% CPU reduction
    
    Design Patterns:
        - Facade: Simplifies complex subsystem (aligner + scorer + recognizer)
        - Strategy: Swappable components via factories
        - Single Responsibility: Only coordinates recognition workflow
    
    Example:
        >>> config = ConfigManager.get_instance()
        >>> stage = RecognitionStage(config)
        >>> 
        >>> # Process frame with tracks
        >>> result = stage.process(frame, tracks)
        >>> 
        >>> # Access results
        >>> for track in tracks:
        ...     if hasattr(track, 'embedding'):
        ...         print(f"Track {track.track_id}: embedding shape {track.embedding.shape}")
        ...         print(f"  Quality: {track.quality:.3f}")
        ...         print(f"  Angle: {track.angle:.1f}°")
    """
    
    def __init__(self, config):
        """
        Initialize recognition stage.
        
        Args:
            config: ConfigManager instance with full configuration
        """
        self.logger = logging.getLogger('RecognitionStage')
        self.config = config
        
        self.logger.info("="*60)
        self.logger.info("Initializing Recognition Stage")
        self.logger.info("="*60)
        
        # Initialize aligner (Phase 2)
        self.logger.info("Loading face aligner...")
        aligner_config = config.get_section('aligner')
        self.aligner = AlignerFactory.create_aligner(aligner_config)
        self.logger.info(f"✅ Aligner: {self.aligner.__class__.__name__}")
        
        # Initialize recognizer (Phase 3A Day 3-4)
        self.logger.info("Loading face recognizer...")
        recognition_config = config.get_section('recognition')
        self.recognizer = RecognizerFactory.create(recognition_config)
        self.logger.info(f"✅ Recognizer: {self.recognizer.__class__.__name__}")
        self.logger.info(f"   Model: {recognition_config.get('model_path')}")
        self.logger.info(f"   Embedding size: {recognition_config.get('embedding_size')}")
        
        # Initialize quality scorer (Phase 3A Day 5)
        self.logger.info("Loading quality scorer...")
        quality_weights = recognition_config.get('quality_weights', {
            'sharpness': 0.30,
            'brightness': 0.20,
            'angle': 0.25,
            'size': 0.15,
            'confidence': 0.10
        })
        self.quality_scorer = QualityScorer(weights=quality_weights)
        self.logger.info(f"✅ Quality Scorer initialized")
        self.logger.info(f"   Weights: {quality_weights}")
        
        # Configuration
        self.min_quality_threshold = recognition_config.get('min_quality_threshold', 0.6)
        self.enable_cache = recognition_config.get('enable_cache', False)
        
        # Note: Caching not implemented yet (Week 2)
        if self.enable_cache:
            self.logger.warning("⚠️  Caching enabled in config but not yet implemented (Week 2)")
            self.logger.warning("    All frames will be processed for now")
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'alignments_success': 0,
            'alignments_failed': 0,
            'recognitions_success': 0,
            'recognitions_failed': 0,
            'quality_too_low': 0,
            'total_alignment_time_ms': 0.0,
            'total_quality_time_ms': 0.0,
            'total_recognition_time_ms': 0.0,
        }
        
        self.logger.info("="*60)
        self.logger.info("✅ Recognition Stage Ready")
        self.logger.info("="*60)
    
    def process(
        self,
        frame: np.ndarray,
        tracks: List[Any]
    ) -> Dict[str, Any]:
        """
        Process frame and tracks through recognition pipeline.
        
        Workflow for each track:
        1. Align face (crop + warp to 112x112)
        2. Assess quality (5 metrics: sharpness, brightness, angle, size, confidence)
        3. Extract embedding if quality >= threshold
        4. Attach results to track object
        
        Args:
            frame: Input frame (BGR format, numpy array)
            tracks: List of Track objects from tracking stage
                    Each track should have:
                    - bbox: Bounding box (x, y, w, h)
                    - track_id: Persistent track ID
                    - confidence: Detection confidence
        
        Returns:
            Dictionary containing:
                - tracks_processed: Number of tracks processed
                - embeddings_extracted: Number of successful embeddings
                - quality_too_low: Number of low-quality faces skipped
                - avg_quality: Average quality score
                - processing_time_ms: Total processing time
                
        Side Effects:
            Modifies track objects by adding:
            - aligned_face: Aligned face image (112, 112, 3) RGB
            - angle: Face yaw angle in degrees
            - quality: Quality score [0.0-1.0]
            - quality_breakdown: Dict with individual metric scores
            - embedding: 512-dim L2-normalized embedding (if quality OK)
            - embedding_extracted: Boolean flag
        """
        start_time = time.time()
        
        if len(tracks) == 0:
            return {
                'tracks_processed': 0,
                'embeddings_extracted': 0,
                'quality_too_low': 0,
                'avg_quality': 0.0,
                'processing_time_ms': 0.0
            }
        
        # Get frame size for quality scoring
        frame_height, frame_width = frame.shape[:2]
        frame_size = (frame_height, frame_width)
        
        embeddings_extracted = 0
        quality_too_low = 0
        quality_scores = []
        
        for track in tracks:
            self.stats['total_processed'] += 1
            
            # Get bbox from track
            bbox = self._get_bbox_from_track(track)
            if bbox is None:
                self.logger.warning(f"Track {track.track_id}: Invalid bbox, skipping")
                continue
            
            confidence = track.confidence if hasattr(track, 'confidence') else 0.0
            
            # Step 1: Align face
            t0 = time.time()
            aligned_face, angle = self._align_face(frame, bbox)
            alignment_time = (time.time() - t0) * 1000
            self.stats['total_alignment_time_ms'] += alignment_time
            
            if aligned_face is None:
                self.stats['alignments_failed'] += 1
                track.embedding_extracted = False
                continue
            
            self.stats['alignments_success'] += 1
            
            # Attach to track for debugging/visualization
            track.aligned_face = aligned_face
            track.angle = angle
            
            # Step 2: Assess quality
            t0 = time.time()
            # Use angle if available, otherwise default to 0.0
            angle_value = angle if angle is not None else 0.0
            quality_breakdown = self.quality_scorer.get_detailed_scores(
                aligned_face, bbox, angle_value, confidence, frame_size
            )
            quality_score = quality_breakdown['combined']
            quality_time = (time.time() - t0) * 1000
            self.stats['total_quality_time_ms'] += quality_time
            
            quality_scores.append(quality_score)
            track.quality = quality_score
            track.quality_breakdown = quality_breakdown
            
            # Step 3: Extract embedding if quality sufficient
            if quality_score >= self.min_quality_threshold:
                t0 = time.time()
                embedding = self._extract_embedding(aligned_face)
                recognition_time = (time.time() - t0) * 1000
                self.stats['total_recognition_time_ms'] += recognition_time
                
                if embedding is not None:
                    self.stats['recognitions_success'] += 1
                    track.embedding = embedding
                    track.embedding_extracted = True
                    embeddings_extracted += 1
                else:
                    self.stats['recognitions_failed'] += 1
                    track.embedding_extracted = False
            else:
                # Quality too low, skip recognition
                self.stats['quality_too_low'] += 1
                quality_too_low += 1
                track.embedding_extracted = False
                
                self.logger.debug(
                    f"Track {track.track_id}: Quality {quality_score:.3f} < "
                    f"threshold {self.min_quality_threshold:.3f}, skipping recognition"
                )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Calculate average quality
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        return {
            'tracks_processed': len(tracks),
            'embeddings_extracted': embeddings_extracted,
            'quality_too_low': quality_too_low,
            'avg_quality': float(avg_quality),
            'processing_time_ms': processing_time_ms
        }
    
    def _get_bbox_from_track(self, track) -> Optional[Tuple[int, int, int, int]]:
        """
        Extract bounding box from track object.
        
        Handles different track formats:
        - Track object with to_tlwh() method (BoT-SORT)
        - Track object with bbox attribute
        - Dictionary format
        
        Args:
            track: Track object
        
        Returns:
            Tuple (x, y, w, h) or None if invalid
        """
        try:
            if hasattr(track, 'to_tlwh'):
                x, y, w, h = track.to_tlwh()
            elif hasattr(track, 'bbox'):
                bbox = track.bbox
                if hasattr(bbox, '__iter__'):
                    x, y, w, h = bbox
                else:
                    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            else:
                # Dictionary format
                x, y, w, h = track['bbox']
            
            return (int(x), int(y), int(w), int(h))
        
        except Exception as e:
            self.logger.warning(f"Failed to extract bbox: {e}")
            return None
    
    def _align_face(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Align face using Phase 2 aligner.
        
        Args:
            frame: Input frame (BGR format)
            bbox: Bounding box (x, y, w, h)
        
        Returns:
            Tuple (aligned_face, angle):
            - aligned_face: (112, 112, 3) RGB uint8, or None if failed
            - angle: Face yaw angle in degrees, or None if failed
        """
        try:
            result = self.aligner.align(frame, bbox)
            if result is None:
                return None, None
            
            aligned_face, angle = result
            return aligned_face, angle
        
        except Exception as e:
            self.logger.warning(f"Alignment failed: {e}")
            return None, None
    
    def _extract_embedding(self, aligned_face: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding using recognizer.
        
        Args:
            aligned_face: Aligned face (112, 112, 3) RGB uint8
        
        Returns:
            512-dim L2-normalized embedding, or None if failed
        """
        try:
            embedding = self.recognizer.get_embedding(aligned_face)
            return embedding
        
        except Exception as e:
            self.logger.warning(f"Embedding extraction failed: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get recognition stage statistics.
        
        Returns:
            Dictionary with performance statistics:
            - total_processed: Total tracks processed
            - alignments_success/failed: Alignment results
            - recognitions_success/failed: Recognition results
            - quality_too_low: Low-quality faces skipped
            - avg_alignment_time_ms: Average alignment time
            - avg_quality_time_ms: Average quality scoring time
            - avg_recognition_time_ms: Average recognition time
        """
        stats = self.stats.copy()
        
        # Calculate averages
        if stats['alignments_success'] > 0:
            stats['avg_alignment_time_ms'] = (
                stats['total_alignment_time_ms'] / stats['alignments_success']
            )
        else:
            stats['avg_alignment_time_ms'] = 0.0
        
        if stats['total_processed'] > 0:
            stats['avg_quality_time_ms'] = (
                stats['total_quality_time_ms'] / stats['total_processed']
            )
        else:
            stats['avg_quality_time_ms'] = 0.0
        
        if stats['recognitions_success'] > 0:
            stats['avg_recognition_time_ms'] = (
                stats['total_recognition_time_ms'] / stats['recognitions_success']
            )
        else:
            stats['avg_recognition_time_ms'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {
            'total_processed': 0,
            'alignments_success': 0,
            'alignments_failed': 0,
            'recognitions_success': 0,
            'recognitions_failed': 0,
            'quality_too_low': 0,
            'total_alignment_time_ms': 0.0,
            'total_quality_time_ms': 0.0,
            'total_recognition_time_ms': 0.0,
        }
    
    def __repr__(self) -> str:
        return (
            f"RecognitionStage("
            f"aligner={self.aligner.__class__.__name__}, "
            f"recognizer={self.recognizer.__class__.__name__}, "
            f"min_quality={self.min_quality_threshold})"
        )
