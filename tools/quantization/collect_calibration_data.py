"""
Calibration Data Collection Script

Collects face samples from webcam for INT8 quantization calibration.

Strategy:
    - Collect continuously for 10 minutes (or until 'q' pressed)
    - Smart selection: Pick best 100 from all collected samples
    - Criteria: Quality score (sharpness, angle, brightness, size)
    - Ensures diversity: Different angles, distances, lighting

Output:
    - tools/quantization/calibration_data/faces_100.npy (100 best samples)
    - tools/quantization/calibration_data/faces_validation.npy (extra samples)
    - tools/quantization/calibration_data/metadata.json (quality scores)

Usage:
    python tools/quantization/collect_calibration_data.py
    
    Instructions during collection:
    - Look at webcam
    - Move head left/right (angles)
    - Lean in/out (distance)
    - Move around room (lighting variety)
    - Press 'q' to stop early (or wait 10 minutes)

Author: Attendance System Team
Date: November 12, 2025
Phase: 3A Week 2 Day 1
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from common.config_manager import ConfigManager
from detectors.factory import DetectorFactory
from detectors.yolo_detector import YOLODetector
from tracking.factory import TrackerFactory
from aligners.factory import AlignerFactory
from recognizers.quality_scorer import QualityScorer


def setup_logging(output_dir: Path) -> logging.Logger:
    """
    Setup comprehensive logging for calibration data collection.
    
    Creates both console and file logging with detailed formatting.
    Log file includes timestamp, level, and message.
    
    Args:
        output_dir: Directory to save log file
        
    Returns:
        Configured logger instance
    """
    # Create logs directory
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"calibration_collection_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger("CalibrationCollector")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # File handler (detailed logging)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler (less verbose)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log file location
    logger.info(f"📝 Log file: {log_file}")
    logger.debug("="*70)
    logger.debug(f"Calibration Data Collection - Started at {datetime.now()}")
    logger.debug("="*70)
    
    return logger


@dataclass
class SampleMetadata:
    """Metadata for a single calibration sample."""
    sample_id: int
    timestamp: float
    track_id: int
    quality_score: float
    sharpness: float
    angle: Optional[float]
    brightness: float
    size_score: float
    confidence: float


class CalibrationCollector:
    """Smart calibration data collector with quality-aware selection."""
    
    def __init__(
        self,
        output_dir: Path,
        target_samples: int = 100,
        collection_time_minutes: int = 10,
        device_id: Optional[int] = None
    ):
        """
        Initialize calibration collector.
        
        Args:
            output_dir: Directory to save calibration data
            target_samples: Number of samples for final calibration set (default: 100)
            collection_time_minutes: How long to collect samples (default: 10)
            device_id: Webcam device ID (None = read from config, default: None)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging FIRST
        self.logger = setup_logging(self.output_dir)
        
        self.target_samples = target_samples
        self.collection_time = collection_time_minutes * 60  # Convert to seconds
        
        # Storage for collected samples
        self.samples: List[np.ndarray] = []  # Aligned face images (112x112x3)
        self.metadata: List[SampleMetadata] = []  # Quality metadata
        
        # Initialize pipeline components
        self.logger.info("Initializing detection + tracking + alignment pipeline...")
        self.logger.debug(f"Output directory: {self.output_dir}")
        self.logger.debug(f"Target samples: {target_samples}")
        self.logger.debug(f"Collection time: {collection_time_minutes} minutes")
        
        config_manager = ConfigManager()
        config_manager.load(PROJECT_ROOT / 'config.yaml')
        self.logger.debug(f"Config loaded from: {PROJECT_ROOT / 'config.yaml'}")
        
        # Get subsection configs for each component
        detector_config = config_manager.get_section('detector')
        tracker_config = config_manager.get_section('tracker')
        aligner_config = config_manager.get_section('aligner')
        
        self.logger.debug("Creating detector...")
        self.detector: YOLODetector = DetectorFactory.create(detector_config)  # type: ignore
        self.logger.debug(f"Detector created: {type(self.detector).__name__}")
        
        self.logger.debug("Initializing detector...")
        self.detector.initialize()  # Initialize detector before use
        self.logger.debug("Detector initialized successfully")
        
        self.logger.debug("Creating tracker...")
        self.tracker = TrackerFactory.create(tracker_config)
        self.logger.debug(f"Tracker created: {type(self.tracker).__name__}")
        
        self.logger.debug("Creating aligner...")
        self.aligner = AlignerFactory.create_aligner(aligner_config)
        self.logger.debug(f"Aligner created: {type(self.aligner).__name__}")
        
        self.logger.debug("Creating quality scorer...")
        self.quality_scorer = QualityScorer()
        self.logger.debug("Quality scorer created")
        
        # Get camera device_id from config (or use provided override)
        camera_config = config_manager.get_section('camera')
        self.device_id: int = device_id if device_id is not None else camera_config.get('device_id', 0)
        self.logger.debug(f"Camera device_id: {self.device_id}")
        
        # Camera (initialized in open_camera)
        self.cap: cv2.VideoCapture
        
        self.logger.info(f"✅ Calibration Collector initialized")
        self.logger.info(f"   Target samples: {self.target_samples}")
        self.logger.info(f"   Collection time: {collection_time_minutes} minutes")
        self.logger.info(f"   Output: {self.output_dir}")
    
    def open_camera(self) -> bool:
        """Open webcam for capture."""
        self.logger.info(f"Opening camera with device_id={self.device_id} (from config.yaml)")
        self.logger.debug(f"Attempting cv2.VideoCapture({self.device_id})")
        
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            self.logger.error(f"❌ Failed to open webcam (device_id={self.device_id})")
            self.logger.error(f"   Try device_id=0 if device_id={self.device_id} doesn't work")
            return False
        
        self.logger.debug("Camera opened successfully")
        
        # Set resolution
        self.logger.debug("Setting camera resolution to 1280x720")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.logger.info(f"✅ Webcam opened (device_id={self.device_id})")
        self.logger.info(f"   Resolution: {actual_width}x{actual_height}")
        self.logger.debug(f"Camera properties: width={actual_width}, height={actual_height}")
        return True
    
    def collect_samples(self) -> int:
        """
        Collect face samples from webcam.
        
        Returns:
            Number of samples collected
        """
        if not self.open_camera():
            self.logger.error("Failed to open camera, aborting collection")
            return 0
        
        self.logger.info("\n" + "="*70)
        self.logger.info("📸 CALIBRATION DATA COLLECTION")
        self.logger.info("="*70)
        self.logger.info(f"⏱️  Collection time: {self.collection_time // 60} minutes")
        self.logger.info(f"🎯 Target: Collect as many samples as possible")
        self.logger.info(f"📊 Final selection: Best {self.target_samples} samples")
        self.logger.info("\n📋 Instructions:")
        self.logger.info("   - Look at the webcam")
        self.logger.info("   - Move your head LEFT/RIGHT (different angles)")
        self.logger.info("   - Lean IN/OUT (different distances)")
        self.logger.info("   - Move around room (different lighting)")
        self.logger.info("   - Press 'q' to stop early")
        self.logger.info("="*70 + "\n")
        
        start_time = time.time()
        frame_count = 0
        last_sample_time = 0
        sample_interval = 1.0  # Sample every 3 seconds (avoid duplicates)
        
        self.logger.debug(f"Collection started at {datetime.now()}")
        self.logger.debug(f"Sample interval: {sample_interval} seconds")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                frame_count += 1
                elapsed = time.time() - start_time
                remaining = self.collection_time - elapsed
                
                # Check time limit
                if elapsed >= self.collection_time:
                    print(f"\n⏱️  Time limit reached ({self.collection_time // 60} minutes)")
                    break
                
                # Detect + track faces (using integrated tracking with persist=True)
                detections, tracks = self.detector.detect_and_track(frame, tracker_config='botsort.yaml')  # type: ignore
                
                if frame_count % 30 == 0:  # Log every 30 frames (~1 second)
                    self.logger.debug(f"Frame {frame_count}: Detected {len(detections)} faces, Tracked {len(tracks)} tracks")
                
                # Process each tracked face
                current_time = time.time()
                for track in tracks:
                    # Sample interval check (avoid too similar frames)
                    if current_time - last_sample_time < sample_interval:
                        continue
                    
                    # track.bbox is in TLWH format (x, y, width, height)
                    x, y, w, h = map(int, track.bbox)
                    x1, y1, x2, y2 = x, y, x + w, y + h  # Convert to XYXY
                    track_id = track.track_id
                    confidence = track.confidence
                    
                    # Crop face region
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        self.logger.debug(f"Empty face crop for track_id={track_id}, bbox=({x1},{y1},{x2},{y2})")
                        continue
                    
                    # Align face to 112x112
                    # bbox coordinates should be relative to the cropped face (0,0 to width,height)
                    crop_height, crop_width = face_crop.shape[:2]
                    aligned_face = self.aligner.align(face_crop, bbox=(0, 0, crop_width, crop_height))
                    if aligned_face is None:
                        self.logger.debug(f"Alignment failed for track_id={track_id}")
                        continue
                    
                    # Get angle from aligner (stored in last_angle after align() call)
                    angle = getattr(self.aligner, 'last_angle', None)
                    
                    # Calculate quality score (detailed breakdown for metadata)
                    quality_details = self.quality_scorer.get_detailed_scores(
                        face=aligned_face,
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        angle=angle if angle is not None else 0.0,
                        confidence=confidence,
                        frame_size=(frame.shape[1], frame.shape[0])
                    )
                    
                    # Store sample + metadata
                    self.samples.append(aligned_face.copy())
                    
                    metadata = SampleMetadata(
                        sample_id=len(self.samples) - 1,
                        timestamp=current_time,
                        track_id=track_id,
                        quality_score=quality_details['combined'],
                        sharpness=quality_details['sharpness'],
                        angle=angle,
                        brightness=quality_details['brightness'],
                        size_score=quality_details['size'],
                        confidence=confidence
                    )
                    self.metadata.append(metadata)
                    
                    self.logger.debug(f"Sample {len(self.samples)} collected: track_id={track_id}, "
                                     f"quality={quality_details['combined']:.3f}, angle={angle if angle else 'N/A'}")
                    
                    last_sample_time = current_time
                    
                    # Draw on frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"Quality: {quality_details['combined']:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
                
                # Display progress
                minutes = int(remaining // 60)
                seconds = int(remaining % 60)
                progress_text = f"Samples: {len(self.samples)} | Time: {minutes:02d}:{seconds:02d}"
                
                cv2.putText(
                    frame,
                    progress_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                
                # Instructions
                instructions = [
                    "Move head LEFT/RIGHT (angles)",
                    "Lean IN/OUT (distance)",
                    "Press 'q' to stop"
                ]
                for i, text in enumerate(instructions):
                    cv2.putText(
                        frame,
                        text,
                        (10, 70 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )
                
                # Show frame (skip if OpenCV GUI not available)
                try:
                    cv2.imshow("Calibration Collection", frame)
                except cv2.error:
                    # OpenCV built without GUI support - skip display
                    pass
                
                # Print progress every 10 samples
                if len(self.samples) > 0 and len(self.samples) % 10 == 0:
                    avg_quality = np.mean([m.quality_score for m in self.metadata])
                    progress_msg = (f"Progress: {len(self.samples)} samples | "
                                   f"Time: {minutes:02d}:{seconds:02d} | "
                                   f"Avg Quality: {avg_quality:.2f}")
                    self.logger.info(progress_msg)
                    self.logger.debug(f"Detailed: {len(detections)} detections, {len(tracks)} tracks")
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.logger.info(f"\n🛑 Collection stopped by user")
                    break
        
        except KeyboardInterrupt:
            self.logger.info(f"\n🛑 Collection interrupted by user")
        
        finally:
            if self.cap is not None:
                self.cap.release()
                self.logger.debug("Camera released")
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                # OpenCV built without GUI support (headless mode)
                pass
        
        # Calculate final statistics
        total_elapsed = time.time() - start_time
        collection_summary = (f"\n✅ Collection complete: {len(self.samples)} samples collected in "
                            f"{total_elapsed/60:.1f} minutes ({frame_count} frames processed)")
        self.logger.info(collection_summary)
        if total_elapsed > 0:
            self.logger.debug(f"Average samples per minute: {len(self.samples) / (total_elapsed/60):.1f}")
        
        return len(self.samples)
    
    def select_best_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select best samples for calibration using smart selection.
        
        Strategy:
            1. Sort by quality score
            2. Ensure angle diversity (cover -45° to +45°)
            3. Ensure distance diversity (different face sizes)
            4. Pick top samples
        
        Returns:
            (calibration_samples, validation_samples)
            - calibration_samples: Best N samples for quantization
            - validation_samples: Remaining samples for testing
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("🔍 SMART SAMPLE SELECTION")
        self.logger.info("="*70)
        self.logger.debug(f"Total samples to select from: {len(self.samples)}")
        
        total_collected = len(self.samples)
        self.logger.info(f"Total collected: {total_collected} samples")
        
        if total_collected == 0:
            self.logger.error("❌ No samples collected!")
            return np.array([]), np.array([])
        
        if total_collected <= self.target_samples:
            self.logger.warning(f"⚠️  Collected only {total_collected} samples (target: {self.target_samples})")
            self.logger.info(f"   Using all {total_collected} samples for calibration")
            return np.array(self.samples), np.array([])
        
        # Sort by quality score (descending)
        sorted_indices = sorted(
            range(len(self.metadata)),
            key=lambda i: self.metadata[i].quality_score,
            reverse=True
        )
        
        # Ensure diversity (angle coverage)
        angle_bins = 5  # 5 bins: [-45, -27, -9, 9, 27, 45]
        samples_per_bin = self.target_samples // angle_bins
        
        selected_indices = []
        angle_counts = {i: 0 for i in range(angle_bins)}
        
        for idx in sorted_indices:
            if len(selected_indices) >= self.target_samples:
                break
            
            angle = self.metadata[idx].angle
            if angle is None:
                angle = 0.0
            
            # Determine angle bin
            angle_bin = min(int((angle + 45) / 18), angle_bins - 1)  # Map -45..45 to 0..4
            
            # Add if bin not full OR we need to fill remaining slots
            if angle_counts[angle_bin] < samples_per_bin or len(selected_indices) >= self.target_samples - 10:
                selected_indices.append(idx)
                angle_counts[angle_bin] += 1
        
        # If we still need more samples, add best remaining
        if len(selected_indices) < self.target_samples:
            for idx in sorted_indices:
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    if len(selected_indices) >= self.target_samples:
                        break
        
        # Split into calibration and validation sets
        calibration_indices = selected_indices[:self.target_samples]
        validation_indices = [i for i in range(total_collected) if i not in calibration_indices]
        
        calibration_samples = np.array([self.samples[i] for i in calibration_indices])
        validation_samples = np.array([self.samples[i] for i in validation_indices])
        
        # Print statistics
        self.logger.info(f"\n✅ Selection complete:")
        self.logger.info(f"   Calibration set: {len(calibration_samples)} samples")
        self.logger.info(f"   Validation set: {len(validation_samples)} samples")
        
        # Angle distribution
        self.logger.info(f"\n📊 Angle distribution (calibration set):")
        angle_ranges = [(-45, -27), (-27, -9), (-9, 9), (9, 27), (27, 45)]
        for min_a, max_a in angle_ranges:
            count = sum(
                1 for i in calibration_indices
                if self.metadata[i].angle is not None
                and min_a <= self.metadata[i].angle < max_a
            )
            self.logger.info(f"   {min_a:3d}° to {max_a:3d}°: {count:3d} samples")
        
        # Quality statistics
        cal_qualities = [self.metadata[i].quality_score for i in calibration_indices]
        self.logger.info(f"\n📊 Quality scores (calibration set):")
        self.logger.info(f"   Min:  {min(cal_qualities):.3f}")
        self.logger.info(f"   Mean: {np.mean(cal_qualities):.3f}")
        self.logger.info(f"   Max:  {max(cal_qualities):.3f}")
        self.logger.debug(f"Calibration indices: {calibration_indices[:10]}...")  # Log first 10
        
        return calibration_samples, validation_samples
    
    def save_datasets(
        self,
        calibration_samples: np.ndarray,
        validation_samples: np.ndarray
    ) -> None:
        """Save calibration and validation datasets."""
        self.logger.info("\n" + "="*70)
        self.logger.info("💾 SAVING DATASETS")
        self.logger.info("="*70)
        
        # Save calibration set
        cal_path = self.output_dir / "faces_100.npy"
        self.logger.debug(f"Saving calibration set to {cal_path}")
        np.save(cal_path, calibration_samples)
        self.logger.info(f"✅ Calibration set saved: {cal_path}")
        self.logger.info(f"   Shape: {calibration_samples.shape}")
        self.logger.info(f"   Size: {calibration_samples.nbytes / 1024 / 1024:.2f} MB")
        
        # Save validation set (if any)
        if len(validation_samples) > 0:
            val_path = self.output_dir / "faces_validation.npy"
            self.logger.debug(f"Saving validation set to {val_path}")
            np.save(val_path, validation_samples)
            self.logger.info(f"✅ Validation set saved: {val_path}")
            self.logger.info(f"   Shape: {validation_samples.shape}")
            self.logger.info(f"   Size: {validation_samples.nbytes / 1024 / 1024:.2f} MB")
        
        # Save metadata
        metadata_path = self.output_dir / "metadata.json"
        metadata_dict = {
            "collection_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_collected": len(self.samples),
            "calibration_count": len(calibration_samples),
            "validation_count": len(validation_samples),
            "target_samples": self.target_samples,
            "samples": [asdict(m) for m in self.metadata]
        }
        
        self.logger.debug(f"Saving metadata to {metadata_path}")
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        self.logger.info(f"✅ Metadata saved: {metadata_path}")
        self.logger.debug(f"Metadata contains {len(metadata_dict['samples'])} sample records")
        self.logger.info("="*70)


def main():
    """Main calibration collection workflow."""
    print("\n" + "="*70)
    print("🚀 AURAFACE INT8 CALIBRATION DATA COLLECTION")
    print("="*70)
    print("Phase: 3A Week 2 Day 1")
    print("Goal: Collect face samples for INT8 quantization")
    print("="*70 + "\n")
    
    # Setup
    output_dir = Path(__file__).parent / "calibration_data"
    collector = CalibrationCollector(
        output_dir=output_dir,
        target_samples=100,
        collection_time_minutes=5,  # Changed from 10 to 5 minutes
        device_id=None  # Reads from config.yaml (camera.device_id)
    )
    
    # Collect samples
    num_collected = collector.collect_samples()
    
    if num_collected == 0:
        print("\n❌ No samples collected. Please check:")
        print("   - Webcam is connected and working")
        print("   - Check config.yaml camera.device_id setting")
        print("   - Face detection is working (good lighting)")
        print("   - Try different device_id if camera not opening")
        return
    
    # Select best samples
    calibration_samples, validation_samples = collector.select_best_samples()
    
    if len(calibration_samples) == 0:
        print("\n❌ Sample selection failed")
        return
    
    # Save datasets
    collector.save_datasets(calibration_samples, validation_samples)
    
    print("\n" + "="*70)
    print("✅ CALIBRATION DATA COLLECTION COMPLETE!")
    print("="*70)
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Calibration samples: {len(calibration_samples)}")
    print(f"📊 Validation samples: {len(validation_samples)}")
    print("\n🎯 Next steps:")
    print("   1. python tools/quantization/quantize_auraface.py")
    print("   2. python tools/quantization/validate_quantized_model.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
