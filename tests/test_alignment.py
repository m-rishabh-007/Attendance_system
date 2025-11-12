#!/usr/bin/env python3
"""
Test script for MediaPipe face alignment.

Tests the aligner module by:
1. Loading a video stream or webcam
2. Detecting faces with YOLO
3. Aligning faces with MediaPipe
4. Displaying aligned faces in a grid

Usage:
    python tests/test_alignment.py
    python tests/test_alignment.py --camera 0
    python tests/test_alignment.py --video path/to/video.mp4
"""

import cv2
import numpy as np
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aligners import AlignerFactory
from detectors import DetectorFactory
from common import ConfigManager


def create_aligned_face_grid(aligned_faces: dict, grid_cols: int = 4) -> np.ndarray:
    """
    Create a grid display of aligned faces.
    
    Args:
        aligned_faces: Dictionary of {track_id: aligned_face_image}
        grid_cols: Number of columns in grid
        
    Returns:
        Grid image showing all aligned faces
    """
    if not aligned_faces:
        # Return blank grid
        return np.zeros((224, 448, 3), dtype=np.uint8)
    
    face_size = 112
    grid_rows = (len(aligned_faces) + grid_cols - 1) // grid_cols
    grid_height = grid_rows * face_size
    grid_width = grid_cols * face_size
    
    # Create blank grid
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Fill grid with faces
    for idx, (track_id, face) in enumerate(aligned_faces.items()):
        row = idx // grid_cols
        col = idx % grid_cols
        
        y1 = row * face_size
        y2 = y1 + face_size
        x1 = col * face_size
        x2 = x1 + face_size
        
        grid[y1:y2, x1:x2] = face
        
        # Add track ID label
        cv2.putText(
            grid,
            f"ID:{track_id}",
            (x1 + 5, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1
        )
    
    return grid


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test MediaPipe face alignment")
    parser.add_argument('--camera', type=int, default=1, help='Camera device ID')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    args = parser.parse_args()
    
    print("=" * 70)
    print("MediaPipe Face Alignment Test")
    print("=" * 70)
    
    # Load configuration
    print("\n[1/4] Loading configuration...")
    config_mgr = ConfigManager()
    config_mgr.load(args.config)
    print(f"✓ Config loaded from: {args.config}")
    
    # Create detector
    print("\n[2/4] Initializing detector...")
    detector_config = config_mgr.get_section('detector')
    detector = DetectorFactory.create(detector_config)
    detector.initialize()  # Initialize the detector
    print(f"✓ Detector: {detector}")
    
    # Create aligner
    print("\n[3/4] Initializing aligner...")
    aligner_config = config_mgr.get_section('aligner')
    aligner = AlignerFactory.create_aligner(aligner_config)
    print(f"✓ Aligner: {aligner}")
    
    # Open video source
    print("\n[4/4] Opening video source...")
    if args.video:
        cap = cv2.VideoCapture(args.video)
        print(f"✓ Video: {args.video}")
    else:
        cap = cv2.VideoCapture(args.camera)
        print(f"✓ Camera: {args.camera}")
    
    if not cap.isOpened():
        print("❌ Failed to open video source!")
        return
    
    print("\n" + "=" * 70)
    print("Processing video... Press 'q' to quit")
    print("=" * 70 + "\n")
    
    frame_count = 0
    aligned_faces_cache = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream")
            break
        
        frame_count += 1
        
        # Detect faces
        detections = detector.detect(frame)
        
        # Align faces
        aligned_faces = {}  # Initialize empty dict
        if len(detections) > 0:
            # Extract bboxes from Detection objects and convert to xyxy format
            bboxes = []
            for det in detections:
                x, y, w, h = det.bbox  # tlwh format
                x1, y1, x2, y2 = x, y, x + w, y + h  # convert to xyxy
                bboxes.append([int(x1), int(y1), int(x2), int(y2)])
            
            bboxes = np.array(bboxes)
            aligned_faces = aligner.align_batch(frame, bboxes)
            
            # Update cache with track IDs (using index as temporary ID)
            aligned_faces_cache = {
                idx: face for idx, face in aligned_faces.items()
            }
        
        # Draw bounding boxes on frame with angle color-coding
        display_frame = frame.copy()
        for idx, detection in enumerate(detections):
            x, y, w, h = detection.bbox  # tlwh format
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            conf = detection.confidence
            
            # Get angle from last alignment (if this face was aligned)
            angle = None
            if idx in aligned_faces and hasattr(aligner, 'last_angle'):
                # Access last_angle from MediaPipeAligner (not all aligners have this)
                angle = getattr(aligner, 'last_angle', None)
            
            # Color-code by angle: Green (frontal) → Yellow (semi) → Red (profile)
            if angle is not None:
                abs_angle = abs(angle)
                if abs_angle < 30:
                    color = (0, 255, 0)      # Green: Frontal
                    angle_label = "frontal"
                elif abs_angle < 60:
                    color = (0, 255, 255)    # Yellow: Semi-profile
                    angle_label = "semi"
                else:
                    color = (0, 0, 255)      # Red: Profile
                    angle_label = "profile"
                
                # Draw bbox with color
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Display confidence and angle
                cv2.putText(
                    display_frame,
                    f"{conf:.2f} | {angle:.1f}° ({angle_label})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
            else:
                # No angle available (alignment failed)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
                cv2.putText(
                    display_frame,
                    f"{conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (128, 128, 128),
                    2
                )
        
        # Add info text
        cv2.putText(
            display_frame,
            f"Frame: {frame_count} | Faces: {len(detections)} | Aligned: {len(aligned_faces_cache)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Add angle legend
        legend_y_start = 60
        legend_items = [
            ("Green: Frontal (0-30°)", (0, 255, 0)),
            ("Yellow: Semi-profile (30-60°)", (0, 255, 255)),
            ("Red: Profile (60-90°)", (0, 0, 255)),
            ("Gray: Alignment failed", (128, 128, 128))
        ]
        
        for i, (text, color) in enumerate(legend_items):
            y = legend_y_start + (i * 25)
            cv2.putText(
                display_frame,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        # Create aligned faces grid
        grid = create_aligned_face_grid(aligned_faces_cache)
        
        # Show windows
        cv2.imshow("Detection", display_frame)
        cv2.imshow("Aligned Faces", grid)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print(f"Test complete! Processed {frame_count} frames")
    print("=" * 70)


if __name__ == "__main__":
    main()
