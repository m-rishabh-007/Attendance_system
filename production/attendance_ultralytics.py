#!/usr/bin/env python3
"""
Production Face Attendance System using Ultralytics API
========================================================

This version prioritizes TRACKING QUALITY over raw inference speed.
Uses Ultralytics' BoT-SORT tracker which handles:
- Motion blur (head shaking, fast movements)
- Temporary occlusions
- Re-identification after detection gaps

Key advantages over custom ByteTrack implementation:
1. Battle-tested by millions of users
2. Superior track persistence (maintains IDs through detection gaps)
3. Minimal code maintenance (10 lines vs 800+)
4. Active development and bug fixes
5. Still uses the same TFLite INT8 model

Performance: ~20-25 FPS with excellent ID persistence
Inference: ~45ms (slightly slower than raw TFLite, but tracking quality worth it)

Author: Rishabh Mishra
Date: October 31, 2025
"""

import cv2
import yaml
from ultralytics import YOLO
from pathlib import Path
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import time


def setup_logging():
    """
    Setup hybrid logging: Terminal + Rotating file logs.
    
    Log Levels:
    - DEBUG: Detailed frame-by-frame info (development)
    - INFO: Important events (startup, shutdown, track events)
    - WARNING: Potential issues (detection gaps, camera problems)
    - ERROR: Critical failures
    """
    # Create logs directory
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Log filename with date
    log_file = log_dir / f"attendance_{datetime.now().strftime('%Y-%m-%d')}.log"
    
    # Create logger
    logger = logging.getLogger('AttendanceSystem')
    logger.setLevel(logging.DEBUG)  # Capture everything, filter in handlers
    
    # Rotating file handler (keeps last 10MB, 7 backups = ~70MB total)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB per file
        backupCount=7,           # Keep 7 days of logs
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    file_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # Console handler (only INFO and above for clean terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_config():
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "research" / "yolov8_face_pipeline" / "pipeline_config.yaml"
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    """Main pipeline execution."""
    # Setup logging
    logger = setup_logging()
    logger.info("="*60)
    logger.info("🚀 Production Face Attendance System Starting")
    logger.info("="*60)
    
    # Load configuration
    cfg = load_config()
    logger.info(f"✅ Configuration loaded")
    
    # Model path (same TFLite model as research pipeline)
    model_path = Path(__file__).parent.parent / "models" / "yolov8n_face_int8.tflite"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    
    # Initialize model
    logger.info("🔄 Loading model...")
    try:
        model = YOLO(model_path, task='detect')
        logger.info(f"✅ Model loaded: {model_path.name}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)
    
    # Camera setup
    camera_id = cfg['deployment_config']['camera_id']
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        logger.error(f"Could not open camera {camera_id}")
        sys.exit(1)
    
    logger.info(f"✅ Camera {camera_id} opened successfully")
    
    # Display configuration
    logger.info("")
    logger.info("📹 Configuration:")
    logger.info(f"   Model:      {model_path.name}")
    logger.info(f"   Camera:     {camera_id}")
    logger.info(f"   Confidence: {cfg['inference_settings']['confidence_threshold']}")
    logger.info(f"   IoU:        {cfg['inference_settings']['iou_threshold']}")
    logger.info(f"   Image Size: {cfg['inference_settings']['input_size']}")
    logger.info(f"   Tracker:    BoT-SORT")
    logger.info("")
    logger.info("💡 Press 'q' to quit")
    logger.info("="*60)
    logger.info("")
    
    # Performance tracking
    frame_count = 0
    start_time = time.time()
    fps_update_interval = 50  # Update stats every 50 frames
    last_fps_update = 0
    active_tracks = {}  # Track ID -> last seen frame
    total_faces_detected = 0
    
    try:
        while cap.isOpened():
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame from camera")
                break
            
            frame_count += 1
            
            # Run inference with tracking
            results = model.track(
                frame,
                persist=True,                                          # Persistent track IDs
                conf=cfg['inference_settings']['confidence_threshold'],  # Confidence threshold
                iou=cfg['inference_settings']['iou_threshold'],          # IoU for NMS
                imgsz=cfg['inference_settings']['input_size'],           # Input size
                verbose=False,                                          # Suppress per-frame output
                tracker='botsort.yaml'                                  # BoT-SORT tracker (best quality)
            )
            
            frame_time = (time.time() - frame_start) * 1000  # Convert to ms
            
            # Extract detection info
            boxes = results[0].boxes
            num_detections = len(boxes) if boxes is not None else 0
            
            # Track management
            current_track_ids = set()
            if boxes is not None and boxes.id is not None:
                for track_id, conf, xyxy in zip(boxes.id, boxes.conf, boxes.xyxy):
                    track_id = int(track_id.item())
                    confidence = float(conf.item())
                    current_track_ids.add(track_id)
                    
                    # New track detected
                    if track_id not in active_tracks:
                        logger.info(f"🆕 New track: ID={track_id}, conf={confidence:.2f}")
                        total_faces_detected += 1
                    
                    active_tracks[track_id] = frame_count
                    
                    # Log detailed frame info to file only
                    logger.debug(f"Frame {frame_count}: Track ID={track_id}, conf={confidence:.2f}, bbox={xyxy.tolist()}")
            
            # Check for lost tracks
            lost_tracks = [tid for tid, last_frame in active_tracks.items() 
                          if tid not in current_track_ids and frame_count - last_frame > 30]
            for tid in lost_tracks:
                logger.info(f"❌ Track lost: ID={tid} (missing for {frame_count - active_tracks[tid]} frames)")
                del active_tracks[tid]
            
            # Periodic terminal stats (every N frames)
            if frame_count % fps_update_interval == 0:
                elapsed = time.time() - start_time
                avg_fps = frame_count / elapsed
                logger.info(f"[Frame {frame_count:4d}] FPS: {avg_fps:5.1f} | Detections: {num_detections} | Active Tracks: {len(current_track_ids)} | Latency: {frame_time:5.1f}ms")
            
            # Get annotated frame with boxes and IDs
            annotated_frame = results[0].plot()
            
            # Add stats overlay
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"FPS: {(frame_count / (time.time() - start_time)):.1f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Tracks: {len(current_track_ids)}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display
            cv2.imshow("Face Attendance - Production (Ultralytics)", annotated_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("\n🛑 User requested quit")
                break
                
    except KeyboardInterrupt:
        logger.info("\n🛑 Interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"\n❌ Error during execution: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Final statistics
        elapsed = time.time() - start_time
        logger.info("")
        logger.info("="*60)
        logger.info("📊 Session Summary")
        logger.info("="*60)
        logger.info(f"   Total Frames:      {frame_count}")
        logger.info(f"   Duration:          {elapsed:.1f}s")
        logger.info(f"   Average FPS:       {frame_count / elapsed if elapsed > 0 else 0:.1f}")
        logger.info(f"   Total Faces:       {total_faces_detected}")
        logger.info(f"   Active Tracks:     {len(active_tracks)}")
        logger.info("="*60)
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        logger.info("✅ Resources released. Shutdown complete.")
        logger.info("")


if __name__ == "__main__":
    main()
