#!/usr/bin/env python3
"""
Async Pipeline Orchestrator - Multiprocessing Architecture

Producer-Consumer pattern with:
- Process 1 (Main Loop): Camera → YOLO → BoT-SORT → Queue (15-25 FPS stable)
- Process 2 (AI Worker): Queue → Align → Recognize → Database (parallel)
- Leaky Bucket queue (maxsize=5, put_nowait) prevents blocking
- Result Queue for recognized names (local cache for instant display)

Architecture prevents "blind spots" that break BoT-SORT tracking.

Date: December 1, 2025
Phase: 4 - Async Implementation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import time
import multiprocessing
import queue
import logging
import numpy as np
from multiprocessing import Process, Queue, Event
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

from common.config_manager import ConfigManager
from pipeline.detection_stage import DetectionStage
from pipeline.recognition_stage import RecognitionStage

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# AI WORKER PROCESS (CONSUMER) - Runs in separate process
# =============================================================================

def ai_worker_process(
    input_queue: Queue,
    result_queue: Queue,
    stop_event: Event,
    config_dict: Dict[str, Any]
):
    """
    Consumer process: extracts faces from queue, runs alignment + recognition.
    
    This runs in a SEPARATE process to prevent blocking the main camera loop.
    Heavy processing (MediaPipe + AuraFace ~300ms) happens here while camera
    continues capturing at 15-25 FPS.
    
    Args:
        input_queue: Receives (face_crop, track_id, timestamp) tuples
        result_queue: Sends (track_id, person_name, confidence, quality) tuples
        stop_event: Signal to gracefully shutdown
        config_dict: Configuration dictionary (recreated in worker process)
    """
    # Re-initialize ConfigManager in worker process
    # Need fresh instance since this is a new process
    config = ConfigManager()
    config._config = config_dict  # Directly set config dict
    
    # Initialize components manually (avoid RecognitionStage complexity in worker)
    from aligners.factory import AlignerFactory
    from recognizers.factory import RecognizerFactory
    
    # Create aligner (MediaPipe initializes automatically in __init__)
    aligner = AlignerFactory.create_aligner(config_dict.get('aligner', {}))
    
    # Create recognizer (ONNX model NOT loaded yet)
    recognizer = RecognizerFactory.create(config_dict.get('recognition', {}))
    
    # CRITICAL: Load the ONNX model before use
    # RecognizerFactory.create() only instantiates, must call load_model() explicitly
    # MediaPipe aligner doesn't need this - it auto-loads in __init__
    # Fix (Dec 2025): Added to prevent "Model not loaded" RuntimeError
    recognizer.load_model()
    
    logger.info("[Worker] AI Process started. Waiting for faces...")
    
    processed_count = 0
    
    while not stop_event.is_set():
        try:
            # Blocking wait for face crops (timeout to check stop_event)
            data = input_queue.get(timeout=1.0)
            
            face_crop, track_id, timestamp = data
            
            worker_start = time.time()
            
            # Get face crop dimensions for bbox
            h, w = face_crop.shape[:2]
            bbox = (0, 0, w, h)  # Full face crop as bbox
            
            # --- ALIGNMENT ---
            aligned_face = aligner.align(face_crop, bbox)
            
            if aligned_face is None:
                continue  # Alignment failed
            
            # --- RECOGNITION ---
            embedding = recognizer.get_embedding(aligned_face)
            
            if embedding is not None:
                # TODO: Database matching (Phase 4 Week 2)
                # For now, simulate recognition
                person_name = f"Person_{track_id}"
                confidence = 0.85  # Placeholder confidence
                quality = 0.75  # Placeholder quality
                
                # Send result back to Main Process
                result_queue.put((track_id, person_name, confidence, quality))
                
                processed_count += 1
                
                worker_time = (time.time() - worker_start) * 1000
                logger.debug(f"[Worker] Processed Track {track_id} in {worker_time:.1f}ms (Total: {processed_count})")
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"[Worker Error] {e}", exc_info=True)
    
    logger.info(f"[Worker] AI Process stopped. Total processed: {processed_count}")


# =============================================================================
# ASYNC ORCHESTRATOR (PRODUCER) - Main class
# =============================================================================

class AsyncOrchestrator:
    """
    Asynchronous Pipeline Orchestrator using multiprocessing.
    
    Architecture:
        Process 1 (Main Loop): Camera → YOLO → BoT-SORT → Display (15-25 FPS)
        Process 2 (AI Worker): Queue → Align → Recognize → Database (parallel)
        
    Key Features:
        - Leaky Bucket queue: Drop frames if worker is slow (fail fast)
        - Result Queue + Local Cache: Instant name display (zero overhead)
        - No blocking: Main loop never waits for recognition
        - Prevents "blind spots" that break BoT-SORT tracking
    """
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize async orchestrator.
        
        Args:
            config: ConfigManager instance (if None, loads from config.yaml)
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        if config is None:
            self.config = ConfigManager()
        else:
            self.config = config
        
        self.running = False
        
        # === MULTIPROCESSING QUEUES ===
        # Leaky Bucket Queue: Max 5 face crops buffered
        # If full, we drop frames (camera never blocks)
        queue_size = self.config.get('pipeline.face_queue_size', 5)
        self.face_queue = Queue(maxsize=queue_size)
        
        # Result Queue: Infinite size (lightweight text data)
        self.result_queue = Queue()
        
        # Stop event for graceful shutdown
        self.stop_event = Event()
        self.worker_process = None
        
        # === LOCAL CACHE (PROCESS 1 ONLY) ===
        # Instant access for displaying names (no IPC overhead)
        self.track_id_name_map = {}  # {track_id: person_name}
        self.track_id_confidence = {}  # {track_id: confidence}
        
        # Optimization: Track which IDs have been queued already
        # Prevents spamming queue with same face
        self.queued_ids = set()
        
        # === PERFORMANCE METRICS ===
        self.stats = {
            'frames_processed': 0,
            'faces_queued': 0,
            'faces_dropped': 0,
            'recognitions_received': 0,
            'queue_depth_max': 0,
        }
        
        # === INITIALIZE DETECTION STAGE (PROCESS 1 ONLY) ===
        # YOLO + BoT-SORT run in main process (lightweight, ~60ms)
        self.detection_stage = DetectionStage(self.config.get_section('detector'))
        
        self.logger.info("="*60)
        self.logger.info("🚀 Async Pipeline Orchestrator Initialized")
        self.logger.info(f"   Queue size: {queue_size}")
        self.logger.info(f"   YOLO runtime: {self.config.get('detector.runtime', 'tflite')}")
        self.logger.info("="*60)
    
    def start(self):
        """Start the background AI worker process."""
        if self.worker_process is not None:
            self.logger.warning("Worker process already running")
            return
        
        self.stop_event.clear()
        
        # Convert config to dict for pickling
        config_dict = self.config.get_all()
        
        # Start AI worker in separate process
        self.worker_process = Process(
            target=ai_worker_process,
            args=(self.face_queue, self.result_queue, self.stop_event, config_dict)
        )
        self.worker_process.start()
        self.running = True
        
        self.logger.info("✅ Async Pipeline Started (Worker Process PID: %d)", self.worker_process.pid)
    
    def stop(self):
        """Graceful shutdown of worker process and queues."""
        self.logger.info("Stopping async pipeline...")
        
        self.running = False
        self.stop_event.set()
        
        if self.worker_process:
            # Wait for worker to finish current task
            self.worker_process.join(timeout=3.0)
            
            if self.worker_process.is_alive():
                self.logger.warning("Worker process did not stop gracefully, terminating...")
                self.worker_process.terminate()
                self.worker_process.join(timeout=1.0)
            
            self.worker_process = None
        
        # Close queues
        try:
            self.face_queue.close()
            self.face_queue.join_thread()
            self.result_queue.close()
            self.result_queue.join_thread()
        except Exception as e:
            self.logger.debug(f"Queue cleanup: {e}")
        
        self.logger.info("✅ Pipeline stopped")
        self._print_stats()
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process single frame through async pipeline.
        
        This is the MAIN LOOP (Process 1). It runs at 15-25 FPS.
        Heavy processing (align + recognize) happens in Process 2.
        
        Flow:
            1. YOLO detection (~60ms with NCNN @ 320x320)
            2. BoT-SORT tracking (~10ms)
            3. Queue face crops (NON-BLOCKING, <1ms)
            4. Drain result queue (NON-BLOCKING, <1ms)
            5. Annotate frame with cached names (instant)
        
        Args:
            frame: Input frame (BGR, numpy array)
            
        Returns:
            Dictionary with:
                - detections: YOLO detections
                - tracks: BoT-SORT tracks (with embeddings if recognized)
                - annotated_frame: Frame with bounding boxes and names
                - processing_time_ms: Main loop time (should be <70ms for 15 FPS)
                - queue_stats: Queue depth and drop count
        """
        start_time = time.time()
        
        # === STAGE 1 + 2: DETECTION + TRACKING ===
        detections, tracks = self.detection_stage.process_with_tracking(
            frame,
            tracker_config=self.config.get('tracker.type', 'botsort') + '.yaml'
        )
        
        # === STAGE 3: QUEUE FACE CROPS (NON-BLOCKING) ===
        for track in tracks:
            track_id = int(track.track_id)
            bbox_tlwh = track.bbox  # [x, y, w, h]
            
            # Convert tlwh to xyxy for cropping
            x1, y1, w, h = bbox_tlwh
            x2, y2 = x1 + w, y1 + h
            
            # Quality check: Only process faces > 10% of frame height
            h_frame, w_frame = frame.shape[:2]
            box_h = h
            
            if box_h < (h_frame * 0.1):
                continue  # Face too small
            
            # Optimization: Skip if already queued and recognized
            if track_id in self.track_id_name_map:
                continue  # Already recognized
            
            if track_id in self.queued_ids:
                continue  # Already in queue
            
            # Crop face
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_frame, x2), min(h_frame, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue  # Invalid crop
            
            face_crop = frame[y1:y2, x1:x2].copy()
            
            # === LEAKY BUCKET: NON-BLOCKING PUT ===
            try:
                self.face_queue.put_nowait((face_crop, track_id, time.time()))
                self.queued_ids.add(track_id)
                self.stats['faces_queued'] += 1
                
                # Track max queue depth
                try:
                    depth = self.face_queue.qsize()
                    self.stats['queue_depth_max'] = max(self.stats['queue_depth_max'], depth)
                except NotImplementedError:
                    pass  # qsize() not available on macOS
                    
            except queue.Full:
                # Queue full → Drop frame (fail fast, camera keeps running)
                self.stats['faces_dropped'] += 1
                pass
        
        # === STAGE 4: UPDATE RESULTS (NON-BLOCKING) ===
        self._drain_result_queue()
        
        # === ANNOTATE FRAME ===
        annotated_frame = self._annotate_frame(frame, tracks)
        
        # === METRICS ===
        processing_time_ms = (time.time() - start_time) * 1000
        self.stats['frames_processed'] += 1
        
        return {
            'detections': detections,
            'tracks': tracks,
            'annotated_frame': annotated_frame,
            'processing_time_ms': processing_time_ms,
            'queue_stats': {
                'queued': self.stats['faces_queued'],
                'dropped': self.stats['faces_dropped'],
                'depth_max': self.stats['queue_depth_max'],
            }
        }
    
    def _drain_result_queue(self):
        """
        Drain result queue and update local cache.
        
        This runs in Process 1 (Main Loop) and is NON-BLOCKING.
        Updates local dictionary for instant name display.
        """
        try:
            while True:
                # Get all available results without waiting
                track_id, person_name, confidence, quality = self.result_queue.get_nowait()
                
                # Update local cache (instant access for display)
                self.track_id_name_map[track_id] = person_name
                self.track_id_confidence[track_id] = confidence
                
                self.stats['recognitions_received'] += 1
                
                self.logger.info(f"✅ Recognized Track {track_id} as '{person_name}' (conf: {confidence:.2f})")
                
        except queue.Empty:
            pass  # No results available
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        tracks: List[Any]
    ) -> np.ndarray:
        """
        Draw bounding boxes and names on frame.
        
        Uses local cache for INSTANT name lookup (zero IPC overhead).
        
        Args:
            frame: Input frame
            tracks: BoT-SORT tracks
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for track in tracks:
            track_id = int(track.track_id)
            bbox_tlwh = track.bbox  # [x, y, w, h]
            
            # Convert tlwh to xyxy for drawing
            x1, y1, w, h = bbox_tlwh
            x2, y2 = x1 + w, y1 + h
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Get name from local cache (instant)
            name = self.track_id_name_map.get(track_id, f"ID: {track_id}")
            confidence = self.track_id_confidence.get(track_id, 0.0)
            
            # Color: Green if recognized, Yellow if tracking
            if track_id in self.track_id_name_map:
                color = (0, 255, 0)  # Green
                label = f"{name} ({confidence:.2f})"
            else:
                color = (0, 255, 255)  # Yellow
                label = name
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )
        
        return annotated
    
    def _print_stats(self):
        """Print performance statistics."""
        self.logger.info("="*60)
        self.logger.info("📊 Async Pipeline Statistics")
        self.logger.info("="*60)
        self.logger.info(f"  Frames processed: {self.stats['frames_processed']}")
        self.logger.info(f"  Faces queued: {self.stats['faces_queued']}")
        self.logger.info(f"  Faces dropped: {self.stats['faces_dropped']}")
        self.logger.info(f"  Recognitions received: {self.stats['recognitions_received']}")
        self.logger.info(f"  Max queue depth: {self.stats['queue_depth_max']}")
        
        if self.stats['faces_queued'] > 0:
            drop_rate = (self.stats['faces_dropped'] / self.stats['faces_queued']) * 100
            self.logger.info(f"  Drop rate: {drop_rate:.1f}%")
        
        self.logger.info("="*60)


# =============================================================================
# MAIN - For Testing
# =============================================================================

if __name__ == "__main__":
    # Test async orchestrator
    config = ConfigManager()
    orchestrator = AsyncOrchestrator(config)
    
    try:
        orchestrator.start()
        
        cap = cv2.VideoCapture(0)
        
        fps_history = []
        frame_count = 0
        fps_start = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = orchestrator.process_frame(frame)
            
            # Calculate FPS
            frame_count += 1
            if time.time() - fps_start > 1.0:
                fps = frame_count / (time.time() - fps_start)
                fps_history.append(fps)
                
                # Print main loop FPS
                queue_stats = result['queue_stats']
                print(f"Main Loop FPS: {fps:.2f} | Queue: {queue_stats['queued']} | Dropped: {queue_stats['dropped']}")
                
                frame_count = 0
                fps_start = time.time()
            
            # Display
            cv2.imshow("Async Attendance System", result['annotated_frame'])
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        orchestrator.stop()
        
        if fps_history:
            print(f"\nAverage FPS: {np.mean(fps_history):.2f}")
            print(f"Min FPS: {np.min(fps_history):.2f}")
            print(f"Max FPS: {np.max(fps_history):.2f}")
