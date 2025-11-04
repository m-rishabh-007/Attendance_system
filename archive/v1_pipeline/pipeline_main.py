#!/usr/bin/env python3
"""
pipeline_main.py - Production Face Detection and Tracking Pipeline

Real-time face detection and tracking using YOLOv8n INT8 TFLite + ByteTrack.
Optimized for Raspberry Pi and laptop deployment with minimal dependencies.

Key Features:
- Direct TFLite inference (no Ultralytics overhead)
- ByteTrack for persistent face IDs across frames
- Threaded camera capture to minimize frame lag
- INT8 quantization for CPU efficiency (~20 FPS laptop, ~5-8 FPS Pi)
- Configurable via pipeline_config.yaml

Quick Start:
    python pipeline_main.py

Configuration:
    Edit pipeline_config.yaml to adjust:
    - Detection thresholds (confidence, NMS)
    - Tracking parameters (track_thresh, match_thresh)
    - Performance settings (num_threads, frame skipping)
    - Camera and model paths

Performance:
    - Raspberry Pi 4: ~5-8 FPS (256x256 input, INT8 model)
    - Desktop/Laptop: ~20-22 FPS (256x256 input, INT8 model)
    - GPU acceleration: Not implemented (CPU inference is sufficient)

Author: Attendance System Project
Date: October 2025
"""

import os
import time
import threading
import queue
import yaml
import numpy as np
import cv2

# =============================================================================
# TFLite Runtime Import
# =============================================================================
# Prefer lightweight tflite-runtime over full TensorFlow for deployment.
# This reduces dependencies and memory footprint, especially on Raspberry Pi.
try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
    from tflite_runtime.interpreter import load_delegate as tflite_load_delegate
    USING_TFLITE_RUNTIME = True
    print("[INFO] Using tflite-runtime (lightweight)")
except Exception:
    # Fallback to full TensorFlow if tflite-runtime not available
    try:
        import tensorflow as tf
        TFLiteInterpreter = tf.lite.Interpreter
        try:
            tflite_load_delegate = tf.lite.experimental.load_delegate
        except AttributeError:
            tflite_load_delegate = None
        USING_TFLITE_RUNTIME = False
        print("[INFO] Using full TensorFlow (fallback)")
    except Exception:
        raise SystemExit(
            "[ERROR] Neither tflite-runtime nor tensorflow found.\n"
            "Install one of them: pip install tflite-runtime  OR  pip install tensorflow"
        )

# Import ByteTrack tracker (must be in same directory or PYTHONPATH)
from face_tracker_bytetrack import ByteTrack

# =============================================================================
# Threaded Camera Capture Class
# =============================================================================
class VideoCaptureAsync:
    """
    Asynchronous video capture using a background thread.
    
    This class solves the frame buffer lag problem by:
    1. Continuously grabbing frames in a background thread
    2. Keeping only the most recent frame (queue size = 1)
    3. Discarding old frames to ensure real-time responsiveness
    
    Without threading, cv2.VideoCapture.read() can introduce latency
    because it reads from an internal buffer that may be several frames old.
    
    Attributes:
        cap: OpenCV VideoCapture object
        q: Queue holding the most recent frame (maxsize=1)
        running: Flag to control the background thread
        thread: Background thread for frame capture
    
    Example:
        vcap = VideoCaptureAsync(src=0).start()
        frame = vcap.read()  # Always get the latest frame
        vcap.stop()
    """
    
    def __init__(self, src=0, width=None, height=None, buffer_size=1):
        """
        Initialize video capture.
        
        Args:
            src: Camera index (0 for default camera) or video file path
            width: Desired frame width (None = use camera default)
            height: Desired frame height (None = use camera default)
            buffer_size: Driver buffer size (1 = minimal buffering)
        """
        # Initialize OpenCV VideoCapture with default backend
        self.cap = cv2.VideoCapture(src)
        
        # Set resolution if specified
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Minimize driver-level buffering (not supported on all platforms)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        except Exception:
            pass
        
        # Queue to hold the most recent frame (size=1 means always latest)
        self.q = queue.Queue(maxsize=1)
        self.running = False
        self.thread = None

    def start(self):
        """Start the background thread for frame capture."""
        if self.running:
            return self  # Already running
        
        self.running = True
        # Daemon thread exits when main program exits
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        return self

    def _reader(self):
        """
        Background thread function that continuously grabs frames.
        
        This runs in a loop, always grabbing the latest frame from the camera
        and putting it in the queue. Old frames are discarded to keep the queue
        size at 1, ensuring we always get the most recent frame.
        """
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)  # Camera not ready, wait a bit
                continue
            
            # Remove old frame from queue if present
            if not self.q.empty():
                try:
                    _ = self.q.get_nowait()  # Discard old frame
                except queue.Empty:
                    pass
            
            # Add new frame to queue
            try:
                self.q.put_nowait(frame)
            except queue.Full:
                pass  # Should never happen with maxsize=1

    def read(self):
        """
        Get the most recent frame.
        
        Returns:
            numpy.ndarray or None: Latest frame, or None if no frame available
        """
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        """Stop the background thread and release camera resources."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        try:
            self.cap.release()
        except Exception:
            pass
        return self

# =============================================================================
# Helper Functions
# =============================================================================

def make_interpreter(model_path: str, num_threads: int = 4, delegate_path: str | None = None):
    """
    Create a TFLite interpreter. Prefers tflite-runtime.
    
    Args:
        model_path: Path to the .tflite model file
        num_threads: Number of CPU threads for inference
        delegate_path: Optional shared object path for delegates (EdgeTPU etc.)
    
    Returns:
        Initialized TFLite interpreter
    """
    if delegate_path is not None and tflite_load_delegate is None:
        print("Warning: delegate requested but load_delegate not available; ignoring delegate.")

    if USING_TFLITE_RUNTIME:
        # Using tflite-runtime
        if delegate_path is not None and tflite_load_delegate:
            delegates = [tflite_load_delegate(delegate_path)]
            interpreter = TFLiteInterpreter(model_path=model_path, experimental_delegates=delegates)
        else:
            interpreter = TFLiteInterpreter(model_path=model_path, num_threads=num_threads)
    else:
        # Using full TensorFlow
        if delegate_path is not None and tflite_load_delegate:
            delegates = [tflite_load_delegate(delegate_path)]
            interpreter = TFLiteInterpreter(model_path=model_path, experimental_delegates=delegates)
        else:
            interpreter = TFLiteInterpreter(model_path=model_path, num_threads=num_threads)

    interpreter.allocate_tensors()
    return interpreter


def safe_get_predictions(interpreter, output_index):
    """
    Return a 2D numpy array of predictions.
    Handles common TFLite output shapes.
    
    Args:
        interpreter: TFLite interpreter
        output_index: Index of the output tensor
    
    Returns:
        2D numpy array of predictions in format [N, features]
        where features = [x_center, y_center, width, height, objectness, class_probs...]
    """
    raw = interpreter.get_tensor(output_index)
    if raw is None:
        return np.empty((0, 6))
    raw = np.array(raw)
    
    # YOLOv8 TFLite outputs shape [1, features, predictions] - need to transpose!
    if raw.ndim == 3 and raw.shape[1] < raw.shape[2]:
        # Shape is [batch, features, predictions] -> transpose to [batch, predictions, features]
        raw = np.transpose(raw, (0, 2, 1))  # Now [1, predictions, features]
        return raw[0]  # Remove batch dimension -> [predictions, features]
    
    if raw.ndim == 3:
        return raw[0]
    if raw.ndim == 4:
        # collapse leading dims
        return raw.reshape(-1, raw.shape[-1])
    if raw.ndim == 2:
        return raw
    # fallback
    return raw.reshape(-1, raw.shape[-1])


def nms_indices(boxes, scores, conf_thresh, iou_thresh):
    """
    Returns indices after OpenCV NMSBoxes. Accepts boxes as [[x,y,w,h],...].
    
    Args:
        boxes: List of bounding boxes in [x, y, w, h] format
        scores: List of confidence scores
        conf_thresh: Confidence threshold
        iou_thresh: IoU threshold for NMS
    
    Returns:
        numpy array of indices after NMS
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
    if isinstance(indices, (list, tuple, np.ndarray)):
        inds = np.array(indices).reshape(-1)
    else:
        # single value
        inds = np.array([int(indices)])
    # filter invalids
    inds = inds[inds >= 0]
    return inds


def load_config():
    """
    Load configuration from pipeline_config.yaml.
    
    Returns:
        tuple: (config dict, script directory path)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "pipeline_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"pipeline_config.yaml not found at {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg, script_dir


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    """
    Main detection and tracking pipeline.
    
    Loads configuration, initializes model and camera, runs real-time detection
    and tracking, and displays results with FPS statistics.
    """
    cfg, script_dir = load_config()

    # Model path (relative -> absolute)
    model_path = cfg.get('tflite_model', {}).get('path', 'models/tflite_saved_models/yolov8n_face_int8.tflite')
    if not os.path.isabs(model_path):
        model_path = os.path.join(script_dir, model_path)
    if not os.path.exists(model_path):
        print(f"Error: model file not found at {model_path}")
        return

    # Inference settings (with safe defaults)
    INPUT_SIZE = int(cfg.get('inference_settings', {}).get('input_size', 640))
    CONF_THRESH = float(cfg.get('inference_settings', {}).get('confidence_threshold', 0.6))
    IOU_THRESH = float(cfg.get('inference_settings', {}).get('iou_threshold', 0.3))

    # Runtime / latency tuning (optional keys in config -> runtime_settings)
    runtime_cfg = cfg.get('runtime_settings', {}) or {}
    NUM_TFLITE_THREADS = int(runtime_cfg.get('num_threads', 3))
    PROCESS_EVERY_N_FRAMES = int(runtime_cfg.get('process_every_n_frames', 1))  # 1 -> process all
    DESIRED_INPUT = int(runtime_cfg.get('desired_input', INPUT_SIZE))  # you can set 320 for speed
    CAMERA_ID = int(cfg.get('deployment_config', {}).get('camera_id', 0))
    DELEGATE = cfg.get('deployment_config', {}).get('delegate', None)  # optional delegate path

    print(f"Model: {model_path}")
    print(f"Input size: {DESIRED_INPUT}  | Conf thresh: {CONF_THRESH}  | IoU: {IOU_THRESH}")
    print(f"Threads: {NUM_TFLITE_THREADS}  | Skip every N frames: {PROCESS_EVERY_N_FRAMES-1}  | Camera: {CAMERA_ID}")

    # Create interpreter
    interpreter = make_interpreter(model_path=model_path, num_threads=NUM_TFLITE_THREADS, delegate_path=DELEGATE)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    output_index = output_details[0]['index']

    # Initialize tracker with config values
    TRACK_THRESH = cfg['tracking_settings']['track_thresh']
    TRACK_BUFFER = cfg['tracking_settings']['track_buffer']
    MATCH_THRESH = cfg['tracking_settings']['match_thresh']
    FRAME_RATE = cfg['tracking_settings']['frame_rate']
    tracker = ByteTrack(track_thresh=TRACK_THRESH, track_buffer=TRACK_BUFFER, match_thresh=MATCH_THRESH, frame_rate=FRAME_RATE)

    # Start threaded capture
    vcap = VideoCaptureAsync(src=CAMERA_ID, width=None, height=None, buffer_size=1).start()
    print("Camera started (threaded). Press 'q' to quit.")

    frame_counter = 0
    processed_frames = 0
    t_start = time.time()

    try:
        while True:
            frame = vcap.read()
            if frame is None:
                time.sleep(0.003)
                continue

            frame_counter += 1

            # Show latest frame if skipping processing (keeps UI responsive)
            if (frame_counter % PROCESS_EVERY_N_FRAMES) != 0:
                cv2.imshow("Live - ByteTrack", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Preprocess: RGB normalize and resize to DESIRED_INPUT
            original_h, original_w = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(img_rgb, (DESIRED_INPUT, DESIRED_INPUT))
            input_data = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0

            # Set input (handle models that want varying input details)
            try:
                interpreter.set_tensor(input_details[0]['index'], input_data)
            except Exception:
                # If interpreter expects other shapes, try to reshape (rare)
                pass

            t_inf0 = time.time()
            try:
                interpreter.invoke()
            except Exception as e:
                print("Interpreter invoke error:", e)
                break
            t_inf1 = time.time()

            preds = safe_get_predictions(interpreter, output_index)

            # DEBUG: Print first few predictions to understand format
            if processed_frames == 0 and len(preds) > 0:
                print(f"\n[DEBUG] First prediction shape: {preds.shape}")
                print(f"[DEBUG] First 3 predictions:\n{preds[:3]}")
                print(f"[DEBUG] Prediction value ranges:")
                print(f"  X (pred[0]): min={preds[:, 0].min():.2f}, max={preds[:, 0].max():.2f}")
                print(f"  Y (pred[1]): min={preds[:, 1].min():.2f}, max={preds[:, 1].max():.2f}")
                print(f"  W (pred[2]): min={preds[:, 2].min():.2f}, max={preds[:, 2].max():.2f}")
                print(f"  H (pred[3]): min={preds[:, 3].min():.2f}, max={preds[:, 3].max():.2f}")
                if preds.shape[1] > 4:
                    print(f"  Objectness (pred[4]): min={preds[:, 4].min():.2f}, max={preds[:, 4].max():.2f}")

            # Postprocess predictions:
            boxes = []   # [x,y,w,h] in original image coords
            scores = []
            classes = []

            # Typical YOLO (nms=False) layout: [x_c, y_c, w, h, objectness, class_probs...]
            for pred in preds:
                if pred.size < 5:
                    continue
                objectness = float(pred[4])
                if pred.size > 5:
                    class_probs = np.array(pred[5:], dtype=float)
                    best_cls = int(np.argmax(class_probs))
                    class_conf = float(class_probs[best_cls])
                else:
                    # single-class model or missing class probs: treat as class_conf=1
                    best_cls = 0
                    class_conf = 1.0
                conf = objectness * class_conf
                if conf <= CONF_THRESH:
                    continue

                x_c = float(pred[0])
                y_c = float(pred[1])
                w_rel = float(pred[2])
                h_rel = float(pred[3])

                # Map from normalized (0..1) on DESIRED_INPUT to original frame size
                # Note: We used simple resize (no letterbox). If you later use letterbox, change mapping accordingly.
                x1 = int((x_c - w_rel / 2.0) * original_w)
                y1 = int((y_c - h_rel / 2.0) * original_h)
                bw = int(w_rel * original_w)
                bh = int(h_rel * original_h)

                x1 = max(0, min(x1, original_w - 1))
                y1 = max(0, min(y1, original_h - 1))
                bw = max(1, min(bw, original_w - x1))
                bh = max(1, min(bh, original_h - y1))

                boxes.append([float(x1), float(y1), float(bw), float(bh)])  # Keep as floats for tracker
                scores.append(float(conf))
                classes.append(best_cls)

            # NMS & prepare for tracker
            dets_for_tracker = []
            scores_for_tracker = []
            classes_for_tracker = []

            if len(boxes) > 0:
                inds = nms_indices(boxes, scores, CONF_THRESH, IOU_THRESH)
                for i in inds:
                    i = int(i)
                    dets_for_tracker.append(boxes[i])
                    scores_for_tracker.append(scores[i])
                    classes_for_tracker.append(classes[i])

            # Update tracker (expects numpy arrays)
            if len(dets_for_tracker) > 0:
                online_targets = tracker.update(np.array(dets_for_tracker), np.array(classes_for_tracker), np.array(scores_for_tracker))
                

                
                # Draw tracked boxes
                for t in online_targets:
                    tlwh = t.to_tlwh()  # returns [x, y, w, h]
                    x1, y1, w, h = [int(i) for i in tlwh]
                    track_id = t.track_id
                    conf = t.conf  # Get confidence from track
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                    
                    # Draw label with ID and confidence
                    label = f"ID:{track_id} FACE:{conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                pass  # No detections to track

            # Display and profiling
            processed_frames += 1
            if processed_frames % 20 == 0:
                elapsed = time.time() - t_start
                avg_fps = processed_frames / elapsed if elapsed > 0 else 0.0
                print(f"[stats] processed={processed_frames} avg_FPS={avg_fps:.2f} last_infer={(t_inf1-t_inf0)*1000:.1f}ms dets={len(dets_for_tracker)} total_preds={len(preds)}")

            cv2.imshow("Live - ByteTrack", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        vcap.stop()
        cv2.destroyAllWindows()
        print("Exiting cleanly.")


if __name__ == "__main__":
    main()
