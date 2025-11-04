# Detectors Component - Face Detection Implementations

**Purpose**: Implement face detection algorithms with Factory pattern for easy swapping.

**Key Concept**: Detectors provide TWO methods:
- `detect()` - Basic detection (no tracking persistence)
- `detect_and_track()` - ⭐ RECOMMENDED (persistent Track IDs via integrated tracking)

---

## 📋 Files in This Component

| File | Purpose | Current Status | Critical? |
|------|---------|----------------|-----------|
| `base.py` | BaseFaceDetector interface | ✅ Complete | ⭐ CRITICAL |
| `factory.py` | Factory pattern for detector creation | ✅ Complete | ⭐ CRITICAL |
| `yolo_detector.py` | YOLO implementation (production) | ✅ Complete | ⭐ CRITICAL |
| `tflite_detector.py` | Direct TFLite inference (fallback) | ✅ Complete | ⚠️ Fallback |
| `retinaface_detector.py` | (Future) RetinaFace implementation | 🚧 Not implemented | Future |
| `mtcnn_detector.py` | (Future) MTCNN implementation | 🚧 Not implemented | Future |

---

## 🎯 Critical Concept: `detect()` vs `detect_and_track()`

### Problem: Why Track IDs Fluctuate

**Using `detect()` only**:
```python
# ❌ BROKEN PATTERN (Track IDs fluctuate!)
detections = detector.detect(frame)         # No state maintained
tracks = tracker.update(detections)         # Just assigns sequential IDs
# Result: Person sitting still gets ID: 1, 2, 3, 4... (different every frame)
```

**Why It Breaks**:
- `detect()` returns fresh detections each frame (no memory)
- Tracker has NO way to know "this is the same person as last frame"
- Tracker assigns new IDs based on position/overlap (unreliable)

---

### Solution: Integrated Tracking with `detect_and_track()`

**Using `detect_and_track()`**:
```python
# ✅ CORRECT PATTERN (Persistent Track IDs!)
detections, tracks = detector.detect_and_track(frame, tracker_config='botsort.yaml')
# Result: Person sitting still gets ID: 1 (same across all frames)
```

**Why It Works**:
- Uses `model.track(persist=True)` internally
- Maintains tracking state INSIDE the model
- Kalman filter predicts position between frames
- ReID features (appearance similarity) help match across occlusions
- Track IDs persist across entire session

---

## 📄 File Details

### 1. `base.py` - Interface Definition

**Purpose**: Define common interface for all detectors.

```python
from abc import ABC, abstractmethod
from typing import List, Tuple
from common.base_classes import Detection, Track

class BaseFaceDetector(ABC):
    """Base interface for all face detectors."""
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect faces in frame (NO tracking persistence).
        
        Returns:
            List of Detection objects (bbox, confidence, class)
        """
        pass
    
    def detect_and_track(self, frame: np.ndarray, tracker_config: str) -> Tuple[List[Detection], List[Track]]:
        """
        Detect AND track faces (WITH persistent Track IDs).
        
        ⚠️ DEFAULT IMPLEMENTATION: Not all detectors support this!
        
        YOLODetector overrides this with model.track(persist=True).
        TFLiteDetector falls back to detect() + warning.
        
        Returns:
            Tuple of (detections, tracks)
        """
        detections = self.detect(frame)
        return detections, []  # Fallback: no tracking
```

---

### 2. `factory.py` - Factory Pattern ⭐ CRITICAL

**Purpose**: Create detectors based on configuration.

```python
class DetectorFactory:
    """Factory for creating detector instances."""
    
    @staticmethod
    def create(config: dict) -> BaseFaceDetector:
        """
        Create detector based on config.
        
        Supported types:
        - 'yolo': YOLODetector (production, supports integrated tracking)
        - 'tflite': TFLiteDetector (fallback, no integrated tracking)
        - 'retinaface': (future)
        - 'mtcnn': (future)
        """
        detector_type = config.get('detector', {}).get('type', 'yolo')
        
        if detector_type == 'yolo':
            from detectors.yolo_detector import YOLODetector
            return YOLODetector(config)
        
        elif detector_type == 'tflite':
            from detectors.tflite_detector import TFLiteDetector
            return TFLiteDetector(config)
        
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
```

**Usage in Pipeline**:
```python
# detection_stage.py
detector = DetectorFactory.create(config)  # Don't care which one!
detections, tracks = detector.detect_and_track(frame)  # Just works!
```

---

### 3. `yolo_detector.py` - Production Implementation ⭐ CRITICAL

**Purpose**: Face detection using Ultralytics YOLO with integrated BoT-SORT tracking.

**Key Features**:
- ✅ Supports `detect_and_track()` with `persist=True`
- ✅ Uses BoT-SORT tracker built into Ultralytics
- ✅ Persistent Track IDs across session
- ✅ TFLite INT8 quantized model (fast on CPU)

#### `__init__(config)`

```python
def __init__(self, config):
    """
    Initialize YOLO detector.
    
    Loads:
    - TFLite INT8 model (1.5MB, optimized for Pi)
    - Confidence threshold from config
    - Tracking parameters
    """
    model_path = config.get('detector', {}).get('model_path')
    self.model = YOLO(model_path, task='detect')
    self.conf_threshold = config.get('detector', {}).get('confidence_threshold', 0.5)
```

#### `detect(frame)` - Basic Detection

```python
def detect(self, frame: np.ndarray) -> List[Detection]:
    """
    Detect faces WITHOUT tracking persistence.
    
    ⚠️ WARNING: Use detect_and_track() for stable Track IDs!
    
    This method:
    - Runs inference
    - Returns fresh detections each frame
    - NO memory of previous frames
    
    Use Case: When you don't need tracking (e.g., single-frame analysis)
    """
    results = self.model.predict(
        source=frame,
        conf=self.conf_threshold,
        verbose=False
    )
    
    detections = []
    for box in results[0].boxes:
        detection = Detection(
            bbox=box.xyxy[0].cpu().numpy(),
            confidence=float(box.conf[0]),
            class_id=int(box.cls[0])
        )
        detections.append(detection)
    
    return detections
```

#### `detect_and_track(frame, tracker_config)` - ⭐ RECOMMENDED

```python
def detect_and_track(self, frame: np.ndarray, tracker_config: str = 'botsort.yaml') -> Tuple[List[Detection], List[Track]]:
    """
    Detect AND track faces using YOLO's integrated tracking.
    
    ⭐ THIS IS THE CORRECT METHOD FOR PERSISTENT TRACK IDS!
    
    How It Works:
    1. Calls model.track(persist=True, tracker='botsort.yaml')
    2. BoT-SORT runs INSIDE Ultralytics (not in tracking/botsort_tracker.py!)
    3. Maintains Kalman filter state between frames
    4. Uses ReID features for appearance matching
    5. Returns same Track ID for same person across frames
    
    Key Parameter:
        persist=True  ← CRITICAL! Without this, IDs reset each frame
    
    Returns:
        Tuple of:
        - detections: List[Detection] (bbox, confidence, class)
        - tracks: List[Track] (bbox, confidence, track_id, state)
    """
    results = self.model.track(
        source=frame,
        conf=self.conf_threshold,
        persist=True,           # ← CRITICAL FOR PERSISTENT IDS!
        tracker=tracker_config,  # 'botsort.yaml' or 'bytetrack.yaml'
        verbose=False
    )
    
    detections = []
    tracks = []
    
    for box in results[0].boxes:
        # Create Detection object
        detection = Detection(
            bbox=box.xyxy[0].cpu().numpy(),
            confidence=float(box.conf[0]),
            class_id=int(box.cls[0])
        )
        detections.append(detection)
        
        # Create Track object (if track ID available)
        if hasattr(box, 'id') and box.id is not None:
            track = Track(
                track_id=int(box.id[0]),
                bbox=box.xyxy[0].cpu().numpy(),
                confidence=float(box.conf[0]),
                state='tracked',
                frames_since_update=0
            )
            tracks.append(track)
    
    return detections, tracks
```

**Where BoT-SORT Actually Runs**:
```
yolo_detector.detect_and_track()
    ↓
model.track(persist=True, tracker='botsort.yaml')
    ↓
Ultralytics internal tracking pipeline
    ├─ Load tracker config from ultralytics/cfg/trackers/botsort.yaml
    ├─ Initialize BoT-SORT with Kalman filter
    ├─ Run detection
    ├─ Run BoT-SORT matching (IoU + ReID features)
    ├─ Update track states
    └─ Return results with persistent track IDs
```

**Key Point**: BoT-SORT runs INSIDE Ultralytics, NOT in `tracking/botsort_tracker.py`!

---

### 4. `tflite_detector.py` - Fallback Implementation

**Purpose**: Direct TFLite inference (for custom models, research).

**Limitations**:
- ❌ No integrated tracking support (must use separate tracking stage)
- ❌ Manual pre/post-processing required
- ❌ Track IDs will fluctuate (no persist=True option)

**When To Use**:
- Custom TFLite models not supported by Ultralytics
- Research/experimentation
- Edge devices with memory constraints

**Fallback Behavior**:
```python
def detect_and_track(self, frame, tracker_config):
    """
    Fallback: TFLite doesn't support integrated tracking.
    
    ⚠️ WARNING: This will NOT give persistent Track IDs!
    
    Returns detections only. Tracking must be done separately
    using tracking/botsort_tracker.py (which also lacks persistence).
    """
    print("⚠️ TFLiteDetector doesn't support integrated tracking!")
    print("   Track IDs may fluctuate. Consider using YOLODetector.")
    
    detections = self.detect(frame)
    return detections, []
```

---

## 🎯 Design Patterns

### 1. **Factory Pattern**

Easily swap detectors via configuration:

```python
# config.yaml
detector:
  type: yolo  # Change to 'tflite', 'retinaface', etc.

# Code doesn't change!
detector = DetectorFactory.create(config)
detections, tracks = detector.detect_and_track(frame)
```

### 2. **Strategy Pattern**

Different detection algorithms for different scenarios:
- **YOLO**: Fast, accurate, supports tracking (production)
- **TFLite**: Custom models, research (fallback)
- **RetinaFace**: (Future) High accuracy for small faces
- **MTCNN**: (Future) Lightweight, cascade detection

### 3. **Interface Segregation**

Base class defines BOTH methods:
- `detect()` - All detectors must implement
- `detect_and_track()` - Optional (has default fallback)

---

## 🔧 Configuration

```yaml
# config.yaml

detector:
  type: yolo                              # 'yolo', 'tflite', etc.
  model_path: models/detection/yolov8n_face_int8.tflite
  confidence_threshold: 0.5               # Min detection confidence
  
tracker:
  type: botsort                           # Used by detect_and_track()
  track_thresh: 0.4                       # Min confidence to start tracking
  track_buffer: 90                        # Frames to keep lost tracks (3 sec @ 30fps)
  match_thresh: 0.4                       # IoU threshold for matching
```

---

## 🚀 Adding New Detectors

### Example: Adding RetinaFace

1. **Create implementation**: `detectors/retinaface_detector.py`

```python
from detectors.base import BaseFaceDetector
import numpy as np

class RetinaFaceDetector(BaseFaceDetector):
    """RetinaFace detection implementation."""
    
    def __init__(self, config):
        model_path = config.get('detector', {}).get('model_path')
        # Load RetinaFace model
        self.model = load_retinaface(model_path)
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run RetinaFace detection."""
        faces, landmarks = self.model.detect_faces(frame)
        
        detections = []
        for face in faces:
            detection = Detection(
                bbox=face['box'],
                confidence=face['confidence'],
                class_id=0,  # face class
                landmarks=face['landmarks']  # 5-point landmarks
            )
            detections.append(detection)
        
        return detections
    
    # Note: detect_and_track() uses default fallback
    # (RetinaFace doesn't support integrated tracking)
```

2. **Register in factory**: `detectors/factory.py`

```python
elif detector_type == 'retinaface':
    from detectors.retinaface_detector import RetinaFaceDetector
    return RetinaFaceDetector(config)
```

3. **Update config**: `config.yaml`

```yaml
detector:
  type: retinaface
  model_path: models/detection/retinaface.onnx
  confidence_threshold: 0.7
```

4. **Test**:

```bash
python attendance_system.py  # Should use RetinaFace automatically!
```

---

## 🧪 Testing

### Unit Tests

```bash
# Test detector factory
python tests/test_detectors/test_factory.py

# Test YOLO detector
python tests/test_detectors/test_yolo_detector.py

# Test integrated tracking
python tests/test_detectors/test_persistent_tracking.py
```

### Manual Testing

```python
# Test detect() only
python -c "
from detectors.yolo_detector import YOLODetector
import cv2

detector = YOLODetector({'detector': {'model_path': 'models/yolov8n_face_int8.tflite'}})
frame = cv2.imread('test_image.jpg')
detections = detector.detect(frame)
print(f'Detected {len(detections)} faces')
"

# Test detect_and_track()
python -c "
from detectors.yolo_detector import YOLODetector
import cv2

detector = YOLODetector({'detector': {'model_path': 'models/yolov8n_face_int8.tflite'}})
cap = cv2.VideoCapture(0)

for i in range(100):
    ret, frame = cap.read()
    detections, tracks = detector.detect_and_track(frame)
    if tracks:
        print(f'Frame {i}: Track IDs = {[t.track_id for t in tracks]}')
        # Should see SAME IDs if person sits still!
"
```

---

## 🐛 Troubleshooting

### Track IDs Still Fluctuating

**Check**:
1. Using `detect_and_track()` with `persist=True`? ✅
2. Using YOLODetector (not TFLiteDetector)? ✅
3. Good lighting and clear faces? ✅
4. Person not moving too fast? ✅

**Verify in code**:
```python
# yolo_detector.py
results = self.model.track(
    source=frame,
    persist=True,  # ← This MUST be True!
    tracker='botsort.yaml',
    verbose=False
)
```

---

### Low Detection Accuracy

**Tune confidence threshold**:
```yaml
detector:
  confidence_threshold: 0.3  # Lower = more detections (more false positives)
```

**Check lighting**:
- Need adequate illumination
- Avoid backlighting
- Clean camera lens

---

### Slow Performance

**Reduce input size**:
```yaml
runtime_settings:
  desired_input: 256  # Try 256 instead of 320/640
  num_threads: 3      # Adjust for your hardware
```

**Enable frame skipping**:
```yaml
runtime_settings:
  process_every_n_frames: 2  # Process every 2nd frame
```

---

## 📊 Performance Comparison

| Detector | Speed (Pi 4) | Accuracy | Tracking Support | Use Case |
|----------|--------------|----------|------------------|----------|
| YOLODetector | ~20 FPS | ⭐⭐⭐⭐ | ✅ YES (persist=True) | **Production** |
| TFLiteDetector | ~15 FPS | ⭐⭐⭐ | ❌ NO | Research/Custom |
| RetinaFace | ~10 FPS | ⭐⭐⭐⭐⭐ | ❌ NO | High accuracy |
| MTCNN | ~8 FPS | ⭐⭐⭐ | ❌ NO | Lightweight |

**Recommendation**: Use YOLODetector for production (best balance of speed + accuracy + tracking).

---

## 📚 Related Documentation

- **Parent Guide**: `docs/DEVELOPER_GUIDE.md`
- **Architecture**: `docs/ARCHITECTURE_V3_HYBRID.md`
- **Pipeline**: `pipeline/README.md`
- **Tracking**: `tracking/README.md` (explains BoT-SORT execution)

---

## 🎓 Key Takeaways

1. **Always use `detect_and_track()` for persistent Track IDs** (not `detect()` + separate tracker)
2. **BoT-SORT runs INSIDE Ultralytics** (not in `tracking/botsort_tracker.py`)
3. **`persist=True` is CRITICAL** (without it, IDs reset each frame)
4. **YOLODetector is production-ready** (TFLiteDetector is fallback)
5. **Factory pattern makes swapping easy** (change config, not code)

---

**Questions?** See `docs/DEVELOPER_GUIDE.md` or create an issue.

**Last Updated**: November 5, 2025
