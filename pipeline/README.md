# Pipeline Component - V3 HYBRID Orchestration Layer

**Purpose**: Coordinates all pipeline stages without implementing detection/tracking logic.

**Design Principle**: Separation of "what to do" (orchestration) from "how to do it" (implementation).

---

## 📋 Files in This Component

| File | Purpose | Runs Every Frame? | Critical? |
|------|---------|-------------------|-----------||
| `orchestrator.py` | Sequential coordinator (V3 HYBRID) | ✅ YES | ⭐ CRITICAL (baseline) |
| `async_orchestrator.py` | **Async multiprocessing coordinator** | ✅ YES | ⭐⭐ PRODUCTION |
| `detection_stage.py` | Detection orchestration | ✅ YES | ⭐ CRITICAL |
| `tracking_stage.py` | Tracking orchestration (fallback) | ❌ NO (bypassed in V3) | ⚠️ Fallback only |
| `recognition_stage.py` | Recognition orchestration | ✅ YES | ⭐ CRITICAL |
| `attendance_stage.py` | (Future) Attendance marking | 🚧 Not implemented | Future |

---

## 🔄 Execution Flow

```
attendance_system.py
    ↓
orchestrator.process_frame(frame)
    ↓
    ├─→ detection_stage.process_with_tracking(frame)  ← Uses integrated tracking
    │       ↓
    │   yolo_detector.detect_and_track(frame, 'botsort.yaml')
    │       ↓
    │   model.track(persist=True, tracker='botsort.yaml')  ← BoT-SORT runs here!
    │       ↓
    │   Returns: (detections, tracks)
    │
    ├─→ alignment_stage (future)
    ├─→ recognition_stage (future)
    └─→ attendance_stage (future)
```

---

## 📄 File Details

### 1. `orchestrator.py` ⭐ CRITICAL

**Purpose**: Main coordinator for all pipeline stages.

**Responsibilities**:
- Initialize all pipeline stages
- Run stages in sequence
- Pass data between stages
- Annotate frames for display
- Return comprehensive results

**Key Methods**:

#### `__init__(config)`
```python
def __init__(self, config):
    """
    Initialize all pipeline stages.
    
    Creates:
    - DetectionStage (always)
    - TrackingStage (fallback, not used in normal operation)
    - AlignmentStage (future, if config.alignment.enabled)
    - RecognitionStage (future, if config.recognition.enabled)
    - AttendanceStage (future, if config.attendance.enabled)
    """
```

#### `process_frame(frame)` ⭐ CRITICAL
```python
def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
    """
    Process single frame through all stages.
    
    V3 HYBRID Flow:
    1. Detection + Tracking (integrated via detect_and_track)
    2. Alignment (future)
    3. Recognition (future)
    4. Attendance (future)
    5. Annotation (draw bounding boxes + labels)
    
    Returns:
        {
            'detections': List[Detection],
            'tracks': List[Track],
            'annotated_frame': np.ndarray,
            'processing_time_ms': float
        }
    """
```

**When It Runs**: Every frame

**Critical Implementation Detail**:
```python
# Uses integrated tracking (NOT separate tracking stage!)
detections, tracks = self.detection_stage.process_with_tracking(
    frame,
    tracker_config='botsort.yaml'
)
```

---

### 2. `detection_stage.py` ⭐ CRITICAL

**Purpose**: Orchestrates face detection (delegates to DetectorFactory).

**Responsibilities**:
- Create detector via Factory pattern
- Call detection methods
- Support both `detect()` and `detect_and_track()`

**Key Methods**:

#### `process(frame)` - Basic Detection (No Tracking)
```python
def process(self, frame) -> List[Detection]:
    """
    Detect faces without tracking.
    
    ⚠️ WARNING: This does NOT include tracking persistence!
    Track IDs will fluctuate if you use this + separate tracking stage.
    
    Use process_with_tracking() for stable Track IDs.
    """
    return self.detector.detect(frame)
```

#### `process_with_tracking(frame, tracker_config)` ⭐ RECOMMENDED
```python
def process_with_tracking(self, frame, tracker_config='botsort.yaml'):
    """
    Detect AND track faces using built-in YOLO tracking.
    
    This is the CORRECT way to get persistent Track IDs!
    
    Bypasses separate tracking stage and uses YOLO's model.track()
    with persist=True for stable IDs.
    
    Returns:
        Tuple of (detections, tracks) with persistent track IDs
    """
    if hasattr(self.detector, 'detect_and_track'):
        return self.detector.detect_and_track(frame, tracker_config)
    else:
        # Fallback: detector doesn't support integrated tracking
        detections = self.detector.detect(frame)
        return detections, []
```

**When It Runs**: Every frame

**Critical Choice**: V3 HYBRID uses `process_with_tracking()` for stable Track IDs.

---

### 3. `tracking_stage.py` ⚠️ FALLBACK ONLY

**Purpose**: Orchestrates tracking using TrackerFactory (fallback mechanism).

**Responsibilities**:
- Create tracker via Factory pattern
- Update tracker with detections
- Return tracks

**When It Runs**:
- ❌ **NOT in normal V3 HYBRID operation** (bypassed by integrated tracking)
- ✅ Only when detector doesn't support integrated tracking (e.g., TFLiteDetector)
- ✅ Testing/debugging

**Key Method**:

```python
def process(self, detections: List[Detection]) -> List[Track]:
    """
    Update tracker with new detections.
    
    ⚠️ NOTE: In V3 HYBRID, this is bypassed!
    
    Tracking happens in detection_stage.process_with_tracking()
    via YOLO's model.track(persist=True).
    
    This stage is only used when:
    1. Detector doesn't support integrated tracking (TFLite)
    2. Old pattern: detect() + update() (incorrect usage)
    3. Testing without full pipeline
    """
    return self.tracker.update(detections)
```

**Why It's Bypassed**: Integrated tracking (`model.track(persist=True)`) provides:
- ✅ Persistent Track IDs
- ✅ Better tracking quality (BoT-SORT with ReID features)
- ✅ Maintained state between frames

**Read More**: See `../tracking/README.md#execution-flow`

---

## 🎯 Design Patterns

### 1. **Stage Pattern** (Orchestration)

Each stage:
- Accepts input (frame, detections, etc.)
- Delegates to implementations (via Factory)
- Returns standardized output
- Doesn't care about implementation details

```python
# Stage doesn't implement logic - just orchestrates
class DetectionStage:
    def __init__(self, config):
        self.detector = DetectorFactory.create(config)  # Delegation
    
    def process(self, frame):
        return self.detector.detect(frame)  # Orchestration
```

### 2. **Pipeline Pattern** (Sequential Processing)

```python
# Orchestrator runs stages in sequence
detections, tracks = detection_stage.process_with_tracking(frame)
aligned_faces = alignment_stage.process(frame, tracks)  # Future
identities = recognition_stage.process(aligned_faces)   # Future
attendance_stage.process(identities)                    # Future
```

---

## 🔧 Configuration

Stages are controlled by `config.yaml`:

```yaml
# Detection Stage (always enabled)
detector:
  type: yolo
  model_path: models/detection/yolov8n_face_int8.tflite
  confidence_threshold: 0.5

# Tracking (integrated with detection in V3 HYBRID)
tracker:
  type: botsort
  track_thresh: 0.4
  track_buffer: 90

# Future stages
alignment:
  enabled: false  # Set to true when implementing

recognition:
  enabled: false  # Set to true when implementing

attendance:
  enabled: false  # Set to true when implementing
```

---

## 🚀 Adding New Stages

### Example: Adding Alignment Stage

1. **Create stage file**: `pipeline/alignment_stage.py`

```python
from aligners.factory import AlignerFactory

class AlignmentStage:
    """Face alignment orchestration."""
    
    def __init__(self, config):
        self.aligner = AlignerFactory.create(config)
    
    def process(self, frame, tracks):
        """Align faces from tracks."""
        aligned_faces = []
        for track in tracks:
            aligned = self.aligner.align(frame, track.bbox)
            aligned_faces.append(aligned)
        return aligned_faces
```

2. **Update orchestrator**: `orchestrator.py`

```python
# In __init__
if config.get('alignment.enabled', False):
    self.alignment_stage = AlignmentStage(config['alignment'])

# In process_frame
if self.alignment_stage:
    aligned_faces = self.alignment_stage.process(frame, tracks)
```

3. **Update config**: `config.yaml`

```yaml
alignment:
  enabled: true
  type: mtcnn  # or 'blazeface'
  target_size: 112
```

4. **Create implementation**: `aligners/mtcnn_aligner.py`

See `docs/ARCHITECTURE_V3_HYBRID.md#adding-components` for complete guide.

---

## 🧪 Testing

### Unit Tests

```bash
# Test orchestrator
python tests/test_pipeline/test_orchestrator.py

# Test detection stage
python tests/test_pipeline/test_detection_stage.py

# Test tracking stage (fallback)
python tests/test_pipeline/test_tracking_stage.py
```

### Integration Tests

```bash
# Test full pipeline
python tests/test_pipeline/test_integration.py
```

---

## 🐛 Troubleshooting

### Track IDs Fluctuating

**Problem**: Track IDs change even when person sits still.

**Cause**: Using `detection_stage.process()` instead of `process_with_tracking()`.

**Fix**: Check `orchestrator.py`:

```python
# ❌ WRONG
detections = self.detection_stage.process(frame)
tracks = self.tracking_stage.process(detections)

# ✅ CORRECT
detections, tracks = self.detection_stage.process_with_tracking(frame)
```

---

### Slow Performance

**Problem**: Pipeline runs slowly.

**Check**:
1. Input size too large: `config.yaml` → `detector.input_size`
2. Too many threads: `config.yaml` → `runtime_settings.num_threads`
3. Frame skipping disabled: `config.yaml` → `runtime_settings.process_every_n_frames`

---

## 🔥 NEW: Async Orchestrator (Phase 3B - Production)

### `async_orchestrator.py` ⭐⭐ PRODUCTION

**Purpose**: Multiprocessing pipeline with Producer-Consumer pattern for 19.36 FPS performance.

**Architecture**:
```
Process 1 (Main Loop - 19.36 FPS):
    Camera → YOLO NCNN → BoT-SORT → Queue → Display
    (Never blocks - 50ms frame time)

Process 2 (AI Worker - Parallel):
    Queue → MediaPipe Align → AuraFace Recognition → Result Queue
    (~90ms total - doesn't affect camera FPS)
```

**Key Benefits**:
- **55% jitter reduction** - Stable FPS prevents BoT-SORT ID switching
- **4x better ID stability** - 0.55 switches/100 frames vs 2.27 sequential
- **0% frame drops** - Leaky bucket queue (maxsize=5) with put_nowait
- **Zero IPC overhead** - Result Queue + Local Cache for instant name display

**Performance**:
- Laptop (AMD Ryzen 5 3500U): 19.36 FPS mean
- Pi 5 (expected): 15-20 FPS with NCNN @ 320x320

**Usage**:
```python
from pipeline.async_orchestrator import AsyncOrchestrator

config = ConfigManager()
orchestrator = AsyncOrchestrator(config)

# Start worker process
orchestrator.start()

# Main loop (never blocks)
while True:
    ret, frame = cap.read()
    result = orchestrator.process_frame(frame)
    cv2.imshow("Attendance", result['annotated_frame'])

# Graceful shutdown
orchestrator.stop()
```

**See**: `tests/benchmark_multiprocessing.py` for performance comparison

---

## 📚 Related Documentation

- **Parent Guide**: `docs/DEVELOPER_GUIDE.md`
- **Architecture**: `docs/ARCHITECTURE_V3_HYBRID.md`
- **Detectors**: `detectors/README.md`
- **Tracking**: `tracking/README.md`

---

**Questions?** See `docs/DEVELOPER_GUIDE.md` or create an issue.

**Last Updated**: December 2, 2025 - Added AsyncOrchestrator (Phase 3B)
