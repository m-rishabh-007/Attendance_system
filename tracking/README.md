# Tracking Component - Multi-Object Tracking Implementations

**Purpose**: Implement tracking algorithms with Strategy pattern for swappable trackers.

**Critical Understanding**: In V3 HYBRID, this component is mostly a FALLBACK!

---

## ⚠️ IMPORTANT: When Does Tracking Actually Run?

### Two Execution Paths

| Scenario | Where Tracking Runs | Persistent IDs? | Quality |
|----------|---------------------|-----------------|---------|
| **Normal Operation** (V3 HYBRID) | Inside `model.track(persist=True)` in YOLODetector | ✅ YES | ⭐⭐⭐⭐⭐ Excellent |
| **Fallback** (TFLite/Testing) | In `tracking/botsort_tracker.py` | ❌ NO | ⭐⭐ Poor |

### Visual Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    NORMAL OPERATION (V3 HYBRID)                 │
└─────────────────────────────────────────────────────────────────┘

orchestrator.process_frame(frame)
    ↓
detection_stage.process_with_tracking(frame, 'botsort.yaml')
    ↓
yolo_detector.detect_and_track(frame, 'botsort.yaml')
    ↓
model.track(persist=True, tracker='botsort.yaml')
    ↓
╔═══════════════════════════════════════════════════════════════╗
║  ULTRALYTICS INTERNAL TRACKING (BoT-SORT)                     ║
║  - Loads config from ultralytics/cfg/trackers/botsort.yaml   ║
║  - Initializes Kalman filter                                  ║
║  - Runs detection                                             ║
║  - Performs IoU + ReID matching                               ║
║  - Maintains state between frames                             ║
║  - Returns results with PERSISTENT Track IDs                  ║
╚═══════════════════════════════════════════════════════════════╝
    ↓
Returns: (detections, tracks) with stable IDs

⚠️ tracking/botsort_tracker.py is NOT used in this flow!


┌─────────────────────────────────────────────────────────────────┐
│                  FALLBACK OPERATION (TFLite/Testing)            │
└─────────────────────────────────────────────────────────────────┘

orchestrator.process_frame(frame)
    ↓
detection_stage.process(frame)  ← No integrated tracking!
    ↓
tflite_detector.detect(frame)
    ↓
Returns: detections only (no tracks)
    ↓
tracking_stage.process(detections)
    ↓
╔═══════════════════════════════════════════════════════════════╗
║  tracking/botsort_tracker.py (FALLBACK WRAPPER)               ║
║  - Converts detections to tracker format                      ║
║  - Calls minimal BoT-SORT matching                            ║
║  - NO Kalman filter state maintained!                         ║
║  - NO ReID features!                                          ║
║  - Returns tracks with FLUCTUATING IDs                        ║
╚═══════════════════════════════════════════════════════════════╝
    ↓
Returns: tracks with unstable IDs ❌
```

---

## 📋 Files in This Component

| File | Purpose | Used in Normal Operation? | Critical? |
|------|---------|---------------------------|-----------|
| `base.py` | BaseTracker interface | ❌ NO (bypassed) | ⚠️ Interface only |
| `factory.py` | Factory pattern for tracker creation | ❌ NO (bypassed) | ⚠️ Fallback only |
| `botsort_tracker.py` | BoT-SORT wrapper (fallback) | ❌ NO (bypassed) | ⚠️ Fallback only |
| `no_tracker.py` | Dummy tracker (testing) | ❌ NO | Testing only |
| `bytetrack_tracker.py` | (Future) ByteTrack wrapper | ❌ NO | Future fallback |
| `deepsort_tracker.py` | (Future) DeepSORT wrapper | ❌ NO | Future fallback |

---

## 🎯 Key Concept: Why This Component Exists

### Historical Context

**V1.0 Architecture** (Old):
```python
# Sequential pattern - tracking in separate stage
detections = detector.detect(frame)
tracks = tracker.update(detections)
```

**Problem**: Track IDs fluctuate because tracker has no memory of model's internal state.

**V3 HYBRID Solution**:
```python
# Integrated pattern - tracking inside detector
detections, tracks = detector.detect_and_track(frame)
# Uses model.track(persist=True) internally
```

**Result**: Track IDs persist because model maintains state.

### So Why Keep This Component?

**Answer**: Fallback for detectors that DON'T support integrated tracking:
- TFLiteDetector (custom models)
- Future detectors (RetinaFace, MTCNN)
- Research/testing scenarios

---

## 📄 File Details

### 1. `base.py` - Interface Definition

```python
from abc import ABC, abstractmethod
from typing import List
from common.base_classes import Detection, Track

class BaseTracker(ABC):
    """Base interface for all tracking algorithms."""
    
    @abstractmethod
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracker with new detections.
        
        ⚠️ WARNING: This is a FALLBACK method!
        
        In normal operation, tracking happens via:
        detector.detect_and_track(frame)  ← Uses model.track(persist=True)
        
        This method is only called when:
        1. Detector doesn't support integrated tracking (TFLite)
        2. Old pattern: detect() + update() (incorrect)
        3. Testing without full pipeline
        
        Returns:
            List of Track objects (may have fluctuating IDs!)
        """
        pass
```

---

### 2. `factory.py` - Factory Pattern

```python
class TrackerFactory:
    """Factory for creating tracker instances."""
    
    @staticmethod
    def create(config: dict) -> BaseTracker:
        """
        Create tracker based on config.
        
        ⚠️ NOTE: In V3 HYBRID, this is only used for fallback scenarios!
        
        Normal tracking happens in YOLODetector.detect_and_track().
        
        Supported types:
        - 'botsort': BoTSORT wrapper (fallback)
        - 'bytetrack': ByteTrack wrapper (future)
        - 'deepsort': DeepSORT wrapper (future)
        - 'none': No tracking (testing)
        """
        tracker_type = config.get('tracker', {}).get('type', 'botsort')
        
        if tracker_type == 'botsort':
            from tracking.botsort_tracker import BoTSORTTracker
            return BoTSORTTracker(config)
        
        elif tracker_type == 'none':
            from tracking.no_tracker import NoTracker
            return NoTracker()
        
        else:
            raise ValueError(f"Unknown tracker type: {tracker_type}")
```

---

### 3. `botsort_tracker.py` - BoT-SORT Wrapper (FALLBACK) ⚠️

**Purpose**: Minimal BoT-SORT implementation for fallback scenarios.

**Limitations**:
- ❌ No Kalman filter state between frames
- ❌ No ReID features (appearance similarity)
- ❌ No tracking persistence
- ❌ Track IDs fluctuate

**When It Runs**:
1. Using TFLiteDetector (doesn't support integrated tracking)
2. Someone incorrectly uses old pattern: `detect()` + `update()`
3. Unit testing without full pipeline

#### Key Implementation

```python
class BoTSORTTracker(BaseTracker):
    """
    BoT-SORT wrapper for fallback scenarios.
    
    ⚠️ CRITICAL LIMITATION: This is NOT the same BoT-SORT that runs
    in YOLODetector.detect_and_track()!
    
    Differences:
    
    | Feature | This Wrapper | Integrated BoT-SORT |
    |---------|--------------|---------------------|
    | Kalman Filter | ❌ NO | ✅ YES |
    | ReID Features | ❌ NO | ✅ YES |
    | Persistent State | ❌ NO | ✅ YES |
    | Track ID Stability | ❌ Poor | ✅ Excellent |
    
    Use Case: ONLY when detector doesn't support integrated tracking.
    """
    
    def __init__(self, config):
        self.track_thresh = config.get('tracker', {}).get('track_thresh', 0.4)
        self.track_buffer = config.get('tracker', {}).get('track_buffer', 30)
        self.match_thresh = config.get('tracker', {}).get('match_thresh', 0.8)
        
        # Minimal state tracking (just for IoU matching)
        self.tracked_tracks = []
        self.next_id = 0
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracker with new detections.
        
        ⚠️ WARNING: This is a FALLBACK implementation with NO tracking persistence!
        
        Track IDs WILL fluctuate because this doesn't maintain state between frames.
        
        Real BoT-SORT (with Kalman filter, ReID features, persistent IDs) happens
        when you use:
            detector.detect_and_track(frame)  ← Uses model.track(persist=True)
        
        This method is only called when:
        1. Someone incorrectly uses detector.detect() + tracker.update()
        2. Testing without full pipeline
        3. Using TFLiteDetector (which doesn't support integrated tracking)
        
        What This Does:
        - Simple IoU matching (no Kalman prediction)
        - Assigns new IDs if IoU match fails
        - NO appearance features (ReID)
        - NO state persistence between frames
        
        Result: Track IDs will change frequently, even for stationary faces!
        """
        if not detections:
            return []
        
        # Convert detections to tracker format (tlwh)
        det_boxes = []
        det_scores = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            w, h = x2 - x1, y2 - y1
            det_boxes.append([x1, y1, w, h])  # tlwh format
            det_scores.append(det.confidence)
        
        det_boxes = np.array(det_boxes)
        det_scores = np.array(det_scores)
        
        # Simple IoU matching (no Kalman filter!)
        tracks = []
        for i, (box, score) in enumerate(zip(det_boxes, det_scores)):
            # Try to match with existing tracks using IoU
            matched = False
            for existing_track in self.tracked_tracks:
                iou = self._compute_iou(box, existing_track['bbox'])
                if iou > self.match_thresh:
                    # Matched! Use existing ID
                    track = Track(
                        track_id=existing_track['id'],
                        bbox=detections[i].bbox,
                        confidence=score,
                        state='tracked',
                        frames_since_update=0
                    )
                    tracks.append(track)
                    matched = True
                    break
            
            if not matched:
                # New track - assign new ID
                track = Track(
                    track_id=self.next_id,
                    bbox=detections[i].bbox,
                    confidence=score,
                    state='new',
                    frames_since_update=0
                )
                tracks.append(track)
                self.next_id += 1
        
        # Update tracked_tracks (simple replacement - no state!)
        self.tracked_tracks = [
            {'id': t.track_id, 'bbox': self._to_tlwh(t.bbox)}
            for t in tracks
        ]
        
        return tracks
    
    def _compute_iou(self, box1, box2):
        """Compute IoU between two boxes (tlwh format)."""
        # Basic IoU calculation
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _to_tlwh(self, bbox):
        """Convert xyxy to tlwh format."""
        x1, y1, x2, y2 = bbox
        return [x1, y1, x2 - x1, y2 - y1]
```

---

### 4. `no_tracker.py` - Dummy Tracker (Testing)

```python
class NoTracker(BaseTracker):
    """Dummy tracker that just assigns sequential IDs (for testing)."""
    
    def __init__(self):
        self.next_id = 0
    
    def update(self, detections):
        """Convert detections to tracks with sequential IDs."""
        tracks = []
        for det in detections:
            track = Track(
                track_id=self.next_id,
                bbox=det.bbox,
                confidence=det.confidence,
                state='new',
                frames_since_update=0
            )
            tracks.append(track)
            self.next_id += 1
        
        return tracks
```

---

## 🔄 Execution Scenarios

### Scenario 1: Normal Operation ✅ RECOMMENDED

```python
# config.yaml
detector:
  type: yolo

# Execution flow
orchestrator.process_frame(frame)
    ↓
detection_stage.process_with_tracking(frame, 'botsort.yaml')
    ↓
yolo_detector.detect_and_track(frame, 'botsort.yaml')
    ↓
model.track(persist=True, tracker='botsort.yaml')
    ↓ BoT-SORT runs INSIDE Ultralytics!
detections, tracks = results

# Result
✅ Persistent Track IDs
✅ Excellent tracking quality
✅ tracking/botsort_tracker.py NOT used
```

### Scenario 2: TFLite Fallback ⚠️

```python
# config.yaml
detector:
  type: tflite

# Execution flow
orchestrator.process_frame(frame)
    ↓
detection_stage.process(frame)
    ↓
tflite_detector.detect(frame)
    ↓ Returns detections only
tracking_stage.process(detections)
    ↓
botsort_tracker.update(detections)
    ↓ tracking/botsort_tracker.py runs!
tracks = results

# Result
❌ Fluctuating Track IDs
⚠️ Poor tracking quality
✅ tracking/botsort_tracker.py IS used (fallback)
```

### Scenario 3: Incorrect Usage ❌ AVOID

```python
# WRONG CODE
detections = detector.detect(frame)          # ❌ Missing tracking!
tracks = tracker.update(detections)          # ❌ No persistence!

# Result
❌ Fluctuating Track IDs (same as Scenario 2)
```

**Fix**: Use `detect_and_track()` instead!

```python
# CORRECT CODE
detections, tracks = detector.detect_and_track(frame)  # ✅ Persistent IDs!
```

---

## 🎯 Design Patterns

### Strategy Pattern

Different tracking algorithms can be swapped:

```yaml
# config.yaml
tracker:
  type: botsort  # or 'bytetrack', 'deepsort', 'none'
```

**But remember**: In V3 HYBRID, this only affects fallback scenarios!

Normal operation uses tracker specified in `detect_and_track('botsort.yaml')`.

---

## 🔧 Configuration

```yaml
# config.yaml

tracker:
  type: botsort                  # 'botsort', 'bytetrack', 'none'
  track_thresh: 0.4              # Min confidence to start tracking
  track_buffer: 90               # Frames to keep lost tracks (3 sec @ 30fps)
  match_thresh: 0.4              # IoU threshold for matching
```

**Note**: These parameters affect:
- **Integrated tracking**: When using `detect_and_track()` (main usage)
- **Fallback tracking**: When using `tracking/botsort_tracker.py` (fallback)

---

## 🚀 Adding New Trackers

### Example: Adding ByteTrack Wrapper

1. **Create implementation**: `tracking/bytetrack_tracker.py`

```python
from tracking.base import BaseTracker
from common.base_classes import Track

class ByteTrackTracker(BaseTracker):
    """ByteTrack wrapper for fallback scenarios."""
    
    def __init__(self, config):
        self.track_thresh = config.get('tracker', {}).get('track_thresh', 0.5)
        # Initialize ByteTrack
    
    def update(self, detections):
        """Update ByteTrack with new detections."""
        # ByteTrack matching logic
        # (Still has same limitations as BoTSORTTracker!)
        return tracks
```

2. **Register in factory**: `tracking/factory.py`

```python
elif tracker_type == 'bytetrack':
    from tracking.bytetrack_tracker import ByteTrackTracker
    return ByteTrackTracker(config)
```

3. **Update config**: `config.yaml`

```yaml
tracker:
  type: bytetrack  # Use ByteTrack wrapper (fallback)
```

**Note**: This only affects TFLite fallback! For YOLODetector, change `detect_and_track('bytetrack.yaml')`.

---

## 🧪 Testing

### Test Persistent Tracking (Most Important!)

```python
# tests/test_tracking/test_persistent_tracking.py

def test_track_ids_persist_with_integrated_tracking():
    """Test that Track IDs stay stable using detect_and_track()."""
    detector = YOLODetector(config)
    
    # Simulate 100 frames of same scene
    track_ids_per_frame = []
    for i in range(100):
        detections, tracks = detector.detect_and_track(frame)
        if tracks:
            track_ids_per_frame.append([t.track_id for t in tracks])
    
    # Verify IDs stay consistent
    first_ids = set(track_ids_per_frame[0])
    for frame_ids in track_ids_per_frame[1:]:
        assert set(frame_ids) == first_ids, "Track IDs fluctuated!"
```

### Test Fallback Tracking

```python
def test_botsort_wrapper_fallback():
    """Test BoT-SORT wrapper for TFLite fallback."""
    tracker = BoTSORTTracker(config)
    
    # Simulate detections
    detections = [Detection(...), Detection(...)]
    
    tracks = tracker.update(detections)
    
    assert len(tracks) == len(detections)
    assert all(isinstance(t, Track) for t in tracks)
```

---

## 🐛 Troubleshooting

### Track IDs Fluctuating (Most Common Issue!)

**Symptoms**:
- Person sitting still gets different ID each frame (1, 2, 3, 4...)
- IDs change when person moves slightly

**Diagnosis**:

1. **Check which execution path is used**:
   ```python
   # Add logging to orchestrator.py
   print("Using integrated tracking" if hasattr(detector, 'detect_and_track') else "Using fallback tracking")
   ```

2. **If using integrated tracking** (should be stable):
   - Verify `persist=True` in `yolo_detector.py`:
     ```python
     results = self.model.track(persist=True, ...)  # ← This MUST be True!
     ```
   - Check lighting (poor lighting → poor tracking)
   - Check camera FPS (low FPS → tracking struggles)

3. **If using fallback tracking** (will be unstable):
   - Switch to YOLODetector: `config.yaml` → `detector.type: yolo`
   - If you MUST use TFLite, accept that IDs will fluctuate

**Root Cause Summary**:

| Cause | Solution |
|-------|----------|
| Using `detect()` + `update()` pattern | Use `detect_and_track()` |
| Using TFLiteDetector | Switch to YOLODetector |
| `persist=False` in `model.track()` | Change to `persist=True` |
| Poor lighting | Improve lighting conditions |

---

### Tracker Not Matching Well

**Tune IoU threshold**:
```yaml
tracker:
  match_thresh: 0.4  # Lower = more lenient matching
```

**Tune track buffer**:
```yaml
tracker:
  track_buffer: 90  # Higher = keep lost tracks longer
```

---

## 📊 Performance Comparison

| Implementation | Persistent IDs | Quality | Speed | Use Case |
|----------------|----------------|---------|-------|----------|
| **Integrated BoT-SORT** (in Ultralytics) | ✅ YES | ⭐⭐⭐⭐⭐ | Fast | **Production (default)** |
| `tracking/botsort_tracker.py` (wrapper) | ❌ NO | ⭐⭐ | Fast | TFLite fallback only |
| `tracking/no_tracker.py` (dummy) | ❌ NO | ⭐ | Instant | Testing only |

**Recommendation**: Always use integrated tracking (YOLODetector) for production.

---

## 📚 Related Documentation

- **Parent Guide**: `docs/DEVELOPER_GUIDE.md`
- **Architecture**: `docs/ARCHITECTURE_V3_HYBRID.md`
- **Detectors**: `detectors/README.md` (explains integrated tracking)
- **Pipeline**: `pipeline/README.md` (shows execution flow)

---

## 🎓 Key Takeaways

1. **In V3 HYBRID, this component is mostly FALLBACK!**
   - Normal operation: Tracking runs in `model.track(persist=True)`
   - Fallback: Tracking runs in `tracking/botsort_tracker.py`

2. **Integrated tracking (YOLODetector) is ALWAYS better**:
   - ✅ Persistent Track IDs
   - ✅ Kalman filter
   - ✅ ReID features
   - ✅ Better quality

3. **Fallback wrapper has limitations**:
   - ❌ No persistence
   - ❌ No Kalman filter
   - ❌ No ReID
   - ❌ IDs fluctuate

4. **When to use each**:
   - Production: YOLODetector + integrated tracking
   - TFLite: Fallback wrapper (accept limitations)
   - Testing: NoTracker or fallback wrapper

5. **Common mistake**: Using `detect()` + `update()` instead of `detect_and_track()`

---

**Questions?** See `docs/DEVELOPER_GUIDE.md` or create an issue.

**Last Updated**: November 5, 2025
