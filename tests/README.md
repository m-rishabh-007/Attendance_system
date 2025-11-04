# Tests - Tracking Verification Suite

**Purpose**: Verify that persistent tracking works correctly and detect regressions.

---

## 🧪 Test Files

| Test File | Purpose | When to Run |
|-----------|---------|-------------|
| `test_persistent_tracking.py` | **Smoke Test** - Verify Track IDs persist | After ANY tracking-related changes |
| `test_tracking_scenarios.py` | **Comprehensive** - Test all tracking scenarios | Before releases, after major changes |
| `run_all_tests.py` | Run all unit tests | Regular development |

---

## 🎯 Test 1: Smoke Test (Most Important!)

### `test_persistent_tracking.py`

**What It Tests**:
- Track IDs remain stable when person doesn't move
- Detects the critical bug where IDs fluctuate (1, 2, 3, 4...)
- Verifies `model.track(persist=True)` is working

**How to Run**:
```bash
cd /home/rishabh/Attendance_system
source venv/bin/activate
python tests/test_persistent_tracking.py
```

**Test Flow**:
1. Opens camera
2. Detects and tracks faces for 100 frames (~3 seconds)
3. Collects Track IDs from each frame
4. Analyzes stability

**Expected Output (PASS)**:
```
✅ PASSED: Track IDs are persistent!
   Only 1-3 unique IDs seen - indicates stable tracking
   Track ID stability: 95.2%
   ⭐ Excellent stability!
```

**Expected Output (FAIL)**:
```
❌ FAILED: Track IDs are fluctuating!
   Saw 47 unique IDs - expected ≤3
   
   This indicates the bug where Track IDs change every frame:
   - Person sitting still gets IDs: 1, 2, 3, 4, 5...
   - Root cause: NOT using model.track(persist=True)
```

**When to Run**:
- ⭐ After modifying `detectors/yolo_detector.py`
- ⭐ After modifying `pipeline/orchestrator.py`
- ⭐ After modifying `pipeline/detection_stage.py`
- Before committing any tracking changes
- Before releases

---

## 🔬 Test 2: Comprehensive Scenarios

### `test_tracking_scenarios.py`

**What It Tests**:

#### Scenario 1: YOLO Integrated Tracking
- Tests `detector.detect_and_track()` with `persist=True`
- Verifies this is the CORRECT approach
- Expected: Stable Track IDs

#### Scenario 2: TFLite Fallback Tracking
- Tests `detector.detect()` + `tracker.update()`
- Demonstrates why fallback has poor performance
- Expected: Fluctuating Track IDs (this is normal for fallback)

#### Scenario 3: Multiple Faces
- Tests tracking with 2+ people
- Verifies each person gets unique stable ID
- Expected: Each face has its own persistent ID

**How to Run**:
```bash
# Run all scenarios
python tests/test_tracking_scenarios.py

# Run specific scenario
python tests/test_tracking_scenarios.py --scenario yolo
python tests/test_tracking_scenarios.py --scenario tflite
python tests/test_tracking_scenarios.py --scenario multiple
```

**When to Run**:
- Before major releases
- After implementing new tracking features
- When debugging complex tracking issues
- Monthly regression testing

---

## 📊 Understanding Test Results

### Metrics Explained

**Unique Track IDs**:
- Good: 1-3 (same people keep same IDs)
- Bad: 10+ (IDs changing frequently)

**Stability Rate**:
- Excellent: >90%
- Good: 70-90%
- Poor: <70%

**ID Changes**:
- Good: 0-5 changes (brief occlusions only)
- Bad: >20 changes (IDs fluctuating)

---

## 🐛 Troubleshooting Test Failures

### Smoke Test Fails

**Symptom**: Many unique IDs seen (>10)

**Diagnosis Steps**:

1. **Check YOLODetector implementation**:
   ```bash
   grep -n "persist=" detectors/yolo_detector.py
   ```
   Should see: `persist=True`

2. **Check orchestrator is using integrated tracking**:
   ```bash
   grep -n "process_with_tracking" pipeline/orchestrator.py
   ```
   Should be calling `detection_stage.process_with_tracking()`

3. **Check detection_stage implementation**:
   ```bash
   grep -n "detect_and_track" pipeline/detection_stage.py
   ```
   Should be calling `detector.detect_and_track()`

**Common Fixes**:
```python
# ❌ WRONG (in orchestrator.py)
detections = self.detection_stage.process(frame)
tracks = self.tracking_stage.process(detections)

# ✅ CORRECT
detections, tracks = self.detection_stage.process_with_tracking(frame)
```

---

### No Faces Detected

**Symptom**: "No faces detected during test"

**Fixes**:
1. Check lighting (need adequate illumination)
2. Ensure you're visible to camera
3. Clean camera lens
4. Test camera: `ffplay /dev/video1`
5. Lower confidence threshold in `config.yaml`:
   ```yaml
   detector:
     confidence_threshold: 0.3  # Try lower value
   ```

---

### Camera Access Issues

**Symptom**: "Could not open camera"

**Fixes**:
1. Check camera permissions:
   ```bash
   ls -l /dev/video*
   sudo usermod -a -G video $USER
   # Log out and back in
   ```

2. Test camera directly:
   ```bash
   v4l2-ctl --list-devices
   ffplay /dev/video1
   ```

3. Update camera_id in `config.yaml`

---

## 🎓 Test Interpretation Guide

### Scenario 1: YOLO Integrated (CRITICAL!)

**This MUST pass** - It tests the production tracking approach.

If this fails:
- ❌ Production tracking is broken
- ❌ Track IDs will fluctuate in real usage
- ❌ Bug needs immediate fixing

### Scenario 2: TFLite Fallback

**This SHOULD show poor results** - It demonstrates why we need integrated tracking.

If this passes with stable IDs:
- ⚠️ Unexpected (probably no faces detected)
- ✅ Not a failure, just informational

### Scenario 3: Multiple Faces

**Nice to have** - Tests advanced scenario.

If this fails:
- ⚠️ May need tuning (match_thresh, etc.)
- ✅ Not critical if Scenario 1 passes

---

## 🚀 Adding New Tests

### Template for New Tracking Test

```python
"""Test description."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from detectors.yolo_detector import YOLODetector
from common.config_manager import ConfigManager

def test_your_scenario():
    """
    Test your specific scenario.
    
    Expected Result:
    - Describe what should happen
    """
    config = ConfigManager.get_instance()
    config.load('config.yaml')
    
    detector = YOLODetector(config.get_all())
    
    # Your test logic here
    
    # Assert expectations
    assert condition, "Failure message"
    
    return True

if __name__ == '__main__':
    success = test_your_scenario()
    sys.exit(0 if success else 1)
```

---

## 📚 Related Documentation

- **Parent Guide**: `docs/DEVELOPER_GUIDE.md`
- **Architecture**: `docs/ARCHITECTURE_V3_HYBRID.md`
- **Detectors**: `detectors/README.md` (explains persistent tracking)
- **Tracking**: `tracking/README.md` (explains execution scenarios)

---

## 📋 Testing Checklist

Before committing tracking changes:

- [ ] Run smoke test: `python tests/test_persistent_tracking.py`
- [ ] Verify Track IDs are stable (≤3 unique IDs)
- [ ] Check stability rate (>90% expected)
- [ ] Test with good lighting
- [ ] Test with poor lighting
- [ ] Test with head movement
- [ ] Test with multiple people (if applicable)

Before major releases:

- [ ] Run comprehensive tests: `python tests/test_tracking_scenarios.py`
- [ ] All 3 scenarios pass
- [ ] No camera access issues
- [ ] Test on Raspberry Pi (if deploying to Pi)
- [ ] Test on laptop (if deploying to laptop)

---

**Questions?** See `docs/DEVELOPER_GUIDE.md#troubleshooting` or create an issue.

**Last Updated**: November 5, 2025
