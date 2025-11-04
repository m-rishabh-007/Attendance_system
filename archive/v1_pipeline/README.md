# Version 1.0 Pipeline (Archived)

This directory contains the original procedural implementation of the face detection and tracking pipeline.

## Files Overview

### Core Pipeline
- **pipeline_main.py**
  - Direct TFLite inference (yolov8n_face_int8.tflite)
  - Manual preprocessing (resize, normalize)
  - Custom post-processing (NMS filtering)
  - Optimized for Raspberry Pi (low latency)
  - **Status**: Educational reference only

### Tracking Implementation
- **face_tracker_bytetrack.py**
  - Custom ByteTrack algorithm implementation
  - Kalman filter for motion prediction
  - IoU-based data association
  - Track lifecycle management (new → tracked → lost → deleted)
  - **Status**: Reference implementation (BoT-SORT preferred in v2.0)

### Legacy Demos
- **attendance_ultralytics.py**
  - Simple demo using Ultralytics YOLO API
  - High-level inference (slower but easier)
  - No explicit tracking configuration
  - **Status**: Replaced by attendance_v2.py

- **attendance_prototype.py**
  - Early prototype/experimental code
  - **Status**: Historical reference

### Configuration
- **pipeline_config.yaml** → Moved to root as `config.yaml`
- **models/calibration.yaml** → Training reference (not used at runtime)

## Why This Code Was Archived

### Problems with v1.0
1. **Procedural Code**: Functions scattered across files, hard to extend
2. **Tight Coupling**: Direct dependencies, difficult to swap components
3. **No Design Patterns**: Harder to maintain and test
4. **Configuration**: Mixed hardcoded values and config file
5. **Testing**: No unit tests, only manual testing

### v2.0 Improvements
1. **Object-Oriented**: Classes with clear responsibilities
2. **Singleton Pattern**: Centralized configuration (ConfigManager)
3. **Observer Pattern**: Event-driven architecture (EventSystem)
4. **Factory Pattern**: Easy component swapping (DetectorFactory, TrackerFactory)
5. **Strategy Pattern**: Interchangeable algorithms (BaseTracker implementations)
6. **Comprehensive Tests**: 20 unit tests validating all patterns

## Migration Guide

If migrating code from v1.0 to v2.0 patterns:

### Old (v1.0 Procedural)
```python
# Direct function calls, tight coupling
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
detections = detect_faces(frame, interpreter)
tracked_faces = tracker.update(detections)
```

### New (v2.0 OOP)
```python
# Factory pattern, loose coupling
config = ConfigManager()
detector = DetectorFactory.create(config)
tracker = TrackerFactory.create(config)

detections = detector.detect(frame)
tracked_faces = tracker.update(detections)
```

See `/docs/MIGRATION_GUIDE.md` for complete details.

## ByteTrack vs BoT-SORT

**ByteTrack** (this implementation):
- Simple IoU matching + Kalman filter
- Fast but less robust with occlusions
- Good for simple scenarios

**BoT-SORT** (v2.0 default):
- Motion + appearance features
- Camera motion compensation
- Better handling of occlusions and ID switches
- **Proven superior in testing** → This is why v2.0 uses BoT-SORT

## Can I Still Use This Code?

**Short answer**: No, use v2.0 instead.

**Long answer**: This code still works, but:
- ❌ No longer maintained
- ❌ Missing modern design patterns
- ❌ Harder to extend with new features
- ❌ No test coverage
- ✅ Good for learning how tracking algorithms work
- ✅ Useful reference for understanding TFLite inference

## Learning Resources

If you want to understand the concepts:

1. **ByteTrack Algorithm**: Read `face_tracker_bytetrack.py`
   - Kalman filter motion model
   - IoU-based matching with Hungarian algorithm
   - Two-stage association (high/low confidence)

2. **TFLite Inference**: Read `pipeline_main.py`
   - Input preprocessing (resize, normalize to [0,1])
   - Interpreter allocation and invocation
   - Output post-processing (NMS, coordinate scaling)

3. **Design Pattern Evolution**: Compare this with `/production/attendance_v2.py`
   - See how factories simplify component creation
   - See how observers decouple event handling
   - See how strategy pattern allows algorithm swapping

## Performance Benchmarks (Historical)

**ByteTrack Performance** (Raspberry Pi 4, 2GB):
- Inference: ~120ms (TFLite INT8, 256×256 input)
- Tracking: ~15ms (Kalman + Hungarian)
- Total FPS: ~7-8 FPS

**BoT-SORT Performance** (v2.0, same hardware):
- Inference: ~120ms (same model)
- Tracking: ~20ms (motion + appearance)
- Total FPS: ~7 FPS
- **ID switches: 40% fewer than ByteTrack**

Trade-off: Slightly slower but much more robust.

---

**Archived**: November 2025  
**Reason**: Replaced by v2.0 OOP architecture  
**Maintainer**: No longer actively maintained  
**Questions**: Refer to `/docs/ARCHITECTURE_V2.md` for current design
