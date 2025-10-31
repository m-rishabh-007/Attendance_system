# Research Pipeline - Custom TFLite Implementation

## Overview

This directory contains the **custom-built face detection and tracking pipeline** using raw TFLite inference and ByteTrack tracking implementation. 

**Purpose**: Educational, research, and learning reference.

## What's Inside

### Core Files

- **`pipeline_main.py`** (20 KB) - Main pipeline with direct TFLite inference
  - Custom post-processing (NMS, coordinate decoding)
  - ByteTrack integration
  - Performance profiling
  - ~25ms inference latency

- **`face_tracker_bytetrack.py`** (17 KB) - Custom ByteTrack implementation
  - Kalman filter tracking (predict, update, covariance)
  - IoU-based data association
  - Track lifecycle management (new → tracked → lost → deleted)
  - Motion prediction during detection gaps

- **`pipeline_config.yaml`** (3.3 KB) - Configuration file
  - Model paths (relative to project root)
  - Inference settings (confidence, IoU, input size)
  - Tracking parameters (track_buffer, match_thresh)
  - Camera configuration

- **`models/calibration.yaml`** - Reference for model quantization
  - Used during TFLite INT8 export
  - Documents calibration dataset structure
  - **Keep as documentation**, not used at runtime

## Learning Outcomes

Building this pipeline taught:

✅ **YOLO Post-Processing**
- How to decode YOLO predictions
- NMS (Non-Maximum Suppression) implementation
- Coordinate system transformations

✅ **Object Tracking**
- Kalman filter mathematics
- Track state management (new → tracked → lost → deleted)
- IoU-based data association
- Hungarian algorithm for optimal matching

✅ **TFLite Optimization**
- INT8 quantization effects
- Model input/output tensor handling
- Thread optimization for inference
- Performance profiling

✅ **System Design**
- Multi-stage pipeline architecture
- Configuration management
- Error handling and debugging
- Performance vs. accuracy trade-offs

## When to Use This Pipeline

### ✅ Ideal For:

1. **Embedded Deployment** (RAM < 512MB)
   - Microcontrollers (ESP32-S3)
   - Low-power ARM devices
   - Minimal dependency footprint

2. **Research & Experimentation**
   - Custom NMS algorithms (soft-NMS, DIoU-NMS)
   - Modified tracking strategies
   - A/B testing post-processing
   - Algorithm development

3. **Performance Profiling**
   - Stage-by-stage timing analysis
   - Memory footprint optimization
   - Custom hardware integration

4. **Learning & Portfolio**
   - Understanding ML pipelines deeply
   - Technical interview preparation
   - Academic projects

### ❌ NOT Recommended For:

- Production deployment (use Ultralytics instead)
- When tracking quality is critical
- Limited development time
- Maintenance-heavy projects

## Key Implementation Details

### Detection Flow
```
Camera Frame
    ↓
Resize to 256x256
    ↓
TFLite Inference (INT8)
    ↓
Transpose [1,5,1344] → [1344,5]
    ↓
Confidence Filtering (>0.5)
    ↓
NMS (IoU threshold 0.3)
    ↓
Detections [x,y,w,h]
```

### Tracking Flow
```
Detections (tlwh)
    ↓
Convert to xyah format
    ↓
Predict existing tracks (Kalman)
    ↓
Compute IoU distance matrix
    ↓
Hungarian algorithm matching
    ↓
Update matched tracks
    ↓
Create new tracks (unmatched dets)
    ↓
Mark lost tracks
    ↓
Output tracked faces with IDs
```

## Known Issues & Limitations

### Tracking Quality
- ⚠️ Less robust than BoT-SORT during:
  - Fast motion (head shaking)
  - Motion blur
  - Temporary occlusions
- ⚠️ Basic Kalman filter (no appearance features)
- ⚠️ IoU-only matching (position-based)

### Root Causes
1. **Detection gaps** → Track IDs lost
2. **Basic ByteTrack** → No re-identification features
3. **No appearance model** → Can't recover from longer occlusions

### Improvements Made
✅ Increased track_buffer (30 → 90 frames)
✅ Lowered match_thresh (0.5 → 0.4) 
✅ Added Kalman prediction during `dets=0`
✅ Display lost tracks with predicted positions
✅ Fixed multiple critical bugs (see git history)

## Performance Benchmarks

### Laptop (Intel i5, 8GB RAM)
- **FPS**: 25-27 FPS
- **Inference**: 22-28ms
- **Total latency**: ~40ms

### Raspberry Pi 4 (4GB)
- **FPS**: 15-20 FPS (estimated)
- **Inference**: 40-50ms (estimated)
- **Total latency**: ~60-70ms (estimated)

## Debugging Tips

### No Detections (dets=0)
1. Check camera is working: `cap.isOpened()`
2. Verify model path is correct
3. Lower confidence_threshold in config
4. Check lighting conditions
5. Verify face is within camera view

### Tracking Issues
1. Check ByteTrack debug output (every 100 frames)
2. Monitor IoU distances (`min_dist`, `max_dist`)
3. Verify match_thresh is appropriate
4. Check track_buffer duration
5. Look for "matches=X" in logs

### Performance Issues
1. Increase `process_every_n_frames` (skip frames)
2. Reduce `num_threads` if CPU overloaded
3. Check for memory leaks in logs
4. Profile with `cProfile` for bottlenecks

## Comparison: Research vs Production

| Aspect | Research (This) | Production (Ultralytics) |
|--------|-----------------|--------------------------|
| **Code Lines** | 800+ | 30 |
| **Tracking** | Basic ByteTrack | BoT-SORT |
| **Inference** | 25ms (faster) | 45ms (slower) |
| **ID Persistence** | Good | Excellent |
| **Motion Robustness** | Moderate | Excellent |
| **Maintenance** | High | Low |
| **Dependencies** | Minimal | Heavy |
| **Use Case** | Learning/Embedded | Production |

## Next Steps for Improvement

### To Match Production Quality:
1. **Add appearance features** (ReID model)
2. **Implement BoT-SORT** algorithm
3. **Add camera motion compensation**
4. **Improve Kalman filter** (velocity model)
5. **Add track recovery** (after long occlusions)

### Alternatively:
**Use the production pipeline** (`production/attendance_ultralytics.py`) which already has these features! 🚀

## References

- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)
- [BoT-SORT Paper](https://arxiv.org/abs/2206.14651)
- [Ultralytics Documentation](https://docs.ultralytics.com)
- [TFLite Optimization Guide](https://www.tensorflow.org/lite/performance/best_practices)

---

**Remember**: Building this from scratch was valuable learning. Using Ultralytics in production is smart engineering. Both have their place! 🎯
