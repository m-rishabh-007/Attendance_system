# Quantization Tools

**Purpose**: Convert FP32 models to INT8 quantized models for faster inference on Raspberry Pi.

**Location**: All quantization scripts live here to keep recognizers/ folder clean.

---

## Overview

This folder contains tools for INT8 quantization of face recognition models:
- **Week 1**: FP32 baseline (no quantization)
- **Week 2**: INT8 quantization workflow

### Why Separate Folder?

✅ Keeps `recognizers/` clean (only runtime code)  
✅ Groups all quantization tools in one place  
✅ Makes it clear these are build-time tools, not runtime  
✅ Easier to maintain and document

---

## Folder Structure

```
tools/quantization/
├── README.md                           ✅ This file
├── collect_calibration_data.py        ✅ Collect 100 samples (Week 2 Day 1 - COMPLETE)
├── calibration_data_reader.py         ⏸️ Custom calibration reader (Week 2 Day 2)
├── quantize_auraface.py               ⏸️ FP32 → INT8 converter (Week 2 Day 2-3)
├── validate_quantized_model.py        ⏸️ Compare FP32 vs INT8 (Week 2 Day 4)
└── calibration_data/                  🚧 Generated during collection
    ├── faces_100.npy                  📊 100 best calibration samples
    ├── faces_validation.npy           📊 Extra samples for testing
    ├── metadata.json                  📋 Quality scores, timestamps
    └── logs/                          📝 Collection session logs
        └── calibration_collection_*.log
```

---

## Week 2 Implementation Plan

### Step 1: Collect Calibration Data ✅ COMPLETE

Script: `collect_calibration_data.py`

```bash
# Smart collection: 5 minutes, select best 100 from all collected
python tools/quantization/collect_calibration_data.py

# Collection strategy:
# - Runs for 5 minutes (or press 'q' to stop early)
# - Collects ~80-150 samples automatically
# - Smart selection: Best 100 based on quality + angle diversity
# - Saves validation set (extras) for testing
# - Comprehensive logging for debugging
```

**Features**:
1. **Multi-stage Pipeline**: Detection (YOLO INT8) → Tracking (BoT-SORT) → Alignment (MediaPipe)
2. **Quality Scoring**: Sharpness (30%), Angle (25%), Brightness (20%), Size (15%), Confidence (10%)
3. **Angle Estimation**: Yaw angle detection (-90° to +90°) for diversity
4. **Smart Selection**: 
   - Sorts by quality score
   - Ensures angle diversity (20 samples per bin: profile, semi, frontal)
   - Picks best 100 from all collected
5. **Per-Track Sampling**: Each person sampled independently (3-second interval per track_id)
6. **Comprehensive Logging**: Debug logs saved to `calibration_data/logs/`

**What happens**:
1. Webcam opens (device_id from config.yaml, default: 1)
2. Detection + Tracking + Alignment runs in real-time
3. Samples collected every 3 seconds per person (avoids duplicates)
4. Quality scored: combined weighted score + individual metrics
5. Angle estimated: MediaPipe yaw angle for pose diversity
6. After 5 min: Selects best 100 diverse samples
7. Saves `faces_100.npy` (calibration) + `faces_validation.npy` (testing) + `metadata.json`

**Output Files**:
- `calibration_data/faces_100.npy`: (100, 112, 112, 3) - Best calibration samples
- `calibration_data/faces_validation.npy`: (N, 112, 112, 3) - Extra samples for validation
- `calibration_data/metadata.json`: Quality scores, angles, timestamps for all samples
- `calibration_data/logs/calibration_collection_YYYYMMDD_HHMMSS.log`: Detailed debug log

**Sample Collection Tips**:
- Move your head LEFT/RIGHT for angle diversity
- Lean IN/OUT for distance variation
- Move around for lighting variety
- Use mobile camera for better mobility (optional)
- Collect from multiple people for robustness

**Purpose**: High-quality, diverse calibration set for accurate INT8 quantization.

### Step 2: Custom Calibration Data Reader

File: `calibration_data_reader.py`

```python
# Custom reader for onnxruntime.quantization
class FaceCalibrationReader:
    def __init__(self, calibration_data_path):
        self.data = np.load(calibration_data_path)
        self.index = 0
    
    def get_next(self):
        if self.index >= len(self.data):
            return None
        batch = self.data[self.index]
        self.index += 1
        return {'input': batch}
```

**Purpose**: Required by `onnxruntime.quantization` (cannot use auraface library).

### Step 3: Quantize FP32 → INT8

Script: `quantize_auraface.py`

```bash
# Quantize AuraFace model
python tools/quantization/quantize_auraface.py \
    --model models/recognition/auraface_resnet100_fp32.onnx \
    --calibration tools/quantization/calibration_data/faces.npy \
    --output models/recognition/auraface_resnet100_int8.onnx
```

**Purpose**: Convert FP32 model to INT8 for 2x speedup.

### Step 4: Validate INT8 Model

Script: `validate_quantized_model.py`

```bash
# Compare FP32 vs INT8 accuracy
python tools/quantization/validate_quantized_model.py \
    --fp32_model models/recognition/auraface_resnet100_fp32.onnx \
    --int8_model models/recognition/auraface_resnet100_int8.onnx \
    --test_data data/validation/
```

**Expected Results**:
- Accuracy drop: <1% (99.83% → 99.5%+)
- Speed improvement: 2x (80ms → 40ms on Pi)
- Model size: 4x smaller (65MB → 16MB)

---

## Quantization Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ Week 2: INT8 Quantization Workflow                          │
└─────────────────────────────────────────────────────────────┘

Day 1: Collect Calibration Data
    ↓
[collect_calibration_data.py]
    ↓
tools/quantization/calibration_data/faces.npy (100 samples)
    ↓
Day 2: Create Custom Calibration Reader
    ↓
[calibration_data_reader.py]
    ↓
Day 3: Quantize Model
    ↓
[quantize_auraface.py]
    ↓
models/recognition/auraface_resnet100_int8.onnx
    ↓
Day 4: Validate INT8 Model
    ↓
[validate_quantized_model.py]
    ↓
✅ INT8 model ready for production
```

---

## Requirements

```bash
# Core quantization tools
pip install onnx onnxruntime-tools

# Optional: TFLite conversion
pip install onnx-tf tensorflow

# For calibration data collection
pip install opencv-python numpy
```

---

## Usage Examples

### Collect Calibration Data

```python
# tools/quantization/collect_calibration_data.py usage
import cv2
import numpy as np
from pipeline.orchestrator import PipelineOrchestrator

# Initialize pipeline
config = load_config('config.yaml')
orchestrator = PipelineOrchestrator(config)

# Collect 100 aligned faces
calibration_faces = []
cap = cv2.VideoCapture(0)

while len(calibration_faces) < 100:
    ret, frame = cap.read()
    results = orchestrator.process_frame(frame)
    
    for track_id, aligned_face in results['aligned_faces'].items():
        if aligned_face is not None:
            # Preprocess for model (CRITICAL: must match training)
            preprocessed = preprocess_face(aligned_face)
            calibration_faces.append(preprocessed)
            print(f"Collected {len(calibration_faces)}/100 faces")

# Save calibration data
np.save('tools/quantization/calibration_data/faces.npy', 
        np.array(calibration_faces))
```

### Quantize Model

```python
# tools/quantization/quantize_auraface.py usage
from onnxruntime.quantization import quantize_static, QuantType
from calibration_data_reader import FaceCalibrationReader

# Load calibration reader
calibration_reader = FaceCalibrationReader(
    'tools/quantization/calibration_data/faces.npy'
)

# Quantize FP32 → INT8
quantize_static(
    model_input='models/recognition/auraface_resnet100_fp32.onnx',
    model_output='models/recognition/auraface_resnet100_int8.onnx',
    calibration_data_reader=calibration_reader,
    quant_format=QuantType.QInt8,
    per_channel=True,
    reduce_range=False,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8
)

print("✅ INT8 model saved!")
```

### Validate Quantized Model

```python
# tools/quantization/validate_quantized_model.py usage
import onnxruntime as ort
import numpy as np

# Load both models
fp32_session = ort.InferenceSession('models/recognition/auraface_resnet100_fp32.onnx')
int8_session = ort.InferenceSession('models/recognition/auraface_resnet100_int8.onnx')

# Test on validation data
similarities_fp32 = []
similarities_int8 = []

for face1, face2, label in validation_pairs:
    # FP32 inference
    emb1_fp32 = fp32_session.run(None, {'input': preprocess(face1)})[0]
    emb2_fp32 = fp32_session.run(None, {'input': preprocess(face2)})[0]
    sim_fp32 = cosine_similarity(emb1_fp32, emb2_fp32)
    
    # INT8 inference
    emb1_int8 = int8_session.run(None, {'input': preprocess(face1)})[0]
    emb2_int8 = int8_session.run(None, {'input': preprocess(face2)})[0]
    sim_int8 = cosine_similarity(emb1_int8, emb2_int8)
    
    similarities_fp32.append((sim_fp32, label))
    similarities_int8.append((sim_int8, label))

# Compare accuracy at threshold 0.6
accuracy_fp32 = compute_accuracy(similarities_fp32, threshold=0.6)
accuracy_int8 = compute_accuracy(similarities_int8, threshold=0.6)

print(f"FP32 Accuracy: {accuracy_fp32:.2%}")
print(f"INT8 Accuracy: {accuracy_int8:.2%}")
print(f"Accuracy Drop: {accuracy_fp32 - accuracy_int8:.2%}")
```

---

## Critical Notes

### Preprocessing Must Match Training

**CRITICAL**: Quantization calibration preprocessing must EXACTLY match:
1. FP32 model training preprocessing
2. Runtime inference preprocessing

```python
# Standard ArcFace preprocessing (use everywhere)
def preprocess_face(face):
    # 1. Resize
    face = cv2.resize(face, (112, 112))
    
    # 2. Normalize to [0, 1]
    face = face.astype(np.float32) / 255.0
    
    # 3. Mean/std normalization
    face = (face - 0.5) / 0.5  # [-1, 1]
    
    # 4. Transpose HWC → CHW
    face = np.transpose(face, (2, 0, 1))
    
    # 5. Add batch dimension
    return np.expand_dims(face, axis=0)
```

**Why Critical?**
- Mismatch = 20%+ accuracy drop
- Match = <1% accuracy drop

### Calibration Data Quality

- **Quantity**: 100 samples minimum
- **Diversity**: Different people, angles, lighting
- **Source**: From deployment camera (same as production)
- **Quality**: Use best quality faces (from quality scorer)

### Expected Performance

| Metric | FP32 | INT8 | Change |
|--------|------|------|--------|
| **Speed (Pi 4)** | 80-100ms | 40-50ms | **2x faster** ✅ |
| **Model Size** | 65MB | 16MB | **4x smaller** ✅ |
| **Accuracy (LFW)** | 99.83% | 99.5%+ | **-0.3%** ✅ |
| **Memory Usage** | 500MB | 300MB | **40% less** ✅ |

---

## Troubleshooting

### Calibration Data Collection Fails

```
Error: Not enough faces collected
```

**Solutions**:
1. Lower detection confidence threshold temporarily
2. Move camera closer to capture faces
3. Collect from video recording instead of live camera

### Quantization Fails

```
Error: Input shape mismatch
```

**Solutions**:
1. Verify calibration data shape: `(N, 1, 3, 112, 112)`
2. Check calibration reader returns correct format
3. Ensure preprocessing matches FP32 model

### INT8 Accuracy Too Low

```
FP32: 99.83%, INT8: 85% (drop: 14.83%)
```

**Solutions**:
1. **Check preprocessing**: Must match exactly!
2. Collect more calibration data (200-500 samples)
3. Use `per_channel=True` in quantization
4. Verify calibration data quality (not blurry/dark)

### INT8 Model Slower Than Expected

```
Expected: 40-50ms, Got: 70ms
```

**Solutions**:
1. Ensure ONNX Runtime using INT8 operators (not casting)
2. Check CPU threads: `num_threads=4` in config
3. Verify no debug/profiling enabled
4. Test on Raspberry Pi (not x86 laptop)

---

## Status

**Current**: 🚧 Week 2 Day 1 Complete - Calibration Data Collection Ready  
**Week 1**: ✅ FP32 baseline complete (recognizers/ module)  
**Week 2 Day 1**: ✅ Calibration data collection script complete (545 lines)

**Completed (Week 2 Day 1)**:
- ✅ `collect_calibration_data.py` - Smart calibration data collection
  - Detection + Tracking + Alignment pipeline integration
  - Quality-aware sampling (5 metrics: sharpness, angle, brightness, size, confidence)
  - Angle estimation via MediaPipe (yaw: -90° to +90°)
  - Smart selection: Quality sorting + angle diversity (5 bins)
  - Per-track sampling (independent timers per person)
  - Comprehensive logging (debug + info logs)
  - Configurable collection time (default: 5 minutes)

**Files to Create (Week 2 Days 2-7)**:
- ⏸️ `calibration_data_reader.py` (Day 2)
- ⏸️ `quantize_auraface.py` (Days 2-3)
- ⏸️ `validate_quantized_model.py` (Day 4)

**Key Achievements**:
- Fixed bbox format conversion (TLWH → XYXY)
- Enabled angle estimation for diversity
- Implemented per-track sampling (no global timer conflict)
- Added comprehensive debug logging
- Tested: 9 samples in 30 seconds with angle diversity confirmed

**Expected Week 2 Results**:
- Day 1: ✅ Calibration data collection (100 samples)
- Days 2-3: INT8 quantization (FP32 → INT8)
- Day 4: Validation (accuracy drop < 1%)
- Days 5-7: Quality-aware caching + Pi testing (target: 7 FPS)

---

**Next Steps**: 
1. Run full 5-minute calibration collection
2. Create calibration data reader for onnxruntime
3. Implement INT8 quantization script

**Last Updated**: November 12, 2025  
**Author**: Attendance System Team
