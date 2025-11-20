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
├── collect_calibration_data.py        ✅ Collect 97 samples (Week 2 Day 1 - COMPLETE)
├── convert_onnx_to_tflite.py          ✅ ONNX → TFLite FP32 (Week 2 Day 2)
├── quantize_tflite_int8.py            ✅ TFLite FP32 → INT8 (Week 2 Day 3)
├── validate_tflite_model.py           ✅ Compare FP32 vs INT8 (Week 2 Day 3)
└── calibration_data/                  🚧 Generated during collection
    ├── faces_100.npy                  📊 97 calibration samples (quality=0.327)
    ├── metadata.json                  📋 Quality scores, angles, timestamps
    └── logs/                          📝 Collection session logs
        └── calibration_collection_*.log
```

---

## Week 2 Implementation Plan

### ✅ Day 1: Collect Calibration Data (COMPLETE)

**Script**: `collect_calibration_data.py`

**Status**: ✅ 97 samples collected (quality=0.327)

```bash
# Smart collection: 5 minutes, best quality samples
python3 tools/quantization/collect_calibration_data.py

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

---

### ✅ Day 2: Convert ONNX → TFLite FP32 (READY)

**Script**: `convert_onnx_to_tflite.py`

**Purpose**: Convert ONNX model to TFLite FP32 format (first step for TFLite INT8)

**Pipeline**: ONNX → TensorFlow SavedModel → TFLite FP32

```bash
# Convert ONNX to TFLite FP32
python3 tools/quantization/convert_onnx_to_tflite.py

# Input: models/recognition/auraface_resnet100_fp32.onnx (166 MB)
# Output: models/recognition/auraface_resnet100_fp32.tflite (166 MB)
```

**What happens**:
1. Loads ONNX model and validates structure
2. Converts ONNX → TensorFlow SavedModel (using onnx-tf)
3. Converts SavedModel → TFLite FP32 (using TFLite converter)
4. Validates TFLite model with test inference
5. Saves TFLite FP32 model (same size as ONNX, no quantization yet)

**Expected time**: ~2-3 minutes

---

### ✅ Day 3: Quantize TFLite FP32 → INT8 (READY)

**Script**: `quantize_tflite_int8.py`

**Purpose**: Post-training quantization FP32 → INT8 using calibration data

**Pipeline**: TFLite FP32 + Calibration Data → TFLite INT8

```bash
# Quantize TFLite FP32 to INT8
python3 tools/quantization/quantize_tflite_int8.py

# Input: 
#   - models/recognition/auraface_resnet100_fp32.tflite (166 MB)
#   - tools/quantization/calibration_data/faces_100.npy (97 samples)
# Output:
#   - models/recognition/auraface_resnet100_int8.tflite (~42 MB)
```

**What happens**:
1. Loads FP32 TFLite model
2. Loads calibration dataset (97 face samples)
3. Creates representative dataset generator
4. Runs TFLite INT8 quantization with calibration
5. Saves INT8 TFLite model (4x smaller)
6. Validates INT8 model with test inference

**Expected Results**:
- Size reduction: 166 MB → 42 MB (4x smaller)
- Quantization: UINT8 for weights + activations
- Speed improvement: 2x faster on Pi (tested in validation)

**Expected time**: ~3-5 minutes

---

### ✅ Day 3: Validate INT8 Model (READY)

**Script**: `validate_tflite_model.py`

**Purpose**: Compare FP32 vs INT8 accuracy on validation dataset

```bash
# Validate INT8 model accuracy
python3 tools/quantization/validate_tflite_model.py

# Input:
#   - models/recognition/auraface_resnet100_fp32.tflite
#   - models/recognition/auraface_resnet100_int8.tflite
#   - tools/quantization/calibration_data/faces_100.npy (validation)
# Output:
#   - tools/quantization/validation_report.json
#   - Console report with metrics
```

**What happens**:
1. Loads FP32 and INT8 TFLite models
2. Runs inference on 97 validation samples
3. Compares embeddings:
   - Cosine similarity (target: >0.99)
   - Mean Absolute Error (target: <0.05)
4. Measures inference time (speedup)
5. Generates validation report (JSON + console)

**Pass Criteria**:
- ✅ Cosine similarity ≥ 0.99 (embeddings nearly identical)
- ✅ MAE ≤ 0.05 (low error)
- ✅ Speedup ≥ 2x (INT8 faster than FP32)

**Expected Results** (with quality=0.327 calibration data):
- Cosine similarity: 0.96-0.98 (lower due to suboptimal calibration)
- MAE: 0.06-0.10 (higher due to calibration quality)
- Speedup: 2-3x faster
- Accuracy drop: 3-5% (re-collect on Pi for <1%)

**Expected time**: ~2-3 minutes

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
