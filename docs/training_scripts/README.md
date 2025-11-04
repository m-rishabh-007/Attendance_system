# Training Scripts

This directory contains scripts used for model training, export, and calibration.

## Contents

### yolo_int8.py
Script used to export the face detection model with INT8 quantization.

**Purpose**: Convert YOLOv8 PyTorch model to TFLite INT8 format for Raspberry Pi deployment

**Usage** (Historical reference - model already exported):
```python
from ultralytics import YOLO

model = YOLO('yolov8n_face.pt')  # Trained model
model.export(
    format='tflite',
    int8=True,              # Enable INT8 quantization
    data='calibration.yaml', # Calibration dataset config
    imgsz=256,              # Input size for Pi optimization
    nms=False,              # NMS done in post-processing
    dynamic=False           # Fixed input shape
)
```

**Key Settings Explained**:
- `int8=True`: Quantize weights and activations to 8-bit integers
  - **Benefit**: 4x smaller model, 2-3x faster inference on Pi
  - **Trade-off**: ~1-2% accuracy loss (acceptable for face detection)
  
- `data='calibration.yaml'`: Points to calibration dataset
  - **Required**: INT8 quantization needs representative images
  - **Purpose**: Determine optimal quantization ranges
  
- `imgsz=256`: Small input size for Pi
  - **Options**: 256 (fastest), 320 (balanced), 640 (most accurate)
  - **Choice**: 256 chosen for real-time Pi performance
  
- `nms=False`: **CRITICAL SETTING**
  - **Reason**: NMS inside TFLite model causes shape issues
  - **Solution**: Manual NMS in post-processing code
  - **See**: `detectors/tflite_detector.py` for implementation

**Output**:
- `yolov8n_face_int8.tflite` → Deployed to `/models/` directory
- Size: ~1.5MB (vs ~6MB FP32)
- Inference: ~120ms on Pi 4 vs ~350ms FP32

## Calibration Dataset

For INT8 quantization, you need a calibration dataset (100-1000 representative images).

**Requirements**:
- Diverse lighting conditions (indoor, outdoor, dim, bright)
- Various face angles (frontal, profile, ±30°)
- Different distances (close-up, medium, far)
- Multiple people per image (if detecting crowds)

**Sources Used** (see `/archive/v1_pipeline/models/calibration.yaml`):
- MOT17 dataset: Crowd tracking sequences
- WIDER Face: Face detection benchmark
- SCUT Head: Head detection dataset

**Total**: ~500 images selected to represent real-world deployment scenarios

## Model Training (Not Included)

The original face detection model was trained separately using:
- **Base Model**: YOLOv8n (nano variant for speed)
- **Dataset**: WIDER Face + custom face dataset
- **Training**: ~100 epochs with standard YOLO training pipeline
- **Validation**: 95%+ mAP@0.5 on validation set

Training code is not included in this repository (proprietary dataset used).

## Re-exporting the Model

If you need to re-export with different settings:

1. **Install dependencies**:
   ```bash
   pip install ultralytics torch onnx onnx2tf
   ```

2. **Prepare calibration dataset** (if changing dataset):
   - Create `calibration.yaml` pointing to images
   - Ensure diverse representative samples
   - See example in `/archive/v1_pipeline/models/calibration.yaml`

3. **Run export**:
   ```bash
   python yolo_int8.py
   ```

4. **Test exported model**:
   ```bash
   cd /home/rishabh/Attendance_system
   source venv/bin/activate
   python attendance_system.py
   ```

5. **Verify performance**:
   - Check inference time (target: <150ms on Pi)
   - Check detection accuracy (should be >90% for frontal faces)
   - Check false positive rate (should be <5%)

## Different Export Options

### Option 1: Larger Input (Better Accuracy)
```python
model.export(format='tflite', int8=True, imgsz=320, nms=False)
# Result: ~160ms inference, better small face detection
```

### Option 2: FP16 Instead of INT8 (Better Accuracy, Larger Size)
```python
model.export(format='tflite', half=True, imgsz=256, nms=False)
# Result: ~3MB model, ~200ms inference, <1% accuracy loss
```

### Option 3: ONNX for Desktop Deployment
```python
model.export(format='onnx', imgsz=640, nms=False)
# Result: ONNX Runtime for x86/x64 CPUs, ~30ms inference
```

### Option 4: TensorRT for NVIDIA GPU
```python
model.export(format='engine', imgsz=640, nms=False)
# Result: <10ms inference on NVIDIA GPU
```

## Common Issues

### Issue 1: "No calibration images found"
**Solution**: Check `calibration.yaml` paths are correct and images exist

### Issue 2: Export fails with shape errors
**Solution**: Ensure `dynamic=False` and `nms=False` are set

### Issue 3: Exported model runs but gives wrong results
**Solution**: Check input preprocessing (must be RGB, normalized to [0,1])

### Issue 4: Model too slow on Raspberry Pi
**Solution**: Try smaller `imgsz` (256 instead of 320) or reduce `num_threads`

## Performance Optimization Tips

1. **Input Size**: Smaller = faster, but worse for small/distant faces
   - 256: Best for Pi, single person close-up
   - 320: Balanced, multiple people at medium distance
   - 640: Best accuracy, requires desktop GPU

2. **Quantization**: INT8 vs FP16 vs FP32
   - INT8: Fastest, smallest, slight accuracy loss (recommended for Pi)
   - FP16: Middle ground, good for GPU deployment
   - FP32: Slowest, largest, best accuracy (development only)

3. **CPU Threads** (at runtime, not export):
   - Pi 4: 3 threads optimal (leave 1 for OS)
   - Desktop: 4-6 threads depending on CPU

4. **Frame Skipping** (at runtime, not export):
   - Process every frame: Real-time, high CPU
   - Process every 2nd frame: 2x faster, slight tracking lag
   - Process every 3rd frame: 3x faster, noticeable lag

---

**Model Version**: yolov8n_face_int8.tflite  
**Export Date**: October 2025  
**Maintained By**: See main README for contact  
**Production Model**: `/models/yolov8n_face_int8.tflite`
