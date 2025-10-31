# Model Training & Export Reference

This document serves as historical reference for how the YOLOv8 face detection model was created and exported to TFLite INT8 format.

## Model Details

- **Base Model**: YOLOv8n (nano variant for edge devices)
- **Task**: Face detection (single class)
- **Final Format**: TFLite INT8 quantized
- **Input Size**: 256x256 (configurable at runtime to 320 or 640)
- **Location**: `/home/rishabh/Attendance_system/models/yolov8n_face_int8.tflite`

## Export Process

### Requirements
```bash
pip install ultralytics torch onnx onnx2tf
```

### Export Command
```python
from ultralytics import YOLO 

model_path = '/path/to/yolov8n_face.pt'  # Trained PyTorch weights
data = '/path/to/calibration.yaml'        # Calibration dataset config

model = YOLO(model_path) 
model.export( 
    format='tflite',   
    int8=True,         # Enable INT8 quantization
    data=data,         # Required for calibration
    imgsz=256,         # Input size for export
    nms=False,         # CRITICAL: Disable NMS (done in post-processing)
    dynamic=False      # Fixed input shape for edge devices
) 
```

### Calibration Dataset Structure

The `calibration.yaml` file specified the dataset format:

```yaml
path: /path/to/calibration_dataset
train: images
val: images

names:
  0: face
```

**Calibration Images**: Random samples from:
- SCUT-HEAD dataset (~200 images)
- WIDER Face validation (~75 images)
- MOT17 tracking sequences (selected frames)

Total: ~300-400 representative images for INT8 quantization calibration.

## Critical Export Settings Explained

### Why `nms=False`?
- TFLite models with NMS baked-in can have compatibility issues
- Our pipeline performs NMS in Python post-processing using OpenCV's `cv2.dnn.NMSBoxes`
- Gives more control over confidence and IoU thresholds at runtime

### Why `int8=True`?
- Reduces model size from ~6MB to ~1.5MB
- Faster inference on CPU (3-5x speedup on Raspberry Pi)
- Minimal accuracy loss with proper calibration

### Why `dynamic=False`?
- Edge devices (Raspberry Pi) work better with fixed input shapes
- Enables better optimization by TFLite compiler
- Input can still be resized at runtime before feeding to interpreter

## Dataset Sources (Historical)

### 1. SCUT-HEAD Dataset (Part A)
- **Source**: https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release
- **Content**: Head detection in crowd scenes with XML annotations
- **Used for**: Training and calibration
- **Extraction**: `datasets/scut_head_part_a/data_extract.py` (randomly selected 200 images)

### 2. WIDER Face Validation
- **Source**: http://shuoyang1213.me/WIDERFACE/
- **Content**: Face detection benchmark with diverse scenarios
- **Used for**: Calibration
- **Extraction**: `datasets/wider_val/wider_face_data.py` (randomly selected 75 images)

### 3. MOT17 Multi-Object Tracking
- **Source**: https://motchallenge.net/data/MOT17/
- **Content**: Video sequences for multi-object tracking
- **Used for**: Calibration (selected frames from train split)
- **Extraction**: `datasets/mot17/train/select_frames_all.py`

## Model Variants Generated

During export, Ultralytics generates multiple variants:
- `model_float32.tflite` - Full precision (largest, slowest)
- `model_float16.tflite` - Half precision
- `model_int8.tflite` - **Used in production** (smallest, fastest)
- `model_full_integer_quant.tflite` - Full integer ops
- `*.onnx` - Intermediate ONNX format

**Only `model_int8.tflite` is kept in the final deployment.**

## Performance Expectations

### Raspberry Pi 4 (4GB)
- **Input**: 256x256
- **Threads**: 3
- **FPS**: ~15-20 (with tracking)
- **Inference time**: ~40-60ms

### Laptop (CPU)
- **Input**: 320x320 or 640x640
- **Threads**: 4
- **FPS**: ~30-40 (with tracking)
- **Inference time**: ~20-30ms

## Lessons Learned

1. **Calibration is critical**: Using diverse images (crowd scenes, varied lighting, angles) improved quantization accuracy
2. **NMS separation**: Doing NMS in Python gives flexibility to tune thresholds without re-exporting
3. **Input size tradeoff**: 256px is fast but misses small faces; 320-640px is slower but more accurate
4. **Frame skipping**: Processing every 2nd or 3rd frame maintains responsiveness without sacrificing tracking

## Re-Exporting the Model

If you need to re-export with different settings (e.g., larger input size):

1. Ensure you have the original `.pt` file (PyTorch weights)
2. Prepare calibration dataset (or reuse existing)
3. Run the export command with modified `imgsz` parameter
4. Test the new model with `pipeline_main.py`
5. Update `pipeline_config.yaml` with new model path

---

**Note**: As of October 2025, all datasets and intermediate files have been removed to save space. Only the final `yolov8n_face_int8.tflite` model is retained in the repository.
