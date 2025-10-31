# Models Directory

This directory contains the production-ready face detection model.

## Current Model

### `yolov8n_face_int8.tflite` (1.5 MB)

**Format**: TensorFlow Lite INT8 Quantized  
**Base Model**: YOLOv8n (nano - smallest YOLO variant)  
**Purpose**: Face detection optimized for Raspberry Pi and laptops  
**Input Size**: 256×256 RGB  
**Output**: Face bounding boxes (no NMS applied in model)

#### Technical Details

| Property | Value |
|----------|-------|
| Quantization | INT8 (8-bit integers) |
| Size | 1.5 MB (vs 6 MB FP32) |
| Inference Speed | ~25ms (Pi), ~20ms (laptop) |
| Input Shape | `[1, 256, 256, 3]` (NHWC format) |
| Output Shape | `[1, 5, 1344]` (needs transpose to `[1344, 5]`) |
| Output Format | `[x_center, y_center, width, height, objectness]` |
| NMS | Applied in post-processing (not in model) |

#### Model Creation History

This model was created using the Ultralytics export pipeline:

```python
from ultralytics import YOLO

# Load base YOLOv8n model
model = YOLO('yolov8n.pt')

# Export to TFLite INT8
model.export(
    format='tflite',
    int8=True,                    # Enable INT8 quantization
    data='calibration.yaml',      # Calibration dataset (required for INT8)
    imgsz=256,                    # Input size
    nms=False,                    # Disable NMS (done in post-processing)
    dynamic=False                 # Fixed input shape
)
```

**Original output**: `yolov8n_saved_model/yolov8n_full_integer_quant.tflite`  
**Renamed to**: `yolov8n_face_int8.tflite` (for clarity)

See [`docs/model_training_reference.md`](../docs/model_training_reference.md) for complete export documentation.

## Usage

### Production Pipeline (Ultralytics API)

```python
from ultralytics import YOLO

model = YOLO('models/yolov8n_face_int8.tflite', task='detect')
results = model.track(frame, persist=True, conf=0.5, iou=0.3)
```

**Used by**: [`production/attendance_ultralytics.py`](../production/attendance_ultralytics.py)

### Research Pipeline (Direct TFLite)

```python
from tensorflow.lite.python.interpreter import Interpreter

interpreter = Interpreter(model_path='models/yolov8n_face_int8.tflite', num_threads=3)
interpreter.allocate_tensors()

# Run inference
input_data = preprocess_frame(frame)  # Resize to 256×256, normalize to [0,1]
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
predictions = interpreter.get_tensor(output_details[0]['index'])
```

**Used by**: [`research/yolov8_face_pipeline/pipeline_main.py`](../research/yolov8_face_pipeline/pipeline_main.py)

## Performance Benchmarks

### Raspberry Pi 4 (4GB)
- **Inference**: 40-50ms
- **Post-processing**: 5-10ms
- **Total FPS**: 15-20 FPS
- **Threads**: 3 (optimal for Pi)

### Laptop (Intel i5, 8GB RAM)
- **Inference**: 20-28ms
- **Post-processing**: 2-5ms
- **Total FPS**: 25-30 FPS
- **Threads**: 4 (optimal for laptop)

## Model Limitations

### Known Issues
- ⚠️ **Output transposed**: Model outputs `[1, 5, 1344]` instead of `[1344, 5]` (needs transpose in post-processing)
- ⚠️ **No NMS**: Non-Maximum Suppression must be done in post-processing
- ⚠️ **Fixed input size**: 256×256 only (no dynamic shapes)
- ⚠️ **No letterboxing**: Simple resize (aspect ratio not preserved)

### Training Details
- **Dataset**: Face detection datasets (WIDER FACE, SCUT-HEAD, MOT17)
- **Calibration**: Representative dataset for INT8 quantization
- **Classes**: Single class (`face`)
- **Augmentation**: Standard YOLO augmentations

## Re-exporting the Model

If you need to modify or re-export the model:

1. **Ensure calibration dataset** is available:
   ```bash
   ls datasets/wider_face_calibration/  # Should have images
   ```

2. **Update calibration config**:
   ```yaml
   # research/yolov8_face_pipeline/models/calibration.yaml
   path: ../../../datasets/wider_face_calibration
   train: images
   val: images
   nc: 1
   names: ['face']
   ```

3. **Run export script**:
   ```bash
   cd docs
   python yolo_int8.py
   ```

4. **Rename output**:
   ```bash
   mv yolov8n_saved_model/yolov8n_full_integer_quant.tflite \
      ../models/yolov8n_face_int8.tflite
   ```

See [`docs/model_training_reference.md`](../docs/model_training_reference.md) for detailed instructions.

## Upgrading the Model

### To YOLOv8s (Small) - Better Accuracy
```python
model = YOLO('yolov8s.pt')  # Larger model
model.export(format='tflite', int8=True, data='calibration.yaml', imgsz=320)
# Result: ~5MB model, ~30-40ms inference, better detection
```

### To YOLOv8m (Medium) - Best Accuracy
```python
model = YOLO('yolov8m.pt')  # Even larger
model.export(format='tflite', int8=True, data='calibration.yaml', imgsz=384)
# Result: ~12MB model, ~50-70ms inference, excellent detection
```

### Trade-offs

| Model | Size | Speed (Pi) | Accuracy | Use Case |
|-------|------|------------|----------|----------|
| YOLOv8n | 1.5MB | 40-50ms | Good | Embedded, real-time |
| YOLOv8s | ~5MB | 60-80ms | Better | Balanced |
| YOLOv8m | ~12MB | 100-150ms | Best | Accuracy-critical |

## Files NOT in Git

These are excluded via `.gitignore`:

- `*.pt` - PyTorch models (6-50MB)
- `*.onnx` - ONNX intermediate format
- `*/tflite_saved_models/` - Export artifacts
- `saved_model/` - TensorFlow SavedModel directories
- `*.pb` - Protocol buffer files

Only the final `yolov8n_face_int8.tflite` is version controlled.

## Verification

**Check model is working**:
```bash
cd production
python attendance_ultralytics.py
# Should see: "✅ Model loaded: yolov8n_face_int8.tflite"
```

**Check model file**:
```bash
ls -lh models/yolov8n_face_int8.tflite
# Should show: ~1.5M file
```

---

**Model is shared by both production and research pipelines!** 🎯
