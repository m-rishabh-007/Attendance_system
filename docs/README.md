# Documentation Directory

This directory contains historical references and documentation for model creation, training, and export processes.

## Files

### `model_training_reference.md`
**Purpose**: Historical reference for YOLOv8 model creation and TFLite export process

**Contents**:
- Model specifications (YOLOv8n INT8, 256×256 input)
- Export command and critical settings explanation
- Dataset sources used for training/calibration
- Performance expectations on different hardware
- Re-export instructions for different configurations

**When to read**:
- Understanding how the model was created
- Re-exporting model with different settings (input size, quantization)
- Troubleshooting model-related issues
- Learning about INT8 quantization process

### `yolo_int8.py`
**Purpose**: Reference script for exporting YOLOv8 models to TFLite INT8 format

**Contents**:
- Minimal working example of model export
- Shows critical export parameters (`nms=False`, `int8=True`, etc.)
- Includes error handling

**When to use**:
- Template for exporting new models
- Quick reference for export settings
- Copy-paste starting point for custom exports

**Note**: This script uses hardcoded absolute paths and is for reference only. Update paths before running.

## Related Documentation

### Model Documentation
- **`../models/README.md`** - Complete model technical documentation, usage examples, and benchmarks

### Pipeline Documentation  
- **`../production/README.md`** - Production pipeline guide (Ultralytics + BoT-SORT)
- **`../research/README.md`** - Research pipeline guide (Custom TFLite + ByteTrack)

### Deployment
- **`../DEPLOYMENT.md`** - Complete deployment guide (venv, Docker, Raspberry Pi)
- **`../ARCHITECTURE.md`** - Future system architecture plans

### Main Documentation
- **`../README.md`** - Project overview and quick start guide

## Why This Directory Exists

As the project evolved:
1. **Large datasets were removed** (~6GB) to save space
2. **Intermediate model files were cleaned** (PyTorch, ONNX variants)
3. **Only the final TFLite model was kept** for production use

This documentation directory preserves:
- ✅ Knowledge of how models were created
- ✅ Ability to reproduce the export process
- ✅ Understanding of dataset sources
- ✅ Reference scripts for future exports

Without these references, it would be difficult to re-create or modify the model pipeline.

## Quick Links

### Want to...
- **Use the model?** → See `../models/README.md`
- **Run the pipeline?** → See `../README.md` (Quick Start)
- **Deploy to Raspberry Pi?** → See `../DEPLOYMENT.md`
- **Export a new model?** → Read `model_training_reference.md` + use `yolo_int8.py` as template
- **Understand tracking?** → See `../research/README.md` (ByteTrack explanation)
- **Improve performance?** → See `../DEPLOYMENT.md` (Performance Tuning)

## Contributing

When adding new model variants or export processes:

1. **Document the process** in `model_training_reference.md`
2. **Save export scripts** with meaningful names (e.g., `yolo_v8s_640_export.py`)
3. **Update model README** in `../models/` with new model specs
4. **Test on target hardware** and document performance benchmarks
5. **Keep calibration datasets** for reproducibility (or document sources)

---

**Last Updated**: October 31, 2025
