# Documentation Directory

This directory contains historical references and documentation for model creation, training, and export processes.

## Files

### `ARCHITECTURE_V3_HYBRID.md` ⭐
**Purpose**: Complete V3 HYBRID architecture documentation (651 lines)

**Contents**:
- V3 HYBRID design philosophy (perfect 25/25 score)
- Complete folder structure and data flow
- All 4 design patterns explained
- Code examples for each component
- How to add new stages (alignment, recognition, attendance, API)
- 9-week migration plan
- Scoring comparison with previous architectures

**When to read**:
- Understanding the complete system architecture
- Adding new features (alignment, recognition, etc.)
- Learning about design patterns in practice
- Planning development work

### `ARCHITECTURE_V2.md`
**Purpose**: Previous V2 architecture documentation (historical reference)

**Contents**:
- OOP design patterns (Singleton, Factory, Strategy, Observer)
- v2.0 architecture explanation
- Why it was superseded by V3 HYBRID

### `QUICKSTART.md`
**Purpose**: 5-minute setup and run guide

**Contents**:
- Quick installation steps
- Running the system
- Common troubleshooting
- Next steps

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

### `training_scripts/yolo_int8.py`
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

### Architecture Documentation
- **`ARCHITECTURE_V3_HYBRID.md`** ⭐ - Complete V3 HYBRID guide (651 lines)
- **`../ARCHITECTURE.md`** - Architecture overview (redirects to V3_HYBRID)
- **`ARCHITECTURE_V2.md`** - Previous V2 architecture (historical)

### Model Documentation
- **`../models/README.md`** - Complete model technical documentation, usage examples, and benchmarks
- **`model_training_reference.md`** - How models were created and exported
- **`training_scripts/yolo_int8.py`** - Export script reference

### Deployment & Setup
- **`../DEPLOYMENT.md`** - Complete deployment guide (venv, Docker, Raspberry Pi)
- **`QUICKSTART.md`** - 5-minute setup guide
- **`../README.md`** - Project overview and quick start

### Historical Implementations
- **`../archive/v1_pipeline/README.md`** - v1.0 custom implementation (TFLite + ByteTrack from scratch)

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
- **Understand the architecture?** → Read `ARCHITECTURE_V3_HYBRID.md` ⭐
- **Run the system quickly?** → See `QUICKSTART.md` (5 minutes)
- **Use the model?** → See `../models/README.md`
- **Deploy to Raspberry Pi?** → See `../DEPLOYMENT.md`
- **Export a new model?** → Read `model_training_reference.md` + use `training_scripts/yolo_int8.py` as template
- **Add new features?** → See `ARCHITECTURE_V3_HYBRID.md` (Adding Components section)
- **Improve performance?** → See `../DEPLOYMENT.md` (Performance Tuning)
- **See historical implementations?** → Check `../archive/v1_pipeline/`

## Contributing

When adding new model variants or export processes:

1. **Document the process** in `model_training_reference.md`
2. **Save export scripts** with meaningful names (e.g., `yolo_v8s_640_export.py`)
3. **Update model README** in `../models/` with new model specs
4. **Test on target hardware** and document performance benchmarks
5. **Keep calibration datasets** for reproducibility (or document sources)

---

**Last Updated**: November 4, 2025 (V3 HYBRID Architecture Complete)
