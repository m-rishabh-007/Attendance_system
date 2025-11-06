# Face Attendance System

Real-time face detection and tracking pipeline optimized for Raspberry Pi and laptops.

**Current Status**: ✅ Phase 1 Complete (Detection + Tracking) | 🚧 Phase 2 (Face Alignment - In Progress)

---

## ⚠️ CRITICAL: Python 3.11 Required

**This project requires Python 3.11** due to:
- `tflite-runtime` is NOT available for Python 3.12+ (PyPI limitation)
- TensorFlow 2.16+ has circular dependency issues with MediaPipe on Python 3.12
- NumPy/MediaPipe version conflicts on Python 3.12

### Ubuntu 24.04 Installation (Python 3.12 default)

```bash
# Install Python 3.11 via deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# Create Python 3.11 virtual environment
cd /home/rishabh/Attendance_system
python3.11 -m venv venv_py311
source venv_py311/bin/activate

# Install dependencies (locked versions)
pip install -r requirements.txt
```

### Raspberry Pi Installation (Python 3.11 available in default repos)

```bash
# Install Python 3.11 (should be in default repos)
sudo apt update
sudo apt install python3.11 python3.11-venv

# Create virtual environment
python3.11 -m venv venv_py311
source venv_py311/bin/activate
pip install -r requirements.txt
```

### Verify Installation

```bash
python --version  # Should show: Python 3.11.x
python -c "import numpy, cv2, mediapipe, tflite_runtime; print('✅ All dependencies OK')"
```

---

## 🚀 Quick Start

### Run the System

```bash
cd /home/rishabh/Attendance_system
source venv_py311/bin/activate  # IMPORTANT: Use Python 3.11 environment
python attendance_system.py
```

**Features:**
- ✅ Superior tracking quality (maintains IDs during head shaking, motion blur)
- ✅ BoT-SORT tracker (better than basic ByteTrack)
- ✅ Minimal maintenance (10 lines of core logic)
- ✅ Battle-tested by millions of users

---

## 📁 Project Structure

```
Attendance_system/
├── attendance_system.py           # 🎯 MAIN ENTRY POINT (V3 HYBRID architecture)
├── config.yaml                    # 🔧 Runtime configuration
│
├── common/                        # 🛠️ SHARED COMPONENTS
│   ├── config_manager.py         # Singleton pattern
│   ├── event_system.py           # Observer pattern
│   └── base_classes.py           # Abstract interfaces
│
├── detectors/                     # 🔍 DETECTION MODULES
│   ├── factory.py                # Factory pattern
│   ├── yolo_detector.py          # YOLOv8n INT8 TFLite detector
│   └── base_detector.py          # Abstract detector interface
│
├── tracking/                      # 🎯 TRACKING MODULES
│   ├── factory.py                # Factory pattern (Strategy)
│   ├── botsort_tracker.py        # BoT-SORT tracker (Phase 1 complete)
│   └── base_tracker.py           # Abstract tracker interface
│
├── aligners/                      # 📐 ALIGNMENT MODULES (Phase 2 - In Progress)
│   ├── factory.py                # Factory pattern
│   ├── mediapipe_aligner.py      # MediaPipe Face Mesh alignment
│   └── base_aligner.py           # Abstract aligner interface
│
├── pipeline/                      # 🚀 PIPELINE ORCHESTRATION
│   ├── orchestrator.py           # Main pipeline controller
│   └── detection_stage.py        # Detection pipeline stage
│
├── tests/                         # ✅ TEST SUITE
│   ├── test_config_manager.py    # Singleton tests
│   ├── test_event_system.py      # Observer tests
│   ├── test_factories.py         # Factory + Strategy tests
│   ├── test_persistent_tracking.py # Tracking smoke test
│   └── run_all_tests.py          # Master test runner
│
├── models/                        # 🤖 MODEL FILES
│   └── detection/
│       └── yolov8n_face_int8.tflite  # INT8 quantized YOLOv8n (1.5MB)
│
├── docs/                          # 📄 DOCUMENTATION
│   ├── DEVELOPER_GUIDE.md        # Master developer reference
│   ├── ARCHITECTURE_V3_HYBRID.md # V3 architecture explained
│   ├── ADR_001_ALIGNMENT_MODEL_SELECTION.md # MediaPipe decision
│   └── QUICKSTART.md             # 5-minute setup guide
│
├── logs/                          # 📊 RUNTIME LOGS
│   └── attendance_YYYY-MM-DD.log # Daily log files
│
├── tools/                         # � UTILITY SCRIPTS
│   └── check_keypoints.py        # Verify YOLO model capabilities
│
├── venv_py311/                    # ⚠️ PYTHON 3.11 VIRTUAL ENVIRONMENT (required)
│
└── archive/                       # 📚 HISTORICAL CODE (v1.0)
    └── v1_pipeline/              # Old pipeline implementation
```

---

## 🎯 Architecture Highlights

### v2.0 Design Patterns (Production-Ready)

✅ **Singleton Pattern** - `ConfigManager` ensures single configuration source  
✅ **Observer Pattern** - `EventSystem` enables event-driven architecture  
✅ **Factory Pattern** - Easy detector/tracker swapping via configuration  
✅ **Strategy Pattern** - Interchangeable tracking algorithms (BoT-SORT, ByteTrack)

### Key Features

- **Modular Design**: Clean separation (detection, tracking, events, config)
- **OOP Architecture**: Maintainable, extensible, testable
- **Superior Tracking**: BoT-SORT maintains IDs during motion/occlusion
- **Configuration-Driven**: Change models/trackers without code changes
- **Comprehensive Tests**: 20 unit tests covering all design patterns

---

## 🛠️ Installation

### Prerequisites

**Python 3.11 is REQUIRED** - See the [Python 3.11 installation section](#️-critical-python-311-required) above.

```bash
# System packages
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv python3.11-dev v4l-utils

# Check camera
v4l2-ctl --list-devices
```

### Setup Python 3.11 Environment

```bash
cd /home/rishabh/Attendance_system

# Create Python 3.11 virtual environment
python3.11 -m venv venv_py311

# Activate environment
source venv_py311/bin/activate

# Verify Python version
python --version  # Should show: Python 3.11.x

# Install dependencies (locked versions for stability)
pip install -r requirements.txt

# Verify installation
python -c "import numpy, cv2, mediapipe, tflite_runtime; print('✅ All dependencies OK')"
```

### Daily Usage

```bash
# Activate Python 3.11 environment
cd /home/rishabh/Attendance_system
source venv_py311/bin/activate

# Run the system
python attendance_system.py

# When done
deactivate
```

---

## ⚙️ Configuration

Edit `config.yaml` at project root:

```yaml
# Detector settings
detector:
  type: yolo                        # or 'tflite'
  confidence_threshold: 0.5         # Detection confidence
  model_path: models/yolov8n_face_int8.tflite

# Tracker settings
tracker:
  type: botsort                     # or 'bytetrack', 'none'  
track_thresh: 0.4            # Minimum confidence for tracking
track_buffer: 90             # Frames to keep lost tracks (3 sec @ 30fps)
match_thresh: 0.4            # IoU threshold for matching

# Camera
camera_id: 1                 # Change to 0 for default camera
```

---

## 🧪 Testing & Validation

### Test Face Detection
```bash
cd production
python attendance_ultralytics.py
```

**Expected Behavior:**
- Green boxes around detected faces
- Persistent track IDs (e.g., "ID:1", "ID:2")
- IDs maintained during head movement
- New faces get incremental IDs

### Stress Test (Head Shaking)
1. Position yourself in front of camera
2. Shake head vigorously
3. **Expected**: Same ID maintained throughout
4. **If ID changes**: Check lighting, camera focus

---

## 🐛 Troubleshooting

### No Camera Access
```bash
# Check permissions
ls -l /dev/video*
sudo usermod -a -G video $USER
# Log out and back in

# Test camera
ffplay /dev/video1
```

### No Face Detected
- Check lighting (need adequate illumination)
- Lower `confidence_threshold` to 0.4
- Verify camera is not covered
- Check distance from camera (2-6 feet optimal)

### Poor Tracking Quality
- Production pipeline already optimal
- Check camera FPS: `v4l2-ctl --device=/dev/video1 --all`
- Ensure good lighting
- Clean camera lens

---

## 📚 Documentation

### Core Documentation
- **Developer Guide**: [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md) ⭐ (Master reference)
- **Complete Architecture**: [`docs/ARCHITECTURE_V3_HYBRID.md`](docs/ARCHITECTURE_V3_HYBRID.md) ⭐
- **Quick Start Guide**: [`docs/QUICKSTART.md`](docs/QUICKSTART.md)

### Architecture Decision Records (ADRs)
- **ADR-001**: [Face Alignment Model Selection](docs/ADR_001_ALIGNMENT_MODEL_SELECTION.md) (Why MediaPipe?)

### Additional Resources
- **Model Documentation**: [`models/README.md`](models/README.md)
- **Training Reference**: [`docs/model_training_reference.md`](docs/model_training_reference.md)
- **Deployment Guide**: [`DEPLOYMENT.md`](DEPLOYMENT.md)
- **V2 Architecture (Previous)**: [`docs/ARCHITECTURE_V2.md`](docs/ARCHITECTURE_V2.md)
- **Historical Implementations**: [`archive/v1_pipeline/`](archive/v1_pipeline/)

---

## 🎓 Learning Outcomes

### What You Built
✅ YOLOv8 TFLite INT8 inference from scratch  
✅ Custom ByteTrack tracker with Kalman filtering  
✅ Multi-stage detection → tracking pipeline  
✅ Configuration management system  
✅ Performance profiling and optimization

### What You Learned
✅ YOLO post-processing internals  
✅ Object tracking algorithms  
✅ TFLite quantization and optimization  
✅ System design trade-offs  
✅ **When to use custom vs. off-the-shelf solutions** ⭐

---

## 🚧 Roadmap

### Phase 1: Face Recognition (Next)
- [ ] Face alignment module
- [ ] ArcFace embedding extraction
- [ ] Face database (SQLite)
- [ ] Similarity matching
- [ ] Recognition confidence scoring

### Phase 2: Attendance System
- [ ] Attendance logging
- [ ] Daily/monthly reports
- [ ] User management
- [ ] Time tracking
- [ ] Export to CSV/Excel

### Phase 3: Production Features
- [ ] REST API server
- [ ] Web dashboard
- [ ] Multi-camera support
- [ ] Cloud deployment
- [ ] Monitoring and alerts

---

## 📄 License

Educational/Research project. Use responsibly and ethically.

---

## 🙏 Acknowledgments

- **Ultralytics** - YOLO models and tracking
- **ByteTrack** - Multi-object tracking algorithm
- **OpenCV** - Computer vision library
- **TensorFlow Lite** - Inference runtime

---

## 💡 Key Takeaway

**Building from scratch taught you deeply. Using Ultralytics in production is smart engineering. Both have their place!** 🎯

- **Research pipeline**: Valuable learning, embedded use cases
- **Production pipeline**: Battle-tested, superior quality, maintainable

Choose the right tool for the job! 🚀
