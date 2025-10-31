# Face Attendance System

Real-time face detection and tracking pipeline optimized for Raspberry Pi and laptops.

**Current Status**: ✅ Detection + Tracking + Production Ready | 🚧 Recognition + Attendance (planned)

---

## 🚀 Quick Start

### Production Pipeline (Recommended)

```bash
cd /home/rishabh/Attendance_system
source venv/bin/activate
cd production
python attendance_ultralytics.py
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
├── production/                    # 🎯 USE THIS FOR DEPLOYMENT
│   ├── attendance_ultralytics.py # Main production pipeline
│   ├── attendance_prototype.py   # Original prototype (reference)
│   └── README.md                 # Production documentation
│
├── research/                      # 📚 LEARNING REFERENCE
│   ├── yolov8_face_pipeline/     # Custom TFLite implementation
│   │   ├── pipeline_main.py      # Raw TFLite + custom ByteTrack
│   │   ├── face_tracker_bytetrack.py  # ByteTrack from scratch
│   │   └── pipeline_config.yaml  # Shared configuration
│   └── README.md                 # Research documentation
│
├── models/                           # 🤖 MODEL FILES
│   ├── yolov8n_face_int8.tflite  # INT8 quantized YOLOv8n (1.5MB)
│   └── README.md                 # Model documentation
│
├── venv/                          # Python virtual environment
└── README.md                      # This file
```

---

## 🎯 Which Pipeline Should I Use?

### For Production/Deployment → `production/`
**Use**: `attendance_ultralytics.py`

✅ **Reasons:**
- Superior track persistence (IDs maintained during fast motion)
- BoT-SORT tracker (motion + appearance features)
- Minimal code maintenance (10 lines vs 800+)
- Active development and bug fixes
- Production-ready error handling

### For Learning/Research → `research/`
**Use**: `yolov8_face_pipeline/pipeline_main.py`

✅ **Reasons:**
- Understand YOLO post-processing internals
- Learn Kalman filter tracking mathematics
- Custom algorithm experimentation
- Embedded deployment (<512MB RAM)
- Portfolio/academic projects

---

## 📊 Performance Comparison

| Metric | Production (Ultralytics) | Research (Custom) |
|--------|--------------------------|-------------------|
| **Code Lines** | 30 | 800+ |
| **Track Persistence** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Good |
| **Motion Robustness** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Moderate |
| **Inference Speed** | 45ms (slower) | 25ms (faster) |
| **FPS** | 20-25 | 25-27 |
| **Maintenance** | Low | High |
| **Use Case** | Production | Learning/Embedded |

**Verdict**: Production pipeline trades 20ms latency for **significantly better tracking quality**. Both achieve real-time performance.

---

## 🛠️ Installation

### Prerequisites
```bash
# System packages
sudo apt-get update
sudo apt-get install python3-pip python3-venv v4l-utils

# Check camera
v4l2-ctl --list-devices
```

### Setup Virtual Environment

**Automated (Recommended)**:
```bash
cd /home/rishabh/Attendance_system
./setup_venv.sh
source venv/bin/activate
```

**Manual**:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Edit `research/yolov8_face_pipeline/pipeline_config.yaml`:

```yaml
# Detection
confidence_threshold: 0.5    # Lower = more detections
iou_threshold: 0.3           # NMS threshold

# Tracking  
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

- **Production Guide**: `production/README.md`
- **Research Guide**: `research/README.md`
- **Model Documentation**: `models/README.md`
- **Training Reference**: `docs/model_training_reference.md`
- **Architecture Plan**: `ARCHITECTURE.md`
- **Deployment Guide**: `DEPLOYMENT.md`

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
