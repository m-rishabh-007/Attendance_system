# Face Attendance System v3.0 - Quick Start Guide

⚠️ **IMPORTANT**: This project requires **Python 3.11**. See [README.md](../README.md) for installation.

## 🚀 5-Minute Setup

### Prerequisites

- Python 3.11 (required - see README.md)
- Webcam
- Ubuntu/Raspberry Pi OS

### Step 1: Clone & Install

```bash
# Navigate to project
cd /home/rishabh/Attendance_system

# Activate Python 3.11 virtual environment (IMPORTANT!)
source venv_py311/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Verify Model Exists

```bash
ls -lh models/yolov8n_face_int8.tflite
# Should show: yolov8n_face_int8.tflite (1.5M)
```

### Step 3: Run Tests (Optional but Recommended)

```bash
# Activate virtual environment first
source venv_py311/bin/activate

# Test all design patterns
python tests/run_all_tests.py
```

Expected output:
```
✅ PASS  Singleton Pattern (ConfigManager)
✅ PASS  Observer Pattern (EventSystem)
✅ PASS  Factory + Strategy Patterns (Detectors & Trackers)

🎉 ALL TESTS PASSED!
```

### Step 4: Run the System

```bash
# Activate virtual environment first
source venv_py311/bin/activate

# Run the system
python attendance_system.py
```

You should see:
```
================================================================
Face Attendance System v2.0 Starting
================================================================
[INFO] Loading configuration...
✅ Configuration loaded from config.yaml
[INFO] Setting up event system...
✅ Event system initialized
[INFO] Creating face detector...
✅ Detector created: YOLODetector(...)
[INFO] Creating tracker...
✅ Tracker created: BotSORTTracker(...)
[INFO] Opening camera...
✅ Camera 0 opened successfully

================================================================
🚀 System initialized successfully!
================================================================

Press 'q' to quit
```

### Step 5: Use the System

- **Webcam window appears** showing live feed
- **Green boxes** appear around detected faces
- **Track IDs** shown above each face (e.g., "ID:1 0.95")
- **Press 'q'** to quit

---

## ⚙️ Configuration

Edit `config.yaml` to customize:

### Switch Detector

```yaml
detector:
  type: yolo      # Or 'tflite' for lightweight
  model_path: models/yolov8n_face_int8.tflite
  confidence_threshold: 0.5  # Lower = more detections, more false positives
```

### Switch Tracker

```yaml
tracker:
  type: botsort   # Or 'none' to disable tracking
  track_thresh: 0.5
  track_buffer: 30  # Frames to keep lost tracks
```

### Camera Settings

```yaml
camera:
  device_id: 0    # Change if using external webcam (1, 2, etc.)
  width: 1280
  height: 720
```

### Display Options

```yaml
display:
  show_window: true       # Set false for headless
  show_fps: true          # Show FPS counter
  show_track_ids: true    # Show track IDs
  show_confidence: true   # Show confidence scores
```

---

## 🧪 Testing Individual Components

### Test Configuration Loading

```bash
python tests/test_config_manager.py
```

### Test Event System

```bash
python tests/test_event_system.py
```

### Test Factories

```bash
python tests/test_factories.py
```

---

## 🔧 Troubleshooting

### Problem: "Camera not found"

**Solution 1**: Try different camera ID
```yaml
camera:
  device_id: 1  # Try 1, 2, etc.
```

**Solution 2**: Check camera access
```bash
ls -l /dev/video*
# You should see /dev/video0, /dev/video1, etc.
```

### Problem: "No module named 'ultralytics'"

**Solution**: Install dependencies
```bash
pip install ultralytics
# Or use tflite detector instead (lighter)
```

### Problem: Low FPS

**Solution 1**: Use TFLite detector (faster)
```yaml
detector:
  type: tflite
```

**Solution 2**: Lower resolution
```yaml
camera:
  width: 640
  height: 480
```

**Solution 3**: Process fewer frames
```yaml
performance:
  process_every_n_frames: 2  # Process every 2nd frame
```

### Problem: "Model not found"

**Solution**: Verify model path
```bash
ls models/yolov8n_face_int8.tflite
```

If missing, check `docs/model_training_reference.md` for export instructions.

---

## 📊 Performance Expectations

### Desktop/Laptop
- **FPS**: 30-40 (YOLO), 40-50 (TFLite)
- **Latency**: ~25ms per frame
- **Memory**: ~250MB

### Raspberry Pi 4
- **FPS**: 10-15 (YOLO), 15-20 (TFLite)
- **Latency**: ~60-80ms per frame
- **Memory**: ~200MB

---

## 🎯 Next Steps

### 1. Understand Architecture

Read `docs/ARCHITECTURE_V2.md` to understand:
- Design patterns used
- How to extend the system
- Adding new features

### 2. Compare Detectors

```bash
# Test YOLO
# Edit config.yaml: detector.type = yolo
python attendance_system.py

# Test TFLite
# Edit config.yaml: detector.type = tflite
python attendance_system.py
```

### 3. Add Custom Event Handler

```python
# Create my_handler.py
from common.event_system import EventSystem, EventType

def my_handler(event):
    print(f"Face detected: {event.data}")

events = EventSystem()
events.subscribe(EventType.FACE_DETECTED, my_handler)
```

### 4. Future Development

- **Face Recognition**: Add embedding extraction + database lookup
- **Attendance Logging**: Save attendance to database
- **Web Dashboard**: View attendance statistics
- **REST API**: Access system remotely

See `docs/ARCHITECTURE_V2.md` for detailed future plans.

---

## 📚 Documentation

- **`docs/ARCHITECTURE_V2.md`** - Complete architecture documentation
- **`README.md`** - Project overview
- **`config.yaml`** - Configuration reference (with comments)
- **`.github/copilot-instructions.md`** - Development guidelines

---

## 🆘 Getting Help

1. **Check logs**: System logs all operations to terminal
2. **Run tests**: `python tests/run_all_tests.py`
3. **Read docs**: `docs/ARCHITECTURE_V2.md` has detailed troubleshooting
4. **Check config**: Verify all paths in `config.yaml` are correct

---

## ✅ Success Checklist

- [ ] Tests pass (`python tests/run_all_tests.py`)
- [ ] Camera opens successfully
- [ ] Faces detected (green boxes appear)
- [ ] Track IDs assigned correctly
- [ ] FPS displayed in window
- [ ] Can quit with 'q' key

If all checked, you're ready to develop! 🎉

---

**Last Updated**: October 31, 2025  
**Version**: 2.0.0
