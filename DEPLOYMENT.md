# Deployment Guide - Face Attendance System

This guide covers deployment options for the Face Attendance System, with special focus on Raspberry Pi.

## Table of Contents
- [Quick Start (Development)](#quick-start-development)
- [Virtual Environment Setup](#virtual-environment-setup)
- [Docker Deployment](#docker-deployment)
- [Raspberry Pi Specific Setup](#raspberry-pi-specific-setup)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

---

## Quick Start (Development)

### Prerequisites
- Python 3.8 or higher
- Camera (USB webcam or Raspberry Pi Camera Module)
- 2GB+ RAM recommended

### Run Without Installation
```bash
# Install dependencies
pip install opencv-python numpy scipy pyyaml tflite-runtime

# Run quick demo
python attendance_system.py

# Or run production pipeline
cd yolov8_face_pipeline
python pipeline_main.py
```

---

## Virtual Environment Setup

### Automated Setup (Recommended)
```bash
# Run the setup script (auto-detects platform)
./setup_venv.sh

# Activate the environment
source venv/bin/activate

# Run the pipeline
cd yolov8_face_pipeline
python pipeline_main.py

# Deactivate when done
deactivate
```

### Manual Setup

#### On Desktop/Laptop:
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install opencv-python numpy scipy pyyaml

# Install TFLite Runtime
pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite-runtime

# Or install full TensorFlow (larger but includes more tools)
# pip install tensorflow

# Run the pipeline
cd yolov8_face_pipeline
python pipeline_main.py
```

#### On Raspberry Pi:
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip

# Install lightweight OpenCV (no GUI)
pip install opencv-python-headless

# Install core packages
pip install numpy scipy pyyaml

# Install TFLite Runtime (optimized for ARM)
pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite-runtime

# Run the pipeline
cd yolov8_face_pipeline
python pipeline_main.py
```

---

## Docker Deployment

### Build the Docker Image
```bash
# Build for current architecture (ARM on Pi, x86 on desktop)
docker build -t face-attendance:latest .

# Check image size
docker images face-attendance
```

### Run with Docker

#### Basic Run:
```bash
docker run --rm \
  --device /dev/video0:/dev/video0 \
  face-attendance:latest
```

#### With Display (X11 forwarding):
```bash
# Allow X11 access (run once per session)
xhost +local:docker

# Run with display
docker run --rm \
  --device /dev/video0:/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  face-attendance:latest

# Revoke access when done
xhost -local:docker
```

#### Using Docker Compose (Recommended):
```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down

# Restart after config changes
docker-compose restart
```

### Modify Configuration Without Rebuild
```bash
# Edit the config file
nano yolov8_face_pipeline/pipeline_config.yaml

# Restart the container
docker-compose restart
```

---

## Raspberry Pi Specific Setup

### Prerequisites
- Raspberry Pi 4 (4GB+ RAM recommended)
- Raspberry Pi OS (64-bit recommended for better performance)
- Camera Module 3 or USB webcam

### Enable Camera Module
```bash
# For Raspberry Pi Camera Module
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable

# Verify camera works
libcamera-hello

# List available cameras
v4l2-ctl --list-devices
```

### System Optimization
```bash
# Increase GPU memory (edit /boot/config.txt)
sudo nano /boot/config.txt
# Add or modify: gpu_mem=256

# Reboot
sudo reboot
```

### Performance Configuration
Edit `yolov8_face_pipeline/pipeline_config.yaml`:
```yaml
runtime_settings:
  num_threads: 3              # Use 3 cores on Pi 4
  process_every_n_frames: 2   # Process every 2nd frame for better FPS
  desired_input: 256          # Smaller input for faster inference

deployment_config:
  camera_id: 0                # Usually 0 for Pi Camera Module
  delegate: null              # Or "xnnpack" for optimization
```

### Run on Boot (systemd service)
```bash
# Create service file
sudo nano /etc/systemd/system/face-attendance.service
```

```ini
[Unit]
Description=Face Attendance System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/Attendance_system/yolov8_face_pipeline
Environment="PATH=/home/pi/Attendance_system/venv/bin"
ExecStart=/home/pi/Attendance_system/venv/bin/python pipeline_main.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable face-attendance.service
sudo systemctl start face-attendance.service

# Check status
sudo systemctl status face-attendance.service

# View logs
journalctl -u face-attendance.service -f
```

---

## Performance Tuning

### Configuration Parameters

| Parameter | Low-End Device | Balanced | High Performance |
|-----------|---------------|----------|------------------|
| `desired_input` | 256 | 320 | 640 |
| `num_threads` | 2 | 3 | 4 |
| `process_every_n_frames` | 3 | 2 | 1 |
| `confidence_threshold` | 0.7 | 0.6 | 0.5 |

### Expected Performance

#### Raspberry Pi 4 (4GB):
- **256x256 input**: ~20-25 FPS
- **320x320 input**: ~15-20 FPS
- **640x640 input**: ~5-10 FPS

#### Desktop/Laptop (4-core CPU):
- **256x256 input**: ~40-50 FPS
- **320x320 input**: ~30-40 FPS
- **640x640 input**: ~15-25 FPS

### Optimization Tips
1. **Reduce input size**: Biggest impact on FPS
2. **Skip frames**: Process every 2nd or 3rd frame
3. **Lower confidence threshold**: Fewer detections = faster NMS
4. **Use XNNPACK delegate**: Can provide 2-3x speedup on some devices
5. **Disable display**: Run headless for production

---

## Troubleshooting

### Camera Not Found
```bash
# List video devices
v4l2-ctl --list-devices

# Test camera access
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"

# Try different camera indices in pipeline_config.yaml
camera_id: 0  # Try 0, 1, 2, etc.
```

### Low FPS / Laggy Performance
1. Reduce `desired_input` to 256 or 320
2. Increase `process_every_n_frames` to 2 or 3
3. Reduce `num_threads` if CPU is overheating
4. Close other applications

### High CPU Usage
```bash
# Monitor CPU usage
htop

# Check if thermal throttling (Raspberry Pi)
vcgencmd measure_temp
vcgencmd get_throttled

# If throttled, improve cooling or reduce workload
```

### Docker Container Issues
```bash
# Check logs
docker-compose logs -f

# Verify camera access
docker run --rm --device /dev/video0 face-attendance:latest python3 -c "import cv2; print(cv2.VideoCapture(0).isOpened())"

# Enter container for debugging
docker-compose exec face-attendance bash
```

### Memory Issues (Raspberry Pi)
```bash
# Check memory usage
free -h

# Increase swap if needed
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

---

## Production Checklist

- [ ] Virtual environment or Docker container set up
- [ ] Camera tested and working
- [ ] Config file tuned for your hardware
- [ ] Performance tested (FPS, latency)
- [ ] Systemd service configured (for auto-start)
- [ ] Logging configured
- [ ] Backup plan for power failures
- [ ] Security: firewall configured if exposing API
- [ ] Documentation updated with any custom changes

---

## Next Steps

For production deployment, you'll need to implement:
1. **Face Recognition**: Add embedding + database lookup
2. **Attendance Logging**: Store attendance records
3. **API Server**: REST API for remote access
4. **Web Dashboard**: View attendance in real-time

See `docs/ARCHITECTURE_V3_HYBRID.md` for the complete V3 HYBRID architecture and implementation patterns.
