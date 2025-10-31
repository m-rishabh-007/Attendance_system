#!/bin/bash
# setup_venv.sh
# Creates and configures a Python virtual environment for the Face Attendance System
# Designed for both development (laptops) and deployment (Raspberry Pi)

set -e  # Exit on any error

echo "🔧 Face Attendance System - Virtual Environment Setup"
echo "======================================================"
echo ""

# Detect platform
if grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null || grep -q "BCM" /proc/cpuinfo 2>/dev/null; then
    PLATFORM="raspberry_pi"
    echo "📟 Platform detected: Raspberry Pi"
else
    PLATFORM="desktop"
    echo "💻 Platform detected: Desktop/Laptop"
fi

echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "🐍 Python version: $PYTHON_VERSION"

# Require Python 3.8+
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "❌ Error: Python 3.8 or higher is required"
    exit 1
fi

# Create virtual environment
VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
    echo "⚠️  Virtual environment already exists at ./$VENV_DIR"
    read -p "Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "ℹ️  Using existing virtual environment"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment at ./$VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies based on platform
echo ""
echo "📥 Installing dependencies..."

if [ "$PLATFORM" = "raspberry_pi" ]; then
    echo "   Installing Raspberry Pi optimized packages..."
    
    # Install TFLite Runtime (lightweight, no TensorFlow needed)
    echo "   - Installing tflite-runtime..."
    pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite-runtime
    
    # Install OpenCV with minimal dependencies
    echo "   - Installing opencv-python-headless (no GUI, lighter)..."
    pip install opencv-python-headless
    
    # Install core dependencies
    echo "   - Installing core packages..."
    pip install numpy scipy pyyaml
    
else
    echo "   Installing desktop/development packages..."
    
    # For development, prefer full TensorFlow or tflite-runtime
    echo "   - Installing tflite-runtime (or use 'pip install tensorflow' for full package)..."
    pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite-runtime || pip install tensorflow
    
    # Install OpenCV with GUI support
    echo "   - Installing opencv-python..."
    pip install opencv-python
    
    # Install core dependencies
    echo "   - Installing core packages..."
    pip install numpy scipy pyyaml
    
    # Optional: Install Ultralytics for model export/training
    read -p "Install Ultralytics for model training/export? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   - Installing ultralytics and dependencies..."
        pip install ultralytics torch onnx onnx2tf
    fi
fi

echo ""
echo "✨ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate the environment:"
echo "      source venv/bin/activate"
echo ""
echo "   2. Test the pipeline:"
if [ "$PLATFORM" = "raspberry_pi" ]; then
    echo "      cd yolov8_face_pipeline"
    echo "      python pipeline_main.py"
else
    echo "      python attendance_system.py  # Quick demo"
    echo "      cd yolov8_face_pipeline && python pipeline_main.py  # Production"
fi
echo ""
echo "   3. Deactivate when done:"
echo "      deactivate"
echo ""
echo "💡 Tip: Add 'venv/' to .gitignore to avoid committing the virtual environment"
