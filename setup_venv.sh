#!/bin/bash
# setup_venv.sh
# Creates and configures a Python 3.11 virtual environment for the Face Attendance System
# ⚠️  IMPORTANT: Requires Python 3.11 (see README.md for installation)

set -e  # Exit on any error

echo "🔧 Face Attendance System - Virtual Environment Setup"
echo "======================================================"
echo ""

# Check if Python 3.11 is available
if ! command -v python3.11 &> /dev/null; then
    echo "❌ Error: Python 3.11 not found!"
    echo ""
    echo "Please install Python 3.11 first:"
    echo ""
    echo "Ubuntu 24.04:"
    echo "  sudo add-apt-repository ppa:deadsnakes/ppa"
    echo "  sudo apt update"
    echo "  sudo apt install python3.11 python3.11-venv python3.11-dev"
    echo ""
    echo "Raspberry Pi:"
    echo "  sudo apt update"
    echo "  sudo apt install python3.11 python3.11-venv"
    echo ""
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3.11 --version 2>&1 | awk '{print $2}')
echo "🐍 Python version: $PYTHON_VERSION"
echo ""

# Create virtual environment
VENV_DIR="venv_py311"
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
    python3.11 -m venv "$VENV_DIR"
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies based on requirements.txt
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
echo "      source venv_py311/bin/activate"
echo ""
echo "   2. Test the system:"
echo "      python attendance_system.py"
echo ""
echo "   3. Deactivate when done:"
echo "      deactivate"
echo ""
echo "💡 Tip: venv_py311/ is already in .gitignore"
