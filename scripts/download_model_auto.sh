#!/bin/bash
"""
Download AuraFace Model Using InsightFace Library

This script uses the insightface library to automatically download
the AuraFace/ArcFace ResNet100 model.

It will:
1. Activate py311 virtual environment
2. Install required packages (insightface, onnx)
3. Download the model using insightface API
4. Copy model to correct location
5. Verify model structure

Usage:
    bash scripts/download_model_auto.sh

Author: Attendance System Team
Created: November 2025
Phase: 3A - Core Recognition (Day 3)
"""

# Change to project root
cd "$(dirname "$0")/.." || exit 1

echo "======================================================================="
echo "AuraFace Model Auto-Download Script"
echo "======================================================================="
echo ""

# Check if venv exists
if [ ! -d "venv_py311" ]; then
    echo "❌ Virtual environment not found: venv_py311"
    echo "   Please create it first: python3.11 -m venv venv_py311"
    exit 1
fi

echo "✅ Found virtual environment: venv_py311"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv_py311/bin/activate || exit 1
echo "✅ Virtual environment activated"
echo ""

# Verify Python version
PYTHON_VERSION=$(python --version 2>&1)
echo "Python version: $PYTHON_VERSION"
echo ""

# Install required packages
echo "Installing required packages..."
echo "  - insightface (for model download)"
echo "  - onnx (for model verification)"
echo "  - onnxruntime (for inference)"
echo ""

pip install --quiet insightface onnx onnxruntime || {
    echo "❌ Failed to install packages"
    exit 1
}

echo "✅ Packages installed"
echo ""

# Run Python script to download model
echo "Downloading AuraFace model..."
echo "(This may take a few minutes depending on your internet speed)"
echo ""

python3 << 'PYTHON_SCRIPT'
import os
import sys
from pathlib import Path
import shutil

try:
    from insightface.app import FaceAnalysis
    import onnx
    print("✅ insightface and onnx imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Try: pip install insightface onnx")
    sys.exit(1)

# Project paths
project_root = Path.cwd()
models_dir = project_root / "models" / "recognition"
models_dir.mkdir(parents=True, exist_ok=True)

target_path = models_dir / "auraface_resnet100_fp32.onnx"

print("\n" + "=" * 70)
print("Downloading model using InsightFace...")
print("=" * 70)

try:
    # Initialize FaceAnalysis (downloads models automatically)
    # Using 'buffalo_l' which includes ArcFace R100
    print("\nInitializing FaceAnalysis with 'buffalo_l' model pack...")
    print("(Models will be downloaded to ~/.insightface/models/)")
    
    app = FaceAnalysis(
        name='buffalo_l',
        providers=['CPUExecutionProvider']
    )
    
    print("✅ Models downloaded successfully!")
    print("\nSearching for recognition model...")
    
    # Find the downloaded model
    insightface_dir = Path.home() / ".insightface" / "models" / "buffalo_l"
    
    if not insightface_dir.exists():
        print(f"❌ Model directory not found: {insightface_dir}")
        sys.exit(1)
    
    print(f"✅ Found model directory: {insightface_dir}")
    
    # Look for recognition model (usually w600k_r50.onnx or similar)
    model_files = list(insightface_dir.glob("*.onnx"))
    
    print(f"\nFound {len(model_files)} ONNX models:")
    for model_file in model_files:
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"  - {model_file.name} ({size_mb:.1f} MB)")
    
    # Find recognition model (usually largest one, ~60-100MB)
    recognition_model = None
    for model_file in model_files:
        size_mb = model_file.stat().st_size / (1024 * 1024)
        if 50 < size_mb < 150:  # Recognition models are typically 60-100MB
            # Verify it's a recognition model by checking structure
            try:
                model = onnx.load(str(model_file))
                input_shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
                output_shape = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]
                
                # Recognition models have input (1,3,112,112) and output (1,512)
                if (input_shape[1:] == [3, 112, 112] or input_shape[1:] == [-1, 112, 112]) and \
                   (output_shape[-1] == 512 or output_shape[-1] == -1):
                    recognition_model = model_file
                    print(f"\n✅ Found recognition model: {model_file.name}")
                    print(f"   Input shape: {input_shape}")
                    print(f"   Output shape: {output_shape}")
                    break
            except Exception as e:
                print(f"   ⚠️  Could not verify {model_file.name}: {e}")
                continue
    
    if recognition_model is None:
        print("\n⚠️  Could not automatically identify recognition model")
        print("   Using largest .onnx file as recognition model...")
        recognition_model = max(model_files, key=lambda f: f.stat().st_size)
    
    # Copy to target location
    print(f"\nCopying model to: {target_path}")
    shutil.copy2(recognition_model, target_path)
    
    # Verify copied model
    if target_path.exists():
        size_mb = target_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model copied successfully!")
        print(f"   Location: {target_path}")
        print(f"   Size: {size_mb:.1f} MB")
        
        # Verify structure
        print("\nVerifying model structure...")
        model = onnx.load(str(target_path))
        
        input_info = model.graph.input[0]
        input_shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
        print(f"✅ Input shape: {input_shape}")
        
        output_info = model.graph.output[0]
        output_shape = [dim.dim_value for dim in output_info.type.tensor_type.shape.dim]
        print(f"✅ Output shape: {output_shape}")
        
        if (input_shape[1:] == [3, 112, 112] or input_shape[1:] == [-1, 112, 112]):
            print("✅ Input shape matches AuraFace standard")
        else:
            print(f"⚠️  Warning: Unexpected input shape {input_shape}")
        
        if (output_shape[-1] == 512 or output_shape[-1] == -1):
            print("✅ Output shape matches AuraFace standard")
        else:
            print(f"⚠️  Warning: Unexpected output shape {output_shape}")
        
        print("\n" + "=" * 70)
        print("✅ MODEL DOWNLOAD COMPLETE!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Run tests: python3 tests/test_auraface_basic.py")
        print("2. Verify recognition works with real faces")
        print("3. Proceed to Day 5: Implement QualityScorer")
        
    else:
        print("❌ Failed to copy model")
        sys.exit(1)
        
except Exception as e:
    print(f"\n❌ Error during download: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

PYTHON_SCRIPT

EXIT_CODE=$?

# Deactivate virtual environment
deactivate

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "🎉 Success! Model is ready to use."
    echo ""
    echo "Run tests with:"
    echo "  source venv_py311/bin/activate"
    echo "  python3 tests/test_auraface_basic.py"
    exit 0
else
    echo ""
    echo "❌ Download failed. Exit code: $EXIT_CODE"
    echo ""
    echo "Manual download options:"
    echo "  python3 scripts/download_auraface_model.py"
    exit 1
fi
