#!/usr/bin/env python3
"""
ONNX to TFLite FP32 Converter
==============================

Converts AuraFace ONNX model to TensorFlow Lite FP32 format.

Conversion Pipeline:
    ONNX → TensorFlow SavedModel → TFLite FP32

Usage:
    python3 tools/quantization/convert_onnx_to_tflite.py

Input:
    models/recognition/auraface_resnet100_fp32.onnx (166 MB)

Output:
    models/recognition/auraface_resnet100_fp32.tflite (166 MB)
    models/recognition/auraface_saved_model/ (TF SavedModel, intermediate)

Phase: Week 2 Day 2 - TFLite Conversion
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
import onnx2tf
import tensorflow as tf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ONNXToTFLiteConverter:
    """
    Converts ONNX model to TFLite FP32 format.
    
    Workflow:
        1. Load ONNX model
        2. Convert ONNX → TensorFlow SavedModel (onnx-tf)
        3. Convert SavedModel → TFLite FP32 (TFLite converter)
        4. Validate output model
    """
    
    def __init__(
        self,
        onnx_model_path: str,
        tflite_output_path: str,
        saved_model_dir: str = "models/recognition/auraface_saved_model"
    ):
        """
        Initialize converter.
        
        Args:
            onnx_model_path: Path to input ONNX model
            tflite_output_path: Path to output TFLite model
            saved_model_dir: Directory for intermediate TF SavedModel
        """
        self.onnx_model_path = Path(onnx_model_path)
        self.tflite_output_path = Path(tflite_output_path)
        self.saved_model_dir = Path(saved_model_dir)
        
        # Validate paths
        if not self.onnx_model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_model_path}")
        
        # Create output directories
        self.tflite_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.saved_model_dir.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Converter initialized")
        logger.info(f"  Input ONNX: {self.onnx_model_path}")
        logger.info(f"  Output TFLite: {self.tflite_output_path}")
        logger.info(f"  Intermediate SavedModel: {self.saved_model_dir}")
    
    def load_onnx_model(self) -> onnx.ModelProto:
        """Load and validate ONNX model."""
        logger.info("=" * 70)
        logger.info("STEP 1: Loading ONNX Model")
        logger.info("=" * 70)
        
        logger.info(f"Loading ONNX model from: {self.onnx_model_path}")
        onnx_model = onnx.load(str(self.onnx_model_path))
        
        # Validate ONNX model
        logger.info("Validating ONNX model...")
        onnx.checker.check_model(onnx_model)
        logger.info("✅ ONNX model is valid")
        
        # Log model info
        input_info = onnx_model.graph.input[0]
        output_info = onnx_model.graph.output[0]
        
        input_shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
        output_shape = [dim.dim_value for dim in output_info.type.tensor_type.shape.dim]
        
        logger.info(f"  Input: {input_info.name}, shape: {input_shape}")
        logger.info(f"  Output: {output_info.name}, shape: {output_shape}")
        
        # Check file size
        size_mb = self.onnx_model_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Model size: {size_mb:.2f} MB")
        
        return onnx_model
    
    def convert_onnx_to_savedmodel(self, onnx_model: onnx.ModelProto) -> None:
        """Convert ONNX model to TensorFlow SavedModel."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 2: Converting ONNX → TensorFlow SavedModel")
        logger.info("=" * 70)
        
        logger.info("Converting ONNX to TensorFlow using onnx2tf...")
        
        # Remove existing SavedModel if present
        if self.saved_model_dir.exists():
            import shutil
            logger.info("  Removing existing SavedModel...")
            shutil.rmtree(self.saved_model_dir)
        
        logger.info(f"Exporting SavedModel to: {self.saved_model_dir}")
        
        # Use onnx2tf for conversion
        onnx2tf.convert(
            input_onnx_file_path=str(self.onnx_model_path),
            output_folder_path=str(self.saved_model_dir.parent),
            output_signaturedefs=True,
            non_verbose=True
        )
        
        logger.info("✅ SavedModel exported successfully")
        
        # Check SavedModel size
        size_mb = sum(
            f.stat().st_size for f in self.saved_model_dir.rglob('*') if f.is_file()
        ) / (1024 * 1024)
        logger.info(f"  SavedModel size: {size_mb:.2f} MB")
    
    def convert_savedmodel_to_tflite(self) -> None:
        """Convert TensorFlow SavedModel to TFLite FP32."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 3: Converting SavedModel → TFLite FP32")
        logger.info("=" * 70)
        
        logger.info("Initializing TFLite converter...")
        converter = tf.lite.TFLiteConverter.from_saved_model(str(self.saved_model_dir))
        
        # Optimization settings (FP32, no quantization yet)
        converter.optimizations = []  # No optimizations for FP32
        
        logger.info("Converting to TFLite FP32...")
        tflite_model = converter.convert()
        
        logger.info(f"Saving TFLite model to: {self.tflite_output_path}")
        with open(self.tflite_output_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info("✅ TFLite FP32 model created successfully")
        
        # Check TFLite size
        size_mb = self.tflite_output_path.stat().st_size / (1024 * 1024)
        logger.info(f"  TFLite model size: {size_mb:.2f} MB")
    
    def validate_tflite_model(self) -> None:
        """Validate TFLite model can be loaded and used for inference."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 4: Validating TFLite Model")
        logger.info("=" * 70)
        
        logger.info("Loading TFLite model...")
        interpreter = tf.lite.Interpreter(model_path=str(self.tflite_output_path))
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        
        logger.info(f"  Input: {input_details['name']}, shape: {input_details['shape']}, dtype: {input_details['dtype']}")
        logger.info(f"  Output: {output_details['name']}, shape: {output_details['shape']}, dtype: {output_details['dtype']}")
        
        # Test inference with dummy input
        logger.info("Running test inference...")
        input_shape = input_details['shape']
        dummy_input = np.random.randn(*input_shape).astype(input_details['dtype'])
        
        interpreter.set_tensor(input_details['index'], dummy_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details['index'])
        
        logger.info(f"  Test output shape: {output.shape}")
        logger.info(f"  Test output range: [{output.min():.4f}, {output.max():.4f}]")
        logger.info("✅ TFLite model validation successful")
    
    def convert(self) -> None:
        """Run complete conversion pipeline."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("ONNX → TFLite FP32 CONVERSION PIPELINE")
        logger.info("=" * 70)
        logger.info("")
        
        try:
            # Step 1: Load ONNX model
            onnx_model = self.load_onnx_model()
            
            # Step 2: Convert ONNX → SavedModel
            self.convert_onnx_to_savedmodel(onnx_model)
            
            # Step 3: Convert SavedModel → TFLite FP32
            self.convert_savedmodel_to_tflite()
            
            # Step 4: Validate TFLite model
            self.validate_tflite_model()
            
            logger.info("")
            logger.info("=" * 70)
            logger.info("✅ CONVERSION COMPLETE!")
            logger.info("=" * 70)
            logger.info(f"📁 Output TFLite model: {self.tflite_output_path}")
            logger.info(f"📊 Model size: {self.tflite_output_path.stat().st_size / (1024 * 1024):.2f} MB")
            logger.info("")
            logger.info("🎯 Next step:")
            logger.info("   python3 tools/quantization/quantize_tflite_int8.py")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error("")
            logger.error("=" * 70)
            logger.error("❌ CONVERSION FAILED")
            logger.error("=" * 70)
            logger.error(f"Error: {e}")
            logger.error("")
            raise


def main():
    """Main entry point."""
    # Paths
    onnx_model_path = "models/recognition/auraface_resnet100_fp32.onnx"
    tflite_output_path = "models/recognition/auraface_resnet100_fp32.tflite"
    saved_model_dir = "models/recognition/auraface_saved_model"
    
    # Create converter
    converter = ONNXToTFLiteConverter(
        onnx_model_path=onnx_model_path,
        tflite_output_path=tflite_output_path,
        saved_model_dir=saved_model_dir
    )
    
    # Run conversion
    converter.convert()


if __name__ == "__main__":
    main()
