#!/usr/bin/env python3
"""
TFLite INT8 Post-Training Quantization
=======================================

Quantizes TFLite FP32 model to INT8 using calibration data.

Quantization Pipeline:
    TFLite FP32 → Calibrate with 97 face samples → TFLite INT8

Usage:
    python3 tools/quantization/quantize_tflite_int8.py

Input:
    models/recognition/auraface_resnet100_fp32.tflite (166 MB)
    tools/quantization/calibration_data/faces_100.npy (97 samples)

Output:
    models/recognition/auraface_resnet100_int8.tflite (~42 MB)

Expected Results:
    - Size reduction: 166 MB → 42 MB (4x smaller)
    - Speed improvement: 2x faster on Pi
    - Accuracy drop: <1% (with good calibration data)

Phase: Week 2 Day 3 - INT8 Quantization
"""

import os
import sys
import logging
from pathlib import Path
from typing import Generator, List

import numpy as np
import tensorflow as tf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TFLiteINT8Quantizer:
    """
    Quantizes TFLite FP32 model to INT8 using calibration data.
    
    Post-Training Quantization Process:
        1. Load FP32 TFLite model
        2. Load calibration dataset (face images)
        3. Create representative dataset generator
        4. Run TFLite quantization
        5. Validate INT8 model
    """
    
    def __init__(
        self,
        fp32_model_path: str,
        calibration_data_path: str,
        int8_output_path: str
    ):
        """
        Initialize quantizer.
        
        Args:
            fp32_model_path: Path to FP32 TFLite model
            calibration_data_path: Path to calibration .npy file (97 samples)
            int8_output_path: Path to output INT8 TFLite model
        """
        self.fp32_model_path = Path(fp32_model_path)
        self.calibration_data_path = Path(calibration_data_path)
        self.int8_output_path = Path(int8_output_path)
        
        # Validate paths
        if not self.fp32_model_path.exists():
            raise FileNotFoundError(f"FP32 TFLite model not found: {self.fp32_model_path}")
        
        if not self.calibration_data_path.exists():
            raise FileNotFoundError(f"Calibration data not found: {self.calibration_data_path}")
        
        # Create output directory
        self.int8_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Quantizer initialized")
        logger.info(f"  Input FP32 TFLite: {self.fp32_model_path}")
        logger.info(f"  Calibration data: {self.calibration_data_path}")
        logger.info(f"  Output INT8 TFLite: {self.int8_output_path}")
    
    def load_calibration_data(self) -> np.ndarray:
        """Load calibration dataset from .npy file."""
        logger.info("=" * 70)
        logger.info("STEP 1: Loading Calibration Data")
        logger.info("=" * 70)
        
        logger.info(f"Loading calibration data from: {self.calibration_data_path}")
        calibration_data = np.load(str(self.calibration_data_path))
        
        logger.info(f"  Calibration samples: {calibration_data.shape[0]}")
        logger.info(f"  Sample shape: {calibration_data.shape[1:]}")
        logger.info(f"  Data type: {calibration_data.dtype}")
        logger.info(f"  Value range: [{calibration_data.min():.4f}, {calibration_data.max():.4f}]")
        
        # Check if data is normalized
        if calibration_data.max() <= 1.0:
            logger.info("  ✅ Data appears to be normalized [0, 1]")
        else:
            logger.warning("  ⚠️  Data may not be normalized (max > 1.0)")
        
        size_mb = self.calibration_data_path.stat().st_size / (1024 * 1024)
        logger.info(f"  File size: {size_mb:.2f} MB")
        
        return calibration_data
    
    def create_representative_dataset(
        self, 
        calibration_data: np.ndarray
    ) -> Generator[List[np.ndarray], None, None]:
        """
        Create representative dataset generator for quantization.
        
        TFLite quantizer needs a generator that yields batches of input data.
        This is used to calculate quantization parameters (min/max ranges).
        
        Args:
            calibration_data: Numpy array of face images (N, H, W, C)
        
        Yields:
            List containing single input tensor (batch_size=1)
        """
        def representative_dataset_gen():
            for i in range(len(calibration_data)):
                # Get single sample
                sample = calibration_data[i:i+1]  # Keep batch dimension
                
                # Ensure float32 dtype
                sample = sample.astype(np.float32)
                
                # Yield as list (TFLite expects list of inputs)
                yield [sample]
        
        return representative_dataset_gen
    
    def quantize_model(self, calibration_data: np.ndarray) -> bytes:
        """
        Quantize FP32 TFLite model to INT8.
        
        Args:
            calibration_data: Calibration dataset for quantization
        
        Returns:
            Quantized INT8 TFLite model (bytes)
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 2: Quantizing FP32 → INT8")
        logger.info("=" * 70)
        
        logger.info("Initializing TFLite converter...")
        converter = tf.lite.TFLiteConverter.from_saved_model(
            str(self.fp32_model_path).replace('.tflite', '_saved_model')
        )
        
        # Actually, we need to load from the TFLite file directly
        # Let me fix this - we need to use a different approach
        logger.info(f"Loading FP32 TFLite model from: {self.fp32_model_path}")
        
        # For quantizing existing TFLite model, we need the SavedModel
        # But we already have the SavedModel from step 1!
        saved_model_dir = "models/recognition/auraface_saved_model"
        
        if not Path(saved_model_dir).exists():
            raise FileNotFoundError(
                f"SavedModel not found: {saved_model_dir}\n"
                f"Please run convert_onnx_to_tflite.py first!"
            )
        
        logger.info(f"Loading SavedModel from: {saved_model_dir}")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        
        # INT8 quantization settings
        logger.info("Configuring INT8 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Set representative dataset for calibration
        logger.info(f"Setting representative dataset ({len(calibration_data)} samples)...")
        converter.representative_dataset = self.create_representative_dataset(calibration_data)
        
        # Force full integer quantization (INT8 for weights AND activations)
        logger.info("Enabling full INT8 quantization (weights + activations)...")
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8  # Quantized input
        converter.inference_output_type = tf.uint8  # Quantized output
        
        logger.info("")
        logger.info("⚙️  Quantization Configuration:")
        logger.info("  - Optimization: DEFAULT")
        logger.info("  - Target ops: INT8")
        logger.info("  - Input type: UINT8")
        logger.info("  - Output type: UINT8")
        logger.info("  - Calibration samples: 97")
        logger.info("")
        
        logger.info("Running quantization (this may take 1-2 minutes)...")
        logger.info("  🔄 Analyzing activation ranges...")
        logger.info("  🔄 Calculating quantization parameters...")
        logger.info("  🔄 Quantizing weights and activations...")
        
        tflite_quant_model = converter.convert()
        
        logger.info("✅ Quantization complete!")
        
        return tflite_quant_model
    
    def save_quantized_model(self, tflite_quant_model: bytes) -> None:
        """Save quantized INT8 model to file."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 3: Saving INT8 Model")
        logger.info("=" * 70)
        
        logger.info(f"Saving INT8 TFLite model to: {self.int8_output_path}")
        with open(self.int8_output_path, 'wb') as f:
            f.write(tflite_quant_model)
        
        logger.info("✅ INT8 model saved successfully")
        
        # Compare file sizes
        fp32_size_mb = self.fp32_model_path.stat().st_size / (1024 * 1024)
        int8_size_mb = self.int8_output_path.stat().st_size / (1024 * 1024)
        reduction = (1 - int8_size_mb / fp32_size_mb) * 100
        
        logger.info("")
        logger.info("📊 Model Size Comparison:")
        logger.info(f"  FP32 model: {fp32_size_mb:.2f} MB")
        logger.info(f"  INT8 model: {int8_size_mb:.2f} MB")
        logger.info(f"  Reduction: {reduction:.1f}% (expected: ~75%)")
    
    def validate_quantized_model(self) -> None:
        """Validate INT8 model can be loaded and used for inference."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 4: Validating INT8 Model")
        logger.info("=" * 70)
        
        logger.info("Loading INT8 TFLite model...")
        interpreter = tf.lite.Interpreter(model_path=str(self.int8_output_path))
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        
        logger.info(f"  Input: {input_details['name']}")
        logger.info(f"    Shape: {input_details['shape']}")
        logger.info(f"    Dtype: {input_details['dtype']}")
        logger.info(f"    Quantization: {input_details['quantization']}")
        
        logger.info(f"  Output: {output_details['name']}")
        logger.info(f"    Shape: {output_details['shape']}")
        logger.info(f"    Dtype: {output_details['dtype']}")
        logger.info(f"    Quantization: {output_details['quantization']}")
        
        # Test inference with dummy input
        logger.info("")
        logger.info("Running test inference...")
        input_shape = input_details['shape']
        
        # Create quantized input (UINT8)
        if input_details['dtype'] == np.uint8:
            dummy_input = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
        else:
            dummy_input = np.random.randn(*input_shape).astype(input_details['dtype'])
        
        interpreter.set_tensor(input_details['index'], dummy_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details['index'])
        
        logger.info(f"  Test output shape: {output.shape}")
        logger.info(f"  Test output dtype: {output.dtype}")
        logger.info(f"  Test output range: [{output.min()}, {output.max()}]")
        logger.info("✅ INT8 model validation successful")
    
    def quantize(self) -> None:
        """Run complete quantization pipeline."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("TFLite FP32 → INT8 QUANTIZATION PIPELINE")
        logger.info("=" * 70)
        logger.info("")
        
        try:
            # Step 1: Load calibration data
            calibration_data = self.load_calibration_data()
            
            # Step 2: Quantize FP32 → INT8
            tflite_quant_model = self.quantize_model(calibration_data)
            
            # Step 3: Save INT8 model
            self.save_quantized_model(tflite_quant_model)
            
            # Step 4: Validate INT8 model
            self.validate_quantized_model()
            
            logger.info("")
            logger.info("=" * 70)
            logger.info("✅ QUANTIZATION COMPLETE!")
            logger.info("=" * 70)
            logger.info(f"📁 Output INT8 model: {self.int8_output_path}")
            logger.info(f"📊 Model size: {self.int8_output_path.stat().st_size / (1024 * 1024):.2f} MB")
            logger.info("")
            logger.info("🎯 Next step:")
            logger.info("   python3 tools/quantization/validate_tflite_model.py")
            logger.info("")
            logger.info("⚠️  Note: Accuracy validation recommended to check <1% drop")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error("")
            logger.error("=" * 70)
            logger.error("❌ QUANTIZATION FAILED")
            logger.error("=" * 70)
            logger.error(f"Error: {e}")
            logger.error("")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main entry point."""
    # Paths
    fp32_model_path = "models/recognition/auraface_resnet100_fp32.tflite"
    calibration_data_path = "tools/quantization/calibration_data/faces_100.npy"
    int8_output_path = "models/recognition/auraface_resnet100_int8.tflite"
    
    # Create quantizer
    quantizer = TFLiteINT8Quantizer(
        fp32_model_path=fp32_model_path,
        calibration_data_path=calibration_data_path,
        int8_output_path=int8_output_path
    )
    
    # Run quantization
    quantizer.quantize()


if __name__ == "__main__":
    main()
