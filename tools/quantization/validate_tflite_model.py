#!/usr/bin/env python3
"""
TFLite Model Validator
=======================

Compares FP32 vs INT8 TFLite model accuracy on validation dataset.

Validation Metrics:
    - Cosine similarity between FP32 and INT8 embeddings
    - Mean Absolute Error (MAE)
    - Inference time comparison
    - Accuracy drop estimation

Usage:
    python3 tools/quantization/validate_tflite_model.py

Input:
    models/recognition/auraface_resnet100_fp32.tflite
    models/recognition/auraface_resnet100_int8.tflite
    tools/quantization/calibration_data/faces_100.npy (validation samples)

Output:
    Validation report (console + JSON file)

Expected Results:
    - Cosine similarity: >0.99 (closer to 1.0 = better)
    - MAE: <0.05
    - Accuracy drop: <1%
    - Speed improvement: 2x faster

Phase: Week 2 Day 3 - Validation
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cosine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TFLiteModelValidator:
    """
    Validates INT8 TFLite model against FP32 baseline.
    
    Validation Process:
        1. Load FP32 and INT8 models
        2. Load validation dataset
        3. Run inference on both models
        4. Compare embeddings (cosine similarity, MAE)
        5. Measure inference time
        6. Generate validation report
    """
    
    def __init__(
        self,
        fp32_model_path: str,
        int8_model_path: str,
        validation_data_path: str,
        report_output_path: str = "tools/quantization/validation_report.json"
    ):
        """
        Initialize validator.
        
        Args:
            fp32_model_path: Path to FP32 TFLite model
            int8_model_path: Path to INT8 TFLite model
            validation_data_path: Path to validation .npy file
            report_output_path: Path to save validation report JSON
        """
        self.fp32_model_path = Path(fp32_model_path)
        self.int8_model_path = Path(int8_model_path)
        self.validation_data_path = Path(validation_data_path)
        self.report_output_path = Path(report_output_path)
        
        # Validate paths
        if not self.fp32_model_path.exists():
            raise FileNotFoundError(f"FP32 model not found: {self.fp32_model_path}")
        
        if not self.int8_model_path.exists():
            raise FileNotFoundError(f"INT8 model not found: {self.int8_model_path}")
        
        if not self.validation_data_path.exists():
            raise FileNotFoundError(f"Validation data not found: {self.validation_data_path}")
        
        self.report_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Validator initialized")
        logger.info(f"  FP32 model: {self.fp32_model_path}")
        logger.info(f"  INT8 model: {self.int8_model_path}")
        logger.info(f"  Validation data: {self.validation_data_path}")
    
    def load_models(self) -> Tuple[tf.lite.Interpreter, tf.lite.Interpreter]:
        """Load FP32 and INT8 TFLite models."""
        logger.info("=" * 70)
        logger.info("STEP 1: Loading Models")
        logger.info("=" * 70)
        
        logger.info("Loading FP32 TFLite model...")
        fp32_interpreter = tf.lite.Interpreter(model_path=str(self.fp32_model_path))
        fp32_interpreter.allocate_tensors()
        
        logger.info("Loading INT8 TFLite model...")
        int8_interpreter = tf.lite.Interpreter(model_path=str(self.int8_model_path))
        int8_interpreter.allocate_tensors()
        
        # Get model details
        fp32_input_details = fp32_interpreter.get_input_details()[0]
        fp32_output_details = fp32_interpreter.get_output_details()[0]
        
        int8_input_details = int8_interpreter.get_input_details()[0]
        int8_output_details = int8_interpreter.get_output_details()[0]
        
        logger.info("")
        logger.info("FP32 Model:")
        logger.info(f"  Input: {fp32_input_details['shape']}, dtype: {fp32_input_details['dtype']}")
        logger.info(f"  Output: {fp32_output_details['shape']}, dtype: {fp32_output_details['dtype']}")
        
        logger.info("")
        logger.info("INT8 Model:")
        logger.info(f"  Input: {int8_input_details['shape']}, dtype: {int8_input_details['dtype']}")
        logger.info(f"  Output: {int8_output_details['shape']}, dtype: {int8_output_details['dtype']}")
        
        return fp32_interpreter, int8_interpreter
    
    def load_validation_data(self) -> np.ndarray:
        """Load validation dataset."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 2: Loading Validation Data")
        logger.info("=" * 70)
        
        logger.info(f"Loading validation data from: {self.validation_data_path}")
        validation_data = np.load(str(self.validation_data_path))
        
        # Use all samples for validation (97 samples)
        num_samples = len(validation_data)
        
        logger.info(f"  Validation samples: {num_samples}")
        logger.info(f"  Sample shape: {validation_data.shape[1:]}")
        logger.info(f"  Data type: {validation_data.dtype}")
        
        return validation_data
    
    def run_inference(
        self,
        interpreter: tf.lite.Interpreter,
        input_data: np.ndarray,
        model_name: str
    ) -> Tuple[np.ndarray, float]:
        """
        Run inference on TFLite model.
        
        Args:
            interpreter: TFLite interpreter
            input_data: Input images (N, H, W, C)
            model_name: Model name for logging (FP32/INT8)
        
        Returns:
            Tuple of (embeddings, avg_inference_time_ms)
        """
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        
        embeddings = []
        inference_times = []
        
        num_samples = len(input_data)
        logger.info(f"Running {model_name} inference on {num_samples} samples...")
        
        for i, sample in enumerate(input_data):
            # Prepare input
            input_tensor = sample[np.newaxis, ...].astype(input_details['dtype'])
            
            # Handle INT8 quantization (if needed)
            if input_details['dtype'] == np.uint8:
                # Quantize FP32 input to UINT8
                scale, zero_point = input_details['quantization']
                if scale > 0:  # Check if quantization params are valid
                    input_tensor = (input_tensor / scale + zero_point).astype(np.uint8)
                else:
                    # No quantization params, assume input is already in [0, 255]
                    input_tensor = (input_tensor * 255).astype(np.uint8)
            
            # Run inference
            interpreter.set_tensor(input_details['index'], input_tensor)
            
            start_time = time.time()
            interpreter.invoke()
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            output = interpreter.get_tensor(output_details['index'])
            
            # Handle INT8 output (dequantize to FP32)
            if output_details['dtype'] == np.uint8:
                scale, zero_point = output_details['quantization']
                if scale > 0:
                    output = (output.astype(np.float32) - zero_point) * scale
            
            embeddings.append(output.flatten())
            inference_times.append(inference_time)
            
            if (i + 1) % 20 == 0:
                logger.info(f"  Processed {i + 1}/{num_samples} samples...")
        
        embeddings = np.array(embeddings)
        avg_inference_time = np.mean(inference_times)
        
        logger.info(f"  ✅ {model_name} inference complete")
        logger.info(f"    Embeddings shape: {embeddings.shape}")
        logger.info(f"    Avg inference time: {avg_inference_time:.2f} ms")
        
        return embeddings, avg_inference_time
    
    def compare_embeddings(
        self,
        fp32_embeddings: np.ndarray,
        int8_embeddings: np.ndarray
    ) -> Dict[str, float]:
        """
        Compare FP32 and INT8 embeddings.
        
        Args:
            fp32_embeddings: FP32 model embeddings (N, embedding_dim)
            int8_embeddings: INT8 model embeddings (N, embedding_dim)
        
        Returns:
            Dictionary with comparison metrics
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 4: Comparing Embeddings")
        logger.info("=" * 70)
        
        num_samples = len(fp32_embeddings)
        
        # Calculate cosine similarities
        logger.info("Calculating cosine similarities...")
        cosine_similarities = []
        
        for i in range(num_samples):
            fp32_emb = fp32_embeddings[i]
            int8_emb = int8_embeddings[i]
            
            # Cosine similarity = 1 - cosine distance
            similarity = 1 - cosine(fp32_emb, int8_emb)
            cosine_similarities.append(similarity)
        
        cosine_similarities = np.array(cosine_similarities)
        
        # Calculate Mean Absolute Error (MAE)
        logger.info("Calculating Mean Absolute Error...")
        mae = np.mean(np.abs(fp32_embeddings - int8_embeddings))
        
        # Calculate Mean Squared Error (MSE)
        mse = np.mean((fp32_embeddings - int8_embeddings) ** 2)
        
        # Calculate metrics
        metrics = {
            'mean_cosine_similarity': float(np.mean(cosine_similarities)),
            'min_cosine_similarity': float(np.min(cosine_similarities)),
            'max_cosine_similarity': float(np.max(cosine_similarities)),
            'std_cosine_similarity': float(np.std(cosine_similarities)),
            'mae': float(mae),
            'mse': float(mse),
            'num_samples': int(num_samples)
        }
        
        logger.info("")
        logger.info("📊 Embedding Comparison Results:")
        logger.info(f"  Cosine Similarity (mean): {metrics['mean_cosine_similarity']:.6f}")
        logger.info(f"  Cosine Similarity (min):  {metrics['min_cosine_similarity']:.6f}")
        logger.info(f"  Cosine Similarity (max):  {metrics['max_cosine_similarity']:.6f}")
        logger.info(f"  Cosine Similarity (std):  {metrics['std_cosine_similarity']:.6f}")
        logger.info(f"  Mean Absolute Error (MAE): {metrics['mae']:.6f}")
        logger.info(f"  Mean Squared Error (MSE):  {metrics['mse']:.6f}")
        
        return metrics
    
    def generate_report(
        self,
        metrics: Dict[str, float],
        fp32_time: float,
        int8_time: float
    ) -> Dict[str, Any]:
        """Generate validation report."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 5: Generating Report")
        logger.info("=" * 70)
        
        # Calculate model sizes
        fp32_size_mb = self.fp32_model_path.stat().st_size / (1024 * 1024)
        int8_size_mb = self.int8_model_path.stat().st_size / (1024 * 1024)
        size_reduction = (1 - int8_size_mb / fp32_size_mb) * 100
        
        # Calculate speed improvement
        speedup = fp32_time / int8_time
        
        report = {
            'model_comparison': {
                'fp32_model_size_mb': float(fp32_size_mb),
                'int8_model_size_mb': float(int8_size_mb),
                'size_reduction_percent': float(size_reduction)
            },
            'inference_time': {
                'fp32_avg_ms': float(fp32_time),
                'int8_avg_ms': float(int8_time),
                'speedup': float(speedup)
            },
            'accuracy_metrics': metrics,
            'pass_criteria': {
                'cosine_similarity_threshold': 0.99,
                'mae_threshold': 0.05,
                'cosine_similarity_pass': metrics['mean_cosine_similarity'] >= 0.99,
                'mae_pass': metrics['mae'] <= 0.05
            },
            'overall_result': 'PASS' if (
                metrics['mean_cosine_similarity'] >= 0.99 and 
                metrics['mae'] <= 0.05
            ) else 'FAIL'
        }
        
        # Save report to JSON
        logger.info(f"Saving report to: {self.report_output_path}")
        with open(self.report_output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("✅ Report saved")
        
        return report
    
    def print_summary(self, report: Dict[str, Any]) -> None:
        """Print validation summary."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 70)
        logger.info("")
        
        logger.info("📊 Model Size:")
        logger.info(f"  FP32: {report['model_comparison']['fp32_model_size_mb']:.2f} MB")
        logger.info(f"  INT8: {report['model_comparison']['int8_model_size_mb']:.2f} MB")
        logger.info(f"  Reduction: {report['model_comparison']['size_reduction_percent']:.1f}%")
        logger.info("")
        
        logger.info("⚡ Inference Speed:")
        logger.info(f"  FP32: {report['inference_time']['fp32_avg_ms']:.2f} ms")
        logger.info(f"  INT8: {report['inference_time']['int8_avg_ms']:.2f} ms")
        logger.info(f"  Speedup: {report['inference_time']['speedup']:.2f}x faster")
        logger.info("")
        
        logger.info("🎯 Accuracy:")
        logger.info(f"  Cosine Similarity: {report['accuracy_metrics']['mean_cosine_similarity']:.6f}")
        logger.info(f"  Mean Absolute Error: {report['accuracy_metrics']['mae']:.6f}")
        logger.info("")
        
        logger.info("✅ Pass Criteria:")
        cosine_pass = report['pass_criteria']['cosine_similarity_pass']
        mae_pass = report['pass_criteria']['mae_pass']
        
        logger.info(f"  Cosine Similarity ≥ 0.99: {'✅ PASS' if cosine_pass else '❌ FAIL'}")
        logger.info(f"  MAE ≤ 0.05: {'✅ PASS' if mae_pass else '❌ FAIL'}")
        logger.info("")
        
        overall = report['overall_result']
        if overall == 'PASS':
            logger.info("=" * 70)
            logger.info("🎉 VALIDATION PASSED! INT8 model is production-ready.")
            logger.info("=" * 70)
        else:
            logger.info("=" * 70)
            logger.info("⚠️  VALIDATION FAILED! Consider re-collecting calibration data.")
            logger.info("=" * 70)
        
        logger.info("")
        logger.info(f"📁 Full report saved to: {self.report_output_path}")
        logger.info("")
    
    def validate(self) -> Dict[str, Any]:
        """Run complete validation pipeline."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("TFLite MODEL VALIDATION PIPELINE")
        logger.info("=" * 70)
        logger.info("")
        
        try:
            # Step 1: Load models
            fp32_interpreter, int8_interpreter = self.load_models()
            
            # Step 2: Load validation data
            validation_data = self.load_validation_data()
            
            # Step 3: Run inference
            logger.info("")
            logger.info("=" * 70)
            logger.info("STEP 3: Running Inference")
            logger.info("=" * 70)
            
            fp32_embeddings, fp32_time = self.run_inference(
                fp32_interpreter, validation_data, "FP32"
            )
            
            int8_embeddings, int8_time = self.run_inference(
                int8_interpreter, validation_data, "INT8"
            )
            
            # Step 4: Compare embeddings
            metrics = self.compare_embeddings(fp32_embeddings, int8_embeddings)
            
            # Step 5: Generate report
            report = self.generate_report(metrics, fp32_time, int8_time)
            
            # Print summary
            self.print_summary(report)
            
            return report
            
        except Exception as e:
            logger.error("")
            logger.error("=" * 70)
            logger.error("❌ VALIDATION FAILED")
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
    int8_model_path = "models/recognition/auraface_resnet100_int8.tflite"
    validation_data_path = "tools/quantization/calibration_data/faces_100.npy"
    report_output_path = "tools/quantization/validation_report.json"
    
    # Create validator
    validator = TFLiteModelValidator(
        fp32_model_path=fp32_model_path,
        int8_model_path=int8_model_path,
        validation_data_path=validation_data_path,
        report_output_path=report_output_path
    )
    
    # Run validation
    validator.validate()


if __name__ == "__main__":
    main()
