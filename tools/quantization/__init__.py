"""
Quantization Tools - INT8 Model Quantization for Raspberry Pi

This package contains tools for converting FP32 face recognition models
to INT8 quantized models for faster inference on edge devices.

Week 1: Use FP32 baseline models (no quantization)
Week 2: Run quantization workflow to create INT8 models

Modules (Week 2):
    - calibration_data_reader: Custom reader for onnxruntime.quantization
    - collect_calibration_data: Collect 100 face samples for calibration
    - quantize_auraface: FP32 → INT8 model converter
    - validate_quantized_model: Compare FP32 vs INT8 accuracy

Author: Attendance System Team
Created: November 2025
Phase: 3B - Quantization (Week 2)
"""

__version__ = '0.1.0'
__author__ = 'Attendance System Team'
__phase__ = '3B - Quantization (Week 2)'
