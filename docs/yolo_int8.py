"""
YOLOv8 TFLite INT8 Export Script - Reference Template

This script exports a trained YOLOv8 PyTorch model (.pt) to TFLite INT8 format.

Usage:
    1. Update model_path to point to your trained .pt file
    2. Update data path to point to your calibration.yaml
    3. Run: python yolo_int8.py

Critical Settings:
    - format='tflite': Export to TensorFlow Lite
    - int8=True: Enable INT8 quantization (reduces size, increases speed)
    - data=calibration.yaml: Required for INT8 calibration
    - imgsz=256: Input size (change to 320 or 640 if needed)
    - nms=False: CRITICAL - NMS done in post-processing, not baked into model
    - dynamic=False: Fixed input shape for edge device optimization

Output:
    - model_int8.tflite (or similar name based on input)
    - This is the file used in production pipeline

See: docs/model_training_reference.md for complete documentation
"""

from ultralytics import YOLO 

# NOTE: Update these paths before running!
model_path = '/home/rishabh/Attendance_system/new/models/model.pt' 
data = '/home/rishabh/Attendance_system/new/models/calibration.yaml'  

try: 
    model = YOLO(model_path) 
    print("Model loaded successfully. Starting TFLite INT8 export...") 
     
    model.export( 
        format='tflite',   
        int8=True,   
        data=data,  
        imgsz=256,   
        nms=False,      
        dynamic=False
    ) 
    print(f"completed....file name '{model_path.replace('.pt', '_int8_raw.tflite')}'") 

except Exception as e: 
    print(f"\nerror occurred: {e}") 
    print("Please check that the model path and data path are correct.")