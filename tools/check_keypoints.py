#!/usr/bin/env python3
"""
Probe script to check if YOLOv8n TFLite face model outputs keypoints.
This will determine if we need an additional landmark detector for alignment.
"""

from ultralytics import YOLO
import cv2
import sys

def check_model_keypoints():
    """Check if the TFLite model provides facial keypoints."""
    
    print("=" * 60)
    print("YOLOv8 Face Model - Keypoint Detection Check")
    print("=" * 60)
    
    # Load the TFLite model
    model_path = "models/detection/yolov8n_face_int8.tflite"
    print(f"\n1. Loading model: {model_path}")
    
    try:
        model = YOLO(model_path)
        print("   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return False
    
    # Try to open camera
    print("\n2. Opening camera (ID: 1)...")
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("   ⚠️  Camera 1 not available, trying camera 0...")
        cap = cv2.VideoCapture(0)
        
    if not cap.isOpened():
        print("   ❌ No camera available")
        return False
    
    print("   ✅ Camera opened successfully")
    
    # Capture a frame
    print("\n3. Capturing frame...")
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("   ❌ Failed to capture frame")
        return False
    
    print(f"   ✅ Frame captured: {frame.shape}")
    
    # Run detection + tracking
    print("\n4. Running model.track() with BoT-SORT...")
    try:
        results = model.track(frame, persist=True, tracker='botsort.yaml', verbose=False)
        print("   ✅ Inference successful")
    except Exception as e:
        print(f"   ❌ Inference failed: {e}")
        return False
    
    # Check for keypoints
    print("\n5. Checking for keypoints in results...")
    r = results[0]
    
    # Check if keypoints attribute exists
    has_kpts = (getattr(r, "keypoints", None) is not None and r.keypoints is not None)
    
    if not has_kpts:
        print("   ❌ No keypoints attribute found")
        print("\n" + "=" * 60)
        print("RESULT: Model does NOT provide keypoints")
        print("=" * 60)
        print("\nRECOMMENDATION:")
        print("  - Use MediaPipe Face Mesh for 5-point landmarks")
        print("  - Or use a separate landmark detector (dlib, RetinaFace)")
        print("  - Alignment stage will need an external landmark model")
        return False
    
    # Check keypoints shape
    shape = None
    if r.keypoints is not None and hasattr(r.keypoints, "xyn"):
        xyn_data = getattr(r.keypoints, "xyn", None)
        if xyn_data is not None:
            shape = tuple(xyn_data.shape)
    
    print(f"   ✅ Keypoints found!")
    print(f"   Shape: {shape}")
    
    # Check number of detections
    num_faces = shape[0] if shape else 0
    num_kpts = shape[1] if shape and len(shape) > 1 else 0
    
    print(f"\n   Faces detected: {num_faces}")
    print(f"   Keypoints per face: {num_kpts}")
    
    # Show first face keypoints if available
    if num_faces > 0 and num_kpts > 0 and r.keypoints is not None:
        print("\n6. First face keypoints (normalized [0..1] coordinates):")
        print("   Format: (x, y)")
        xyn_data = getattr(r.keypoints, "xyn", None)
        if xyn_data is not None:
            kpts = xyn_data[0]
            
            landmark_names = [
                "Left eye",
                "Right eye", 
                "Nose tip",
                "Left mouth corner",
                "Right mouth corner"
            ]
            
            for i, (x, y) in enumerate(kpts):
                name = landmark_names[i] if i < len(landmark_names) else f"Point {i}"
                print(f"   {name:20s}: ({x:.4f}, {y:.4f})")
    
    print("\n" + "=" * 60)
    if num_kpts == 5:
        print("RESULT: Model provides 5-point facial landmarks! ✅")
        print("=" * 60)
        print("\nRECOMMENDATION:")
        print("  ✅ No external landmark detector needed")
        print("  ✅ Proceed with alignment using model's keypoints")
        print("  ✅ Implement AlignmentStage with similarity transform")
        print("  ✅ Use 112x112 canonical template for ArcFace")
    elif num_kpts > 0:
        print(f"RESULT: Model provides {num_kpts}-point landmarks")
        print("=" * 60)
        print("\nRECOMMENDATION:")
        print(f"  - Model outputs {num_kpts} points (not standard 5-point)")
        print("  - Map to 5-point format or use all available points")
        print("  - Alignment is still possible")
    else:
        print("RESULT: Keypoints exist but empty (no faces detected)")
        print("=" * 60)
        print("\nRECOMMENDATION:")
        print("  - Re-run with a face visible in frame")
        print("  - Model appears to support keypoints")
    
    return True

if __name__ == "__main__":
    print("\n🔍 Checking if YOLOv8n TFLite model supports facial keypoints...\n")
    
    try:
        success = check_model_keypoints()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error during check: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
