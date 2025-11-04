"""
Face Attendance Prototype - Original Working Version

This is the original prototype that validated the Ultralytics + ByteTrack approach.
It demonstrated superior tracking quality compared to the custom ByteTrack implementation.

Status: LEGACY / REFERENCE ONLY
Use: production/attendance_ultralytics.py for actual deployment

Key Features:
- Uses Ultralytics YOLO API with TFLite model
- ByteTrack tracker configuration
- Simple 10-line core logic
- Proof of concept that led to production pipeline

History:
- Created as quick test of Ultralytics approach
- Revealed superior tracking quality vs custom implementation
- Validated strategic decision to use Ultralytics for production
- Preserved as reference for minimal working example
"""

import cv2
from ultralytics import YOLO
import numpy as np

# --- Configuration ---
MODEL_PATH = 'models/yolov8n_face_int8.tflite'
TRACKER_CONFIG = 'bytetrack.yaml' # Explicitly define the tracker
CAMERA_INDEX = 1
CONFIDENCE_THRESHOLD = 0.6
IOU_THRESHOLD = 0.3

def main():
    # Load the YOLOv8 model
    print("Loading model...")
    try:
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Open a connection to the camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {CAMERA_INDEX}")
        return
    print("Camera initialized.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        # --- Detection and Tracking with Explicit Tracker ---
        # The 'tracker' argument ensures ByteTrack is used.
        results = model.track(
            frame,
            imgsz=256,
            tracker=TRACKER_CONFIG,
            persist=True,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD
        )

        # Get the bounding boxes and track IDs
        if results[0].boxes is not None and results[0].boxes.id is not None:
            # Plot the results on the frame for visualization
            annotated_frame = results[0].plot()
        else:
            # If no tracks, show the original frame
            annotated_frame = frame

        # Display the annotated frame
        cv2.imshow("Face Detection and Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released.")

if __name__ == "__main__":
    main()