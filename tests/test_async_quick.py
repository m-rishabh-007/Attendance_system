#!/usr/bin/env python3
"""
Quick Test - Async Pipeline with NCNN

Verifies that the async pipeline works correctly:
- NCNN model loads successfully
- Multiprocessing starts without errors
- Camera feed displays with face detection
- FPS is reported correctly

Run: python tests/test_async_quick.py

Date: December 1, 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import time
import numpy as np
from common.config_manager import ConfigManager
from pipeline.async_orchestrator import AsyncOrchestrator

def main():
    print("="*60)
    print("Async Pipeline Quick Test")
    print("="*60)
    
    # Load config
    config = ConfigManager()
    config.load('config.yaml')
    
    # Verify NCNN model exists
    model_path = Path(config.get('detector.model_path'))
    print(f"\n📂 Model: {model_path}")
    print(f"   Runtime: {config.get('detector.runtime')}")
    print(f"   Input size: {config.get('detector.input_size')}")
    
    if not model_path.exists():
        print(f"\n❌ ERROR: Model not found at {model_path}")
        print("Run: yolo export model=yolov8n.pt format=ncnn imgsz=320 half=True")
        return 1
    
    print(f"   ✅ Model found")
    
    # Create async orchestrator
    print(f"\n🚀 Starting async orchestrator...")
    orchestrator = AsyncOrchestrator(config)
    
    try:
        # Start worker process
        orchestrator.start()
        print(f"   ✅ Worker process started (PID: {orchestrator.worker_process.pid})")
        
        # Open camera
        print(f"\n📷 Opening camera...")
        camera_id = config.get('camera.device_id', 0)
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"   ❌ Failed to open camera")
            return 1
        
        print(f"   ✅ Camera opened")
        
        # Process frames
        print(f"\n🎬 Processing frames (press 'q' to quit)...\n")
        
        fps_history = []
        frame_count = 0
        fps_start = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Process frame
            result = orchestrator.process_frame(frame)
            
            # Calculate FPS
            frame_count += 1
            if time.time() - fps_start > 1.0:
                fps = frame_count / (time.time() - fps_start)
                fps_history.append(fps)
                
                # Print stats
                queue_stats = result.get('queue_stats', {})
                tracks_count = len(result.get('tracks', []))
                
                print(f"FPS: {fps:5.2f} | "
                      f"Tracks: {tracks_count:2d} | "
                      f"Queue: {queue_stats.get('queued', 0):3d} | "
                      f"Dropped: {queue_stats.get('dropped', 0):3d} | "
                      f"Frame time: {result.get('processing_time_ms', 0):5.1f}ms")
                
                frame_count = 0
                fps_start = time.time()
            
            # Display
            cv2.imshow("Async Pipeline Test", result.get('annotated_frame', frame))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        if fps_history:
            print(f"\n📊 Summary:")
            print(f"   Mean FPS: {np.mean(fps_history):.2f}")
            print(f"   Min FPS: {np.min(fps_history):.2f}")
            print(f"   Max FPS: {np.max(fps_history):.2f}")
            print(f"   Std Dev: {np.std(fps_history):.2f}")
        
        print(f"\n✅ Test completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️ Interrupted by user")
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        print(f"\n🛑 Stopping orchestrator...")
        orchestrator.stop()
        print(f"   ✅ Shutdown complete")


if __name__ == "__main__":
    exit(main())
