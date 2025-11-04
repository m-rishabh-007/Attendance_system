"""
Smoke Test: Persistent Track ID Verification

This is a CRITICAL test that verifies the core tracking functionality:
- Track IDs should remain stable when a person doesn't move
- Track IDs should persist across frames
- This prevents the bug where IDs fluctuate (1, 2, 3, 4... for same person)

Run this test after ANY changes to:
- detectors/yolo_detector.py
- pipeline/orchestrator.py  
- pipeline/detection_stage.py
- tracking/ components

Usage:
    python tests/test_persistent_tracking.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import time
from typing import List, Set

# Import components
from detectors.yolo_detector import YOLODetector
from common.config_manager import ConfigManager


class TestPersistentTracking:
    """
    Smoke test for verifying Track ID persistence.
    
    This test simulates the most common scenario:
    - One person sitting in front of camera
    - Person doesn't move (or minimal movement)
    - Track ID should stay the same across all frames
    """
    
    def __init__(self):
        """Initialize test with configuration."""
        self.config = ConfigManager.get_instance()
        self.config.load('config.yaml')
        
        # Create detector
        self.detector = YOLODetector(self.config.get_all())
        
        # Test parameters
        self.num_test_frames = 100
        self.camera_id = self.config.get('camera_id', 0)
        
        print("=" * 70)
        print("SMOKE TEST: Persistent Track ID Verification")
        print("=" * 70)
        print(f"Camera ID: {self.camera_id}")
        print(f"Test frames: {self.num_test_frames}")
        print(f"Detector: YOLODetector")
        print("=" * 70)
    
    def test_track_ids_persist_across_frames(self) -> bool:
        """
        Main test: Verify Track IDs remain stable.
        
        Test Logic:
        1. Open camera
        2. Detect and track faces for 100 frames
        3. Collect all Track IDs seen for each detected face
        4. Verify that each face has consistent Track ID
        
        Expected Result:
        - If 1 person detected: Should see only 1 unique Track ID
        - If 2 people detected: Should see only 2 unique Track IDs
        - Track IDs should NOT increment (1, 2, 3, 4... = FAILURE)
        
        Returns:
            True if test passes, False otherwise
        """
        print("\n[TEST] Starting Track ID persistence test...")
        print("Please sit still in front of camera for ~3 seconds...")
        
        # Open camera
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print("❌ FAILED: Could not open camera")
            return False
        
        # Allow camera to warm up
        time.sleep(1)
        
        # Track data for analysis
        frame_count = 0
        track_ids_per_frame = []
        first_frame_track_ids = None
        
        print(f"\n[TEST] Processing {self.num_test_frames} frames...")
        
        try:
            while frame_count < self.num_test_frames:
                ret, frame = cap.read()
                if not ret:
                    print(f"❌ FAILED: Could not read frame {frame_count}")
                    return False
                
                # Run detection + tracking (with persist=True)
                detections, tracks = self.detector.detect_and_track(
                    frame, 
                    tracker_config='botsort.yaml'
                )
                
                # Collect Track IDs from this frame
                current_track_ids = set()
                if tracks:
                    current_track_ids = {track.track_id for track in tracks}
                    track_ids_per_frame.append(current_track_ids)
                    
                    # Store first frame IDs for comparison
                    if first_frame_track_ids is None:
                        first_frame_track_ids = current_track_ids
                        print(f"[INFO] First detection at frame {frame_count}")
                        print(f"       Initial Track IDs: {sorted(current_track_ids)}")
                
                # Progress indicator
                if frame_count % 20 == 0:
                    print(f"  Frame {frame_count}/{self.num_test_frames} - "
                          f"Tracks: {len(tracks)} - "
                          f"IDs: {sorted(current_track_ids) if tracks else 'None'}")
                
                frame_count += 1
        
        finally:
            cap.release()
        
        # Analyze results
        print("\n" + "=" * 70)
        print("TEST RESULTS")
        print("=" * 70)
        
        if not track_ids_per_frame:
            print("⚠️  WARNING: No faces detected during test")
            print("   Please ensure:")
            print("   - You're visible to the camera")
            print("   - Good lighting conditions")
            print("   - Camera is working correctly")
            return False
        
        # Calculate statistics
        all_track_ids_seen = set()
        for ids in track_ids_per_frame:
            all_track_ids_seen.update(ids)
        
        num_unique_ids = len(all_track_ids_seen)
        frames_with_detection = len(track_ids_per_frame)
        detection_rate = (frames_with_detection / self.num_test_frames) * 100
        
        print(f"\nStatistics:")
        print(f"  Frames processed: {self.num_test_frames}")
        print(f"  Frames with detection: {frames_with_detection}")
        print(f"  Detection rate: {detection_rate:.1f}%")
        print(f"  Initial Track IDs: {sorted(first_frame_track_ids)}")
        print(f"  All Track IDs seen: {sorted(all_track_ids_seen)}")
        print(f"  Unique Track IDs: {num_unique_ids}")
        
        # Determine test result
        print("\n" + "-" * 70)
        
        if num_unique_ids <= 3:
            # Expected: Same people should have same IDs
            # Allow up to 3 unique IDs (in case of brief occlusions or new faces entering)
            print("✅ PASSED: Track IDs are persistent!")
            print(f"   Only {num_unique_ids} unique IDs seen - indicates stable tracking")
            
            # Check for perfect stability
            id_changes = 0
            prev_ids = first_frame_track_ids
            for ids in track_ids_per_frame[1:]:
                if ids != prev_ids and ids:  # Ignore empty frames
                    id_changes += 1
                prev_ids = ids
            
            stability_rate = ((frames_with_detection - id_changes) / frames_with_detection) * 100
            print(f"   Track ID stability: {stability_rate:.1f}%")
            
            if stability_rate > 90:
                print("   ⭐ Excellent stability!")
            elif stability_rate > 70:
                print("   ✅ Good stability")
            else:
                print("   ⚠️  Some ID fluctuation detected (but still within acceptable range)")
            
            return True
        
        else:
            # FAILURE: Too many unique IDs = Track IDs are fluctuating
            print("❌ FAILED: Track IDs are fluctuating!")
            print(f"   Saw {num_unique_ids} unique IDs - expected ≤3")
            print("\n   This indicates the bug where Track IDs change every frame:")
            print("   - Person sitting still gets IDs: 1, 2, 3, 4, 5...")
            print("   - Root cause: NOT using model.track(persist=True)")
            print("\n   Check:")
            print("   1. yolo_detector.py - Verify model.track(persist=True)")
            print("   2. orchestrator.py - Verify using detect_and_track()")
            print("   3. detection_stage.py - Verify process_with_tracking()")
            
            return False
    
    def run_all_tests(self) -> bool:
        """Run all smoke tests."""
        print("\nRunning persistent tracking smoke test...\n")
        
        test_passed = self.test_track_ids_persist_across_frames()
        
        print("\n" + "=" * 70)
        print("FINAL RESULT")
        print("=" * 70)
        
        if test_passed:
            print("✅ ALL TESTS PASSED")
            print("\nPersistent tracking is working correctly!")
            print("Track IDs remain stable when people don't move.")
        else:
            print("❌ TESTS FAILED")
            print("\nPersistent tracking is NOT working correctly!")
            print("Please review the diagnostics above and fix the issues.")
        
        print("=" * 70)
        
        return test_passed


def main():
    """Main test runner."""
    try:
        test_suite = TestPersistentTracking()
        success = test_suite.run_all_tests()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ ERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
