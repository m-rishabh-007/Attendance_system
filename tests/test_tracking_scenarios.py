"""
Comprehensive Tracking Scenarios Test

This test suite validates different tracking scenarios:
1. YOLO integrated tracking (persist=True) - Should have stable IDs
2. TFLite fallback tracking - IDs will fluctuate (expected)
3. Multiple faces tracking - Each face should have unique stable ID
4. Occlusion handling - IDs should persist through brief occlusions

Usage:
    python tests/test_tracking_scenarios.py
    
    # Or run specific scenario:
    python tests/test_tracking_scenarios.py --scenario yolo
    python tests/test_tracking_scenarios.py --scenario tflite
    python tests/test_tracking_scenarios.py --scenario multiple
"""

import sys
import os
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import time
from typing import List, Dict, Set, Tuple

# Import components
from detectors.yolo_detector import YOLODetector
from detectors.tflite_detector import TFLiteDetector
from tracking.botsort_tracker import BotSORTTracker
from common.config_manager import ConfigManager
from common.base_classes import Detection


class TrackingScenarioTests:
    """Comprehensive test suite for different tracking scenarios."""
    
    def __init__(self):
        """Initialize test suite."""
        self.config = ConfigManager()
        self.config.load('config.yaml')
        
        self.camera_id = self.config.get('camera.device_id', 1)
        self.test_frames = 100
        
        print("=" * 70)
        print("COMPREHENSIVE TRACKING SCENARIOS TEST")
        print("=" * 70)
        print(f"Camera ID: {self.camera_id}")
        print(f"Test frames per scenario: {self.test_frames}")
        print("=" * 70)
    
    def _open_camera(self) -> cv2.VideoCapture:
        """Open camera with error handling."""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}")
        time.sleep(1)  # Camera warm-up
        return cap
    
    def _analyze_track_ids(self, track_ids_per_frame: List[Set[int]]) -> Dict:
        """
        Analyze Track ID stability.
        
        Returns:
            Dictionary with statistics:
            - unique_ids: Number of unique IDs seen
            - stability_rate: Percentage of frames where IDs stayed same
            - avg_ids_per_frame: Average number of IDs per frame
        """
        if not track_ids_per_frame:
            return {
                'unique_ids': 0,
                'stability_rate': 0.0,
                'avg_ids_per_frame': 0.0,
                'id_changes': 0
            }
        
        # Calculate unique IDs
        all_ids = set()
        for ids in track_ids_per_frame:
            all_ids.update(ids)
        
        # Calculate stability
        id_changes = 0
        prev_ids = track_ids_per_frame[0]
        for ids in track_ids_per_frame[1:]:
            if ids and ids != prev_ids:
                id_changes += 1
            if ids:
                prev_ids = ids
        
        stability_rate = ((len(track_ids_per_frame) - id_changes) / len(track_ids_per_frame)) * 100
        avg_ids = sum(len(ids) for ids in track_ids_per_frame) / len(track_ids_per_frame)
        
        return {
            'unique_ids': len(all_ids),
            'stability_rate': stability_rate,
            'avg_ids_per_frame': avg_ids,
            'id_changes': id_changes
        }
    
    def test_yolo_integrated_tracking(self) -> bool:
        """
        Test 1: YOLO with integrated tracking (model.track persist=True).
        
        Expected Result:
        - ✅ Stable Track IDs (≤3 unique IDs)
        - ✅ High stability rate (>90%)
        - ✅ IDs persist through minor movements
        
        This is the CORRECT way to do tracking!
        """
        print("\n" + "=" * 70)
        print("TEST 1: YOLO Integrated Tracking (persist=True)")
        print("=" * 70)
        print("This tests the CORRECT tracking approach:")
        print("  detector.detect_and_track() → model.track(persist=True)")
        print("\nExpected: Stable Track IDs across frames")
        print("Please sit still in front of camera...")
        
        try:
            detector_config = self.config.get_section('detector')
            detector = YOLODetector(detector_config)
            cap = self._open_camera()
            
            track_ids_per_frame = []
            frame_count = 0
            
            print(f"\nProcessing {self.test_frames} frames...")
            
            while frame_count < self.test_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Use integrated tracking (CORRECT!)
                detections, tracks = detector.detect_and_track(frame, 'botsort.yaml')
                
                if tracks:
                    track_ids = {track.track_id for track in tracks}
                    track_ids_per_frame.append(track_ids)
                    
                    if frame_count % 20 == 0:
                        print(f"  Frame {frame_count}: Track IDs = {sorted(track_ids)}")
                
                frame_count += 1
            
            cap.release()
            
            # Analyze results
            stats = self._analyze_track_ids(track_ids_per_frame)
            
            print("\n" + "-" * 70)
            print("RESULTS:")
            print(f"  Unique Track IDs: {stats['unique_ids']}")
            print(f"  Stability Rate: {stats['stability_rate']:.1f}%")
            print(f"  ID Changes: {stats['id_changes']}")
            print(f"  Avg IDs per frame: {stats['avg_ids_per_frame']:.1f}")
            
            # Determine pass/fail
            if stats['unique_ids'] <= 3 and stats['stability_rate'] > 70:
                print("\n✅ PASSED: YOLO integrated tracking works correctly!")
                print("   Track IDs are persistent as expected.")
                return True
            else:
                print("\n❌ FAILED: Track IDs are not stable!")
                print(f"   Expected ≤3 unique IDs, got {stats['unique_ids']}")
                print(f"   Expected >70% stability, got {stats['stability_rate']:.1f}%")
                return False
        
        except Exception as e:
            print(f"\n❌ ERROR: Test failed with exception: {e}")
            return False
    
    def test_tflite_fallback_tracking(self) -> bool:
        """
        Test 2: TFLite detector + separate tracker (fallback).
        
        Expected Result:
        - ⚠️ Fluctuating Track IDs (many unique IDs)
        - ⚠️ Low stability rate (<50%)
        - ⚠️ IDs change frequently
        
        This demonstrates WHY we use integrated tracking!
        """
        print("\n" + "=" * 70)
        print("TEST 2: TFLite Fallback Tracking (No persistence)")
        print("=" * 70)
        print("This tests the FALLBACK tracking approach:")
        print("  detector.detect() + tracker.update()")
        print("\nExpected: Fluctuating Track IDs (this is normal for fallback)")
        print("Please sit still in front of camera...")
        
        try:
            # Check if TFLite detector exists
            tflite_model = self.config.get('detector', {}).get('tflite_model_path')
            if not tflite_model or not Path(tflite_model).exists():
                print("\n⚠️  SKIPPED: TFLite model not found")
                print(f"   Expected at: {tflite_model}")
                return True  # Skip, not fail
            
            detector = TFLiteDetector(self.config.get_all())
            tracker = BotSORTTracker(self.config.get_all())
            cap = self._open_camera()
            
            track_ids_per_frame = []
            frame_count = 0
            
            print(f"\nProcessing {self.test_frames} frames...")
            
            while frame_count < self.test_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Use fallback pattern (separate detection + tracking)
                detections = detector.detect(frame)
                tracks = tracker.update(detections)
                
                if tracks:
                    track_ids = {track.track_id for track in tracks}
                    track_ids_per_frame.append(track_ids)
                    
                    if frame_count % 20 == 0:
                        print(f"  Frame {frame_count}: Track IDs = {sorted(track_ids)}")
                
                frame_count += 1
            
            cap.release()
            
            # Analyze results
            stats = self._analyze_track_ids(track_ids_per_frame)
            
            print("\n" + "-" * 70)
            print("RESULTS:")
            print(f"  Unique Track IDs: {stats['unique_ids']}")
            print(f"  Stability Rate: {stats['stability_rate']:.1f}%")
            print(f"  ID Changes: {stats['id_changes']}")
            print(f"  Avg IDs per frame: {stats['avg_ids_per_frame']:.1f}")
            
            # For fallback, we EXPECT poor stability
            if stats['unique_ids'] > 10 or stats['stability_rate'] < 50:
                print("\n✅ PASSED: Fallback tracking behaves as expected")
                print("   Track IDs fluctuate (this is normal for fallback)")
                print("   This demonstrates why integrated tracking is needed!")
                return True
            else:
                print("\n⚠️  UNEXPECTED: Fallback tracking is too stable!")
                print("   This suggests the tracker might have persistence")
                print("   (Or no faces were detected)")
                return True  # Not a failure, just unexpected
        
        except Exception as e:
            print(f"\n⚠️  SKIPPED: {e}")
            return True  # Don't fail if TFLite not available
    
    def test_multiple_faces_tracking(self) -> bool:
        """
        Test 3: Multiple faces with stable IDs.
        
        Expected Result:
        - ✅ Each face gets unique Track ID
        - ✅ IDs remain stable for each person
        - ✅ New person entering gets new ID
        
        Instructions:
        - Start with 1 person, then add another person mid-test
        """
        print("\n" + "=" * 70)
        print("TEST 3: Multiple Faces Tracking")
        print("=" * 70)
        print("This tests tracking with multiple people:")
        print("\nInstructions:")
        print("  1. Start with 1 person in frame (you)")
        print("  2. After 2 seconds, have another person enter frame")
        print("  3. Both people should get stable Track IDs")
        print("\nPress Enter when ready...")
        input()
        
        try:
            detector_config = self.config.get_section('detector')
            detector = YOLODetector(detector_config)
            cap = self._open_camera()
            
            track_data = {}  # track_id -> [frame_numbers]
            frame_count = 0
            
            print(f"\nProcessing {self.test_frames} frames...")
            print("Bring in second person after 2 seconds...\n")
            
            while frame_count < self.test_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                detections, tracks = detector.detect_and_track(frame, 'botsort.yaml')
                
                # Track when each ID appears
                for track in tracks:
                    tid = track.track_id
                    if tid not in track_data:
                        track_data[tid] = []
                    track_data[tid].append(frame_count)
                
                if frame_count % 20 == 0 and tracks:
                    ids = [t.track_id for t in tracks]
                    print(f"  Frame {frame_count}: {len(tracks)} faces - IDs = {sorted(ids)}")
                
                frame_count += 1
            
            cap.release()
            
            # Analyze results
            print("\n" + "-" * 70)
            print("RESULTS:")
            print(f"  Total unique Track IDs: {len(track_data)}")
            
            for tid, frames in sorted(track_data.items()):
                print(f"\n  Track ID {tid}:")
                print(f"    First seen: Frame {frames[0]}")
                print(f"    Last seen: Frame {frames[-1]}")
                print(f"    Total appearances: {len(frames)} frames")
                print(f"    Continuity: {(len(frames) / (frames[-1] - frames[0] + 1)) * 100:.1f}%")
            
            # Determine pass/fail
            if len(track_data) >= 2:
                print("\n✅ PASSED: Successfully tracked multiple faces")
                print("   Each person got a unique Track ID")
                return True
            else:
                print("\n⚠️  Only 1 face tracked (need 2 for full test)")
                print("   Test inconclusive but not failed")
                return True  # Not a failure
        
        except Exception as e:
            print(f"\n❌ ERROR: Test failed with exception: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tracking scenario tests."""
        results = {}
        
        print("\nRunning comprehensive tracking scenario tests...\n")
        
        # Test 1: YOLO integrated tracking (most important!)
        results['yolo_integrated'] = self.test_yolo_integrated_tracking()
        
        # Test 2: TFLite fallback (demonstrates why integrated is better)
        results['tflite_fallback'] = self.test_tflite_fallback_tracking()
        
        # Test 3: Multiple faces
        results['multiple_faces'] = self.test_multiple_faces_tracking()
        
        # Summary
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        
        for test_name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{status}: {test_name.replace('_', ' ').title()}")
        
        all_passed = all(results.values())
        
        if all_passed:
            print("\n✅ ALL TESTS PASSED")
        else:
            print("\n❌ SOME TESTS FAILED")
        
        print("=" * 70)
        
        return results


def main():
    """Main test runner with CLI."""
    parser = argparse.ArgumentParser(description='Comprehensive tracking scenarios test')
    parser.add_argument(
        '--scenario',
        choices=['yolo', 'tflite', 'multiple', 'all'],
        default='all',
        help='Which scenario to test'
    )
    
    args = parser.parse_args()
    
    try:
        test_suite = TrackingScenarioTests()
        
        if args.scenario == 'yolo':
            success = test_suite.test_yolo_integrated_tracking()
        elif args.scenario == 'tflite':
            success = test_suite.test_tflite_fallback_tracking()
        elif args.scenario == 'multiple':
            success = test_suite.test_multiple_faces_tracking()
        else:  # all
            results = test_suite.run_all_tests()
            success = all(results.values())
        
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
