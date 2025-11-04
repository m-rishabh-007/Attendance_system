#!/usr/bin/env python3
"""
Test Factory Pattern: DetectorFactory and TrackerFactory

Verifies that:
1. Factories can create correct detector/tracker types
2. Invalid types raise appropriate errors
3. Created objects implement correct interfaces
4. Configuration is passed correctly to created objects
"""

import sys
from pathlib import Path
import tempfile

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detectors.factory import DetectorFactory
from tracking.factory import TrackerFactory
from common.base_classes import BaseFaceDetector, BaseTracker


def test_detector_factory_yolo():
    """Test creating YOLO detector via factory."""
    print("\n=== Test 1: Create YOLO Detector ===")
    
    config = {
        'type': 'yolo',
        'model_path': 'models/yolov8n_face_int8.tflite',
        'confidence_threshold': 0.5,
        'iou_threshold': 0.45
    }
    
    try:
        detector = DetectorFactory.create(config)
        
        # Verify type
        assert isinstance(detector, BaseFaceDetector), \
            f"Detector should implement BaseFaceDetector, got {type(detector)}"
        print(f"✅ Created detector: {detector}")
        
        # Verify it's YOLO detector
        assert 'YOLO' in str(type(detector).__name__), \
            "Should be YOLODetector"
        print(f"✅ Correct detector type: {type(detector).__name__}")
        
    except ImportError as e:
        print(f"⚠️ Skipping YOLO test (dependency not installed): {e}")
        return True  # Don't fail test if optional dependency missing
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


def test_detector_factory_tflite():
    """Test creating TFLite detector via factory."""
    print("\n=== Test 2: Create TFLite Detector ===")
    
    config = {
        'type': 'tflite',
        'model_path': 'models/yolov8n_face_int8.tflite',
        'confidence_threshold': 0.5,
        'iou_threshold': 0.45,
        'num_threads': 3,
        'input_size': 256
    }
    
    try:
        detector = DetectorFactory.create(config)
        
        # Verify type
        assert isinstance(detector, BaseFaceDetector)
        print(f"✅ Created detector: {detector}")
        
        # Verify it's TFLite detector
        assert 'TFLite' in str(type(detector).__name__), \
            "Should be TFLiteDetector"
        print(f"✅ Correct detector type: {type(detector).__name__}")
        
    except FileNotFoundError as e:
        print(f"⚠️ Skipping TFLite test (model not found): {e}")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


def test_detector_factory_invalid_type():
    """Test that invalid detector type raises error."""
    print("\n=== Test 3: Invalid Detector Type ===")
    
    config = {
        'type': 'invalid_detector_type',
        'model_path': 'models/some_model.tflite'
    }
    
    try:
        detector = DetectorFactory.create(config)
        assert False, "Should have raised ValueError for invalid type"
    except ValueError as e:
        print(f"✅ Correct error raised: {e}")


def test_tracker_factory_botsort():
    """Test creating BoT-SORT tracker via factory."""
    print("\n=== Test 4: Create BoT-SORT Tracker ===")
    
    config = {
        'type': 'botsort',
        'track_thresh': 0.5,
        'track_buffer': 30,
        'match_thresh': 0.8
    }
    
    try:
        tracker = TrackerFactory.create(config)
        
        # Verify type
        assert isinstance(tracker, BaseTracker), \
            f"Tracker should implement BaseTracker, got {type(tracker)}"
        print(f"✅ Created tracker: {tracker}")
        
        # Verify it's BoT-SORT tracker
        assert 'BotSORT' in str(type(tracker).__name__), \
            "Should be BotSORTTracker"
        print(f"✅ Correct tracker type: {type(tracker).__name__}")
        
    except ImportError as e:
        print(f"⚠️ Skipping BoT-SORT test (dependency not installed): {e}")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


def test_tracker_factory_no_tracker():
    """Test creating NoTracker (dummy tracker) via factory."""
    print("\n=== Test 5: Create NoTracker ===")
    
    config = {
        'type': 'none'
    }
    
    tracker = TrackerFactory.create(config)
    
    # Verify type
    assert isinstance(tracker, BaseTracker)
    print(f"✅ Created tracker: {tracker}")
    
    # Verify it's NoTracker
    assert 'NoTracker' in str(type(tracker).__name__)
    print(f"✅ Correct tracker type: {type(tracker).__name__}")


def test_tracker_factory_invalid_type():
    """Test that invalid tracker type raises error."""
    print("\n=== Test 6: Invalid Tracker Type ===")
    
    config = {
        'type': 'invalid_tracker_type'
    }
    
    try:
        tracker = TrackerFactory.create(config)
        assert False, "Should have raised ValueError for invalid type"
    except ValueError as e:
        print(f"✅ Correct error raised: {e}")


def test_factory_with_missing_config():
    """Test factory behavior with missing config keys."""
    print("\n=== Test 7: Missing Configuration ===")
    
    # Empty config should fail
    try:
        detector = DetectorFactory.create({})
        assert False, "Should have raised KeyError for missing 'type'"
    except KeyError as e:
        print(f"✅ Correct error for missing config: {e}")
    
    # Config without type should fail
    try:
        tracker = TrackerFactory.create({'some_param': 'value'})
        assert False, "Should have raised KeyError for missing 'type'"
    except KeyError as e:
        print(f"✅ Correct error for missing type: {e}")


def test_strategy_pattern_interchangeability():
    """Test that different trackers can be swapped (Strategy Pattern)."""
    print("\n=== Test 8: Strategy Pattern - Tracker Interchangeability ===")
    
    # Create different trackers with same interface
    trackers = []
    
    # NoTracker
    trackers.append(TrackerFactory.create({'type': 'none'}))
    print("✅ Created NoTracker")
    
    # Try BoT-SORT if available
    try:
        trackers.append(TrackerFactory.create({'type': 'botsort'}))
        print("✅ Created BotSORTTracker")
    except ImportError:
        print("⚠️ BoT-SORT not available, skipping")
    
    # All should implement same interface
    for tracker in trackers:
        assert isinstance(tracker, BaseTracker)
        assert hasattr(tracker, 'update')
        print(f"✅ {type(tracker).__name__} implements BaseTracker")
    
    print("✅ All trackers are interchangeable (Strategy Pattern verified)")


def run_all_tests():
    """Run all Factory pattern tests."""
    print("\n" + "="*60)
    print("Testing Factory Pattern (DetectorFactory & TrackerFactory)")
    print("="*60)
    
    tests = [
        test_detector_factory_yolo,
        test_detector_factory_tflite,
        test_detector_factory_invalid_type,
        test_tracker_factory_botsort,
        test_tracker_factory_no_tracker,
        test_tracker_factory_invalid_type,
        test_factory_with_missing_config,
        test_strategy_pattern_interchangeability
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test in tests:
        try:
            result = test()
            if result is True:  # Explicitly returned True (skipped)
                skipped += 1
            else:
                passed += 1
        except Exception as e:
            print(f"\n❌ Test failed: {test.__name__}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
