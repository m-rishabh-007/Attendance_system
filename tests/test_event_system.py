#!/usr/bin/env python3
"""
Test Observer Pattern: EventSystem

Verifies that:
1. Event subscription works correctly
2. Event publishing triggers callbacks
3. Multiple subscribers receive events
4. Unsubscribe works correctly
5. Error handling in callbacks doesn't crash system
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.event_system import EventSystem, EventType, Event


def test_basic_subscription():
    """Test basic subscribe and publish."""
    print("\n=== Test 1: Basic Subscription ===")
    
    events = EventSystem()
    received_events = []
    
    # Define callback
    def callback(event: Event):
        received_events.append(event)
    
    # Subscribe
    events.subscribe(EventType.FACE_DETECTED, callback)
    print("✅ Subscribed to FACE_DETECTED")
    
    # Publish event
    events.publish(
        EventType.FACE_DETECTED,
        {'track_id': 1, 'confidence': 0.95},
        source='test'
    )
    
    # Verify
    assert len(received_events) == 1, f"Expected 1 event, got {len(received_events)}"
    event = received_events[0]
    assert event.event_type == EventType.FACE_DETECTED
    assert event.data['track_id'] == 1
    assert event.source == 'test'
    print(f"✅ Event received: {event}")


def test_multiple_subscribers():
    """Test that multiple subscribers receive the same event."""
    print("\n=== Test 2: Multiple Subscribers ===")
    
    events = EventSystem()
    received_1 = []
    received_2 = []
    received_3 = []
    
    # Create three subscribers
    events.subscribe(EventType.FACE_LOST, lambda e: received_1.append(e))
    events.subscribe(EventType.FACE_LOST, lambda e: received_2.append(e))
    events.subscribe(EventType.FACE_LOST, lambda e: received_3.append(e))
    print("✅ Three subscribers registered")
    
    # Publish one event
    events.publish(EventType.FACE_LOST, {'track_id': 5}, source='test')
    
    # All should receive it
    assert len(received_1) == 1
    assert len(received_2) == 1
    assert len(received_3) == 1
    print("✅ All three subscribers received the event")


def test_multiple_event_types():
    """Test subscribing to different event types."""
    print("\n=== Test 3: Multiple Event Types ===")
    
    events = EventSystem()
    face_detected = []
    face_lost = []
    pipeline_started = []
    
    # Subscribe to different event types
    events.subscribe(EventType.FACE_DETECTED, lambda e: face_detected.append(e))
    events.subscribe(EventType.FACE_LOST, lambda e: face_lost.append(e))
    events.subscribe(EventType.PIPELINE_STARTED, lambda e: pipeline_started.append(e))
    print("✅ Subscribed to 3 different event types")
    
    # Publish different events
    events.publish(EventType.FACE_DETECTED, {}, 'test')
    events.publish(EventType.FACE_LOST, {}, 'test')
    events.publish(EventType.PIPELINE_STARTED, {}, 'test')
    events.publish(EventType.FACE_DETECTED, {}, 'test')  # Second face detected
    
    # Verify correct counts
    assert len(face_detected) == 2, f"Expected 2 FACE_DETECTED, got {len(face_detected)}"
    assert len(face_lost) == 1, f"Expected 1 FACE_LOST, got {len(face_lost)}"
    assert len(pipeline_started) == 1, f"Expected 1 PIPELINE_STARTED, got {len(pipeline_started)}"
    print("✅ Each subscriber received only its event type")


def test_unsubscribe():
    """Test unsubscribe functionality."""
    print("\n=== Test 4: Unsubscribe ===")
    
    events = EventSystem()
    received = []
    
    def callback(event: Event):
        received.append(event)
    
    # Subscribe
    events.subscribe(EventType.FACE_DETECTED, callback)
    
    # Publish first event
    events.publish(EventType.FACE_DETECTED, {}, 'test')
    assert len(received) == 1
    print("✅ Event received before unsubscribe")
    
    # Unsubscribe
    events.unsubscribe(EventType.FACE_DETECTED, callback)
    print("✅ Unsubscribed")
    
    # Publish second event
    events.publish(EventType.FACE_DETECTED, {}, 'test')
    assert len(received) == 1, "Should still be 1 after unsubscribe"
    print("✅ No event received after unsubscribe")


def test_callback_error_handling():
    """Test that errors in callbacks don't crash the system."""
    print("\n=== Test 5: Callback Error Handling ===")
    
    events = EventSystem()
    good_callback_count = [0]
    
    def bad_callback(event: Event):
        raise ValueError("Intentional error in callback")
    
    def good_callback(event: Event):
        good_callback_count[0] += 1
    
    # Subscribe both callbacks
    events.subscribe(EventType.FACE_DETECTED, bad_callback)
    events.subscribe(EventType.FACE_DETECTED, good_callback)
    print("✅ Subscribed bad and good callbacks")
    
    # Publish event
    events.publish(EventType.FACE_DETECTED, {}, 'test')
    
    # Good callback should still execute
    assert good_callback_count[0] == 1, "Good callback should have executed"
    print("✅ Good callback executed despite bad callback error")


def test_event_history():
    """Test event history tracking."""
    print("\n=== Test 6: Event History ===")
    
    events = EventSystem()
    
    # Publish some events
    events.publish(EventType.FACE_DETECTED, {'id': 1}, 'test')
    events.publish(EventType.FACE_LOST, {'id': 1}, 'test')
    events.publish(EventType.FACE_DETECTED, {'id': 2}, 'test')
    
    # Check history
    history = events.get_event_history(limit=10)
    assert len(history) == 3, f"Expected 3 events in history, got {len(history)}"
    print(f"✅ Event history contains {len(history)} events")
    
    # Verify order
    assert history[0].event_type == EventType.FACE_DETECTED
    assert history[1].event_type == EventType.FACE_LOST
    assert history[2].event_type == EventType.FACE_DETECTED
    print("✅ Event history preserves order")
    
    # Test limit
    limited = events.get_event_history(limit=2)
    assert len(limited) == 2
    print("✅ History limit works")
    
    # Test clear history
    events.clear_history()
    history_after_clear = events.get_event_history(limit=10)
    assert len(history_after_clear) == 0
    print("✅ Clear history works")


def test_event_data_structure():
    """Test that Event dataclass works correctly."""
    print("\n=== Test 7: Event Data Structure ===")
    
    events = EventSystem()
    received = []
    
    events.subscribe(EventType.FACE_RECOGNIZED, lambda e: received.append(e))
    
    # Publish complex event data
    complex_data = {
        'track_id': 123,
        'person_name': 'John Doe',
        'confidence': 0.95,
        'bbox': [100, 200, 50, 50],
        'metadata': {
            'age': 25,
            'timestamp': '2025-10-31T10:00:00'
        }
    }
    
    events.publish(EventType.FACE_RECOGNIZED, complex_data, source='recognizer')
    
    # Verify event structure
    event = received[0]
    assert event.event_type == EventType.FACE_RECOGNIZED
    assert event.source == 'recognizer'
    assert event.data == complex_data
    assert isinstance(event.timestamp, datetime)
    print("✅ Event dataclass preserves all data correctly")
    print(f"   Event: {event}")


def run_all_tests():
    """Run all EventSystem tests."""
    print("\n" + "="*60)
    print("Testing EventSystem (Observer Pattern)")
    print("="*60)
    
    tests = [
        test_basic_subscription,
        test_multiple_subscribers,
        test_multiple_event_types,
        test_unsubscribe,
        test_callback_error_handling,
        test_event_history,
        test_event_data_structure
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n❌ Test failed: {test.__name__}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
