#!/usr/bin/env python3
"""
Master Test Runner for Face Attendance System v2.0

Runs all test suites for design patterns and components:
- Singleton Pattern (ConfigManager)
- Observer Pattern (EventSystem)
- Factory Pattern (DetectorFactory, TrackerFactory)
- Strategy Pattern (Tracker implementations)

Usage:
    python tests/run_all_tests.py
    python tests/run_all_tests.py --verbose
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import test modules
from tests import test_config_manager, test_event_system, test_factories


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description='Run all tests for Face Attendance System v2.0')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    print_header("Face Attendance System v2.0 - Test Suite")
    print("Testing all design patterns and components...")
    print()
    
    results = {}
    
    # Test 1: Singleton Pattern
    print_header("1/3: Singleton Pattern (ConfigManager)")
    try:
        results['singleton'] = test_config_manager.run_all_tests()
    except Exception as e:
        print(f"❌ Singleton tests crashed: {e}")
        results['singleton'] = False
    
    # Test 2: Observer Pattern
    print_header("2/3: Observer Pattern (EventSystem)")
    try:
        results['observer'] = test_event_system.run_all_tests()
    except Exception as e:
        print(f"❌ Observer tests crashed: {e}")
        results['observer'] = False
    
    # Test 3: Factory + Strategy Patterns
    print_header("3/3: Factory & Strategy Patterns (Detectors & Trackers)")
    try:
        results['factory'] = test_factories.run_all_tests()
    except Exception as e:
        print(f"❌ Factory tests crashed: {e}")
        results['factory'] = False
    
    # Final Summary
    print_header("FINAL RESULTS")
    
    pattern_names = {
        'singleton': 'Singleton Pattern (ConfigManager)',
        'observer': 'Observer Pattern (EventSystem)',
        'factory': 'Factory + Strategy Patterns (Detectors & Trackers)'
    }
    
    all_passed = True
    for pattern, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}  {pattern_names[pattern]}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("All design patterns are correctly implemented.")
        print("="*70 + "\n")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("Please review the errors above.")
        print("="*70 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
