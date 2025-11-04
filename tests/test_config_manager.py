#!/usr/bin/env python3
"""
Test Singleton Pattern: ConfigManager

Verifies that:
1. Only one instance is created (singleton behavior)
2. Configuration loading works correctly
3. get() and get_section() methods work
4. Reload functionality works
"""

import sys
from pathlib import Path
import tempfile

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.config_manager import ConfigManager


def test_singleton_behavior():
    """Test that ConfigManager is truly a singleton."""
    print("\n=== Test 1: Singleton Behavior ===")
    
    # Create two instances
    config1 = ConfigManager()
    config2 = ConfigManager()
    
    # They should be the same object
    assert config1 is config2, "ConfigManager should be a singleton"
    print("✅ Singleton behavior verified: Both instances are the same object")
    
    # Test with reset
    ConfigManager._instance = None  # Force reset for testing
    config3 = ConfigManager()
    assert config3 is not config1, "After reset, new instance should be created"
    print("✅ Singleton reset works correctly")


def test_config_loading():
    """Test configuration loading from YAML."""
    print("\n=== Test 2: Configuration Loading ===")
    
    # Create temporary config file
    config_content = """
detector:
  type: yolo
  confidence_threshold: 0.5

tracker:
  type: botsort
  track_thresh: 0.6

camera:
  device_id: 0
  width: 1280
  height: 720
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_path = Path(f.name)
    
    try:
        # Reset singleton
        ConfigManager._instance = None
        config = ConfigManager()
        
        # Load config
        config.load(config_path)
        print(f"✅ Config loaded from {config_path}")
        
        # Test get() method
        detector_type = config.get('detector.type')
        assert detector_type == 'yolo', f"Expected 'yolo', got '{detector_type}'"
        print(f"✅ get('detector.type') = '{detector_type}'")
        
        confidence = config.get('detector.confidence_threshold')
        assert confidence == 0.5, f"Expected 0.5, got {confidence}"
        print(f"✅ get('detector.confidence_threshold') = {confidence}")
        
        # Test get() with default
        missing_value = config.get('nonexistent.key', default='default_value')
        assert missing_value == 'default_value'
        print(f"✅ get() with default works: '{missing_value}'")
        
        # Test get_section()
        camera_section = config.get_section('camera')
        assert camera_section['device_id'] == 0
        assert camera_section['width'] == 1280
        print(f"✅ get_section('camera') = {camera_section}")
        
    finally:
        # Cleanup
        config_path.unlink()


def test_nested_keys():
    """Test deeply nested configuration keys."""
    print("\n=== Test 3: Nested Keys ===")
    
    config_content = """
level1:
  level2:
    level3:
      value: "deep_value"
  another_key: 42
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_path = Path(f.name)
    
    try:
        ConfigManager._instance = None
        config = ConfigManager()
        config.load(config_path)
        
        # Test nested access
        value = config.get('level1.level2.level3.value')
        assert value == "deep_value"
        print(f"✅ Nested access works: {value}")
        
        # Test partial path
        level2 = config.get('level1.level2')
        assert isinstance(level2, dict)
        assert 'level3' in level2
        print(f"✅ Partial path returns dict: {level2}")
        
    finally:
        config_path.unlink()


def test_reload():
    """Test configuration reload functionality."""
    print("\n=== Test 4: Reload Functionality ===")
    
    # Create initial config
    config_content_v1 = "version: 1"
    config_content_v2 = "version: 2"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content_v1)
        config_path = Path(f.name)
    
    try:
        ConfigManager._instance = None
        config = ConfigManager()
        config.load(config_path)
        
        assert config.get('version') == 1
        print("✅ Initial version: 1")
        
        # Modify file
        with open(config_path, 'w') as f:
            f.write(config_content_v2)
        
        # Reload
        config.reload()
        assert config.get('version') == 2
        print("✅ After reload: 2")
        
    finally:
        config_path.unlink()


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\n=== Test 5: Error Handling ===")
    
    ConfigManager._instance = None
    config = ConfigManager()
    
    # Test get() before initialization returns default
    result = config.get('some.key', default='default_value')
    assert result == 'default_value', "Should return default when config not loaded"
    print(f"✅ Returns default when config not loaded: '{result}'")
    
    # Test invalid file
    try:
        config.load(Path('/nonexistent/path.yaml'))
        assert False, "Should raise error for invalid path"
    except FileNotFoundError:
        print("✅ Correct error for invalid path")


def run_all_tests():
    """Run all ConfigManager tests."""
    print("\n" + "="*60)
    print("Testing ConfigManager (Singleton Pattern)")
    print("="*60)
    
    tests = [
        test_singleton_behavior,
        test_config_loading,
        test_nested_keys,
        test_reload,
        test_error_handling
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
