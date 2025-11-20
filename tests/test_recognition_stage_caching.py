#!/usr/bin/env python3
"""
Test script for RecognitionStage Quality-Aware Caching Integration.

NOTE: This is a SIMPLIFIED integration test that verifies:
1. Cache initialization from config  
2. Cache statistics tracking works
3. Config integration is correct
4. Direct cache API integration

It does NOT test end-to-end face processing (alignment + recognition) 
because that requires real face images, not synthetic data.

For comprehensive cache behavior testing, see: tests/test_quality_cache.py

Author: AI Assistant
Date: November 8, 2025
Phase: 3A Week 2 (Quality-Aware Caching)
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.recognition_stage import RecognitionStage
from common.config_manager import ConfigManager


def print_section(title):
    """Print section header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def test_cache_initialization():
    """Test 1: Verify cache initialization from config."""
    print_section("TEST 1: Cache Initialization from Config")
    
    # Load config
    config_manager = ConfigManager()
    config_manager.load('config.yaml')
    
    # Create recognition stage
    recognition_stage = RecognitionStage(config_manager)
    
    # Verify cache is enabled
    assert recognition_stage.cache is not None, "❌ Cache should be initialized!"
    assert recognition_stage.enable_cache is True, "❌ Cache should be enabled!"
    
    # Verify cache parameters from config
    assert recognition_stage.cache.ttl_seconds == 300, "❌ TTL should be 300s"
    assert recognition_stage.cache.sample_size == 10, "❌ Sample size should be 10"
    assert recognition_stage.cache.upgrade_threshold == 0.1, "❌ Upgrade threshold should be 0.1"
    
    print("✅ Cache initialized correctly from config:")
    print(f"   TTL: {recognition_stage.cache.ttl_seconds}s")
    print(f"   Sample size: {recognition_stage.cache.sample_size}")
    print(f"   Upgrade threshold: {recognition_stage.cache.upgrade_threshold}")
    print(f"   Max size: {recognition_stage.cache.max_size}")
    
    return recognition_stage


def test_direct_cache_api(recognition_stage):
    """Test 2: Verify cache API works directly."""
    print_section("TEST 2: Direct Cache API Integration")
    
    cache = recognition_stage.cache
    
    # Test 1: Cache miss
    print("\n📊 Test cache MISS:")
    result = cache.get(track_id=999)
    assert result is None, "❌ Should return None for non-existent track_id!"
    print(f"   ✅ cache.get(999) = None (expected)")
    
    # Test 2: Cache put
    print("\n📊 Test cache PUT:")
    embedding = np.random.rand(512).astype(np.float32)
    accepted = cache.put(track_id=999, embedding=embedding, quality=0.7)
    assert accepted is True, "❌ First sample should be accepted!"
    print(f"   ✅ cache.put(999, embedding, 0.7) = {accepted}")
    
    # Test 3: Cache hit
    print("\n📊 Test cache HIT:")
    cached_embedding = cache.get(track_id=999)
    assert cached_embedding is not None, "❌ Should return cached embedding!"
    assert np.allclose(cached_embedding, embedding), "❌ Cached embedding doesn't match!"
    print(f"   ✅ cache.get(999) = embedding (hit)")
    
    # Test 4: Quality upgrade
    print("\n📊 Test quality upgrade:")
    better_embedding = np.random.rand(512).astype(np.float32)
    accepted = cache.put(track_id=999, embedding=better_embedding, quality=0.85)  # +0.15 > 0.1 threshold
    assert accepted is True, "❌ Better quality should be accepted!"
    print(f"   ✅ cache.put(999, better_embedding, 0.85) = {accepted} (upgraded)")
    
    # Verify the better embedding is now cached
    cached_embedding = cache.get(track_id=999)
    assert np.allclose(cached_embedding, better_embedding), "❌ Should have upgraded to better embedding!"
    print(f"   ✅ Cached embedding upgraded to better quality")
    
    # Test 5: Cache statistics
    print("\n📊 Test cache statistics:")
    stats = cache.get_stats()
    print(f"   Hits: {stats['hits']}")
    print(f"   Misses: {stats['misses']}")
    print(f"   Upgrades: {stats['upgrades']}")
    print(f"   Hit rate: {stats['hit_rate']:.1%}")
    print(f"   Cache size: {stats['size']}")
    
    assert stats['hits'] == 2, "❌ Should have 2 cache hits!"
    assert stats['misses'] == 1, "❌ Should have 1 cache miss!"
    assert stats['upgrades'] == 1, "❌ Should have 1 upgrade!"
    assert stats['size'] == 1, "❌ Should have 1 cached track!"
    
    print("\n✅ Direct cache API working correctly!")


def test_statistics_integration(recognition_stage):
    """Test 3: Verify statistics integration."""
    print_section("TEST 3: Statistics Integration")
    
    # Reset statistics
    recognition_stage.reset_stats()
    
    # Verify statistics structure
    stats = recognition_stage.get_stats()
    
    print("\n📊 Testing statistics structure:")
    required_fields = ['total_processed', 'alignments_success', 'alignments_failed',
                      'recognitions_success', 'recognitions_failed', 'quality_too_low',
                      'cache_hits', 'cache_misses', 'cache_hit_rate', 'cache_total_hits',
                      'cache_total_misses', 'cache_upgrades', 'cache_evictions', 'cache_size']
    
    for field in required_fields:
        assert field in stats, f"❌ Missing statistics field: {field}"
        print(f"   ✅ {field}: {stats[field]}")
    
    print("\n✅ Statistics integration working correctly!")


def main():
    """Run all tests."""
    print("="*60)
    print("  RecognitionStage Quality-Aware Caching Integration Test")
    print("="*60)
    print("\nPhase 3A Week 2: Testing cache integration with config")
    print("NOTE: For comprehensive cache behavior, see test_quality_cache.py\n")
    
    try:
        # Test 1: Cache initialization from config
        recognition_stage = test_cache_initialization()
        
        # Test 2: Direct cache API integration
        test_direct_cache_api(recognition_stage)
        
        # Test 3: Statistics integration
        test_statistics_integration(recognition_stage)
        
        # All tests passed
        print_section("ALL TESTS PASSED ✅")
        print("\n✨ Quality-Aware Caching integration is working correctly!")
        print("   ✅ Config integration verified")
        print("   ✅ Cache API accessible from RecognitionStage")
        print("   ✅ Statistics tracking integrated")
        print("\nNext Steps:")
        print("   1. Run test_quality_cache.py for comprehensive cache behavior")
        print("   2. Test end-to-end with real face images")
        print("   3. Benchmark performance on Raspberry Pi")
        
        return 0
    
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
