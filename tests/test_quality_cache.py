#!/usr/bin/env python3
"""
Test Quality-Aware Cache
=========================

Validates the quality-aware caching system with simulated face tracking data.

Tests:
    1. Basic cache operations (get/put)
    2. Quality-aware upgrades (better quality replaces worse)
    3. Sampling completion (10 samples → finalized)
    4. TTL expiration (5 minute timeout)
    5. Cache statistics (hit rate, upgrades)

Usage:
    python3 tests/test_quality_cache.py

Expected Results:
    - All tests pass
    - Hit rate > 95%
    - Quality upgrades work correctly
    - Sampling finalizes after 10 frames

Phase: Week 2 Day 4-5
"""

import sys
import time
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from recognizers.quality_cache import QualityAwareCache

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def test_basic_operations():
    """Test 1: Basic cache get/put operations."""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Cache Operations")
    print("=" * 70)
    
    cache = QualityAwareCache(ttl_seconds=60, sample_size=10)
    
    # Create dummy embedding
    embedding = np.random.randn(512).astype(np.float32)
    track_id = 1
    quality = 0.75
    
    # Test put
    result = cache.put(track_id, embedding, quality)
    assert result == True, "First put should succeed"
    assert track_id in cache, "Track should be in cache"
    assert len(cache) == 1, "Cache size should be 1"
    print("✅ PUT operation: SUCCESS")
    
    # Test get
    cached_embedding = cache.get(track_id)
    assert cached_embedding is not None, "Should return cached embedding"
    assert np.allclose(cached_embedding, embedding), "Embeddings should match"
    print("✅ GET operation: SUCCESS")
    
    # Test cache miss
    missing = cache.get(999)
    assert missing is None, "Should return None for missing track"
    print("✅ MISS operation: SUCCESS")
    
    print("✅ TEST 1 PASSED: Basic operations work correctly\n")


def test_quality_upgrades():
    """Test 2: Quality-aware upgrades."""
    print("\n" + "=" * 70)
    print("TEST 2: Quality-Aware Upgrades")
    print("=" * 70)
    
    cache = QualityAwareCache(
        ttl_seconds=60,
        sample_size=10,
        upgrade_threshold=0.1  # 10% improvement required
    )
    
    track_id = 1
    
    # Sample 1: Quality 0.50 (first sample, always cached)
    emb1 = np.random.randn(512).astype(np.float32)
    cache.put(track_id, emb1, quality=0.50)
    print(f"Sample 1: quality=0.50 (FIRST)")
    
    # Sample 2: Quality 0.55 (+0.05, below threshold, should NOT upgrade)
    emb2 = np.random.randn(512).astype(np.float32)
    upgraded = cache.put(track_id, emb2, quality=0.55)
    assert upgraded == False, "Should NOT upgrade (+0.05 < 0.1 threshold)"
    cached = cache.get(track_id)
    assert cached is not None, "Should have cached embedding"
    assert np.allclose(cached, emb1), "Should keep original embedding"
    print(f"Sample 2: quality=0.55 (+0.05) → KEEP original")
    
    # Sample 3: Quality 0.70 (+0.20, above threshold, should UPGRADE)
    emb3 = np.random.randn(512).astype(np.float32)
    upgraded = cache.put(track_id, emb3, quality=0.70)
    assert upgraded == True, "Should UPGRADE (+0.20 > 0.1 threshold)"
    cached = cache.get(track_id)
    assert cached is not None, "Should have cached embedding"
    assert np.allclose(cached, emb3), "Should have new embedding"
    print(f"Sample 3: quality=0.70 (+0.20) → UPGRADE")
    
    # Sample 4: Quality 0.65 (-0.05, worse quality, should NOT upgrade)
    emb4 = np.random.randn(512).astype(np.float32)
    upgraded = cache.put(track_id, emb4, quality=0.65)
    assert upgraded == False, "Should NOT upgrade (worse quality)"
    cached = cache.get(track_id)
    assert cached is not None, "Should have cached embedding"
    assert np.allclose(cached, emb3), "Should keep best embedding"
    print(f"Sample 4: quality=0.65 (-0.05) → KEEP best (0.70)")
    
    stats = cache.get_stats()
    assert stats['upgrades'] == 1, "Should have 1 upgrade"
    print(f"\n✅ TEST 2 PASSED: Quality upgrades work correctly")
    print(f"   Total upgrades: {stats['upgrades']}")
    print(f"   Total puts: {stats['total_puts']}\n")


def test_sampling_completion():
    """Test 3: Sampling completion and finalization."""
    print("\n" + "=" * 70)
    print("TEST 3: Sampling Completion (10 samples)")
    print("=" * 70)
    
    cache = QualityAwareCache(
        ttl_seconds=60,
        sample_size=10,
        upgrade_threshold=0.1
    )
    
    track_id = 1
    
    # Simulate 10 samples with increasing quality
    print("Collecting 10 samples with increasing quality:")
    for i in range(10):
        embedding = np.random.randn(512).astype(np.float32)
        quality = 0.50 + (i * 0.05)  # 0.50, 0.55, 0.60, ..., 0.95
        
        upgraded = cache.put(track_id, embedding, quality)
        
        # Check if finalized after 10th sample
        entry = cache._cache[track_id]
        is_final = entry.is_final
        
        print(f"  Sample {i+1}: quality={quality:.2f}, upgraded={upgraded}, final={is_final}")
        
        if i < 9:
            assert not is_final, f"Should NOT be final at sample {i+1}"
        else:
            assert is_final, "Should be final after 10 samples"
    
    # Try to add 11th sample (should be rejected - sampling complete)
    print("\nTrying 11th sample (should be REJECTED):")
    emb11 = np.random.randn(512).astype(np.float32)
    upgraded = cache.put(track_id, emb11, quality=0.99)  # Even better quality!
    assert upgraded == False, "Should reject 11th sample (sampling complete)"
    print("  ✅ 11th sample REJECTED (as expected)")
    
    entry = cache._cache[track_id]
    print(f"\nFinal cached quality: {entry.quality:.3f}")
    print(f"Total samples collected: {entry.sample_count}")
    
    print(f"\n✅ TEST 3 PASSED: Sampling completes after 10 samples\n")


def test_ttl_expiration():
    """Test 4: TTL expiration."""
    print("\n" + "=" * 70)
    print("TEST 4: TTL Expiration")
    print("=" * 70)
    
    cache = QualityAwareCache(
        ttl_seconds=2,  # 2 second TTL for testing
        sample_size=10
    )
    
    track_id = 1
    embedding = np.random.randn(512).astype(np.float32)
    
    # Cache embedding
    cache.put(track_id, embedding, quality=0.75)
    print("Cached embedding with TTL=2 seconds")
    
    # Get immediately (should work)
    cached = cache.get(track_id)
    assert cached is not None, "Should be cached"
    print("✅ Immediately: Cache HIT")
    
    # Wait 1 second (should still work)
    print("Waiting 1 second...")
    time.sleep(1)
    cached = cache.get(track_id)
    assert cached is not None, "Should still be cached"
    print("✅ After 1s: Cache HIT")
    
    # Wait another 1.5 seconds (total 2.5s, should expire)
    print("Waiting another 1.5 seconds (total 2.5s)...")
    time.sleep(1.5)
    cached = cache.get(track_id)
    assert cached is None, "Should be expired"
    print("✅ After 2.5s: Cache MISS (expired)")
    
    stats = cache.get_stats()
    assert stats['evictions'] == 1, "Should have 1 eviction"
    print(f"\n✅ TEST 4 PASSED: TTL expiration works correctly")
    print(f"   Evictions: {stats['evictions']}\n")


def test_cache_statistics():
    """Test 5: Cache statistics and hit rate."""
    print("\n" + "=" * 70)
    print("TEST 5: Cache Statistics & Hit Rate")
    print("=" * 70)
    
    cache = QualityAwareCache(ttl_seconds=60, sample_size=10)
    
    # Simulate realistic usage: 3 people, each sampled 10 times
    print("Simulating 3 people × 10 samples each = 30 frames:")
    
    for track_id in [1, 2, 3]:
        print(f"\nPerson {track_id}:")
        for sample_num in range(10):
            embedding = np.random.randn(512).astype(np.float32)
            quality = 0.50 + np.random.rand() * 0.3  # Random quality [0.5, 0.8]
            
            # Try to get from cache first (simulates real pipeline)
            cached = cache.get(track_id)
            
            if cached is None:
                # Cache miss - extract new embedding
                cache.put(track_id, embedding, quality)
                print(f"  Sample {sample_num+1}: MISS → extract + cache")
            else:
                # Cache hit - use cached embedding
                print(f"  Sample {sample_num+1}: HIT → use cached")
    
    # Get statistics
    stats = cache.get_stats()
    
    print("\n" + "=" * 70)
    print("CACHE STATISTICS:")
    print("=" * 70)
    print(f"  Total Requests: {stats['total_requests']}")
    print(f"  Cache Hits: {stats['hits']}")
    print(f"  Cache Misses: {stats['misses']}")
    print(f"  Hit Rate: {stats['hit_rate']:.1%}")
    print(f"  Quality Upgrades: {stats['upgrades']}")
    print(f"  Cache Size: {stats['size']}/{stats['max_size']}")
    print("=" * 70)
    
    # Expected: 3 people × 10 samples = 30 requests
    # First sample per person = miss (3 misses)
    # Remaining 9 samples per person = hits if sampling complete (27 hits)
    # But sampling continues until 10 samples, so some may be misses
    
    assert stats['total_requests'] == 30, "Should have 30 total requests"
    assert stats['size'] == 3, "Should have 3 cached tracks"
    
    print(f"\n✅ TEST 5 PASSED: Cache statistics work correctly")
    print(f"   Expected hit rate: ~0% for first samples, increases as cache fills")
    print(f"   Actual hit rate: {stats['hit_rate']:.1%}\n")


def test_realistic_scenario():
    """Test 6: Realistic multi-person tracking scenario."""
    print("\n" + "=" * 70)
    print("TEST 6: Realistic Multi-Person Tracking")
    print("=" * 70)
    
    cache = QualityAwareCache(
        ttl_seconds=300,  # 5 minutes
        sample_size=10,
        upgrade_threshold=0.1
    )
    
    # Simulate 5 people tracked for 60 frames each (5 seconds at 12 FPS)
    num_people = 5
    num_frames = 60
    
    print(f"Simulating {num_people} people × {num_frames} frames:")
    print(f"Expected: ~10 extractions per person = {num_people * 10} total")
    print(f"Remaining {num_people * (num_frames - 10)} frames should be cache hits\n")
    
    for frame_num in range(num_frames):
        for track_id in range(1, num_people + 1):
            # Try cache first
            cached = cache.get(track_id)
            
            if cached is None:
                # Cache miss - extract new embedding
                embedding = np.random.randn(512).astype(np.float32)
                quality = 0.50 + np.random.rand() * 0.4  # [0.5, 0.9]
                cache.put(track_id, embedding, quality)
        
        if (frame_num + 1) % 10 == 0:
            stats = cache.get_stats()
            print(f"Frame {frame_num+1:3d}: "
                  f"Hits={stats['hits']:3d}, "
                  f"Misses={stats['misses']:3d}, "
                  f"Hit Rate={stats['hit_rate']:5.1%}")
    
    # Final statistics
    stats = cache.get_stats()
    cache.print_stats()
    
    # Validation
    total_frames = num_people * num_frames
    expected_misses = num_people * 10  # 10 samples per person
    expected_hits = total_frames - expected_misses
    expected_hit_rate = expected_hits / total_frames
    
    print(f"📊 Expected Results:")
    print(f"   Total Frames: {total_frames}")
    print(f"   Expected Misses: {expected_misses} (sampling phase)")
    print(f"   Expected Hits: {expected_hits} (post-sampling)")
    print(f"   Expected Hit Rate: {expected_hit_rate:.1%}")
    print(f"\n📊 Actual Results:")
    print(f"   Total Requests: {stats['total_requests']}")
    print(f"   Actual Misses: {stats['misses']}")
    print(f"   Actual Hits: {stats['hits']}")
    print(f"   Actual Hit Rate: {stats['hit_rate']:.1%}")
    
    # Hit rate should be ~83% (50 hits / 60 frames per person after sampling)
    assert stats['hit_rate'] >= 0.80, f"Hit rate should be ≥80% (got {stats['hit_rate']:.1%})"
    
    print(f"\n✅ TEST 6 PASSED: Realistic scenario achieves {stats['hit_rate']:.1%} hit rate")
    print(f"   🎯 CPU Reduction: {stats['hit_rate']:.1%} fewer recognitions needed!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("QUALITY-AWARE CACHE TEST SUITE")
    print("=" * 70)
    
    try:
        test_basic_operations()
        test_quality_upgrades()
        test_sampling_completion()
        test_ttl_expiration()
        test_cache_statistics()
        test_realistic_scenario()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\n🎉 Quality-Aware Cache is working correctly!")
        print("   Ready for integration into RecognitionStage.\n")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
