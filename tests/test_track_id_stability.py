#!/usr/bin/env python3
"""
Track ID Stability Diagnostic Test

This test checks if BoT-SORT is maintaining persistent track IDs
across frames. If track IDs change every frame, the cache won't work!

Expected behavior:
- Same person should get same track_id across all frames
- Cache hits should occur after sampling window (10 frames)

Actual behavior to check:
- Are track IDs stable?
- Are cache hits happening?
- What's the hit rate?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import logging
from collections import defaultdict
from common.config_manager import ConfigManager
from pipeline.orchestrator import PipelineOrchestrator

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_track_id_stability():
    """Test track ID persistence with real webcam"""
    
    logger.info("="*70)
    logger.info("TRACK ID STABILITY DIAGNOSTIC")
    logger.info("="*70)
    
    # Load config and create pipeline
    config = ConfigManager()
    config.load('config.yaml')
    
    # Try to open webcam
    camera_id = config.get('camera.device_id', 0)
    logger.info(f"\nAttempting to open camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        logger.error("❌ Camera not available!")
        return False
    
    logger.info("✅ Camera opened")
    logger.info("\nInstructions:")
    logger.info("  - Stay in front of camera for 30 frames")
    logger.info("  - We'll track your track_id consistency")
    logger.info("  - Press 'q' to quit early\n")
    
    orchestrator = PipelineOrchestrator(config)
    
    # Track ID history
    track_ids_per_frame = []
    unique_track_ids = set()
    
    # Cache stats
    cache_hits = 0
    cache_misses = 0
    
    frame_count = 0
    max_frames = 30
    
    logger.info("Starting capture (30 frames)...\n")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame")
            break
        
        # Process frame
        result = orchestrator.process_frame(frame)
        frame_count += 1
        
        # Get track IDs in this frame
        current_track_ids = [track.track_id for track in result['tracks']]
        track_ids_per_frame.append(current_track_ids)
        unique_track_ids.update(current_track_ids)
        
        # Get cache stats
        if orchestrator.recognition_stage is not None:
            rec_stats = orchestrator.recognition_stage.get_stats()
            cache_hits = rec_stats.get('cache_hits', 0)
            cache_misses = rec_stats.get('cache_misses', 0)
        
        # Print progress
        if current_track_ids:
            track_id_str = ", ".join([f"ID-{tid}" for tid in current_track_ids])
            logger.info(f"Frame {frame_count:2d}: Tracks=[{track_id_str}]  |  "
                       f"Cache: {cache_hits} hits, {cache_misses} misses")
        else:
            logger.info(f"Frame {frame_count:2d}: No tracks detected")
        
        # Display
        annotated = result['annotated_frame']
        cv2.putText(annotated, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Track ID Stability Test', annotated)
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Analyze results
    logger.info("\n" + "="*70)
    logger.info("DIAGNOSTIC RESULTS")
    logger.info("="*70)
    
    logger.info(f"\nFrames processed: {frame_count}")
    logger.info(f"Unique track IDs seen: {len(unique_track_ids)}")
    logger.info(f"Track IDs: {sorted(unique_track_ids)}")
    
    # Count frames with tracks
    frames_with_tracks = sum(1 for ids in track_ids_per_frame if ids)
    logger.info(f"Frames with tracks: {frames_with_tracks}")
    
    # Analyze ID stability
    if frames_with_tracks > 0:
        logger.info("\n📊 Track ID Stability Analysis:")
        
        # Count how many times each ID appeared
        id_counts = defaultdict(int)
        for ids in track_ids_per_frame:
            for tid in ids:
                id_counts[tid] += 1
        
        for tid, count in sorted(id_counts.items()):
            percentage = (count / frames_with_tracks) * 100
            logger.info(f"  Track ID {tid}: appeared in {count}/{frames_with_tracks} frames ({percentage:.1f}%)")
        
        # Check if IDs are stable
        if len(unique_track_ids) == 1:
            logger.info("\n✅ EXCELLENT: Only 1 track ID (perfectly stable!)")
        elif len(unique_track_ids) <= 3:
            logger.info(f"\n⚠️  WARNING: {len(unique_track_ids)} different track IDs (some flickering)")
        else:
            logger.info(f"\n❌ CRITICAL: {len(unique_track_ids)} different track IDs (severe flickering!)")
    
    # Cache analysis
    total_requests = cache_hits + cache_misses
    if total_requests > 0:
        hit_rate = (cache_hits / total_requests) * 100
        
        logger.info("\n📊 Cache Performance:")
        logger.info(f"  Total requests: {total_requests}")
        logger.info(f"  Cache hits: {cache_hits}")
        logger.info(f"  Cache misses: {cache_misses}")
        logger.info(f"  Hit rate: {hit_rate:.1f}%")
        
        # Expected vs actual
        expected_misses = min(10, frames_with_tracks)  # Sampling window
        expected_hits = max(0, frames_with_tracks - expected_misses)
        expected_hit_rate = (expected_hits / total_requests * 100) if total_requests > 0 else 0
        
        logger.info(f"\n  Expected (with stable IDs):")
        logger.info(f"    Misses: ~{expected_misses} (sampling window)")
        logger.info(f"    Hits: ~{expected_hits} (post-sampling)")
        logger.info(f"    Hit rate: ~{expected_hit_rate:.1f}%")
        
        if hit_rate >= expected_hit_rate * 0.8:
            logger.info("\n✅ Cache is working correctly!")
        else:
            logger.info(f"\n❌ Cache hit rate too low! (expected ~{expected_hit_rate:.1f}%, got {hit_rate:.1f}%)")
    else:
        logger.info("\n⚠️  No cache activity detected")
    
    logger.info("\n" + "="*70)
    
    return True


if __name__ == '__main__':
    try:
        success = test_track_id_stability()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n\nTest interrupted by user")
        sys.exit(1)
