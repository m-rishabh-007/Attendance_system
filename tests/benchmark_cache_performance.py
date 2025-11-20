#!/usr/bin/env python3
"""
Performance Benchmark: Cache vs Baseline
=========================================

Measures FPS and timing for:
1. Baseline (cache disabled)
2. Cached (cache enabled)

Captures:
- Overall FPS
- Detection time
- Alignment time  
- Recognition time
- Cache hit rate

Usage:
    python tests/benchmark_cache_performance.py

Phase: 3A Week 2 Day 6-7
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import time
import numpy as np
import logging
from typing import Dict, List
from common.config_manager import ConfigManager
from pipeline.orchestrator import PipelineOrchestrator

logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CacheBenchmark:
    """Performance benchmark for quality-aware caching."""
    
    def __init__(self):
        """Initialize benchmark."""
        self.config = ConfigManager()
        self.config.load('config.yaml')
        self.camera_id = self.config.get('camera.device_id', 1)
        self.num_frames = 60  # 60 frames = ~2 seconds at 30 FPS
        
    def run_benchmark(self, enable_cache: bool) -> Dict:
        """
        Run benchmark with cache enabled or disabled.
        
        Args:
            enable_cache: Whether to enable cache
            
        Returns:
            Dictionary with timing statistics
        """
        mode = "CACHED" if enable_cache else "BASELINE"
        print(f"\n{'='*70}")
        print(f"Running {mode} Mode")
        print(f"{'='*70}")
        
        # Create fresh config with cache setting
        config = ConfigManager()
        config.load('config.yaml')
        
        # Temporarily override cache setting
        all_config = config.get_all()
        all_config['recognition']['enable_cache'] = enable_cache
        
        # Create fresh orchestrator
        orchestrator = PipelineOrchestrator(config)
        
        # Open camera
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")
        
        # Warm up (skip first 5 frames)
        print("Warming up...")
        for _ in range(5):
            cap.read()
        
        print(f"Capturing {self.num_frames} frames...\n")
        
        # Collect timing data
        frame_times = []
        detection_times = []
        alignment_times = []
        recognition_times = []
        faces_detected = []
        
        start_time = time.time()
        
        for i in range(self.num_frames):
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {i}")
                break
            
            frame_start = time.time()
            result = orchestrator.process_frame(frame)
            frame_end = time.time()
            
            frame_time = (frame_end - frame_start) * 1000  # ms
            frame_times.append(frame_time)
            
            # Extract individual stage times if available
            rec_result = result.get('recognition_result')
            if rec_result:
                detection_times.append(rec_result.get('detection_time_ms', 0))
                alignment_times.append(rec_result.get('avg_alignment_time_ms', 0))
                recognition_times.append(rec_result.get('avg_recognition_time_ms', 0))
            
            faces_detected.append(len(result['tracks']))
            
            if (i + 1) % 20 == 0:
                print(f"  Frame {i+1}/{self.num_frames} - {frame_time:.1f}ms - {len(result['tracks'])} faces")
        
        elapsed = time.time() - start_time
        cap.release()
        
        # Calculate statistics
        stats = {
            'mode': mode,
            'total_frames': len(frame_times),
            'elapsed_time': elapsed,
            'fps': len(frame_times) / elapsed if elapsed > 0 else 0,
            'frame_time_mean': np.mean(frame_times) if frame_times else 0,
            'frame_time_median': np.median(frame_times) if frame_times else 0,
            'frame_time_min': np.min(frame_times) if frame_times else 0,
            'frame_time_max': np.max(frame_times) if frame_times else 0,
            'frame_time_std': np.std(frame_times) if frame_times else 0,
            'avg_faces': np.mean(faces_detected) if faces_detected else 0,
        }
        
        # Get cache stats if enabled
        if enable_cache and orchestrator.recognition_stage and orchestrator.recognition_stage.cache:
            cache_stats = orchestrator.recognition_stage.cache.get_stats()
            stats['cache_hits'] = cache_stats['hits']
            stats['cache_misses'] = cache_stats['misses']
            stats['cache_hit_rate'] = cache_stats['hit_rate']
            stats['cache_upgrades'] = cache_stats['upgrades']
        
        # Get recognition stage stats
        if orchestrator.recognition_stage:
            rec_stats = orchestrator.recognition_stage.get_stats()
            stats['alignments_success'] = rec_stats['alignments_success']
            stats['recognitions_success'] = rec_stats['recognitions_success']
            stats['avg_alignment_time'] = rec_stats.get('avg_alignment_time_ms', 0)
            stats['avg_recognition_time'] = rec_stats.get('avg_recognition_time_ms', 0)
        
        return stats
    
    def print_comparison(self, baseline: Dict, cached: Dict):
        """Print comparison between baseline and cached performance."""
        print("\n" + "="*70)
        print("PERFORMANCE COMPARISON")
        print("="*70)
        
        print(f"\n{'Metric':<35} {'Baseline':<15} {'Cached':<15} {'Improvement'}")
        print("-"*70)
        
        # FPS
        fps_improvement = ((cached['fps'] - baseline['fps']) / baseline['fps']) * 100 if baseline['fps'] > 0 else 0
        print(f"{'FPS (frames/sec)':<35} {baseline['fps']:>14.2f} {cached['fps']:>14.2f} {fps_improvement:>+13.1f}%")
        
        # Frame time
        frame_improvement = ((baseline['frame_time_mean'] - cached['frame_time_mean']) / baseline['frame_time_mean']) * 100 if baseline['frame_time_mean'] > 0 else 0
        print(f"{'Frame Time Mean (ms)':<35} {baseline['frame_time_mean']:>14.1f} {cached['frame_time_mean']:>14.1f} {frame_improvement:>+13.1f}%")
        print(f"{'Frame Time Median (ms)':<35} {baseline['frame_time_median']:>14.1f} {cached['frame_time_median']:>14.1f}")
        print(f"{'Frame Time Min (ms)':<35} {baseline['frame_time_min']:>14.1f} {cached['frame_time_min']:>14.1f}")
        print(f"{'Frame Time Max (ms)':<35} {baseline['frame_time_max']:>14.1f} {cached['frame_time_max']:>14.1f}")
        
        # Alignment time
        if baseline.get('avg_alignment_time', 0) > 0:
            align_improvement = ((baseline['avg_alignment_time'] - cached['avg_alignment_time']) / baseline['avg_alignment_time']) * 100
            print(f"{'Alignment Time (ms)':<35} {baseline['avg_alignment_time']:>14.1f} {cached['avg_alignment_time']:>14.1f} {align_improvement:>+13.1f}%")
        
        # Recognition time
        if baseline.get('avg_recognition_time', 0) > 0:
            recog_improvement = ((baseline['avg_recognition_time'] - cached['avg_recognition_time']) / baseline['avg_recognition_time']) * 100
            print(f"{'Recognition Time (ms)':<35} {baseline['avg_recognition_time']:>14.1f} {cached['avg_recognition_time']:>14.1f} {recog_improvement:>+13.1f}%")
        
        # Alignments/Recognitions
        print(f"\n{'Alignments Performed':<35} {baseline.get('alignments_success', 0):>14} {cached.get('alignments_success', 0):>14}")
        print(f"{'Recognitions Performed':<35} {baseline.get('recognitions_success', 0):>14} {cached.get('recognitions_success', 0):>14}")
        
        # Cache stats
        if 'cache_hit_rate' in cached:
            print(f"\n{'Cache Hit Rate':<35} {'N/A':>14} {cached['cache_hit_rate']:>13.1f}%")
            print(f"{'Cache Hits':<35} {'N/A':>14} {cached['cache_hits']:>14}")
            print(f"{'Cache Misses':<35} {'N/A':>14} {cached['cache_misses']:>14}")
            print(f"{'Cache Upgrades':<35} {'N/A':>14} {cached['cache_upgrades']:>14}")
        
        # CPU Savings
        if baseline.get('recognitions_success', 0) > 0 and cached.get('recognitions_success', 0) > 0:
            cpu_saved = ((baseline['recognitions_success'] - cached['recognitions_success']) / baseline['recognitions_success']) * 100
            print(f"\n{'CPU Savings (recognitions)':<35} {'':>14} {cpu_saved:>13.1f}%")
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"✅ FPS Improvement: {fps_improvement:+.1f}%")
        print(f"✅ Frame Time Reduction: {frame_improvement:+.1f}%")
        if 'cache_hit_rate' in cached:
            print(f"✅ Cache Hit Rate: {cached['cache_hit_rate']:.1f}%")
        print(f"✅ Test Duration: {baseline['elapsed_time']:.1f}s (baseline), {cached['elapsed_time']:.1f}s (cached)")
        print("="*70)
    
    def run(self):
        """Run complete benchmark."""
        print("="*70)
        print("CACHE PERFORMANCE BENCHMARK")
        print("="*70)
        print(f"Camera: {self.camera_id}")
        print(f"Frames: {self.num_frames}")
        print(f"\nPlease sit in front of camera for ~4 seconds...")
        print("="*70)
        
        input("\nPress Enter when ready to start baseline test...")
        
        # Run baseline (cache disabled)
        baseline_stats = self.run_benchmark(enable_cache=False)
        
        print(f"\n✅ Baseline complete: {baseline_stats['fps']:.2f} FPS")
        print(f"   Mean frame time: {baseline_stats['frame_time_mean']:.1f}ms")
        
        input("\nPress Enter when ready to start cached test...")
        
        # Run cached (cache enabled)
        cached_stats = self.run_benchmark(enable_cache=True)
        
        print(f"\n✅ Cached complete: {cached_stats['fps']:.2f} FPS")
        print(f"   Mean frame time: {cached_stats['frame_time_mean']:.1f}ms")
        
        # Print comparison
        self.print_comparison(baseline_stats, cached_stats)


def main():
    """Main benchmark runner."""
    try:
        benchmark = CacheBenchmark()
        benchmark.run()
        return 0
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
