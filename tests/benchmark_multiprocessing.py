#!/usr/bin/env python3
"""
Multiprocessing Pipeline Benchmark

Measures performance of async pipeline vs sequential pipeline:
- Process 1 FPS (min/mean/max/std deviation)
- Queue metrics (drops count, max depth, drop rate)
- Recognition latency (time from queue.put to result appearing)
- BoT-SORT ID stability (ID switches per 100 frames)
- FPS jitter analysis (identifies "blind spots")

Validates that async mode achieves:
- Stable 15-25 FPS (no jitter from recognition blocking)
- <5% queue drop rate (worker keeps up with camera)
- <400ms recognition latency (results appear promptly)
- 0 ID switches per 100 frames (tracking stability)

Date: December 1, 2025
Phase: 4 - Async Performance Validation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import time
import numpy as np
import logging
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Any

from common.config_manager import ConfigManager
from pipeline.async_orchestrator import AsyncOrchestrator
from pipeline.orchestrator import PipelineOrchestrator

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AsyncPipelineBenchmark:
    """
    Comprehensive benchmark for async multiprocessing pipeline.
    
    Metrics:
        - FPS Statistics: min, mean, max, std deviation
        - Queue Performance: drops, max depth, drop rate
        - Recognition Latency: time from detection to recognition
        - ID Stability: track ID switches per 100 frames
        - FPS Jitter: variance in frame times (identifies blocking)
    """
    
    def __init__(self, config: ConfigManager, mode: str = 'async'):
        """
        Initialize benchmark.
        
        Args:
            config: ConfigManager instance
            mode: 'async' or 'sequential'
        """
        self.config = config
        self.mode = mode
        
        # Create appropriate orchestrator
        if mode == 'async':
            self.orchestrator = AsyncOrchestrator(config)
        else:
            self.orchestrator = PipelineOrchestrator(config)
        
        # Metrics storage
        self.frame_times = []  # ms per frame (Process 1 loop time)
        self.fps_history = []  # FPS per second
        self.queue_stats_history = []  # Queue depth samples
        
        # Latency tracking
        self.detection_timestamps = {}  # {track_id: timestamp when queued}
        self.recognition_latencies = []  # ms from queue to result
        
        # ID stability tracking
        self.track_id_history = deque(maxlen=100)  # Last 100 frames
        self.id_switches = []  # Count of switches per 100-frame window
        
        # Performance counters
        self.total_frames = 0
        self.total_faces_detected = 0
        self.total_faces_recognized = 0
    
    def run_benchmark(self, source: int = 0, duration_seconds: int = 30) -> Dict[str, Any]:
        """
        Run benchmark on video source.
        
        Args:
            source: Camera device ID or video path
            duration_seconds: How long to run benchmark
            
        Returns:
            Dictionary with comprehensive performance metrics
        """
        logger.info("="*60)
        logger.info(f"Starting {self.mode.upper()} Pipeline Benchmark")
        logger.info(f"Duration: {duration_seconds} seconds")
        logger.info("="*60)
        
        # Start async orchestrator if needed
        if self.mode == 'async':
            self.orchestrator.start()
        
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")
        
        start_time = time.time()
        last_fps_calc = start_time
        frame_count_for_fps = 0
        
        try:
            while time.time() - start_time < duration_seconds:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame, restarting...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                    continue
                
                # Measure frame processing time
                frame_start = time.time()
                
                result = self.orchestrator.process_frame(frame)
                
                frame_end = time.time()
                frame_time_ms = (frame_end - frame_start) * 1000
                self.frame_times.append(frame_time_ms)
                
                # Track detections and recognitions
                self.total_frames += 1
                frame_count_for_fps += 1
                
                if 'tracks' in result:
                    tracks = result['tracks']
                    self.total_faces_detected += len(tracks)
                    
                    # Track ID stability analysis
                    current_ids = set(int(t.track_id) for t in tracks)
                    self.track_id_history.append(current_ids)
                    
                    # Check for ID switches (same person, different ID)
                    if len(self.track_id_history) >= 2:
                        prev_ids = self.track_id_history[-2]
                        disappeared = prev_ids - current_ids
                        appeared = current_ids - prev_ids
                        
                        # Heuristic: If IDs disappear and new ones appear simultaneously,
                        # it might be an ID switch (not definitive, but indicative)
                        if len(disappeared) > 0 and len(appeared) > 0:
                            self.id_switches.append(len(disappeared))
                    
                    # Track recognition timestamps (async mode only)
                    if self.mode == 'async' and hasattr(self.orchestrator, 'queued_ids'):
                        for track in tracks:
                            track_id = int(track.track_id)
                            
                            # Mark when face was queued
                            if track_id in self.orchestrator.queued_ids:
                                if track_id not in self.detection_timestamps:
                                    self.detection_timestamps[track_id] = time.time()
                            
                            # Measure latency when recognition completes
                            if track_id in self.orchestrator.track_id_name_map:
                                if track_id in self.detection_timestamps:
                                    latency_ms = (time.time() - self.detection_timestamps[track_id]) * 1000
                                    self.recognition_latencies.append(latency_ms)
                                    self.total_faces_recognized += 1
                                    del self.detection_timestamps[track_id]  # Only count once
                
                # Collect queue stats (async mode only)
                if self.mode == 'async' and 'queue_stats' in result:
                    self.queue_stats_history.append(result['queue_stats'].copy())
                
                # Calculate FPS every second
                if time.time() - last_fps_calc >= 1.0:
                    fps = frame_count_for_fps / (time.time() - last_fps_calc)
                    self.fps_history.append(fps)
                    
                    # Print progress
                    elapsed = time.time() - start_time
                    remaining = duration_seconds - elapsed
                    logger.info(f"[{elapsed:.1f}s] FPS: {fps:.2f} | Frames: {self.total_frames} | Remaining: {remaining:.1f}s")
                    
                    frame_count_for_fps = 0
                    last_fps_calc = time.time()
                
                # Optional: Display frame (comment out for headless benchmarking)
                # cv2.imshow("Benchmark", result.get('annotated_frame', frame))
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            if self.mode == 'async':
                self.orchestrator.stop()
        
        # Calculate final metrics
        return self._calculate_metrics()
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        metrics = {
            'mode': self.mode,
            'total_frames': self.total_frames,
            'total_faces_detected': self.total_faces_detected,
            'total_faces_recognized': self.total_faces_recognized,
        }
        
        # === FPS STATISTICS ===
        if self.fps_history:
            metrics['fps'] = {
                'mean': float(np.mean(self.fps_history)),
                'min': float(np.min(self.fps_history)),
                'max': float(np.max(self.fps_history)),
                'std': float(np.std(self.fps_history)),
                'median': float(np.median(self.fps_history)),
            }
        
        # === FRAME TIME STATISTICS (Process 1 loop time) ===
        if self.frame_times:
            metrics['frame_time_ms'] = {
                'mean': float(np.mean(self.frame_times)),
                'min': float(np.min(self.frame_times)),
                'max': float(np.max(self.frame_times)),
                'std': float(np.std(self.frame_times)),
                'p50': float(np.percentile(self.frame_times, 50)),
                'p95': float(np.percentile(self.frame_times, 95)),
                'p99': float(np.percentile(self.frame_times, 99)),
            }
            
            # FPS jitter analysis (identifies blocking)
            jitter = np.std(self.frame_times)
            metrics['fps_jitter_ms'] = float(jitter)
            metrics['fps_stability'] = 'STABLE' if jitter < 20 else 'UNSTABLE'
        
        # === QUEUE STATISTICS (async mode only) ===
        if self.mode == 'async' and self.queue_stats_history:
            total_queued = sum(s['queued'] for s in self.queue_stats_history)
            total_dropped = sum(s['dropped'] for s in self.queue_stats_history)
            
            metrics['queue'] = {
                'total_queued': total_queued,
                'total_dropped': total_dropped,
                'drop_rate_pct': (total_dropped / total_queued * 100) if total_queued > 0 else 0,
                'max_depth': max(s['depth_max'] for s in self.queue_stats_history) if self.queue_stats_history else 0,
            }
        
        # === RECOGNITION LATENCY ===
        if self.recognition_latencies:
            metrics['latency_ms'] = {
                'mean': float(np.mean(self.recognition_latencies)),
                'min': float(np.min(self.recognition_latencies)),
                'max': float(np.max(self.recognition_latencies)),
                'p50': float(np.percentile(self.recognition_latencies, 50)),
                'p95': float(np.percentile(self.recognition_latencies, 95)),
            }
        
        # === ID STABILITY ===
        if self.id_switches:
            total_switches = sum(self.id_switches)
            switches_per_100_frames = (total_switches / self.total_frames * 100) if self.total_frames > 0 else 0
            
            metrics['id_stability'] = {
                'total_switches': total_switches,
                'switches_per_100_frames': float(switches_per_100_frames),
                'stability_rating': 'EXCELLENT' if switches_per_100_frames < 1 else 'GOOD' if switches_per_100_frames < 5 else 'POOR',
            }
        
        return metrics
    
    def print_report(self, metrics: Dict[str, Any]):
        """Print human-readable benchmark report."""
        logger.info("="*80)
        logger.info(f"BENCHMARK REPORT - {metrics['mode'].upper()} MODE")
        logger.info("="*80)
        
        logger.info(f"\n📊 OVERALL STATISTICS")
        logger.info(f"  Total frames: {metrics['total_frames']}")
        logger.info(f"  Faces detected: {metrics['total_faces_detected']}")
        logger.info(f"  Faces recognized: {metrics['total_faces_recognized']}")
        
        if 'fps' in metrics:
            fps = metrics['fps']
            logger.info(f"\n🎯 FPS PERFORMANCE")
            logger.info(f"  Mean FPS: {fps['mean']:.2f}")
            logger.info(f"  Min FPS: {fps['min']:.2f}")
            logger.info(f"  Max FPS: {fps['max']:.2f}")
            logger.info(f"  Std Dev: {fps['std']:.2f}")
            logger.info(f"  Median FPS: {fps['median']:.2f}")
        
        if 'frame_time_ms' in metrics:
            ft = metrics['frame_time_ms']
            logger.info(f"\n⏱️  FRAME TIME (Process 1 Loop)")
            logger.info(f"  Mean: {ft['mean']:.2f}ms")
            logger.info(f"  Min: {ft['min']:.2f}ms")
            logger.info(f"  Max: {ft['max']:.2f}ms")
            logger.info(f"  P50: {ft['p50']:.2f}ms")
            logger.info(f"  P95: {ft['p95']:.2f}ms")
            logger.info(f"  P99: {ft['p99']:.2f}ms")
            logger.info(f"  Jitter: {metrics['fps_jitter_ms']:.2f}ms ({metrics['fps_stability']})")
        
        if 'queue' in metrics:
            q = metrics['queue']
            logger.info(f"\n📦 QUEUE PERFORMANCE")
            logger.info(f"  Total queued: {q['total_queued']}")
            logger.info(f"  Total dropped: {q['total_dropped']}")
            logger.info(f"  Drop rate: {q['drop_rate_pct']:.2f}%")
            logger.info(f"  Max depth: {q['max_depth']}")
            
            if q['drop_rate_pct'] < 5:
                logger.info(f"  ✅ Drop rate < 5% (Worker keeping up!)")
            else:
                logger.warning(f"  ⚠️ Drop rate > 5% (Worker may be overloaded)")
        
        if 'latency_ms' in metrics:
            lat = metrics['latency_ms']
            logger.info(f"\n⏳ RECOGNITION LATENCY")
            logger.info(f"  Mean: {lat['mean']:.2f}ms")
            logger.info(f"  Min: {lat['min']:.2f}ms")
            logger.info(f"  Max: {lat['max']:.2f}ms")
            logger.info(f"  P50: {lat['p50']:.2f}ms")
            logger.info(f"  P95: {lat['p95']:.2f}ms")
            
            if lat['p95'] < 400:
                logger.info(f"  ✅ P95 latency < 400ms (Results appear promptly!)")
            else:
                logger.warning(f"  ⚠️ P95 latency > 400ms (Students may walk away before recognition)")
        
        if 'id_stability' in metrics:
            ids = metrics['id_stability']
            logger.info(f"\n🔒 ID STABILITY (BoT-SORT)")
            logger.info(f"  Total ID switches: {ids['total_switches']}")
            logger.info(f"  Switches per 100 frames: {ids['switches_per_100_frames']:.2f}")
            logger.info(f"  Rating: {ids['stability_rating']}")
            
            if ids['switches_per_100_frames'] < 1:
                logger.info(f"  ✅ Excellent tracking stability!")
            elif ids['switches_per_100_frames'] < 5:
                logger.info(f"  ⚠️ Good tracking, minor ID switches")
            else:
                logger.warning(f"  ❌ Poor tracking stability (FPS jitter breaking Kalman filter)")
        
        logger.info("="*80)


def main():
    """Run benchmark comparison: Sequential vs Async."""
    # Load base config
    config = ConfigManager()
    config.load('config.yaml')
    
    # Get config dict and override for async mode
    config_dict_async = config.get_all().copy()
    config_dict_async['pipeline'] = {
        'async_mode': True,
        'face_queue_size': 5,
        'enable_result_annotations': True
    }
    config_async = ConfigManager()
    config_async._config = config_dict_async
    
    # Get config dict and override for sequential mode
    config_dict_seq = config.get_all().copy()
    config_dict_seq['pipeline'] = {
        'async_mode': False
    }
    config_seq = ConfigManager()
    config_seq._config = config_dict_seq
    
    # === BENCHMARK 1: ASYNC MODE ===
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK 1: ASYNC MODE (Multiprocessing)")
    logger.info("="*80 + "\n")
    
    # Get camera ID from config
    camera_id = config.get('camera.device_id', 1)
    logger.info(f"Using camera device: {camera_id}")
    
    benchmark_async = AsyncPipelineBenchmark(config_async, mode='async')
    metrics_async = benchmark_async.run_benchmark(source=camera_id, duration_seconds=30)
    benchmark_async.print_report(metrics_async)
    
    # Wait a bit before next benchmark
    logger.info("\nWaiting 3 seconds before next benchmark...")
    time.sleep(3)
    
    # === BENCHMARK 2: SEQUENTIAL MODE ===
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK 2: SEQUENTIAL MODE (Baseline)")
    logger.info("="*80 + "\n")
    
    benchmark_seq = AsyncPipelineBenchmark(config_seq, mode='sequential')
    metrics_seq = benchmark_seq.run_benchmark(source=camera_id, duration_seconds=30)
    benchmark_seq.print_report(metrics_seq)
    
    # === COMPARISON ===
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("="*80)
    
    if 'fps' in metrics_async and 'fps' in metrics_seq:
        fps_improvement = ((metrics_async['fps']['mean'] - metrics_seq['fps']['mean']) / metrics_seq['fps']['mean']) * 100
        logger.info(f"\nFPS Improvement: {fps_improvement:+.1f}%")
        logger.info(f"  Async: {metrics_async['fps']['mean']:.2f} FPS")
        logger.info(f"  Sequential: {metrics_seq['fps']['mean']:.2f} FPS")
    
    if 'fps_jitter_ms' in metrics_async and 'fps_jitter_ms' in metrics_seq:
        jitter_reduction = ((metrics_seq['fps_jitter_ms'] - metrics_async['fps_jitter_ms']) / metrics_seq['fps_jitter_ms']) * 100
        logger.info(f"\nJitter Reduction: {jitter_reduction:+.1f}%")
        logger.info(f"  Async jitter: {metrics_async['fps_jitter_ms']:.2f}ms")
        logger.info(f"  Sequential jitter: {metrics_seq['fps_jitter_ms']:.2f}ms")
    
    logger.info("\n" + "="*80)


if __name__ == "__main__":
    main()
