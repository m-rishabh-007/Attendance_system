#!/usr/bin/env python3
"""
Pipeline Performance Benchmark - Phase 3A Day 7

Benchmarks full pipeline performance:
- FPS measurement
- Per-stage timing breakdown
- Quality score distribution
- Memory usage
- Recognition success rate

Date: November 8, 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import time
import logging
from collections import defaultdict
from common.config_manager import ConfigManager
from pipeline.orchestrator import PipelineOrchestrator

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PipelineBenchmark:
    """Benchmark pipeline performance"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.reset_stats()
    
    def reset_stats(self):
        """Reset all statistics"""
        self.frame_times = []
        self.track_counts = []
        self.quality_scores = []
        self.embeddings_extracted = []
        self.processing_times = {
            'total': [],
            'detection': [],
            'tracking': [],
            'recognition': []
        }
    
    def process_frame(self, frame):
        """Process frame and collect metrics"""
        start_time = time.time()
        result = self.orchestrator.process_frame(frame)
        total_time = (time.time() - start_time) * 1000
        
        # Collect metrics
        self.frame_times.append(total_time)
        self.track_counts.append(len(result['tracks']))
        
        # Recognition metrics
        if result['recognition_result'] is not None:
            rec_result = result['recognition_result']
            self.embeddings_extracted.append(rec_result['embeddings_extracted'])
            
            if rec_result['avg_quality'] > 0:
                self.quality_scores.append(rec_result['avg_quality'])
            
            self.processing_times['recognition'].append(rec_result['processing_time_ms'])
        
        return result
    
    def get_stats(self):
        """Calculate benchmark statistics"""
        stats = {}
        
        if self.frame_times:
            stats['fps'] = {
                'mean': 1000 / np.mean(self.frame_times),
                'min': 1000 / np.max(self.frame_times),
                'max': 1000 / np.min(self.frame_times),
            }
            
            stats['frame_time_ms'] = {
                'mean': np.mean(self.frame_times),
                'min': np.min(self.frame_times),
                'max': np.max(self.frame_times),
                'std': np.std(self.frame_times),
            }
        
        if self.track_counts:
            stats['tracks'] = {
                'mean': np.mean(self.track_counts),
                'max': np.max(self.track_counts),
                'total': np.sum(self.track_counts),
            }
        
        if self.quality_scores:
            stats['quality'] = {
                'mean': np.mean(self.quality_scores),
                'min': np.min(self.quality_scores),
                'max': np.max(self.quality_scores),
                'std': np.std(self.quality_scores),
            }
        
        if self.embeddings_extracted:
            stats['embeddings'] = {
                'total': np.sum(self.embeddings_extracted),
                'mean_per_frame': np.mean(self.embeddings_extracted),
            }
        
        return stats
    
    def print_report(self):
        """Print formatted benchmark report"""
        stats = self.get_stats()
        
        logger.info("\n" + "="*70)
        logger.info("BENCHMARK REPORT")
        logger.info("="*70)
        
        logger.info(f"\nFrames Processed: {len(self.frame_times)}")
        
        if 'fps' in stats:
            logger.info(f"\nFPS Performance:")
            logger.info(f"  Mean FPS: {stats['fps']['mean']:.2f}")
            logger.info(f"  Min FPS: {stats['fps']['min']:.2f}")
            logger.info(f"  Max FPS: {stats['fps']['max']:.2f}")
        
        if 'frame_time_ms' in stats:
            logger.info(f"\nFrame Processing Time:")
            logger.info(f"  Mean: {stats['frame_time_ms']['mean']:.1f}ms")
            logger.info(f"  Min: {stats['frame_time_ms']['min']:.1f}ms")
            logger.info(f"  Max: {stats['frame_time_ms']['max']:.1f}ms")
            logger.info(f"  Std Dev: {stats['frame_time_ms']['std']:.1f}ms")
        
        if 'tracks' in stats:
            logger.info(f"\nTracking:")
            logger.info(f"  Total tracks: {int(stats['tracks']['total'])}")
            logger.info(f"  Avg per frame: {stats['tracks']['mean']:.2f}")
            logger.info(f"  Max simultaneous: {int(stats['tracks']['max'])}")
        
        if 'quality' in stats:
            logger.info(f"\nQuality Scores:")
            logger.info(f"  Mean: {stats['quality']['mean']:.3f}")
            logger.info(f"  Min: {stats['quality']['min']:.3f}")
            logger.info(f"  Max: {stats['quality']['max']:.3f}")
            logger.info(f"  Std Dev: {stats['quality']['std']:.3f}")
        
        if 'embeddings' in stats:
            logger.info(f"\nRecognition:")
            logger.info(f"  Faces with embeddings: {int(stats['embeddings']['total'])}")
            logger.info(f"  Avg per frame: {stats['embeddings']['mean_per_frame']:.2f}")
        
        # Recognition stage stats
        if self.orchestrator.recognition_stage is not None:
            rec_stats = self.orchestrator.recognition_stage.get_stats()
            logger.info(f"\nRecognition Stage Details:")
            logger.info(f"  Total processed: {rec_stats['total_processed']}")
            logger.info(f"  Alignments success: {rec_stats['alignments_success']}")
            logger.info(f"  Alignments failed: {rec_stats['alignments_failed']}")
            logger.info(f"  Recognitions success: {rec_stats['recognitions_success']}")
            logger.info(f"  Quality too low: {rec_stats['quality_too_low']}")
            
            # Cache statistics
            if 'cache_hits' in rec_stats:
                cache_hits = rec_stats.get('cache_hits', 0)
                cache_misses = rec_stats.get('cache_misses', 0)
                total_cache_requests = cache_hits + cache_misses
                
                if total_cache_requests > 0:
                    logger.info(f"\nCache Performance:")
                    logger.info(f"  Cache hits: {cache_hits}")
                    logger.info(f"  Cache misses: {cache_misses}")
                    logger.info(f"  Hit rate: {rec_stats.get('cache_hit_rate', 0)*100:.1f}%")
                    logger.info(f"  CPU reduction: {(cache_hits/total_cache_requests)*100:.1f}%")
            
            if rec_stats['alignments_success'] > 0:
                logger.info(f"\nTiming Details:")
                logger.info(f"  Avg alignment time: {rec_stats['avg_alignment_time_ms']:.1f}ms")
            if rec_stats['recognitions_success'] > 0:
                logger.info(f"  Avg recognition time: {rec_stats['avg_recognition_time_ms']:.1f}ms")
            if rec_stats['total_processed'] > 0:
                logger.info(f"  Avg quality scoring: {rec_stats['avg_quality_time_ms']:.1f}ms")
        
        logger.info("\n" + "="*70)


def benchmark_synthetic_video(num_frames=100):
    """Benchmark with synthetic video"""
    logger.info("\n" + "="*70)
    logger.info("Benchmark 1: Synthetic Video")
    logger.info("="*70)
    
    config = ConfigManager()
    config.load('config.yaml')
    orchestrator = PipelineOrchestrator(config)
    benchmark = PipelineBenchmark(orchestrator)
    
    logger.info(f"Processing {num_frames} synthetic frames...")
    
    for i in range(num_frames):
        # Create random frame
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        benchmark.process_frame(frame)
        
        if (i + 1) % 20 == 0:
            logger.info(f"  Processed {i+1}/{num_frames} frames...")
    
    benchmark.print_report()
    return True


def benchmark_webcam(duration_seconds=10):
    """Benchmark with webcam (if available)"""
    logger.info("\n" + "="*70)
    logger.info("Benchmark 2: Webcam (Real Faces)")
    logger.info("="*70)
    
    config = ConfigManager()
    config.load('config.yaml')
    
    # Try to open webcam
    camera_id = config.get('camera.device_id', 0)
    logger.info(f"Attempting to open camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        logger.warning("⚠️  Camera not available, skipping")
        return True
    
    logger.info(f"✅ Camera opened")
    logger.info(f"Running for {duration_seconds} seconds...")
    logger.info("Press 'q' to quit early")
    
    orchestrator = PipelineOrchestrator(config)
    benchmark = PipelineBenchmark(orchestrator)
    
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < duration_seconds:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = benchmark.process_frame(frame)
        frame_count += 1
        
        # Display
        annotated = result['annotated_frame']
        
        # Add FPS overlay
        if benchmark.frame_times:
            fps = 1000 / benchmark.frame_times[-1]
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Benchmark', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    logger.info(f"Processed {frame_count} frames in {duration_seconds}s")
    benchmark.print_report()
    
    return True


def benchmark_static_image():
    """Benchmark with static test image"""
    logger.info("\n" + "="*70)
    logger.info("Benchmark 3: Static Image (Stress Test)")
    logger.info("="*70)
    
    config = ConfigManager()
    config.load('config.yaml')
    orchestrator = PipelineOrchestrator(config)
    benchmark = PipelineBenchmark(orchestrator)
    
    # Create test image with face-like features
    frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
    
    # Add face-like region
    cv2.rectangle(frame, (200, 150), (350, 330), (180, 200, 220), -1)
    cv2.circle(frame, (250, 210), 15, (50, 50, 50), -1)
    cv2.circle(frame, (300, 210), 15, (50, 50, 50), -1)
    
    logger.info("Processing same frame 100 times...")
    
    for i in range(100):
        benchmark.process_frame(frame.copy())
        
        if (i + 1) % 20 == 0:
            logger.info(f"  Processed {i+1}/100 frames...")
    
    benchmark.print_report()
    return True


def run_all_benchmarks():
    """Run all benchmark tests"""
    logger.info("\n" + "="*70)
    logger.info("PIPELINE PERFORMANCE BENCHMARK")
    logger.info("Phase 3A - Week 1 Day 7")
    logger.info("="*70)
    
    benchmarks = [
        ("Synthetic Video (100 frames)", lambda: benchmark_synthetic_video(100)),
        ("Static Image Stress Test", benchmark_static_image),
        ("Webcam Real-time (10s)", lambda: benchmark_webcam(10)),
    ]
    
    passed = 0
    failed = 0
    
    for name, benchmark_func in benchmarks:
        try:
            logger.info(f"\nRunning: {name}")
            if benchmark_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"❌ Benchmark crashed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    logger.info("\n" + "="*70)
    logger.info(f"Benchmarks: {passed}/{len(benchmarks)} completed")
    logger.info("="*70)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_benchmarks()
    sys.exit(0 if success else 1)
