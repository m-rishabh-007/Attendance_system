#!/usr/bin/env python3
"""End-to-end pipeline test with recognition - Phase 3A Day 6"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import logging
from common.config_manager import ConfigManager
from pipeline.orchestrator import PipelineOrchestrator

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_initialization():
    """Test 1: Pipeline initialization with recognition"""
    logger.info("="*60)
    logger.info("Test 1: Pipeline Initialization")
    logger.info("="*60)
    
    config = ConfigManager()
    config.load('config.yaml')
    
    orchestrator = PipelineOrchestrator(config)
    
    if orchestrator.recognition_stage is None:
        logger.error("❌ Recognition stage not initialized")
        return False
    
    logger.info(f"✅ Pipeline: {orchestrator}")
    return True

def test_empty_frame():
    """Test 2: Process empty frame"""
    logger.info("\n" + "="*60)
    logger.info("Test 2: Process Empty Frame")
    logger.info("="*60)
    
    config = ConfigManager()
    config.load('config.yaml')
    orchestrator = PipelineOrchestrator(config)
    
    # Create empty frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    result = orchestrator.process_frame(frame)
    
    logger.info(f"Processing time: {result['processing_time_ms']:.1f}ms")
    logger.info(f"Tracks: {len(result['tracks'])}")
    
    if result['recognition_result'] is not None:
        rec = result['recognition_result']
        logger.info(f"Recognition result:")
        logger.info(f"  Tracks processed: {rec['tracks_processed']}")
        logger.info(f"  Embeddings: {rec['embeddings_extracted']}")
        logger.info(f"  Avg quality: {rec['avg_quality']:.3f}")
    
    logger.info("✅ PASS")
    return True

def test_with_stats():
    """Test 3: Check recognition stage stats"""
    logger.info("\n" + "="*60)
    logger.info("Test 3: Recognition Stage Stats")
    logger.info("="*60)
    
    config = ConfigManager()
    config.load('config.yaml')
    orchestrator = PipelineOrchestrator(config)
    
    # Process a few frames
    for i in range(3):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        orchestrator.process_frame(frame)
    
    # Get stats
    stats = orchestrator.recognition_stage.get_stats()
    
    logger.info(f"Recognition Stats:")
    logger.info(f"  Total processed: {stats['total_processed']}")
    logger.info(f"  Alignments success: {stats['alignments_success']}")
    logger.info(f"  Alignments failed: {stats['alignments_failed']}")
    logger.info(f"  Recognitions success: {stats['recognitions_success']}")
    logger.info(f"  Quality too low: {stats['quality_too_low']}")
    
    if stats['alignments_success'] > 0:
        logger.info(f"  Avg alignment time: {stats['avg_alignment_time_ms']:.1f}ms")
    if stats['recognitions_success'] > 0:
        logger.info(f"  Avg recognition time: {stats['avg_recognition_time_ms']:.1f}ms")
    
    logger.info("✅ PASS")
    return True

def run_all_tests():
    """Run all tests"""
    logger.info("\n" + "="*70)
    logger.info("Pipeline Recognition Test Suite (Phase 3A Day 6)")
    logger.info("="*70)
    
    tests = [
        test_initialization,
        test_empty_frame,
        test_with_stats,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"❌ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    logger.info("\n" + "="*70)
    logger.info(f"Results: {passed}/{len(tests)} passed")
    if failed > 0:
        logger.error(f"❌ {failed} test(s) FAILED")
    else:
        logger.info("✅ All tests PASSED")
    logger.info("="*70)
    
    return failed == 0

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
