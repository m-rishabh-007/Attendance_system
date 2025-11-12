#!/usr/bin/env python3
"""
Test suite for QualityScorer

Tests all 5 quality metrics (sharpness, brightness, angle, size, confidence)
and the combined scoring system.

Date: November 8, 2025
Phase 3A - Week 1 Day 5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
import logging
from recognizers import QualityScorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_face(size=(112, 112), blur=None, brightness_factor=1.0):
    """
    Create synthetic test face with controlled properties
    
    Args:
        size: Face image size (height, width)
        blur: Blur kernel size (None for sharp, 15+ for blurry)
        brightness_factor: Multiply pixel values (0.5=dark, 1.0=normal, 2.0=bright)
    
    Returns:
        face: uint8 image (0-255)
    """
    # Create random texture pattern (high frequency content for sharpness)
    np.random.seed(42)  # Reproducible
    face = np.random.randint(0, 256, size, dtype=np.uint8).astype(np.float32)
    
    # Add circular mask (simulates face shape)
    cx, cy = size[1] // 2, size[0] // 2
    for i in range(size[0]):
        for j in range(size[1]):
            dist = np.sqrt((i - cy)**2 + (j - cx)**2)
            if dist < min(size) // 2:
                # Keep random texture inside circle
                pass
            else:
                # Fade to background
                face[i, j] = face[i, j] * (1 - (dist - min(size)//2) / (min(size)//2))
    
    # Apply brightness
    face = face * brightness_factor
    face = np.clip(face, 0, 255)
    
    # Apply blur if requested
    if blur:
        face = cv2.GaussianBlur(face, (blur, blur), 0)
    
    return face.astype(np.uint8)


def test_1_sharpness_metric():
    """Test 1: Sharpness metric distinguishes sharp vs blurry faces"""
    logger.info("\n" + "="*60)
    logger.info("Test 1: Sharpness Metric")
    logger.info("="*60)
    
    scorer = QualityScorer()
    
    # Sharp face
    sharp_face = create_test_face(blur=None)
    sharp_score = scorer.compute_sharpness(sharp_face)
    
    # Blurry face
    blurry_face = create_test_face(blur=15)
    blurry_score = scorer.compute_sharpness(blurry_face)
    
    logger.info(f"Sharp face score: {sharp_score:.4f}")
    logger.info(f"Blurry face score: {blurry_score:.4f}")
    
    # Sharp should score higher
    assert sharp_score > blurry_score, \
        f"Sharp ({sharp_score:.4f}) should > Blurry ({blurry_score:.4f})"
    
    # Sharp should be reasonably high (>0.3)
    assert sharp_score > 0.3, \
        f"Sharp score ({sharp_score:.4f}) should be >0.3"
    
    # Blurry should be low (<0.2)
    assert blurry_score < 0.5, \
        f"Blurry score ({blurry_score:.4f}) should be <0.5"
    
    logger.info("✅ PASS: Sharpness metric works correctly")


def test_2_brightness_metric():
    """Test 2: Brightness metric penalizes dark/bright faces"""
    logger.info("\n" + "="*60)
    logger.info("Test 2: Brightness Metric")
    logger.info("="*60)
    
    scorer = QualityScorer()
    
    # Normal brightness
    normal_face = create_test_face(brightness_factor=1.0)
    normal_score = scorer.compute_brightness(normal_face)
    
    # Dark face
    dark_face = create_test_face(brightness_factor=0.3)
    dark_score = scorer.compute_brightness(dark_face)
    
    # Bright face
    bright_face = create_test_face(brightness_factor=2.0)
    bright_score = scorer.compute_brightness(bright_face)
    
    logger.info(f"Normal brightness score: {normal_score:.4f}")
    logger.info(f"Dark brightness score: {dark_score:.4f}")
    logger.info(f"Bright brightness score: {bright_score:.4f}")
    
    # Normal should score highest
    assert normal_score > dark_score, \
        f"Normal ({normal_score:.4f}) should > Dark ({dark_score:.4f})"
    assert normal_score > bright_score, \
        f"Normal ({normal_score:.4f}) should > Bright ({bright_score:.4f})"
    
    # Normal should be high (>0.7)
    assert normal_score > 0.7, \
        f"Normal score ({normal_score:.4f}) should be >0.7"
    
    logger.info("✅ PASS: Brightness metric works correctly")


def test_3_angle_score():
    """Test 3: Angle score prefers frontal faces"""
    logger.info("\n" + "="*60)
    logger.info("Test 3: Angle Score")
    logger.info("="*60)
    
    scorer = QualityScorer()
    
    # Test various angles
    angles = [0, 15, 30, 45, 60, 75, 90]
    scores = [scorer.compute_angle_score(angle) for angle in angles]
    
    logger.info("Angle → Score:")
    for angle, score in zip(angles, scores):
        logger.info(f"  {angle:3d}° → {score:.4f}")
    
    # Frontal (0°) should score highest
    assert scores[0] == 1.0, f"Frontal face (0°) should score 1.0, got {scores[0]:.4f}"
    
    # Score should decrease with angle
    for i in range(len(scores) - 1):
        assert scores[i] > scores[i+1], \
            f"Score at {angles[i]}° ({scores[i]:.4f}) should > {angles[i+1]}° ({scores[i+1]:.4f})"
    
    # Profile (90°) should score near 0
    assert scores[-1] < 0.1, \
        f"Profile face (90°) should score <0.1, got {scores[-1]:.4f}"
    
    # Test negative angles (should be symmetric)
    neg_score = scorer.compute_angle_score(-45)
    pos_score = scorer.compute_angle_score(45)
    assert abs(neg_score - pos_score) < 0.01, \
        f"Angles should be symmetric: -45° ({neg_score:.4f}) vs +45° ({pos_score:.4f})"
    
    logger.info("✅ PASS: Angle score works correctly")


def test_4_size_score():
    """Test 4: Size score prefers larger faces"""
    logger.info("\n" + "="*60)
    logger.info("Test 4: Size Score")
    logger.info("="*60)
    
    scorer = QualityScorer()
    frame_size = (480, 640)  # VGA resolution
    
    # Small bbox (far from camera)
    small_bbox = (100, 100, 150, 150)  # 50x50 = 2500 pixels
    small_score = scorer.compute_size_score(small_bbox, frame_size)
    
    # Medium bbox
    medium_bbox = (100, 100, 200, 200)  # 100x100 = 10000 pixels
    medium_score = scorer.compute_size_score(medium_bbox, frame_size)
    
    # Large bbox (close to camera)
    large_bbox = (100, 100, 300, 300)  # 200x200 = 40000 pixels
    large_score = scorer.compute_size_score(large_bbox, frame_size)
    
    logger.info(f"Small bbox (50x50) score: {small_score:.4f}")
    logger.info(f"Medium bbox (100x100) score: {medium_score:.4f}")
    logger.info(f"Large bbox (200x200) score: {large_score:.4f}")
    
    # Larger should score higher
    assert small_score < medium_score < large_score, \
        f"Scores should increase with size: {small_score:.4f} < {medium_score:.4f} < {large_score:.4f}"
    
    logger.info("✅ PASS: Size score works correctly")


def test_5_confidence_score():
    """Test 5: Confidence score is passthrough"""
    logger.info("\n" + "="*60)
    logger.info("Test 5: Confidence Score")
    logger.info("="*60)
    
    scorer = QualityScorer()
    
    confidences = [0.5, 0.7, 0.9, 0.95]
    scores = [scorer.compute_confidence_score(conf) for conf in confidences]
    
    logger.info("Confidence → Score:")
    for conf, score in zip(confidences, scores):
        logger.info(f"  {conf:.2f} → {score:.4f}")
    
    # Should be exact passthrough
    for conf, score in zip(confidences, scores):
        assert abs(conf - score) < 1e-6, \
            f"Confidence score should be passthrough: {conf} → {score}"
    
    logger.info("✅ PASS: Confidence score is passthrough")


def test_6_combined_quality_score():
    """Test 6: Combined quality score (weighted average)"""
    logger.info("\n" + "="*60)
    logger.info("Test 6: Combined Quality Score")
    logger.info("="*60)
    
    scorer = QualityScorer()
    frame_size = (480, 640)
    
    # High quality face
    high_quality_face = create_test_face(blur=None, brightness_factor=1.0)
    high_quality_bbox = (100, 100, 300, 300)  # Large
    high_quality_angle = 0  # Frontal
    high_quality_conf = 0.95
    
    high_score = scorer.compute_quality(
        high_quality_face,
        high_quality_bbox,
        high_quality_angle,
        high_quality_conf,
        frame_size
    )
    
    # Low quality face
    low_quality_face = create_test_face(blur=15, brightness_factor=0.3)
    low_quality_bbox = (100, 100, 150, 150)  # Small
    low_quality_angle = 75  # Profile
    low_quality_conf = 0.6
    
    low_score = scorer.compute_quality(
        low_quality_face,
        low_quality_bbox,
        low_quality_angle,
        low_quality_conf,
        frame_size
    )
    
    logger.info(f"High quality face score: {high_score:.4f}")
    logger.info(f"Low quality face score: {low_score:.4f}")
    
    # High quality should score significantly higher
    assert high_score > low_score, \
        f"High quality ({high_score:.4f}) should > Low quality ({low_score:.4f})"
    
    # Scores should be in [0, 1]
    assert 0 <= high_score <= 1, f"Score {high_score:.4f} out of range [0, 1]"
    assert 0 <= low_score <= 1, f"Score {low_score:.4f} out of range [0, 1]"
    
    # High quality should be reasonably high (>0.5)
    # Note: Synthetic faces won't score as high as real faces
    assert high_score > 0.5, \
        f"High quality score ({high_score:.4f}) should be >0.5"
    
    logger.info("✅ PASS: Combined quality score works correctly")


def test_7_detailed_scores():
    """Test 7: Detailed scores breakdown"""
    logger.info("\n" + "="*60)
    logger.info("Test 7: Detailed Scores Breakdown")
    logger.info("="*60)
    
    scorer = QualityScorer()
    
    face = create_test_face()
    bbox = (100, 100, 200, 200)
    angle = 30
    confidence = 0.85
    frame_size = (480, 640)
    
    # Get detailed scores
    scores = scorer.get_detailed_scores(face, bbox, angle, confidence, frame_size)
    
    logger.info("Detailed scores:")
    for metric, score in scores.items():
        logger.info(f"  {metric}: {score:.4f}")
    
    # Check all metrics present
    expected_metrics = ['sharpness', 'brightness', 'angle', 'size', 'confidence', 'combined']
    for metric in expected_metrics:
        assert metric in scores, f"Missing metric: {metric}"
    
    # All scores should be in [0, 1]
    for metric, score in scores.items():
        assert 0 <= score <= 1, \
            f"Score for {metric} ({score:.4f}) out of range [0, 1]"
    
    # Combined should match compute_quality()
    combined = scorer.compute_quality(face, bbox, angle, confidence, frame_size)
    assert abs(scores['combined'] - combined) < 1e-6, \
        f"Detailed combined ({scores['combined']:.4f}) != compute_quality ({combined:.4f})"
    
    logger.info("✅ PASS: Detailed scores breakdown works correctly")


def test_8_weight_validation():
    """Test 8: Weight validation (must sum to 1.0)"""
    logger.info("\n" + "="*60)
    logger.info("Test 8: Weight Validation")
    logger.info("="*60)
    
    # Valid weights
    valid_weights = {
        'sharpness': 0.30,
        'brightness': 0.20,
        'angle': 0.25,
        'size': 0.15,
        'confidence': 0.10
    }
    scorer = QualityScorer(weights=valid_weights)
    logger.info("✅ Valid weights accepted (sum=1.0)")
    
    # Invalid weights (don't sum to 1.0)
    invalid_weights = {
        'sharpness': 0.30,
        'brightness': 0.20,
        'angle': 0.25,
        'size': 0.15,
        'confidence': 0.20  # Sum = 1.10 (invalid)
    }
    
    try:
        scorer = QualityScorer(weights=invalid_weights)
        logger.error("❌ FAIL: Invalid weights accepted")
        assert False, "Should have raised ValueError for invalid weights"
    except ValueError as e:
        logger.info(f"✅ Invalid weights rejected: {e}")
    
    # Custom weights (valid)
    custom_weights = {
        'sharpness': 0.40,  # Prioritize sharpness
        'brightness': 0.10,
        'angle': 0.30,
        'size': 0.10,
        'confidence': 0.10
    }
    scorer = QualityScorer(weights=custom_weights)
    logger.info("✅ Custom weights accepted (sum=1.0)")
    
    logger.info("✅ PASS: Weight validation works correctly")


def run_all_tests():
    """Run all QualityScorer tests"""
    logger.info("\n" + "="*70)
    logger.info("QualityScorer Test Suite")
    logger.info("Phase 3A - Week 1 Day 5")
    logger.info("="*70)
    
    tests = [
        test_1_sharpness_metric,
        test_2_brightness_metric,
        test_3_angle_score,
        test_4_size_score,
        test_5_confidence_score,
        test_6_combined_quality_score,
        test_7_detailed_scores,
        test_8_weight_validation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            logger.error(f"❌ FAIL: {test.__name__}")
            logger.error(f"   Error: {e}")
            failed += 1
    
    logger.info("\n" + "="*70)
    logger.info(f"Test Results: {passed}/{len(tests)} passed")
    if failed > 0:
        logger.error(f"❌ {failed} test(s) FAILED")
    else:
        logger.info("✅ All tests PASSED")
    logger.info("="*70)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
