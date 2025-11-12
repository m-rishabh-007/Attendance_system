#!/usr/bin/env python3
"""
Test AuraFace Recognizer Implementation

This script tests the AuraFace recognizer implementation with a sample image.
Run this after downloading the model to verify everything works.

Usage:
    python tests/test_auraface_basic.py

Requirements:
    - Model downloaded to: models/recognition/auraface_resnet100_fp32.onnx
    - Test image with face (can be any image with face)

Author: Attendance System Team
Created: November 2025
Phase: 3A - Core Recognition (Week 1, Day 3-4)
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from recognizers import AuraFaceRecognizer, RecognizerFactory


def test_direct_instantiation():
    """Test 1: Direct recognizer instantiation."""
    print("\n" + "=" * 70)
    print("TEST 1: Direct Instantiation")
    print("=" * 70)
    
    model_path = project_root / "models" / "recognition" / "auraface_resnet100_fp32.onnx"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Please download the model first.")
        print("   Run: python3 scripts/download_auraface_model.py")
        return False
    
    try:
        # Create recognizer
        recognizer = AuraFaceRecognizer(
            model_path=str(model_path),
            embedding_size=512,
            input_size=(112, 112),
            quantized=False,
            num_threads=4
        )
        print(f"✅ Created recognizer: {recognizer}")
        
        # Load model
        print("\nLoading model...")
        recognizer.load_model()
        print(f"✅ Model loaded: is_loaded={recognizer.is_loaded}")
        
        # Get model info
        info = recognizer.get_model_info()
        print("\nModel Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_factory_pattern():
    """Test 2: Factory pattern creation."""
    print("\n" + "=" * 70)
    print("TEST 2: Factory Pattern")
    print("=" * 70)
    
    try:
        # Load config (simplified version)
        config = {
            'model_type': 'auraface',
            'model_path': str(project_root / "models" / "recognition" / "auraface_resnet100_fp32.onnx"),
            'embedding_size': 512,
            'input_size': [112, 112],
            'num_threads': 4,
            'quantized': False
        }
        
        # Create recognizer using factory
        print("Creating recognizer via factory...")
        recognizer = RecognizerFactory.create(config)
        print(f"✅ Factory created: {type(recognizer).__name__}")
        
        # Load model
        recognizer.load_model()
        print(f"✅ Model loaded via factory")
        
        # Test get_supported_types
        supported = RecognizerFactory.get_supported_types()
        print(f"\nSupported recognizer types: {supported}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_preprocessing():
    """Test 3: Preprocessing pipeline."""
    print("\n" + "=" * 70)
    print("TEST 3: Preprocessing")
    print("=" * 70)
    
    try:
        # Create recognizer
        model_path = project_root / "models" / "recognition" / "auraface_resnet100_fp32.onnx"
        recognizer = AuraFaceRecognizer(model_path=str(model_path))
        recognizer.load_model()
        
        # Create dummy face image (112x112 RGB)
        dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        print(f"Input face shape: {dummy_face.shape}, dtype: {dummy_face.dtype}")
        
        # Preprocess
        preprocessed = recognizer.preprocess(dummy_face)
        print(f"\nPreprocessed shape: {preprocessed.shape}")
        print(f"Preprocessed dtype: {preprocessed.dtype}")
        print(f"Value range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
        
        # Verify expected properties
        assert preprocessed.shape == (1, 3, 112, 112), f"Expected (1, 3, 112, 112), got {preprocessed.shape}"
        assert preprocessed.dtype == np.float32, f"Expected float32, got {preprocessed.dtype}"
        assert -1.1 <= preprocessed.min() <= -0.9, f"Min value out of range: {preprocessed.min()}"
        assert 0.9 <= preprocessed.max() <= 1.1, f"Max value out of range: {preprocessed.max()}"
        
        print("\n✅ Preprocessing validates correctly")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embedding_extraction():
    """Test 4: Embedding extraction with dummy face."""
    print("\n" + "=" * 70)
    print("TEST 4: Embedding Extraction")
    print("=" * 70)
    
    try:
        # Create recognizer
        model_path = project_root / "models" / "recognition" / "auraface_resnet100_fp32.onnx"
        recognizer = AuraFaceRecognizer(model_path=str(model_path))
        recognizer.load_model()
        
        # Create dummy face
        dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        
        # Extract embedding
        print("Extracting embedding from dummy face...")
        embedding = recognizer.get_embedding(dummy_face)
        
        if embedding is None:
            print("❌ Embedding extraction returned None")
            return False
        
        print(f"\n✅ Embedding extracted!")
        print(f"   Shape: {embedding.shape}")
        print(f"   Dtype: {embedding.dtype}")
        print(f"   L2 norm: {np.linalg.norm(embedding):.6f}")
        print(f"   Value range: [{embedding.min():.4f}, {embedding.max():.4f}]")
        
        # Verify properties
        assert embedding.shape == (512,), f"Expected (512,), got {embedding.shape}"
        assert embedding.dtype == np.float32, f"Expected float32, got {embedding.dtype}"
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5), \
            f"L2 norm should be 1.0, got {np.linalg.norm(embedding)}"
        
        print("\n✅ Embedding properties validated")
        
        # Test consistency (same input = same output)
        print("\nTesting consistency...")
        embedding2 = recognizer.get_embedding(dummy_face)
        if embedding2 is None:
            print("❌ Failed to extract second embedding")
            return False
        
        similarity = np.dot(embedding, embedding2)
        print(f"Self-similarity: {similarity:.6f}")
        assert np.isclose(similarity, 1.0, atol=1e-5), \
            f"Same face should have similarity=1.0, got {similarity}"
        
        print("✅ Consistency validated")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_similarity_comparison():
    """Test 5: Compare embeddings from different faces."""
    print("\n" + "=" * 70)
    print("TEST 5: Similarity Comparison")
    print("=" * 70)
    
    try:
        # Create recognizer
        model_path = project_root / "models" / "recognition" / "auraface_resnet100_fp32.onnx"
        recognizer = AuraFaceRecognizer(model_path=str(model_path))
        recognizer.load_model()
        
        # Create two different dummy faces
        face1 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        face2 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        
        # Extract embeddings
        emb1 = recognizer.get_embedding(face1)
        emb2 = recognizer.get_embedding(face2)
        
        if emb1 is None or emb2 is None:
            print("❌ Failed to extract embeddings")
            return False
        
        # Compute similarity
        similarity = np.dot(emb1, emb2)
        print(f"\nSimilarity between different random faces: {similarity:.4f}")
        print(f"Expected: ~0.0 to 0.3 (random faces should be different)")
        
        # Test self-similarity
        self_sim = np.dot(emb1, emb1)
        print(f"Self-similarity of face1: {self_sim:.6f}")
        assert np.isclose(self_sim, 1.0, atol=1e-5), f"Self-similarity should be 1.0"
        
        print("\n✅ Similarity computation works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("AURAFACE RECOGNIZER IMPLEMENTATION TEST")
    print("=" * 70)
    print("\nThis test verifies the AuraFace recognizer implementation.")
    print("Note: Uses dummy random faces (not real face detection).")
    
    results = []
    
    # Run tests
    results.append(("Direct Instantiation", test_direct_instantiation()))
    results.append(("Factory Pattern", test_factory_pattern()))
    results.append(("Preprocessing", test_preprocessing()))
    results.append(("Embedding Extraction", test_embedding_extraction()))
    results.append(("Similarity Comparison", test_similarity_comparison()))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! AuraFace recognizer is working correctly.")
        print("\nNext steps:")
        print("1. Test with real face images (from Phase 2 alignment)")
        print("2. Implement QualityScorer (Day 5)")
        print("3. Integrate into pipeline (Day 6-7)")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
