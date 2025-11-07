# Phase 3 Implementation Plan: Face Recognition

**Status**: 📋 Planning Complete | 🚧 Implementation Pending

**Created**: November 7, 2025  
**Target Completion**: TBD

---

## 📋 Overview

Phase 3 implements face recognition using ArcFace embeddings with quality-aware caching for optimal performance and accuracy.

**Key Goals:**
1. Generate 512-dim face embeddings using ArcFace model
2. Implement quality-aware caching to use best quality frames
3. Optimize for Raspberry Pi (CPU-only, INT8 quantization)
4. Achieve >95% recognition accuracy on frontal faces

---

## 🎯 Core Requirements

### 1. Face Recognition Module

**Model**: ArcFace (ResNet-50 or MobileFaceNet backbone)
- **Input**: 112x112 RGB aligned face (from Phase 2)
- **Output**: 512-dimensional embedding vector (L2-normalized)
- **Format**: ONNX Runtime (CPU optimized)
- **Quantization**: INT8 for Raspberry Pi deployment

**Files to Create:**
```
recognizers/
├── __init__.py                 # Module exports
├── base_recognizer.py          # Abstract interface (Strategy pattern)
├── arcface_recognizer.py       # ArcFace implementation (~300 lines)
├── factory.py                  # RecognizerFactory (~80 lines)
└── README.md                   # Documentation
```

**Key Methods:**
```python
class ArcFaceRecognizer(BaseRecognizer):
    def get_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """Generate 512-dim embedding from aligned face."""
        
    def get_embeddings_batch(self, aligned_faces: List[np.ndarray]) -> np.ndarray:
        """Batch processing for multiple faces."""
        
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Cosine similarity between embeddings."""
```

---

## 🔥 Critical Feature: Quality-Aware Caching

### Problem Statement

**WITHOUT Quality-Aware Caching:**
```
Frame 1:  Blurry face (quality=0.4) → Align → Recognize → Cache ❌
Frame 30: Sharp face (quality=0.9) → Skip (cached) ❌

Result: Using low-quality embedding → 60% accuracy ⚠️
CPU Waste: 30 alignments/sec × 300 frames = 9,000 redundant operations
```

**WITH Quality-Aware Caching:**
```
Frames 1-10: Sample all → Select best quality (0.9) → Cache ✅
Frames 11+:  Use cached embedding ✅

Result: Using best-quality embedding → 95% accuracy ✅
CPU Savings: 97% (10 vs 300 alignments)
```

---

### Implementation: QualityAwareCache Class

**File**: `pipeline/quality_cache.py` (~400 lines)

**Core Logic:**
```python
class QualityAwareCache:
    """
    Intelligent caching system that selects best quality frames per track.
    
    Strategy:
    1. Sample first N frames per track (default: 10)
    2. Compute quality score for each frame
    3. Cache BEST quality alignment + embedding
    4. Upgrade cache if significantly better quality found
    5. Periodic re-check for lighting changes (every 5 min)
    """
    
    def __init__(self, config):
        self.sampling_frames = config.get('quality_sampling_frames', 10)
        self.min_quality = config.get('min_quality_threshold', 0.6)
        self.quality_improvement_delta = config.get('quality_improvement_delta', 0.1)
        self.periodic_recheck_interval = config.get('periodic_recheck_seconds', 300)
        
        # Cache storage
        self.aligned_faces = {}      # {track_id: best_aligned_face}
        self.embeddings = {}          # {track_id: best_embedding}
        self.quality_scores = {}      # {track_id: quality_score}
        self.sample_counts = {}       # {track_id: frames_sampled}
        self.identities = {}          # {track_id: person_name}
```

---

### Quality Scoring Algorithm

**Metrics** (weighted combination):

| Metric | Weight | Description | Implementation |
|--------|--------|-------------|----------------|
| **Sharpness** | 0.30 | Laplacian variance (blur detection) | `cv2.Laplacian(gray, cv2.CV_64F).var()` |
| **Brightness** | 0.20 | Histogram analysis (avoid over/underexposure) | Optimal range: 100-180 |
| **Face Angle** | 0.25 | Yaw angle from aligner (frontal=best) | From `aligner.last_angle` |
| **Face Size** | 0.15 | Bbox area (larger=more detail) | Normalize to 224×224 ideal |
| **Detection Conf** | 0.10 | YOLO confidence score | From `track.confidence` |

**Formula:**
```python
def compute_quality(aligned_face, track, aligner) -> float:
    scores = []
    
    # 1. Sharpness (most important)
    gray = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = min(1.0, laplacian_var / 500.0)
    scores.append(sharpness * 0.30)
    
    # 2. Brightness
    brightness = np.mean(gray)
    if 100 <= brightness <= 180:
        brightness_score = 1.0
    elif brightness < 100:
        brightness_score = brightness / 100.0
    else:
        brightness_score = max(0.0, 1.0 - (brightness - 180) / 75.0)
    scores.append(brightness_score * 0.20)
    
    # 3. Face angle (frontal=1.0, profile=0.0)
    angle = abs(aligner.last_angle)
    angle_score = max(0.0, 1.0 - angle / 90.0)
    scores.append(angle_score * 0.25)
    
    # 4. Face size
    x1, y1, x2, y2 = track.bbox
    face_area = (x2 - x1) * (y2 - y1)
    size_score = min(1.0, face_area / 50000.0)
    scores.append(size_score * 0.15)
    
    # 5. Detection confidence
    scores.append(track.confidence * 0.10)
    
    return sum(scores)  # Range: 0.0-1.0
```

---

### Caching Strategy

**Sampling Window** (first 10 frames):
```python
if sample_count < 10:
    # Always process during sampling window
    aligned_face = aligner.align(frame, track.bbox)
    quality = compute_quality(aligned_face, track, aligner)
    
    if quality > cached_quality + 0.1:
        # Upgrade cache (10% improvement threshold)
        embedding = recognizer.get_embedding(aligned_face)
        cache_update(track.id, aligned_face, embedding, quality)
```

**After Sampling** (frames 11+):
```python
if track.id in cache:
    # Use cached embedding
    embedding = cache.embeddings[track.id]
    quality = cache.quality_scores[track.id]
    
    # Periodic re-check (every 5 min for lighting changes)
    if time.time() - last_recheck > 300:
        # Re-sample to check if quality improved
        aligned_face = aligner.align(frame, track.bbox)
        new_quality = compute_quality(aligned_face, track, aligner)
        
        if new_quality > quality + 0.1:
            # Lighting improved, upgrade cache
            embedding = recognizer.get_embedding(aligned_face)
            cache_update(track.id, aligned_face, embedding, new_quality)
```

**Poor Quality Handling:**
```python
if cached_quality < 0.8:
    # Quality not ideal, keep trying every 10th frame
    if sample_count % 10 == 0:
        aligned_face = aligner.align(frame, track.bbox)
        quality = compute_quality(aligned_face, track, aligner)
        
        if quality > cached_quality + 0.1:
            # Found better quality, upgrade
            embedding = recognizer.get_embedding(aligned_face)
            cache_update(track.id, aligned_face, embedding, quality)
```

---

## 🏗️ Pipeline Integration

### Updated Orchestrator

**File**: `pipeline/orchestrator.py` (modifications)

```python
class PipelineOrchestrator:
    def __init__(self, config):
        # Existing stages
        self.detection_stage = DetectionStage(config.get_section('detector'))
        self.tracking_stage = TrackingStage(config.get_section('tracker'))
        
        # NEW: Phase 3 stages
        self.alignment_stage = AlignmentStage(config.get_section('aligner'))
        self.recognition_stage = RecognitionStage(config.get_section('recognition'))
        
        # NEW: Quality-aware cache
        self.quality_cache = QualityAwareCache(config.get_section('recognition'))
        
        logger.info("✅ Pipeline with Recognition Ready")
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process frame through full pipeline: Detect → Track → Align → Recognize
        """
        # Stage 1: Detection
        detections = self.detection_stage.process(frame)
        
        # Stage 2: Tracking (with persist=True for stable IDs)
        tracks = self.tracking_stage.process(frame, detections)
        
        # Stage 3+4: Alignment + Recognition (with quality-aware caching)
        results = []
        
        for track in tracks:
            # Check if cached
            if track.id in self.quality_cache.identities:
                # Already identified
                results.append({
                    'track_id': track.id,
                    'bbox': track.bbox,
                    'person_name': self.quality_cache.identities[track.id],
                    'quality': self.quality_cache.quality_scores[track.id],
                    'cached': True
                })
                continue
            
            # Process with quality-aware caching
            result = self.quality_cache.process_track(
                track, frame, 
                self.alignment_stage.aligner,
                self.recognition_stage.recognizer
            )
            
            if result['embedding'] is not None:
                # Match with database (Phase 4)
                person_name = self.match_with_database(result['embedding'])
                self.quality_cache.identities[track.id] = person_name
                
                results.append({
                    'track_id': track.id,
                    'bbox': track.bbox,
                    'person_name': person_name,
                    'quality': result['quality'],
                    'cached': result['is_cached']
                })
        
        return {
            'tracks': results,
            'frame_count': self.frame_count,
            'annotated_frame': self._annotate_frame(frame, results)
        }
```

---

## ⚙️ Configuration

### config.yaml Additions

```yaml
# Face Recognition Configuration
recognition:
  type: arcface
  model_path: models/arcface_mobilefacenet_int8.onnx
  embedding_size: 512
  
  # Quality-aware caching
  enable_cache: true
  quality_sampling_frames: 10        # Sample first 10 frames per track
  min_quality_threshold: 0.6         # Minimum acceptable quality
  quality_improvement_delta: 0.1     # Upgrade if 10% better
  periodic_recheck_seconds: 300      # Re-check every 5 minutes
  cache_ttl_seconds: 600             # Expire cache after 10 min idle
  
  # Quality metrics weights
  quality_weights:
    sharpness: 0.30     # Blur detection (most important)
    brightness: 0.20    # Lighting
    angle: 0.25         # Face pose (from Phase 2 angle estimation)
    size: 0.15          # Face resolution
    confidence: 0.10    # Detection quality
  
  # Performance
  batch_size: 1                      # No batching on Pi (memory limited)
  use_gpu: false                     # CPU-only for Pi
  num_threads: 4                     # Multi-threading
```

---

## 📊 Expected Performance

### Raspberry Pi 4 (with Quality-Aware Caching)

| Scenario | Without Cache | With Cache | Savings |
|----------|---------------|------------|---------|
| **1 person, 10 sec** | 300 align + 300 recognize | 10 align + 10 recognize | 97% |
| **CPU per person** | 27 seconds | 0.9 seconds | 30× faster |
| **10 people, 5 min** | 75 minutes CPU | 10 seconds CPU | 99.8% |
| **Recognition accuracy** | 60-80% (variable) | 95%+ (best quality) | +15-35% |

### Time Breakdown (Single Face, First Frame)

| Operation | Time (Pi 4) | Time (Laptop) |
|-----------|-------------|---------------|
| Detection (YOLO) | 130ms | 80ms |
| Tracking (BoT-SORT) | 10ms | 5ms |
| Alignment (MediaPipe) | 40ms | 25ms |
| **Recognition (ArcFace)** | **50ms** | **30ms** |
| Quality Scoring | 5ms | 3ms |
| Database Match (Phase 4) | 10ms | 5ms |
| **Total (First Frame)** | **245ms (~4 FPS)** | **148ms (~7 FPS)** |
| **Cached Frames** | **140ms (~7 FPS)** | **85ms (~12 FPS)** |

---

## 🧪 Testing Strategy

### Unit Tests

**File**: `tests/test_recognition.py`

```python
def test_arcface_embedding_generation():
    """Test embedding generation for single face."""
    
def test_quality_scoring():
    """Test quality score computation."""
    
def test_cache_upgrade():
    """Test cache upgrades when better quality found."""
    
def test_cache_expiration():
    """Test TTL-based cache cleanup."""
```

### Integration Test

**File**: `tests/test_quality_cache.py`

```python
def test_quality_aware_caching_pipeline():
    """
    Test full pipeline with quality-aware caching.
    
    Scenario:
    - Frame 1: Blurry face (quality=0.4)
    - Frame 5: Sharp face (quality=0.9)
    - Frame 30: Good face (quality=0.85)
    
    Expected:
    - Frame 1: Cached with quality=0.4
    - Frame 5: Upgraded to quality=0.9
    - Frame 30: Keep quality=0.9 (no downgrade)
    """
```

---

## 📋 Implementation Checklist

### Phase 3A: Core Recognition (Week 1)

- [ ] Create `recognizers/base_recognizer.py` (Strategy pattern)
- [ ] Create `recognizers/arcface_recognizer.py` (ONNX Runtime)
- [ ] Create `recognizers/factory.py`
- [ ] Download/convert ArcFace model to ONNX INT8
- [ ] Test embedding generation on sample faces
- [ ] Benchmark performance on Pi vs Laptop
- [ ] Update `config.yaml` with recognition section

### Phase 3B: Quality-Aware Caching (Week 2)

- [ ] Create `pipeline/quality_cache.py`
- [ ] Implement quality scoring algorithm
- [ ] Implement sampling window logic (first 10 frames)
- [ ] Implement cache upgrade logic (delta threshold)
- [ ] Implement periodic re-check (lighting changes)
- [ ] Add cache expiration (TTL-based cleanup)
- [ ] Test with various quality scenarios

### Phase 3C: Pipeline Integration (Week 2)

- [ ] Update `pipeline/orchestrator.py` with recognition stage
- [ ] Integrate quality-aware cache into orchestrator
- [ ] Add AlignmentStage wrapper (if needed)
- [ ] Add RecognitionStage wrapper
- [ ] Test end-to-end pipeline
- [ ] Measure performance (FPS, CPU usage)

### Phase 3D: Testing & Documentation (Week 3)

- [ ] Create `tests/test_recognition.py` (unit tests)
- [ ] Create `tests/test_quality_cache.py` (integration tests)
- [ ] Update `recognizers/README.md` with documentation
- [ ] Update `docs/DEVELOPER_GUIDE.md` with Phase 3
- [ ] Update `.github/copilot-instructions.md`
- [ ] Create demo video showing quality-aware caching

---

## 🚨 Known Challenges & Solutions

### Challenge 1: ArcFace Model Availability

**Problem**: Pre-trained ArcFace INT8 ONNX model may not exist

**Solutions**:
1. Use InsightFace model zoo (has pre-trained models)
2. Convert PyTorch ArcFace to ONNX manually
3. Use MobileFaceNet (lighter, faster for Pi)
4. Fallback to FaceNet if ArcFace unavailable

### Challenge 2: Quality Scoring Calibration

**Problem**: Quality thresholds may need tuning per deployment

**Solutions**:
1. Start with empirical defaults (tested in lab)
2. Add calibration mode (analyze 100 frames, auto-tune)
3. Expose thresholds in config for easy adjustment
4. Log quality distributions for offline analysis

### Challenge 3: Memory Usage on Pi

**Problem**: Caching embeddings for many tracks may exhaust RAM

**Solutions**:
1. Implement LRU eviction (keep only N most recent)
2. TTL-based expiration (remove idle tracks after 10 min)
3. Limit max cache size (e.g., 100 tracks = ~200KB)
4. Serialize to disk if cache exceeds threshold

---

## 🎯 Success Criteria

Phase 3 is complete when:

- ✅ ArcFace model generates 512-dim embeddings
- ✅ Quality-aware caching selects best frames (>0.9 quality)
- ✅ Recognition accuracy >95% on frontal faces
- ✅ CPU usage <30% on Pi (with caching)
- ✅ FPS >5 on Pi, >10 on laptop
- ✅ All unit and integration tests pass
- ✅ Documentation complete

---

## 📚 References

### ArcFace Papers & Implementations

- **Paper**: [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
- **InsightFace**: https://github.com/deepinsight/insightface
- **ONNX Models**: https://github.com/onnx/models/tree/main/vision/body_analysis/arcface

### Related Documentation

- **Phase 2 (Alignment)**: `aligners/README.md`
- **Architecture**: `ARCHITECTURE.md`
- **Developer Guide**: `docs/DEVELOPER_GUIDE.md`
- **Pipeline**: `pipeline/README.md`

---

**Status**: 📋 Planning Document Complete

**Next Step**: Begin Phase 3A implementation (Core Recognition)

**Last Updated**: November 7, 2025
