# AI Agent Instructions - Face Attendance System

## Project Overview

Real-time face detection and tracking pipeline using **YOLOv8n NCNN FP16** + **BoT-SORT** for persistent face IDs. **Async multiprocessing architecture** prevents camera blocking. Optimized for Raspberry Pi 5 and laptops with Docker and virtual environment support.

**Current Status**: ✅ V3 HYBRID Architecture | ✅ Async Multiprocessing (19.36 FPS) | ✅ NCNN Optimization | ✅ Detection + Tracking + Alignment + Recognition | ✅ Quality-Aware Caching | 🚧 Database + Attendance (Phase 4)

**Architecture**: V3 HYBRID + **Producer-Consumer Multiprocessing**
- Process 1 (Main Loop): Camera → YOLO NCNN → BoT-SORT @ 19.36 FPS (never blocks)
- Process 2 (AI Worker): MediaPipe Align → AuraFace Recognition (parallel)
- Leaky Bucket queue prevents camera blocking
- Result Queue + Local Cache for instant name display

**Key Features**: 
- **Async multiprocessing** - 55% jitter reduction, 4x better ID stability
- **NCNN FP16** - 2-3x faster than TFLite on ARM (Pi 5 optimized)
- All 4 design patterns: Singleton, Factory, Strategy, Observer
- Modular folder structure: pipeline/, detectors/, tracking/, aligners/, recognizers/, database/, server/
- Config-driven (NO hardcoded values)

---

## Critical Development Rules

### Rule 1: Documentation for All Major Changes

When introducing new features, modules, or significant changes:

1. Code Comments (Required):
   - Add comprehensive docstrings to all new classes and methods
   - Use clear inline comments for complex logic
   - Explain WHY, not just WHAT the code does

2. README Updates (Required):
   - Update relevant README.md files in affected directories
   - Add usage examples for new features
   - Document configuration changes

3. Changelog (Recommended):
   - Document breaking changes
   - List new features and improvements

### Rule 2: Object-Oriented Programming (OOP) Required

All new code MUST follow OOP principles. Use classes not functions, include type hints, follow design patterns (Factory, Singleton, Observer, Strategy).

### Rule 3: Documentation Maintenance

When modifying any component, UPDATE its documentation! If you modify detectors/yolo_detector.py, update detectors/README.md.

### Rule 4: Testing Requirements

Run smoke test before committing tracking changes:
- python tests/test_persistent_tracking.py
- Files requiring smoke test: detectors/yolo_detector.py, pipeline/orchestrator.py, tracking/* files

See docs/DEVELOPER_GUIDE.md for complete documentation.

---

## Phase Completion Status

### ✅ Phase 1: Face Detection + Tracking (COMPLETE)
**Completed**: November 4, 2025

**Components**:
- ✅ YOLOv8n INT8 TFLite detector (custom trained)
- ✅ BoT-SORT tracking with persist=True
- ✅ Detection + Tracking stages
- ✅ Performance: ~6 FPS on Raspberry Pi 4

**Over-Engineering Assessment**: 17% (EventSystem built for Phase 4, not used yet - acceptable)

---

### ✅ Phase 2: Face Alignment (COMPLETE)
**Completed**: November 7, 2025

**Components**:
- ✅ MediaPipe Face Mesh (468 landmarks)
- ✅ 5-point alignment to 112x112 (ArcFace standard)
- ✅ Angle estimation for debugging (yaw angle -90° to +90°)
- ✅ Test script with color-coded visualization
- ✅ Performance: ~40ms per face on Pi

**Enhancement**: Angle estimation added for deployment debugging:
- Log face angles to diagnose camera placement
- Color-coded visualization (Green=frontal, Yellow=semi-profile, Red=profile)
- Enables future adaptive alignment if needed

**Over-Engineering Assessment**: 23% (Angle estimation + test polish - acceptable for debugging value)

**Files**:
- `aligners/base_aligner.py` - Abstract interface (140 lines)
- `aligners/mediapipe_aligner.py` - Implementation (405 lines)
- `aligners/factory.py` - Factory pattern (65 lines)
- `tests/test_alignment.py` - Test with visualization (266 lines)
- `aligners/README.md` - Complete documentation

---

### ✅ Phase 3A: Face Recognition - Week 1 (COMPLETE)
**Status**: FP32 Baseline Complete | November 8, 2025

**Delivered**:
- ✅ BaseRecognizer abstract interface (367 lines, Strategy pattern)
- ✅ AuraFaceRecognizer FP32 (368 lines, Apache 2.0 license)
- ✅ RecognizerFactory (188 lines, config-driven)
- ✅ QualityScorer (547 lines, 5 metrics)
- ✅ RecognitionStage pipeline integration (410 lines, Facade pattern)
- ✅ Comprehensive testing (16/16 tests passing)
- ✅ Performance benchmarking (~4 FPS laptop, ~3.6 FPS Pi estimate)

**Quality-Aware Sampling**:
- 5 metrics: Sharpness (30%), Angle (25%), Brightness (20%), Size (15%), Confidence (10%)
- Sample first 10 frames, cache BEST quality (Week 2)
- Expected: 95%+ accuracy vs 60% with naive first-frame caching

**Performance Baseline**:
- Laptop: 4.02 FPS mean (249ms/frame)
- Pi 4 (estimated): 3.6 FPS (275ms/frame)
- Week 2 target: 7 FPS with INT8 quantization

**Architecture**: Clean separation of runtime (recognizers/) vs build-time (tools/quantization/)

**See**: 
- `docs/PHASE_3A_WEEK1_COMPLETE.md` - Complete Week 1 summary
- `recognizers/README.md` - Module documentation
- `tools/quantization/README.md` - Week 2 quantization workflow

---

### ✅ Phase 3A: Week 2 - Quality-Aware Caching + Benchmarking (COMPLETE)
**Status**: COMPLETE | November 20, 2025

**Week 2 Deliverables**:
- ✅ Day 1: Calibration data collection (97 samples, quality=0.327)
- ⏸️ Day 2-3: TFLite INT8 quantization DEFERRED to Pi deployment
  - Reason: Dependency conflicts (onnx-tf incompatible with TensorFlow 2.16 + MediaPipe)
  - Plan: Re-quantize on Pi with production calibration data (quality>0.6)
- ✅ Day 4-5: Quality-Aware Caching (COMPLETE)
  - ✅ QualityAwareCache module created (262 lines)
  - ✅ Comprehensive tests (331 lines, 6/6 passing)
  - ✅ RecognitionStage integration (70 lines of changes)
  - ✅ Config integration (config.yaml)
  - ✅ Performance: 98.3% hit rate (5 people × 60 frames)
  - ✅ CPU reduction: 98.3% (from 300 to 5 recognitions)
- ✅ Day 6-7: End-to-end testing + benchmarking (COMPLETE)
  - ✅ Track ID stability verified (100% persistent IDs)
  - ✅ Cache performance validated (96.7% hit rate on webcam)
  - ✅ Benchmark output improved (clear cache statistics)
  - ✅ Real-world testing: 6.59 FPS on laptop with caching

**Delivered**:
- `recognizers/quality_cache.py` (262 lines) - Cache implementation
- `tests/test_quality_cache.py` (331 lines) - Unit tests (6/6 passing)
- `tests/test_recognition_stage_caching.py` (170 lines) - Integration tests (3/3 passing)
- `tests/test_track_id_stability.py` (170 lines) - Track ID diagnostic tool
- `tests/benchmark_pipeline.py` (improved) - Clear cache performance reporting
- `pipeline/recognition_stage.py` (~70 lines of changes) - Cache integration
- `config.yaml` (cache configuration section)
- `docs/PHASE_3A_WEEK2_DAY4-5_COMPLETE.md` - Complete summary

**Final Performance Results**:
- **Cache hit rate**: 96.7% (single person, 30 frames)
- **CPU reduction**: 96.7% (1 recognition vs 30 without cache)
- **Track ID stability**: 100% (BoT-SORT with persist=True working perfectly)
- **FPS (laptop)**: 6.59 FPS mean (real webcam with recognition)
- **FPS (laptop, no faces)**: 8.37 FPS (detection/tracking only)
- **Overhead**: <1ms per frame (cache lookup)

**Performance Breakdown (Webcam Test - 62 frames, 135 tracks)**:
- Faces with embeddings: 18
- Alignments attempted: 73 (54% success rate)
- Recognitions performed: 1 (only once!)
- Quality rejections: 72 (quality < 0.5)
- Cache hits: 17 (saved 17 redundant recognitions)
- Cache effectiveness: 94.4% for recognized faces (1 extraction + 17 cache hits)

---

### ✅ Phase 3B: Async Multiprocessing + NCNN Optimization (COMPLETE)
**Status**: COMPLETE | December 1, 2025

**Delivered**:
- ✅ AsyncOrchestrator with Producer-Consumer pattern (526 lines)
- ✅ Process 1: Camera → YOLO NCNN → BoT-SORT (never blocks camera)
- ✅ Process 2: MediaPipe Align → AuraFace Recognition (parallel)
- ✅ Leaky Bucket queue (maxsize=5, put_nowait for fail-fast)
- ✅ Result Queue + Local Cache (zero IPC overhead for name display)
- ✅ NCNN FP16 @ 320x320 (2-3x faster than TFLite on ARM)
- ✅ YOLODetector runtime switching (ncnn/tflite/pt)
- ✅ Comprehensive benchmarking (async vs sequential comparison)

**Performance Results (Laptop - AMD Ryzen 5 3500U)**:
- **FPS**: 19.36 mean (15.11-21.72 range)
- **Frame time**: 50.78ms mean (P95: 60ms)
- **Jitter**: 10.53ms (55% reduction vs sequential 24.70ms)
- **Track ID stability**: EXCELLENT (0.55 switches/100 frames vs 2.27 sequential)
- **Queue drop rate**: 0.0% (worker keeping up perfectly)
- **Recognition latency**: P95 < 400ms

**Expected Pi 5 Performance**:
- 15-20 FPS with NCNN @ 320x320
- Same stability benefits (async prevents camera blocking)

**Architecture Benefits**:
- Main loop NEVER blocks (queue.put_nowait drops frames if full)
- Heavy AI processing (MediaPipe + AuraFace ~90ms) runs in parallel
- Prevents "blind spots" that cause BoT-SORT ID switching
- 4x better ID stability (0.55 vs 2.27 switches per 100 frames)

**Files**:
- `pipeline/async_orchestrator.py` (526 lines) - Producer-Consumer implementation
- `tests/benchmark_multiprocessing.py` (464 lines) - Async vs sequential benchmarks
- `tests/test_async_quick.py` (140 lines) - Quick functional test
- `detectors/yolo_detector.py` (updated) - Runtime parameter support
- `config.yaml` (updated) - Async mode config, NCNN runtime

**See**: Benchmark output showing 19.36 FPS, 0% drop rate, 55% jitter reduction

---

### 🚀 Phase 4: Database + Attendance System (IN PROGRESS)
**Status**: Starting Week 1 | December 2, 2025

**Goal**: Add database storage, enrollment, matching, and attendance marking

**Components**:
- Database layer (SQLite with 5 tables: persons, embeddings, attendance, sessions, logs)
- Face enrollment workflow (capture + store best quality embeddings)
- Similarity matching (cosine similarity search across stored embeddings)
- Attendance marking with cooldown logic (prevent duplicates within 60 min)
- EventSystem integration (Observer pattern for attendance events)

**Database Schema**:
- `persons`: Store enrolled individuals (name, email, department, status)
- `embeddings`: Multiple 512-dim embeddings per person (quality-scored)
- `attendance`: Log when person recognized (with cooldown)
- `attendance_sessions`: Track check-in/check-out
- `system_logs`: Unknown faces and errors

**New Modules** (to be created):
- `database/person_database.py` - Person CRUD operations
- `database/embedding_database.py` - Embedding storage + similarity search
- `database/attendance_database.py` - Attendance logging with cooldown
- `database/database_manager.py` - Singleton coordinator (Facade pattern)
- `pipeline/enrollment_stage.py` - Enrollment workflow
- `pipeline/attendance_stage.py` - Attendance marking logic

**Design Patterns**:
- Singleton: DatabaseManager (single DB instance)
- Facade: DatabaseManager (unified interface to sub-databases)
- Observer: EventSystem for attendance/enrollment events
- Strategy: Multiple matching algorithms possible

**Performance Considerations**:
- Linear search O(N) acceptable for <1000 people
- Future optimization: FAISS/Annoy for >1000 people
- Database indexing on person_id and timestamp columns

**Implementation Timeline**:
- Week 1: Database foundation (schema + 4 database classes + tests)
- Week 2: Pipeline integration (enrollment + attendance stages)
- Week 3: End-to-end testing + refinement

**See**: `docs/PHASE_4_IMPLEMENTATION_PLAN.md` for complete specification

---

### 🔮 Phase 5: API + Web Interface (FUTURE)
**Status**: Not Started

**Components**:
- FastAPI REST API
- Real-time WebSocket updates
- Admin dashboard
- Attendance reports

---

## Architecture Decisions

### Design Patterns Used

1. **Strategy Pattern**: BaseDetector, BaseAligner, BaseRecognizer (swap implementations)
2. **Factory Pattern**: DetectorFactory, AlignerFactory, RecognizerFactory (config-driven)
3. **Singleton Pattern**: ConfigManager (single config instance)
4. **Observer Pattern**: EventSystem (Phase 4 - attendance events)

### Key Architectural Principles

1. **Pipeline Orchestration**: Separates "what to do" from "how to do it"
2. **Config-Driven**: No hardcoded values, everything in config.yaml
3. **Modular**: Each stage independent, swappable
4. **Quality-Aware**: Cache best quality frames, not first frames (Phase 3)

---

## Performance Targets

### Laptop (Development - AMD Ryzen 5 3500U)

**Async Mode Performance (Achieved)**:
- **Main Loop FPS**: 19.36 mean (15.11-21.72 range)
- **Frame Time**: 50.78ms (P95: 60ms, P99: 79.74ms)
- **Jitter**: 10.53ms (STABLE - 55% reduction vs sequential)
- **Queue Drop Rate**: 0.0% (worker keeping up)
- **Track ID Stability**: 0.55 switches/100 frames (EXCELLENT)

### Raspberry Pi 5 (Production Target)

**Expected Performance with NCNN**:
- **Main Loop FPS**: 15-20 FPS (NCNN @ 320x320)
- **Detection**: 40-50ms (YOLO NCNN FP16, ARM NEON optimized)
- **Tracking**: 5-10ms (BoT-SORT with persist=True)
- **Alignment**: 40ms (MediaPipe, runs in worker process)
- **Recognition**: 50ms (AuraFace, runs in worker process)
- **Queue Drop Rate**: <5% (acceptable worker lag)

**Architecture Benefits**:
- Main loop NEVER blocks (async prevents camera freezing)
- Worker process handles heavy AI (90ms total, parallel to camera)
- Quality-aware caching reduces redundant processing
- Stable FPS prevents BoT-SORT ID switching

---

## Technical Debt Tracking

### Phase 1 + Phase 2 Over-Engineering: ~19%

**Acceptable Debt** (Will be used later):
- EventSystem (290 lines) - Phase 4 dependency ✅
- Angle estimation (130 lines) - Debugging value ✅

**Lesson**: Stricter MVP discipline in Phase 3

---

## Development Workflow

### Before Starting Phase 4

1. ✅ Phase 3A documented and complete
2. ✅ Phase 3B async multiprocessing complete
3. ✅ NCNN optimization complete (19.36 FPS validated)
4. 📋 Review `docs/PHASE_4_IMPLEMENTATION_PLAN.md`
5. 🚀 Begin with database schema and foundation classes

### Commit Guidelines

```bash
# Good commit message format:
git commit -m "feat: <component> - <brief description>

- Bullet list of changes
- Include performance notes if relevant
- Reference issue/phase number

Technical debt: <any over-engineering notes>
Files: X lines of code"
```

---

Last Updated: December 2, 2025 - Phase 3B Complete (Async Multiprocessing + NCNN) | Phase 4 Ready
