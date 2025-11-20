# AI Agent Instructions - Face Attendance System

## Project Overview

Real-time face detection and tracking pipeline using **YOLOv8n INT8 TFLite** + **BoT-SORT** for persistent face IDs. Optimized for Raspberry Pi and laptops with Docker and virtual environment support.

**Current Status**: ✅ V3 HYBRID Architecture | ✅ Detection + Tracking + Alignment + Recognition | ✅ Quality-Aware Caching (96.7% CPU reduction) | 🚧 Database + Attendance + API (Phase 4-5)

**Architecture**: V3 HYBRID - Combines pipeline orchestration (from OLD) + design patterns (from CURRENT) = Perfect 25/25 score

**Key Features**: 
- Pipeline orchestration layer separates "what to do" from "how to do it"
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

### 🚀 Phase 4: Attendance System (READY TO START)
**Status**: Planning Complete | November 20, 2025

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

### Raspberry Pi 4 (Production Target)

| Stage | Time | Notes |
|-------|------|-------|
| Detection | 130ms | YOLO INT8 TFLite |
| Tracking | 10ms | BoT-SORT with persist=True |
| Alignment | 40ms | MediaPipe (once per track) |
| Recognition | 50ms | ArcFace (once per track with caching) |
| **First Frame** | **230ms (~4 FPS)** | Full pipeline |
| **Cached Frames** | **140ms (~7 FPS)** | Skip align+recognize |

**With Quality-Aware Caching**:
- 97% fewer alignments (10 vs 300 per person)
- 97% fewer recognitions (10 vs 300 per person)
- 99.8% CPU savings for multi-person scenarios

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
2. ✅ Phase 4 planning document created
3. 📋 Review `docs/PHASE_4_IMPLEMENTATION_PLAN.md`
4. 🚀 Begin with database schema and foundation classes

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

Last Updated: November 20, 2025 - Phase 3A Complete (Recognition + Quality-Aware Caching) | Phase 4 Ready
