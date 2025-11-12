# AI Agent Instructions - Face Attendance System

## Project Overview

Real-time face detection and tracking pipeline using **YOLOv8n INT8 TFLite** + **BoT-SORT** for persistent face IDs. Optimized for Raspberry Pi and laptops with Docker and virtual environment support.

**Current Status**: ✅ V3 HYBRID Architecture | ✅ Detection + Tracking | ✅ Alignment + Angle Estimation | 🚧 Recognition + Database + API (future phases)

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

### 🚧 Phase 3A: Week 2 - INT8 Quantization + Caching (NEXT)
**Status**: Not Started | Target: November 15, 2025

**Week 2 Objectives**:
1. **INT8 Quantization** (Days 1-3):
   - Collect 100 calibration faces
   - Quantize FP32 → INT8 (auraface_resnet100_int8.onnx)
   - Target: 2x speed improvement (90ms → 45ms)
   - Validate: <1% accuracy drop

2. **Quality-Aware Caching** (Days 4-5):
   - Implement QualityAwareCache class
   - Sample 10 frames per track_id
   - Cache best quality embedding
   - Target: 97% CPU reduction

3. **Integration + Testing** (Days 6-7):
   - End-to-end testing on Pi 4
   - Performance benchmarking
   - Week 2 completion report

**Expected Results**:
- FPS: 3.6 → 7 FPS on Pi (2x from INT8)
- CPU savings: 97% (caching prevents redundant processing)
- Accuracy: 95%+ (quality-aware caching)

---

### 🔮 Phase 4: Attendance System (FUTURE)
**Status**: Not Started

**Components**:
- Database integration (SQLite)
- Face enrollment/registration
- Attendance marking logic
- EventSystem integration (Observer pattern will be used here)

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

### Before Starting Phase 3

1. ✅ Phase 2 documented
2. ✅ Phase 3 planning document created
3. 📋 Review `docs/PHASE_3_IMPLEMENTATION_PLAN.md`
4. 🚀 Begin with quality-aware caching (critical feature)

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

Last Updated: November 8, 2025 - Phase 3A Week 1 Complete (FP32 Baseline Recognition)
