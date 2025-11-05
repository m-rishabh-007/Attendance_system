# Developer Guide - Face Attendance System V3 HYBRID

**Complete reference for understanding, modifying, and extending the codebase.**

Last Updated: November 5, 2025

---

## 📚 Table of Contents

1. [Quick Navigation](#quick-navigation)
2. [Architecture Overview](#architecture-overview)
3. [Architecture Decision Records (ADRs)](#architecture-decision-records-adrs)
4. [Component Documentation](#component-documentation)
5. [Testing Guide](#testing-guide)
6. [Common Development Tasks](#common-development-tasks)
7. [Critical Implementation Details](#critical-implementation-details)
8. [Troubleshooting](#troubleshooting)

---

## 🗺️ Quick Navigation

### Core Components (README in each folder)

| Component | Path | Purpose | Documentation |
|-----------|------|---------|---------------|
| **Pipeline** | `pipeline/` | Orchestration layer | [`pipeline/README.md`](../pipeline/README.md) |
| **Detectors** | `detectors/` | Face detection implementations | [`detectors/README.md`](../detectors/README.md) |
| **Tracking** | `tracking/` | Face tracking implementations | [`tracking/README.md`](../tracking/README.md) |
| **Common** | `common/` | Shared utilities (Singleton, Observer) | [`common/README.md`](../common/README.md) |
| **Aligners** | `aligners/` | (Future) Face alignment | [`aligners/README.md`](../aligners/README.md) |
| **Recognizers** | `recognizers/` | (Future) Face recognition | [`recognizers/README.md`](../recognizers/README.md) |
| **Database** | `database/` | (Future) Data persistence | [`database/README.md`](../database/README.md) |
| **Server** | `server/` | (Future) API deployment | [`server/README.md`](../server/README.md) |

### Key Files

| File | Purpose | When It Runs |
|------|---------|--------------|
| `attendance_system.py` | Main entry point (thin wrapper) | Every time you run the system |
| `config.yaml` | All configuration | Read at startup |
| `pipeline/orchestrator.py` | Main coordinator | Every frame |
| `detectors/yolo_detector.py` | YOLO face detection | Every frame |
| `tracking/botsort_tracker.py` | BoT-SORT wrapper (fallback only) | Only in edge cases |

---

## 🏗️ Architecture Overview

### V3 HYBRID Design

```
attendance_system.py (UI + Camera)
         ↓
pipeline/orchestrator.py (Coordinates stages)
         ↓
   ┌─────┴──────┐
   ↓            ↓
detection_stage  tracking_stage (integrated!)
   ↓
detectors/yolo_detector.py
   ↓
model.track(persist=True, tracker='botsort.yaml')  ← BoT-SORT runs HERE!
```

**Key Principle**: Separation of concerns
- `pipeline/` = "What to do" (orchestration)
- `detectors/`, `tracking/`, etc. = "How to do it" (implementation)

For complete architecture details, see [`ARCHITECTURE_V3_HYBRID.md`](ARCHITECTURE_V3_HYBRID.md).

---

## � Architecture Decision Records (ADRs)

**Purpose**: Documents why we made specific architectural/technology choices.

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](ADR_001_ALIGNMENT_MODEL_SELECTION.md) | Face Alignment Model Selection | ✅ Accepted | Nov 5, 2025 |

**Why Document Decisions?**
- Future developers understand the "why" behind choices
- Prevents rehashing old discussions
- Shows what alternatives were considered
- Makes it easy to revisit decisions if constraints change

**ADR-001 Summary**: Chose MediaPipe Face Mesh for alignment over:
- Retraining YOLO with keypoints (delays implementation)
- MediaPipe-only detection (loses BoT-SORT tracking)
- MTCNN (too slow: 80-120ms)
- RetinaFace (way too slow: 150-250ms)
- No alignment (lower accuracy)

**Read the full analysis**: [`docs/ADR_001_ALIGNMENT_MODEL_SELECTION.md`](ADR_001_ALIGNMENT_MODEL_SELECTION.md)

---

## �📖 Component Documentation

### 1. Pipeline Layer (`pipeline/`)

**Purpose**: Orchestrates all pipeline stages without implementing logic.

**Key Files**:
- `orchestrator.py` - Main coordinator (runs all stages)
- `detection_stage.py` - Detection orchestration
- `tracking_stage.py` - Tracking orchestration (fallback)

**When It Runs**: Every frame

**Read More**: [`pipeline/README.md`](../pipeline/README.md)

---

### 2. Detectors (`detectors/`)

**Purpose**: Face detection implementations using Factory pattern.

**Key Files**:
- `factory.py` - DetectorFactory (creates detectors based on config)
- `yolo_detector.py` - YOLO implementation (currently used)
- `base.py` - BaseFaceDetector interface

**When It Runs**: Every frame (via detection_stage)

**Critical Method**: `detect_and_track()` - Uses YOLO's integrated tracking for persistent IDs

**Read More**: [`detectors/README.md`](../detectors/README.md)

---

### 3. Tracking (`tracking/`)

**Purpose**: Face tracking implementations using Strategy pattern.

**Key Files**:
- `factory.py` - TrackerFactory (creates trackers based on config)
- `botsort_tracker.py` - BoT-SORT wrapper (FALLBACK ONLY!)
- `base.py` - BaseTracker interface

**When It Runs**: 
- **Normal operation**: Never! (tracking happens in `yolo_detector.detect_and_track()`)
- **Fallback scenarios**: TFLiteDetector, old patterns, testing

**Read More**: [`tracking/README.md`](../tracking/README.md)

---

### 4. Common Utilities (`common/`)

**Purpose**: Shared utilities using Singleton and Observer patterns.

**Key Files**:
- `config_manager.py` - Singleton: Global config access
- `event_system.py` - Observer: Event notifications
- `base_classes.py` - Common interfaces (Detection, Track, etc.)

**When It Runs**: Throughout the system

**Read More**: [`common/README.md`](../common/README.md)

---

### 5. Future Components

| Component | Purpose | Status | Documentation |
|-----------|---------|--------|---------------|
| **Aligners** | Face alignment for recognition | 🚧 Not implemented | [`aligners/README.md`](../aligners/README.md) |
| **Recognizers** | Face recognition (ArcFace, etc.) | 🚧 Not implemented | [`recognizers/README.md`](../recognizers/README.md) |
| **Database** | Embeddings + attendance storage | 🚧 Not implemented | [`database/README.md`](../database/README.md) |
| **Server** | REST API + WebSocket | 🚧 Not implemented | [`server/README.md`](../server/README.md) |

---

## 🧪 Testing Guide

### Test Structure

```
tests/
├── test_persistent_tracking.py     # Smoke test: Verify IDs stay stable
├── test_tracking_scenarios.py      # Comprehensive: All tracking scenarios
├── test_config_manager.py          # Singleton pattern tests
├── test_event_system.py            # Observer pattern tests
├── test_factories.py               # Factory pattern tests
└── run_all_tests.py                # Master test runner
```

### Running Tests

```bash
# All tests
python tests/run_all_tests.py

# Specific test
python tests/test_persistent_tracking.py

# With pytest (if installed)
pytest tests/
```

**Read More**: [`tests/README.md`](../tests/README.md)

---

## 🔧 Common Development Tasks

### Adding a New Detector

1. Create implementation: `detectors/new_detector.py`
2. Inherit from `BaseFaceDetector`
3. Implement `detect()` and `initialize()` methods
4. Register in `detectors/factory.py`
5. Update `config.yaml`

**Example**: See [`detectors/README.md#adding-new-detector`](../detectors/README.md)

---

### Adding a New Tracker

1. Create implementation: `tracking/new_tracker.py`
2. Inherit from `BaseTracker`
3. Implement `update()` and `reset()` methods
4. Register in `tracking/factory.py`
5. Update `config.yaml`

**Example**: See [`tracking/README.md#adding-new-tracker`](../tracking/README.md)

---

### Adding Face Recognition (Phase 2)

1. Create `recognizers/base.py` with `BaseRecognizer`
2. Create `recognizers/factory.py`
3. Create `recognizers/arcface_recognizer.py`
4. Create `pipeline/recognition_stage.py`
5. Update `pipeline/orchestrator.py` to include recognition stage
6. Update `config.yaml` with recognition settings

**Example**: See [`ARCHITECTURE_V3_HYBRID.md#adding-components`](ARCHITECTURE_V3_HYBRID.md)

---

## 🔍 Critical Implementation Details

### 1. Persistent Track IDs

**Problem**: Track IDs must persist across frames for attendance tracking.

**Solution**: Use YOLO's integrated tracking with `persist=True`:

```python
# ✅ CORRECT (in yolo_detector.py)
results = self.model.track(
    frame,
    persist=True,           # Maintains state across frames
    tracker='botsort.yaml'  # BoT-SORT algorithm
)
```

**Why It Matters**: Without persistent IDs:
- Can't track who is who
- Face recognition runs hundreds of times per person
- Attendance logs become meaningless

**Read More**: 
- [`detectors/README.md#persistent-tracking`](../detectors/README.md)
- [`tracking/README.md#tracking-scenarios`](../tracking/README.md)

---

### 2. When Does BoT-SORT Actually Run?

**Normal Operation (V3 HYBRID)**:
```
orchestrator → detection_stage.process_with_tracking()
           → yolo_detector.detect_and_track()
           → model.track(persist=True, tracker='botsort.yaml')
             ↑
             BoT-SORT runs HERE inside Ultralytics!
```

**Fallback Scenarios** (when `tracking/botsort_tracker.py` runs):
1. Using TFLiteDetector (no integrated tracking)
2. Someone incorrectly uses old pattern: `detect()` + `update()`
3. Unit testing without full pipeline

**Read More**: [`tracking/README.md#execution-flow`](../tracking/README.md)

---

### 3. Design Patterns in Use

| Pattern | Location | Purpose |
|---------|----------|---------|
| **Singleton** | `common/config_manager.py` | Single config instance |
| **Observer** | `common/event_system.py` | Event notifications |
| **Factory** | `detectors/factory.py`, `tracking/factory.py` | Create components |
| **Strategy** | `tracking/` implementations | Swap tracking algorithms |

**Read More**: [`ARCHITECTURE_V3_HYBRID.md#design-patterns`](ARCHITECTURE_V3_HYBRID.md)

---

### 4. Configuration-Driven Development

**Rule**: NO hardcoded values! Everything in `config.yaml`.

```yaml
# config.yaml
detector:
  type: yolo                    # Easy to swap to 'tflite'
  model_path: models/...
  confidence_threshold: 0.5     # Easy to tune

tracker:
  type: botsort                 # Easy to swap to 'bytetrack'
  track_thresh: 0.4
```

**Read More**: [`common/README.md#configuration`](../common/README.md)

---

## 🐛 Troubleshooting

### Track IDs Fluctuating

**Symptom**: IDs change even when person sits still (1→2→1→3).

**Cause**: Not using integrated tracking with `persist=True`.

**Fix**: Ensure orchestrator calls `detection_stage.process_with_tracking()`.

**Check**:
```bash
# Run system with debug logging
python attendance_system.py

# Should see:
✅ Using YOLO integrated tracking

# Should NOT see:
⚠️ BotSORTTracker.update() called (fallback mode)
```

**Read More**: [`tracking/README.md#troubleshooting`](../tracking/README.md)

---

### No Camera Access

**Symptom**: "Failed to open camera"

**Fix**:
```bash
# Check camera permissions
v4l2-ctl --list-devices

# Update config.yaml
camera:
  device_id: 1  # Change to correct device
```

---

### Low FPS / Slow Performance

**Symptom**: System runs slowly (<10 FPS)

**Causes**:
1. Input size too large
2. Too many threads
3. Wrong camera settings

**Fix**:
```yaml
# config.yaml
detector:
  input_size: 256  # Try 256 instead of 640

runtime_settings:
  num_threads: 3   # 3 for Pi, 4 for laptop
```

---

## 📝 Maintenance Guidelines

### When You Modify Code

1. **Update Component README** in the affected folder
2. **Add/Update Tests** in `tests/`
3. **Update This Guide** if architecture changes
4. **Run Tests** before committing

### Documentation Update Checklist

- [ ] Component README updated
- [ ] Code comments added/updated
- [ ] Tests added/updated
- [ ] DEVELOPER_GUIDE.md updated (if major change)
- [ ] `.github/copilot-instructions.md` updated (if architectural change)

---

## 🔗 Related Documentation

- **[ARCHITECTURE_V3_HYBRID.md](ARCHITECTURE_V3_HYBRID.md)** - Complete architecture guide
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute setup guide
- **[model_training_reference.md](model_training_reference.md)** - Model creation history
- **[DEPLOYMENT.md](../DEPLOYMENT.md)** - Production deployment
- **[.github/copilot-instructions.md](../.github/copilot-instructions.md)** - AI agent guide

---

## 🤝 Contributing

Before contributing, read:
1. This DEVELOPER_GUIDE.md (you are here)
2. [ARCHITECTURE_V3_HYBRID.md](ARCHITECTURE_V3_HYBRID.md)
3. [CONTRIBUTING.md](../CONTRIBUTING.md)

Then explore component READMEs for implementation details.

---

**Questions?** Check component READMEs or create an issue on GitHub.

**Last Updated**: November 5, 2025
