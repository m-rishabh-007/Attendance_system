# Phase 3A Architecture - Clean Separation

**Date**: November 8, 2025  
**Status**: Step 1 Complete - BaseRecognizer Interface

---

## Clean Architecture Principle

✅ **Runtime code in module folders** (detectors/, aligners/, recognizers/)  
✅ **Build-time tools in tools/ folder** (quantization, data collection)  
✅ **Clear separation of concerns**

---

## Folder Structure

```
Attendance_system/
├── recognizers/                    ✅ RUNTIME CODE ONLY
│   ├── __init__.py                 ✅ Module exports
│   ├── base_recognizer.py          ✅ Abstract interface (367 lines)
│   ├── auraface_recognizer.py      ⏸️ Implementation (Day 3-4)
│   ├── factory.py                  ⏸️ Factory pattern (Day 3-4)
│   ├── quality_scorer.py           ⏸️ Quality assessment (Day 5)
│   └── README.md                   ✅ Module documentation
│
├── tools/                          ✅ BUILD-TIME TOOLS ONLY
│   ├── __init__.py                 ✅ Tools package
│   └── quantization/               ✅ INT8 quantization tools
│       ├── __init__.py             ✅ Package init
│       ├── README.md               ✅ Quantization guide
│       ├── calibration_data_reader.py      ⏸️ Week 2
│       ├── collect_calibration_data.py     ⏸️ Week 2
│       ├── quantize_auraface.py            ⏸️ Week 2
│       ├── validate_quantized_model.py     ⏸️ Week 2
│       └── calibration_data/               ⏸️ Week 2
│           └── faces.npy
│
├── models/
│   └── recognition/
│       ├── auraface_resnet100_fp32.onnx    ⏸️ Day 3
│       └── auraface_resnet100_int8.onnx    ⏸️ Week 2
│
└── config.yaml                     ✅ Updated with recognition section
```

---

## Why This Architecture?

### Problem (Before)
```
recognizers/
├── base_recognizer.py              ✅ Runtime
├── auraface_recognizer.py          ✅ Runtime
├── calibration_reader.py           ❌ Build-time (messy!)
├── quantize_model.py               ❌ Build-time (messy!)
└── collect_data.py                 ❌ Build-time (messy!)
```

**Issues**:
- ❌ Mixes runtime and build-time code
- ❌ Confusing for developers (what runs in production?)
- ❌ Hard to maintain (which files for which phase?)

### Solution (Current)
```
recognizers/                        ✅ CLEAN: Only runtime code
└── [runtime recognizer classes]

tools/quantization/                 ✅ CLEAN: Only build-time tools
└── [quantization scripts]
```

**Benefits**:
- ✅ Clear separation: runtime vs build-time
- ✅ Easy to understand: "recognizers = production, tools = development"
- ✅ Maintainable: Each folder has single responsibility
- ✅ Modular: Can delete tools/ after quantization if needed

---

## When to Use Each Folder

### `recognizers/` (Runtime Code)
**Use for**: Code that runs in production pipeline

**Examples**:
- ✅ BaseRecognizer (abstract interface)
- ✅ AuraFaceRecognizer (ONNX inference)
- ✅ RecognizerFactory (config-driven creation)
- ✅ QualityScorer (quality assessment)

**Characteristics**:
- Must be efficient (runs on every frame)
- Must handle both FP32 and INT8 models
- Must be tested thoroughly
- Part of the pipeline

### `tools/quantization/` (Build-Time Tools)
**Use for**: Scripts that prepare models before deployment

**Examples**:
- ✅ collect_calibration_data.py (one-time data collection)
- ✅ calibration_data_reader.py (used by quantization only)
- ✅ quantize_auraface.py (FP32 → INT8 conversion)
- ✅ validate_quantized_model.py (accuracy testing)

**Characteristics**:
- Run ONCE before deployment (not in production)
- Can be slow (doesn't affect runtime performance)
- Creates artifacts (INT8 models, calibration data)
- Not imported by pipeline

---

## Workflow

### Week 1: FP32 Baseline (Current)
```
Day 1-2: ✅ Create recognizers/base_recognizer.py
Day 3:   ⏸️ Download FP32 model → models/recognition/
Day 3-4: ⏸️ Create recognizers/auraface_recognizer.py
Day 4:   ⏸️ Create recognizers/factory.py
Day 5:   ⏸️ Create recognizers/quality_scorer.py
Day 6-7: ⏸️ Unit tests + pipeline integration
```

**Uses**: `recognizers/` folder only  
**Result**: FP32 recognition working (~80-100ms on Pi)

### Week 2: INT8 Quantization
```
Day 1: ⏸️ Create tools/quantization/collect_calibration_data.py
       ⏸️ Run: Collect 100 face samples
       
Day 2: ⏸️ Create tools/quantization/calibration_data_reader.py
       ⏸️ Create tools/quantization/quantize_auraface.py
       ⏸️ Run: FP32 → INT8 conversion
       
Day 3: ⏸️ Create tools/quantization/validate_quantized_model.py
       ⏸️ Run: Validate accuracy (<1% drop)
       
Day 4: ⏸️ Update config.yaml: model_path → INT8
       ⏸️ Test: Verify pipeline works with INT8
```

**Uses**: `tools/quantization/` folder for preparation  
**Result**: INT8 model in `models/recognition/` (~40-50ms on Pi)

### Week 3: Database + Matching
```
Day 1-7: Create database/, enrollment system, similarity matching
```

**Uses**: `recognizers/` for recognition, `database/` for storage  
**Result**: Full attendance system working

---

## Config Switching (Week 1 → Week 2)

### Week 1: FP32 Model
```yaml
recognition:
  model_path: models/recognition/auraface_resnet100_fp32.onnx
  # AuraFaceRecognizer automatically handles FP32
```

### Week 2: INT8 Model
```yaml
recognition:
  model_path: models/recognition/auraface_resnet100_int8.onnx
  # AuraFaceRecognizer automatically handles INT8 (no code changes!)
```

**Magic**: AuraFaceRecognizer detects model type automatically  
**Benefit**: No code changes needed, just update config!

---

## Documentation Structure

### Module Documentation (recognizers/README.md)
**Focus**: How to USE recognizers in production

**Sections**:
- Overview and architecture
- Usage examples (basic, direct, comparison)
- BaseRecognizer interface documentation
- Performance targets
- Testing guide
- Configuration
- **Brief** quantization overview (points to tools/)

### Build Tools Documentation (tools/quantization/README.md)
**Focus**: How to PREPARE models for production

**Sections**:
- Quantization workflow (detailed)
- Why quantization (performance benefits)
- Step-by-step guide (collect, quantize, validate)
- Usage examples (full scripts)
- Troubleshooting (accuracy, speed issues)
- Requirements and dependencies

---

## Key Takeaways

1. **Clean Separation**: Runtime (recognizers/) vs Build-time (tools/)
2. **Single Responsibility**: Each folder has ONE clear purpose
3. **Easy to Understand**: Folder name = what it's for
4. **Maintainable**: No mixed concerns
5. **Professional**: Industry-standard architecture

---

**Architecture Decision**: ✅ Approved and Implemented  
**Reason**: Keeps recognizers/ clean and focused  
**Impact**: Better maintainability, clearer separation of concerns

**Last Updated**: November 8, 2025  
**Next**: Day 3 - Download AuraFace model and implement AuraFaceRecognizer
