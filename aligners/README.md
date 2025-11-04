# Aligners Component - Face Alignment Implementations

**Status**: 🚧 Not Yet Implemented (Future Phase)

**Purpose**: Align detected faces to normalized pose for recognition.

---

## 📋 Planned Implementation

Face alignment is a preprocessing step before recognition that:
- Normalizes face pose (frontal view)
- Corrects rotation and scale
- Aligns facial landmarks to standard positions
- Improves recognition accuracy

---

## 🎯 Planned Files

| File | Purpose | Status |
|------|---------|--------|
| `base.py` | BaseFaceAligner interface | 🚧 Planned |
| `factory.py` | Factory pattern for aligner creation | 🚧 Planned |
| `mtcnn_aligner.py` | MTCNN-based alignment | 🚧 Planned |
| `blazeface_aligner.py` | BlazeFace-based alignment | 🚧 Planned |
| `similarity_aligner.py` | Similarity transform alignment | 🚧 Planned |

---

## 🔄 Planned Integration

### Pipeline Flow (Future)

```
detection_stage.process_with_tracking(frame)
    ↓ (detections, tracks)
alignment_stage.process(frame, tracks)  ← NEW STAGE
    ↓ (aligned_faces)
recognition_stage.process(aligned_faces)
    ↓ (identities)
attendance_stage.process(identities)
```

### Usage Example (Future)

```python
# pipeline/alignment_stage.py
from aligners.factory import AlignerFactory

class AlignmentStage:
    def __init__(self, config):
        self.aligner = AlignerFactory.create(config)
    
    def process(self, frame, tracks):
        aligned_faces = []
        for track in tracks:
            aligned = self.aligner.align(frame, track.bbox)
            aligned_faces.append(aligned)
        return aligned_faces
```

---

## 🔧 Planned Configuration

```yaml
# config.yaml (future)
alignment:
  enabled: true
  type: mtcnn  # 'mtcnn', 'blazeface', 'similarity'
  target_size: 112  # Output face size
  landmarks: 5  # Number of landmarks (5 or 68)
```

---

## 📚 Related Documentation

- **Parent Guide**: `docs/DEVELOPER_GUIDE.md`
- **Architecture**: `docs/ARCHITECTURE_V3_HYBRID.md#future-phases`
- **Pipeline**: `pipeline/README.md#adding-new-stages`

---

## 🚀 Implementation Checklist

When implementing this component:

- [ ] Create `base.py` with BaseFaceAligner interface
- [ ] Create `factory.py` with Factory pattern
- [ ] Implement MTCNN aligner
- [ ] Implement BlazeFace aligner
- [ ] Create `pipeline/alignment_stage.py`
- [ ] Update `pipeline/orchestrator.py` to use alignment
- [ ] Add alignment config to `config.yaml`
- [ ] Create unit tests
- [ ] Update this README with implementation details

---

**Status**: Waiting for recognition phase implementation planning.

**Last Updated**: November 5, 2025
