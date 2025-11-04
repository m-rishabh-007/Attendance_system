# Archive Directory

This directory contains historical code and previous versions of the Face Attendance System.

## Purpose

- **Preservation**: Keep old code for reference and learning
- **Version Control**: Track evolution from v1.0 procedural to v2.0 OOP
- **Documentation**: Understand design decisions and architecture changes

## Contents

### v1_pipeline/
Legacy version 1.0 code (procedural approach):
- `pipeline_main.py` - Original TFLite inference pipeline
- `face_tracker_bytetrack.py` - Custom ByteTrack implementation
- `attendance_ultralytics.py` - Original Ultralytics-based demo
- `attendance_prototype.py` - Early prototype code

See `v1_pipeline/README.md` for detailed information.

## Current Production Code

The active codebase is in the root directory:
- **production/attendance_v2.py** - Current OOP pipeline (use this!)
- **common/** - Singleton ConfigManager, Observer EventSystem
- **detectors/** - Factory pattern for detector creation
- **tracking/** - Factory pattern for tracker creation

## Why Archive Instead of Delete?

1. **Learning**: Understand why v2.0 is better (OOP, design patterns, modularity)
2. **Reference**: ByteTrack implementation useful for education
3. **Debugging**: Compare behavior if issues arise
4. **Portfolio**: Show evolution of your skills

## When to Use Archive Code

- ❌ **Don't use for production** - outdated, unmaintained
- ✅ **Use for learning** - see how tracking algorithms work
- ✅ **Use for comparison** - understand v1.0 vs v2.0 differences
- ✅ **Use for reference** - copy concepts (with improvements)

---

**Last Updated**: November 2025  
**Archival Reason**: Replaced by v2.0 OOP architecture with design patterns
