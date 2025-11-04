# Architecture Documentation

**⚠️ This file is now a redirect. The complete architecture documentation has moved.**

## Current Architecture: V3 HYBRID

The Face Attendance System now uses **V3 HYBRID Architecture** - a perfect combination of pipeline orchestration and design patterns.

**📖 Read the complete architecture guide here:**

➡️ **[docs/ARCHITECTURE_V3_HYBRID.md](docs/ARCHITECTURE_V3_HYBRID.md)**

---

## Quick Overview

**V3 HYBRID** achieves a **perfect 25/25 score** by combining:
- ✅ Pipeline orchestration layer (from OLD architecture)
- ✅ All 4 design patterns: Singleton, Factory, Strategy, Observer (from CURRENT architecture)
- ✅ Modular folder structure
- ✅ Config-driven development (NO hardcoded values)

## Architecture Summary

```
Attendance_system/
├── attendance_system.py      # Thin wrapper (UI + camera)
├── config.yaml                # All configuration
│
├── pipeline/                  # Orchestration layer
│   ├── orchestrator.py        # Main coordinator
│   ├── detection_stage.py     # Detection orchestration
│   └── tracking_stage.py      # Tracking orchestration
│
├── detectors/                 # Detector implementations
│   ├── factory.py             # Factory pattern
│   └── yolo_detector.py
│
├── tracking/                  # Tracker implementations
│   ├── factory.py             # Strategy pattern
│   └── botsort_tracker.py
│
├── common/                    # Singleton + Observer
├── aligners/                  # (future) Alignment
├── recognizers/               # (future) Recognition
├── database/                  # (future) Data persistence
└── server/                    # (future) API deployment
```

## Key Principles

1. **Separation of Concerns**: `pipeline/` = "what to do", component folders = "how to do it"
2. **Design Patterns**: Factory for creation, Strategy for swapping, Singleton for config, Observer for events
3. **Extensibility**: Easy to add new detectors, trackers, or pipeline stages
4. **Config-Driven**: All settings in `config.yaml`, no hardcoded values

## Related Documentation

- **[docs/ARCHITECTURE_V3_HYBRID.md](docs/ARCHITECTURE_V3_HYBRID.md)** - Complete V3 HYBRID guide (651 lines)
- **[docs/ARCHITECTURE_V2.md](docs/ARCHITECTURE_V2.md)** - Previous V2 architecture
- **[README.md](README.md)** - Project overview and quick start
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Deployment guide
- **[.github/copilot-instructions.md](.github/copilot-instructions.md)** - AI agent development guide

---

**For the complete architecture documentation, see [docs/ARCHITECTURE_V3_HYBRID.md](docs/ARCHITECTURE_V3_HYBRID.md)**
