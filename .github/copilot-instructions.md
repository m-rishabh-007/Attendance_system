# AI Agent Instructions - Face Attendance System

## Project Overview

Real-time face detection and tracking pipeline using **YOLOv8n INT8 TFLite** + **BoT-SORT** for persistent face IDs. Optimized for Raspberry Pi and laptops with Docker and virtual environment support.

**Current Status**: ✅ V3 HYBRID Architecture | ✅ Detection + Tracking | 🚧 Recognition + Alignment + Database + API (future phases)

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
Last Updated: November 5, 2025 - Phase 3 Complete
