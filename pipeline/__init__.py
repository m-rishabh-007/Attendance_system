"""
Pipeline orchestration layer.

This package contains the pipeline stages and orchestrator that coordinate
the complete face attendance processing flow.

Modules:
    orchestrator: Main pipeline coordinator
    detection_stage: Stage 1 - Face detection
    tracking_stage: Stage 2 - Face tracking
    alignment_stage: Stage 3 - Face alignment (future)
    recognition_stage: Stage 4 - Face recognition (future)
    attendance_stage: Stage 5 - Attendance marking (future)
"""

__version__ = '3.0.0'
__all__ = [
    'PipelineOrchestrator',
    'DetectionStage',
    'TrackingStage',
]
