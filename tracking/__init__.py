"""
Object Tracking Module - Strategy Pattern

Provides unified interface for different tracking algorithms.
Supports easy swapping between ByteTrack, BoT-SORT, DeepSORT, etc.

Design Pattern: Strategy
Use Cases:
    - Compare tracking performance
    - Switch algorithms based on requirements
    - Platform-specific tracker selection
"""

__version__ = "2.0.0"
