"""
Common utilities and shared components for Face Attendance System.

This module provides core infrastructure used across the entire system:
- ConfigManager: Singleton for centralized configuration management
- EventSystem: Observer pattern for event-driven architecture
- Base classes for detectors, trackers, and other components
"""

from common.config_manager import ConfigManager
from common.event_system import EventSystem

__version__ = "2.0.0"
__author__ = "Rishabh Mishra"
__all__ = ['ConfigManager', 'EventSystem']
