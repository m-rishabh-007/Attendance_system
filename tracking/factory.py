"""
Tracking Strategy Factory

Creates appropriate tracker based on configuration using Strategy pattern.
Allows easy swapping between tracking algorithms.
"""

from typing import Dict, Any
import logging

from common.base_classes import BaseTracker


class TrackerFactory:
    """
    Factory for creating tracker instances.
    
    Supports multiple tracker types:
    - 'botsort': BoT-SORT tracker (best quality, recommended)
    - 'none': No tracking (for testing/debugging only)
    
    Example:
        >>> config = {
        ...     'type': 'botsort',
        ...     'track_thresh': 0.5,
        ...     'track_buffer': 90,
        ...     'match_thresh': 0.4
        ... }
        >>> tracker = TrackerFactory.create(config)
        >>> tracks = tracker.update(detections)
    """
    
    _logger = logging.getLogger('TrackerFactory')
    
    @classmethod
    def create(cls, config: Dict[str, Any]) -> BaseTracker:
        """
        Create tracker based on configuration.
        
        Args:
            config: Tracker configuration dictionary containing:
                - type: Tracker type ('botsort', 'bytetrack', 'none')
                - Other tracker-specific settings
                
        Returns:
            BaseTracker instance
            
        Raises:
            ValueError: If tracker type is unsupported
            KeyError: If required config keys are missing
        """
        tracker_type = config.get('type')
        
        if not tracker_type:
            raise KeyError("'type' field required in tracker config")
        
        # Lazy imports
        if tracker_type == 'botsort':
            from tracking.botsort_tracker import BotSORTTracker
            cls._logger.info("Creating BoT-SORT tracker")
            return BotSORTTracker(config)
        
        elif tracker_type == 'none':
            from tracking.no_tracker import NoTracker
            cls._logger.info("Creating NoTracker (no tracking)")
            return NoTracker(config)
        
        else:
            available = ['botsort', 'none']
            raise ValueError(
                f"Unsupported tracker type: '{tracker_type}'. "
                f"Available types: {available}"
            )
    
    @classmethod
    def get_available_types(cls) -> list:
        """Get list of available tracker types."""
        return ['botsort', 'none']
