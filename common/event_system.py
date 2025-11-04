"""
Event System - Observer Pattern

Provides event-driven architecture allowing loose coupling between components.
Multiple observers can subscribe to events and react independently when events occur.

Design Pattern: Observer (Publish-Subscribe)
Use Cases:
    - Face detected → Log + Alert + Mark Attendance
    - Unknown face → Send notification + Capture photo
    - Track lost → Update database + Log event

Example:
    >>> events = EventSystem()
    >>> events.subscribe('face_detected', lambda data: print(f"Face: {data}"))
    >>> events.publish('face_detected', {'track_id': 1, 'confidence': 0.85})
"""

from typing import Callable, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging


class EventType(Enum):
    """
    Standard event types for Face Attendance System.
    
    Using Enum ensures type safety and prevents typos in event names.
    """
    # Detection Events
    FACE_DETECTED = "face_detected"
    FACE_LOST = "face_lost"
    
    # Recognition Events (Future)
    FACE_RECOGNIZED = "face_recognized"
    UNKNOWN_FACE = "unknown_face"
    
    # Tracking Events
    TRACK_STARTED = "track_started"
    TRACK_ENDED = "track_ended"
    
    # Attendance Events (Future)
    ATTENDANCE_MARKED = "attendance_marked"
    
    # System Events
    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_STOPPED = "pipeline_stopped"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class Event:
    """
    Event data container.
    
    Standardized event structure for all event types, ensuring
    consistency and making it easy to add event logging/debugging.
    
    Attributes:
        event_type: Type of event (from EventType enum)
        timestamp: When the event occurred
        data: Event-specific data (flexible dictionary)
        source: Component that generated the event (optional)
    """
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = field(default="unknown")
    
    def __str__(self) -> str:
        """Human-readable event representation."""
        return (
            f"Event({self.event_type.value} at {self.timestamp.strftime('%H:%M:%S')}, "
            f"source={self.source}, data_keys={list(self.data.keys())})"
        )


class EventSystem:
    """
    Event system implementing Observer pattern for event-driven architecture.
    
    Allows components to publish events and other components to subscribe
    to those events without direct dependencies between them.
    
    Benefits:
        - Loose coupling: Publishers don't know about subscribers
        - Easy extensibility: Add new observers without changing publishers
        - Separation of concerns: Each observer handles one responsibility
    
    Thread Safety:
        Not thread-safe. Use threading.Lock() if needed for multi-threading.
    
    Example:
        >>> # Setup
        >>> events = EventSystem()
        >>> 
        >>> # Subscribe to events
        >>> def log_face(event: Event):
        ...     print(f"[LOG] Face detected: {event.data}")
        >>> 
        >>> def mark_attendance(event: Event):
        ...     print(f"[ATTENDANCE] Marking present: {event.data}")
        >>> 
        >>> events.subscribe(EventType.FACE_DETECTED, log_face)
        >>> events.subscribe(EventType.FACE_DETECTED, mark_attendance)
        >>> 
        >>> # Publish event (both subscribers notified)
        >>> events.publish(EventType.FACE_DETECTED, {
        ...     'track_id': 1,
        ...     'confidence': 0.85,
        ...     'bbox': [100, 100, 200, 200]
        ... })
    """
    
    def __init__(self):
        """Initialize event system with empty subscriber lists."""
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = {}
        self._logger = logging.getLogger('EventSystem')
        self._event_history: List[Event] = []  # For debugging
        self._max_history = 100  # Keep last 100 events
        
        # Initialize subscriber lists for all event types
        for event_type in EventType:
            self._subscribers[event_type] = []
        
        self._logger.info("EventSystem initialized")
    
    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[Event], None]
    ) -> None:
        """
        Subscribe to an event type.
        
        The callback function will be called whenever an event of the
        specified type is published.
        
        Args:
            event_type: Event type to subscribe to (EventType enum)
            callback: Function to call when event occurs.
                      Must accept single Event parameter.
        
        Example:
            >>> def on_face_detected(event: Event):
            ...     print(f"Face {event.data['track_id']} detected!")
            >>> 
            >>> events.subscribe(EventType.FACE_DETECTED, on_face_detected)
        """
        if event_type not in self._subscribers:
            self._logger.warning(f"Unknown event type: {event_type}")
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append(callback)
        self._logger.debug(
            f"Subscribed {callback.__name__} to {event_type.value}. "
            f"Total subscribers: {len(self._subscribers[event_type])}"
        )
    
    def unsubscribe(
        self,
        event_type: EventType,
        callback: Callable[[Event], None]
    ) -> None:
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: Event type to unsubscribe from
            callback: Previously subscribed callback function
        """
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(callback)
                self._logger.debug(f"Unsubscribed {callback.__name__} from {event_type.value}")
            except ValueError:
                self._logger.warning(
                    f"Callback {callback.__name__} not found in {event_type.value} subscribers"
                )
    
    def publish(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        source: str = "unknown"
    ) -> None:
        """
        Publish an event to all subscribers.
        
        Creates an Event object and notifies all subscribers of the
        specified event type.
        
        Args:
            event_type: Type of event to publish
            data: Event data dictionary
            source: Component generating the event (for debugging)
        
        Example:
            >>> events.publish(
            ...     EventType.FACE_DETECTED,
            ...     {'track_id': 1, 'confidence': 0.85},
            ...     source='YOLODetector'
            ... )
        """
        event = Event(
            event_type=event_type,
            timestamp=datetime.now(),
            data=data,
            source=source
        )
        
        # Store in history for debugging
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        # Notify all subscribers
        subscribers = self._subscribers.get(event_type, [])
        self._logger.debug(
            f"Publishing {event_type.value} to {len(subscribers)} subscribers"
        )
        
        for callback in subscribers:
            try:
                callback(event)
            except Exception as e:
                self._logger.error(
                    f"Error in subscriber {callback.__name__} "
                    f"for event {event_type.value}: {e}",
                    exc_info=True
                )
    
    def get_subscriber_count(self, event_type: EventType) -> int:
        """Get number of subscribers for an event type."""
        return len(self._subscribers.get(event_type, []))
    
    def get_event_history(self, limit: int = 10) -> List[Event]:
        """
        Get recent event history.
        
        Useful for debugging and understanding event flow.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of recent events (most recent last)
        """
        return self._event_history[-limit:]
    
    def clear_history(self) -> None:
        """Clear event history (useful for testing)."""
        self._event_history.clear()
        self._logger.debug("Event history cleared")
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        total_subscribers = sum(len(subs) for subs in self._subscribers.values())
        return (
            f"EventSystem(subscribers={total_subscribers}, "
            f"history_size={len(self._event_history)})"
        )
