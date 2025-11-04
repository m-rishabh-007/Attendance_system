# Common Component - Shared Utilities & Design Patterns

**Purpose**: Provide shared utilities, base classes, and design pattern implementations used across all components.

**Key Features**:
- Singleton pattern (ConfigManager)
- Observer pattern (EventSystem)
- Base classes for Detection, Track, and other data structures
- Shared utilities

---

## 📋 Files in This Component

| File | Purpose | Pattern | Critical? |
|------|---------|---------|-----------|
| `config_manager.py` | Global configuration access | Singleton | ⭐ CRITICAL |
| `event_system.py` | Event notifications | Observer | ⭐ CRITICAL |
| `base_classes.py` | Common data structures | Data Classes | ⭐ CRITICAL |
| `utils.py` | Shared utility functions | Utilities | ⚠️ Helper |
| `logger.py` | Logging configuration | Singleton | ⚠️ Helper |

---

## 📄 File Details

### 1. `config_manager.py` - Singleton Pattern ⭐ CRITICAL

**Purpose**: Provide global access to configuration without passing config everywhere.

**Design Pattern**: Singleton (only one instance exists)

**Why Singleton?**:
- Configuration is application-wide state
- Should be loaded once at startup
- All components need access
- Prevents multiple file reads

#### Implementation

```python
import yaml
from pathlib import Path
from typing import Any, Dict

class ConfigManager:
    """
    Singleton configuration manager.
    
    Ensures only one config instance exists and provides global access.
    
    Usage:
        # Get instance (loads config first time)
        config = ConfigManager.get_instance()
        
        # Access configuration
        detector_type = config.get('detector.type')
        threshold = config.get('detector.confidence_threshold', default=0.5)
    """
    
    _instance = None
    _config = None
    
    def __new__(cls):
        """Singleton pattern - only create one instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls, config_path: str = 'config.yaml'):
        """
        Get singleton instance (creates if doesn't exist).
        
        Args:
            config_path: Path to config file (only used on first call)
        
        Returns:
            ConfigManager instance
        """
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_config(config_path)
        return cls._instance
    
    def _load_config(self, config_path: str):
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Examples:
            config.get('detector.type')  # Returns 'yolo'
            config.get('detector.confidence_threshold')  # Returns 0.5
            config.get('nonexistent.key', default='fallback')  # Returns 'fallback'
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_all(self) -> Dict:
        """Get entire configuration dictionary."""
        return self._config.copy()
    
    def reload(self, config_path: str = 'config.yaml'):
        """Reload configuration from file."""
        self._load_config(config_path)
```

#### Usage Examples

```python
# In any component
from common.config_manager import ConfigManager

# Get instance
config = ConfigManager.get_instance()

# Access configuration
detector_type = config.get('detector.type')
conf_threshold = config.get('detector.confidence_threshold', default=0.5)

# Access nested config
num_threads = config.get('runtime_settings.num_threads', default=3)
```

**Benefits**:
- ✅ No need to pass config everywhere
- ✅ Single source of truth
- ✅ Easy to reload without restarting
- ✅ Default values for missing keys

---

### 2. `event_system.py` - Observer Pattern ⭐ CRITICAL

**Purpose**: Notify components about events without tight coupling.

**Design Pattern**: Observer (publish-subscribe)

**Why Observer?**:
- Loose coupling between components
- Multiple listeners for same event
- Easy to add new listeners
- Enables logging, monitoring, webhooks, etc.

#### Implementation

```python
from typing import Callable, Dict, List
from collections import defaultdict

class EventSystem:
    """
    Observer pattern event system.
    
    Allows components to subscribe to events and get notified
    when those events occur.
    
    Usage:
        # Get instance
        events = EventSystem.get_instance()
        
        # Subscribe to event
        def on_face_detected(data):
            print(f"Face detected: {data['track_id']}")
        
        events.subscribe('face_detected', on_face_detected)
        
        # Publish event
        events.publish('face_detected', {'track_id': 1, 'confidence': 0.85})
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._subscribers = defaultdict(list)
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def subscribe(self, event_name: str, callback: Callable):
        """
        Subscribe to an event.
        
        Args:
            event_name: Name of event to listen for
            callback: Function to call when event occurs
                     (should accept one argument: event data dict)
        
        Example:
            def my_handler(data):
                print(f"Got event: {data}")
            
            events.subscribe('face_detected', my_handler)
        """
        self._subscribers[event_name].append(callback)
    
    def unsubscribe(self, event_name: str, callback: Callable):
        """
        Unsubscribe from an event.
        
        Args:
            event_name: Name of event
            callback: Previously subscribed callback
        """
        if event_name in self._subscribers:
            self._subscribers[event_name].remove(callback)
    
    def publish(self, event_name: str, data: Dict = None):
        """
        Publish an event to all subscribers.
        
        Args:
            event_name: Name of event to publish
            data: Event data dictionary
        
        Example:
            events.publish('face_detected', {
                'track_id': 1,
                'confidence': 0.85,
                'bbox': [100, 100, 200, 200]
            })
        """
        if event_name in self._subscribers:
            for callback in self._subscribers[event_name]:
                try:
                    callback(data or {})
                except Exception as e:
                    print(f"Error in event handler for '{event_name}': {e}")
    
    def clear_subscribers(self, event_name: str = None):
        """
        Clear subscribers for specific event or all events.
        
        Args:
            event_name: Event to clear (None = clear all)
        """
        if event_name:
            self._subscribers[event_name].clear()
        else:
            self._subscribers.clear()
```

#### Event Types (Current & Future)

| Event Name | When Published | Data |
|------------|----------------|------|
| `face_detected` | New face detected | `{track_id, confidence, bbox}` |
| `face_lost` | Track lost | `{track_id, frames_lost}` |
| `face_recognized` | (Future) Face identified | `{track_id, identity, confidence}` |
| `attendance_marked` | (Future) Attendance recorded | `{identity, timestamp}` |
| `error_occurred` | Error in pipeline | `{stage, error_message}` |

#### Usage Examples

```python
from common.event_system import EventSystem

# Get instance
events = EventSystem.get_instance()

# Example 1: Log detections
def log_detection(data):
    print(f"[LOG] Face {data['track_id']} detected with conf {data['confidence']:.2f}")

events.subscribe('face_detected', log_detection)

# Example 2: Send webhook
def send_webhook(data):
    requests.post('http://server/webhook', json=data)

events.subscribe('face_detected', send_webhook)

# Example 3: Update database (future)
def update_db(data):
    db.insert_detection(data['track_id'], data['timestamp'])

events.subscribe('face_detected', update_db)

# Publish event (from orchestrator or detector)
events.publish('face_detected', {
    'track_id': 1,
    'confidence': 0.85,
    'bbox': [100, 100, 200, 200],
    'timestamp': time.time()
})
```

**Benefits**:
- ✅ Loose coupling (detector doesn't know about logger)
- ✅ Multiple handlers for same event
- ✅ Easy to add features (webhooks, logging, monitoring)
- ✅ Clean separation of concerns

---

### 3. `base_classes.py` - Data Structures ⭐ CRITICAL

**Purpose**: Define common data structures used across components.

**Data Classes**:
- `Detection`: Face detection result
- `Track`: Tracked face with persistent ID
- `AlignedFace`: (Future) Aligned face for recognition
- `Identity`: (Future) Recognized person

#### Implementation

```python
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

@dataclass
class Detection:
    """
    Face detection result.
    
    Attributes:
        bbox: Bounding box [x1, y1, x2, y2] (xyxy format)
        confidence: Detection confidence [0-1]
        class_id: Class ID (0 for face)
        landmarks: Optional facial landmarks (5 or 68 points)
    """
    bbox: np.ndarray
    confidence: float
    class_id: int = 0
    landmarks: Optional[np.ndarray] = None
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'bbox': self.bbox.tolist(),
            'confidence': float(self.confidence),
            'class_id': int(self.class_id),
            'landmarks': self.landmarks.tolist() if self.landmarks is not None else None
        }


@dataclass
class Track:
    """
    Tracked face with persistent ID.
    
    Attributes:
        track_id: Persistent track ID (same across frames)
        bbox: Bounding box [x1, y1, x2, y2]
        confidence: Detection confidence [0-1]
        state: Track state ('new', 'tracked', 'lost')
        frames_since_update: Frames since last detection
        age: Total frames this track has existed
    """
    track_id: int
    bbox: np.ndarray
    confidence: float
    state: str = 'tracked'
    frames_since_update: int = 0
    age: int = 1
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'track_id': int(self.track_id),
            'bbox': self.bbox.tolist(),
            'confidence': float(self.confidence),
            'state': self.state,
            'frames_since_update': int(self.frames_since_update),
            'age': int(self.age)
        }
    
    def to_tlwh(self) -> Tuple[float, float, float, float]:
        """Convert bbox from xyxy to tlwh (top-left-width-height)."""
        x1, y1, x2, y2 = self.bbox
        return x1, y1, x2 - x1, y2 - y1


@dataclass
class AlignedFace:
    """
    (Future) Aligned face for recognition.
    
    Attributes:
        track_id: Associated track ID
        image: Aligned face image (112x112 typically)
        landmarks: Facial landmarks used for alignment
        alignment_score: Quality score [0-1]
    """
    track_id: int
    image: np.ndarray
    landmarks: np.ndarray
    alignment_score: float
    
    def to_dict(self):
        return {
            'track_id': int(self.track_id),
            'image_shape': self.image.shape,
            'alignment_score': float(self.alignment_score)
        }


@dataclass
class Identity:
    """
    (Future) Recognized person identity.
    
    Attributes:
        track_id: Associated track ID
        person_id: Database person ID
        name: Person name
        confidence: Recognition confidence [0-1]
        embedding: Face embedding vector
    """
    track_id: int
    person_id: int
    name: str
    confidence: float
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self):
        return {
            'track_id': int(self.track_id),
            'person_id': int(self.person_id),
            'name': self.name,
            'confidence': float(self.confidence)
        }
```

#### Usage Examples

```python
from common.base_classes import Detection, Track

# Create detection
detection = Detection(
    bbox=np.array([100, 100, 200, 200]),
    confidence=0.85,
    class_id=0
)

# Create track
track = Track(
    track_id=1,
    bbox=np.array([100, 100, 200, 200]),
    confidence=0.85,
    state='tracked'
)

# Convert to dict for JSON
track_dict = track.to_dict()
# {'track_id': 1, 'bbox': [100, 100, 200, 200], ...}

# Convert bbox format
tlwh = track.to_tlwh()  # (100, 100, 100, 100)
```

---

### 4. `utils.py` - Shared Utilities

**Purpose**: Common utility functions used across components.

```python
import cv2
import numpy as np
from typing import Tuple

def resize_with_aspect_ratio(image: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resize image maintaining aspect ratio.
    
    Args:
        image: Input image
        target_size: Target size for longer edge
    
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    if h > w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))
    
    return cv2.resize(image, (new_w, new_h))


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute Intersection over Union between two boxes.
    
    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]
    
    Returns:
        IoU score [0-1]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def draw_bbox(image: np.ndarray, bbox: np.ndarray, label: str, 
              color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw bounding box with label on image.
    
    Args:
        image: Input image
        bbox: Bounding box [x1, y1, x2, y2]
        label: Label text
        color: Box color (B, G, R)
    
    Returns:
        Image with drawn box
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Draw label background
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
    
    # Draw label text
    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 255, 255), 1)
    
    return image
```

---

### 5. `logger.py` - Logging Configuration

**Purpose**: Centralized logging setup.

```python
import logging
from pathlib import Path

class Logger:
    """Singleton logger configuration."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls, name: str = 'attendance_system', 
                     log_file: str = 'logs/app.log'):
        """Get configured logger instance."""
        if cls._instance is None:
            # Create logs directory
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            
            # Configure logger
            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)
            
            # File handler
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            
            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            logger.addHandler(fh)
            logger.addHandler(ch)
            
            cls._instance = logger
        
        return cls._instance
```

---

## 🎯 Design Pattern Summary

| Pattern | Implementation | Purpose |
|---------|----------------|---------|
| **Singleton** | ConfigManager, Logger, EventSystem | Single instance for global state |
| **Observer** | EventSystem | Decouple event producers from consumers |
| **Data Classes** | Detection, Track | Type-safe data structures |

---

## 🔧 Configuration

Configuration is accessed via ConfigManager:

```python
from common.config_manager import ConfigManager

config = ConfigManager.get_instance('config.yaml')

# All components can now access config
detector_type = config.get('detector.type')
```

---

## 🧪 Testing

```bash
# Test config manager
python tests/test_common/test_config_manager.py

# Test event system
python tests/test_common/test_event_system.py

# Test base classes
python tests/test_common/test_base_classes.py
```

---

## 📚 Related Documentation

- **Parent Guide**: `docs/DEVELOPER_GUIDE.md`
- **Architecture**: `docs/ARCHITECTURE_V3_HYBRID.md`
- **Design Patterns**: `docs/ARCHITECTURE_V3_HYBRID.md#design-patterns`

---

**Questions?** See `docs/DEVELOPER_GUIDE.md` or create an issue.

**Last Updated**: November 5, 2025
