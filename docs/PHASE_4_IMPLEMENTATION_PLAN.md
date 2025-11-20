# Phase 4: Attendance System - Implementation Plan

**Created**: November 20, 2025  
**Status**: 📋 Planning Complete | 🚧 Implementation Pending

---

## 📋 Overview

Phase 4 implements the attendance system to store known faces, match live embeddings, and log attendance records.

**Key Goals:**
1. Store known people's face embeddings (enrollment)
2. Match live embeddings with database (recognition)
3. Mark attendance when recognized (with cooldown)
4. Provide attendance logs and reports

**Prerequisites**: Phases 1-3A complete (Detection, Tracking, Alignment, Recognition, Caching)

---

## 🗄️ Database Schema (SQLite)

### Table: `persons`
Store enrolled individuals.

```sql
CREATE TABLE persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    email TEXT,
    department TEXT,
    enrollment_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'active',  -- 'active', 'inactive', 'suspended'
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Table: `embeddings`
Store multiple embeddings per person (quality-scored).

```sql
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    embedding BLOB NOT NULL,  -- 512-dim float32 array (2048 bytes)
    quality_score FLOAT NOT NULL,
    yaw_angle FLOAT,  -- Face angle when captured
    enrollment_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_primary BOOLEAN DEFAULT 0,  -- Best quality embedding
    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
);
```

### Table: `attendance`
Log when people are recognized.

```sql
CREATE TABLE attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    confidence FLOAT NOT NULL,  -- Similarity score
    quality_score FLOAT,
    track_id INTEGER,
    frame_number INTEGER,
    camera_id TEXT DEFAULT 'default',
    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
);
```

### Table: `attendance_sessions`
Track continuous presence (check-in/check-out).

```sql
CREATE TABLE attendance_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    check_in DATETIME NOT NULL,
    check_out DATETIME,
    duration_minutes INTEGER,
    camera_id TEXT DEFAULT 'default',
    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
);
```

### Table: `system_logs`
Track unknown faces and errors.

```sql
CREATE TABLE system_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,  -- 'unknown_face', 'error', 'warning'
    message TEXT,
    data TEXT,  -- JSON serialized data
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Indexes for Performance

```sql
CREATE INDEX idx_embeddings_person_id ON embeddings(person_id);
CREATE INDEX idx_attendance_person_id ON attendance(person_id);
CREATE INDEX idx_attendance_timestamp ON attendance(timestamp);
CREATE INDEX idx_sessions_person_id ON attendance_sessions(person_id);
CREATE INDEX idx_sessions_check_in ON attendance_sessions(check_in);
```

---

## 📁 File Structure

### New Files to Create

```
database/
├── __init__.py                    # Update: export new classes
├── base_database.py               # Abstract interfaces (NEW)
├── person_database.py             # Person CRUD operations (NEW)
├── embedding_database.py          # Embedding storage + search (NEW)
├── attendance_database.py         # Attendance logging (NEW)
├── database_manager.py            # Singleton coordinator (NEW)
└── schema.sql                     # Database schema (NEW)

pipeline/
├── enrollment_stage.py            # Enrollment workflow (NEW)
└── attendance_stage.py            # Attendance marking logic (NEW)

tests/
├── test_database/
│   ├── __init__.py
│   ├── test_person_database.py    # Person CRUD tests (NEW)
│   ├── test_embedding_database.py # Embedding search tests (NEW)
│   └── test_attendance_database.py # Attendance logic tests (NEW)
├── test_enrollment.py             # Enrollment workflow tests (NEW)
└── test_attendance_stage.py       # Attendance stage tests (NEW)
```

---

## 🏗️ Component Design

### 1. `database/base_database.py` - Abstract Interfaces

**Purpose**: Define data structures and abstract interfaces.

**Key Classes**:
```python
@dataclass
class Person:
    id: Optional[int]
    name: str
    email: Optional[str]
    department: Optional[str]
    enrollment_date: Optional[datetime]
    status: str  # 'active', 'inactive', 'suspended'

@dataclass
class EmbeddingRecord:
    id: Optional[int]
    person_id: int
    embedding: np.ndarray  # (512,) float32
    quality_score: float
    yaw_angle: Optional[float]
    is_primary: bool

@dataclass
class AttendanceRecord:
    id: Optional[int]
    person_id: int
    timestamp: datetime
    confidence: float
    quality_score: Optional[float]
    track_id: Optional[int]
```

### 2. `database/person_database.py` - Person Management

**Purpose**: Handle person CRUD operations.

**Key Methods**:
```python
class PersonDatabase:
    def add_person(name: str, email: str = None, department: str = None) -> int
    def get_person_by_id(person_id: int) -> Optional[Person]
    def get_person_by_name(name: str) -> Optional[Person]
    def list_active_persons() -> List[Person]
    def update_person(person_id: int, **kwargs) -> None
    def delete_person(person_id: int) -> None
```

**Features**:
- Unique name constraint
- Cascade delete (removes embeddings + attendance)
- Status management (active/inactive/suspended)

### 3. `database/embedding_database.py` - Embedding Storage & Search

**Purpose**: Store embeddings and perform similarity search.

**Key Methods**:
```python
class EmbeddingDatabase:
    def add_embedding(person_id: int, embedding: np.ndarray, 
                     quality_score: float, is_primary: bool = False) -> int
    def search_similar(query_embedding: np.ndarray, 
                      threshold: float = 0.6) -> Optional[Tuple[int, float]]
    def get_person_embeddings(person_id: int) -> List[EmbeddingRecord]
    def get_primary_embedding(person_id: int) -> Optional[np.ndarray]
```

**Matching Algorithm**:
- **Cosine similarity**: `similarity = np.dot(query, stored)` (assumes L2-normalized)
- **Threshold**: 0.6 default (configurable)
- **Linear search**: O(N) acceptable for <1000 people
- **Future**: FAISS/Annoy for >1000 people

**Performance**:
```
Search time (N embeddings):
- N=100: ~5ms
- N=1000: ~50ms
- N=10000: ~500ms (consider FAISS)
```

### 4. `database/attendance_database.py` - Attendance Logging

**Purpose**: Log attendance with cooldown prevention.

**Key Methods**:
```python
class AttendanceDatabase:
    def mark_attendance(person_id: int, confidence: float,
                       track_id: int = None, frame_number: int = None) -> bool
    def get_today_attendance() -> List[AttendanceRecord]
    def get_person_attendance(person_id: int, 
                             start_date: datetime, end_date: datetime) -> List[AttendanceRecord]
    def is_within_cooldown(person_id: int, camera_id: str) -> bool
```

**Cooldown Logic**:
```python
# Don't log if person marked within last N minutes
cooldown_threshold = datetime.now() - timedelta(minutes=60)
if last_attendance > cooldown_threshold:
    return False  # Skip, within cooldown
```

**Benefits**:
- Prevent duplicate attendance from same person
- Configurable cooldown period (default 60 min)
- Per-camera cooldown tracking

### 5. `database/database_manager.py` - Singleton Coordinator

**Purpose**: Unified interface to all database operations (Facade pattern).

**Design Pattern**: Singleton + Facade

```python
class DatabaseManager:
    _instance = None
    
    @classmethod
    def get_instance(cls, config: dict = None) -> 'DatabaseManager':
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance
    
    def __init__(self, config: dict):
        self.persons = PersonDatabase(db_path)
        self.embeddings = EmbeddingDatabase(db_path)
        self.attendance = AttendanceDatabase(attendance_path, cooldown)
        self.connect()
```

**Usage**:
```python
# Initialize once
db = DatabaseManager.get_instance(config)

# Use throughout application
person_id = db.persons.add_person("John Doe")
db.embeddings.add_embedding(person_id, embedding, quality)
db.attendance.mark_attendance(person_id, confidence)
```

---

## 🔄 Pipeline Integration

### 6. `pipeline/enrollment_stage.py` - Enrollment Workflow

**Purpose**: Register new faces to the database.

**Workflow**:
1. Start enrollment session (provide name)
2. Capture N frames (10-20) with quality assessment
3. Select best quality frames (top 5)
4. Extract embeddings for selected frames
5. Store person + embeddings in database
6. Publish enrollment event

**API**:
```python
class EnrollmentStage:
    def start_enrollment(name: str, email: str = None, department: str = None)
    def add_sample(embedding: np.ndarray, quality_score: float) -> bool
    def complete_enrollment() -> Optional[int]
```

**Example Usage**:
```python
enrollment = EnrollmentStage(config, db_manager)
enrollment.start_enrollment("John Doe", "john@example.com")

# Capture 10 frames
for _ in range(10):
    result = pipeline.process_frame(frame)
    if result['recognition_result']:
        for track in result['tracks']:
            if track.embedding is not None:
                enrollment.add_sample(track.embedding, track.quality)

# Complete enrollment
person_id = enrollment.complete_enrollment()
print(f"Enrolled: ID {person_id}")
```

### 7. `pipeline/attendance_stage.py` - Attendance Marking

**Purpose**: Match faces and mark attendance.

**Workflow**:
1. Receive tracks with embeddings from recognition stage
2. Search database for matching embeddings
3. If match found (similarity > threshold) → mark attendance
4. If no match → log as unknown face
5. Publish attendance events

**API**:
```python
class AttendanceStage:
    def process(tracks: List[Track], frame_number: int = None) -> Dict[str, Any]
    def get_stats() -> Dict[str, Any]
```

**Integration with Orchestrator**:
```python
# pipeline/orchestrator.py modifications

class PipelineOrchestrator:
    def __init__(self, config):
        # ... existing stages ...
        
        # Initialize attendance stage
        if config.get('attendance.enabled', False):
            self.db_manager = DatabaseManager.get_instance(config.to_dict())
            self.attendance_stage = AttendanceStage(config.to_dict(), self.db_manager)
        else:
            self.attendance_stage = None
    
    def process_frame(self, frame):
        # ... detection, tracking, recognition ...
        
        # Attendance marking
        attendance_result = None
        if self.attendance_stage is not None:
            attendance_result = self.attendance_stage.process(tracks, self.frame_count)
        
        return {
            # ... existing results ...
            'attendance_result': attendance_result
        }
```

**Return Value**:
```python
{
    'matches_found': 2,         # Number of recognized persons
    'attendance_marked': 1,     # Number of attendance records created
    'unknown_faces': 3,         # Number of unrecognized faces
    'cooldown_skipped': 1       # Skipped due to cooldown
}
```

---

## ⚙️ Configuration

### `config.yaml` Additions

```yaml
# Database Configuration
database:
  type: sqlite
  path: data/faces.db
  max_enrolled: 1000
  max_faces_per_person: 5
  enable_fuzzy_match: false

# Attendance Configuration
attendance:
  enabled: true
  database_path: data/attendance.db
  auto_mark: true
  cooldown_minutes: 60

# Recognition (update)
recognition:
  # ... existing config ...
  similarity_threshold: 0.6
  distance_metric: cosine
```

---

## 🎯 Event System Integration

### New Event Types

```python
# common/event_system.py additions

class EventType(Enum):
    # ... existing events ...
    
    # Recognition Events
    FACE_RECOGNIZED = "face_recognized"
    UNKNOWN_FACE = "unknown_face"
    
    # Attendance Events
    ATTENDANCE_MARKED = "attendance_marked"
    ENROLLMENT_COMPLETE = "enrollment_complete"
```

### Event Publishing

```python
# When attendance marked
self.events.publish(
    EventType.ATTENDANCE_MARKED,
    {
        'person_id': person_id,
        'name': person_name,
        'similarity': 0.85,
        'track_id': 123,
        'timestamp': datetime.now()
    }
)

# When unknown face detected
self.events.publish(
    EventType.UNKNOWN_FACE,
    {
        'track_id': 456,
        'quality': 0.72,
        'frame_number': 1234
    }
)
```

---

## 🧪 Testing Strategy

### Unit Tests

**`tests/test_database/test_person_database.py`**:
- ✅ Add person (valid name)
- ✅ Add person (duplicate name → error)
- ✅ Get person by ID
- ✅ Get person by name
- ✅ Update person
- ✅ Delete person (cascade deletes embeddings)
- ✅ List active persons

**`tests/test_database/test_embedding_database.py`**:
- ✅ Add embedding (valid)
- ✅ Add embedding (invalid shape → error)
- ✅ Search similar (match found)
- ✅ Search similar (no match)
- ✅ Search similar (multiple embeddings per person)
- ✅ Get primary embedding

**`tests/test_database/test_attendance_database.py`**:
- ✅ Mark attendance (success)
- ✅ Mark attendance (within cooldown → skip)
- ✅ Get today's attendance
- ✅ Get person attendance (date range)

### Integration Tests

**`tests/test_enrollment.py`**:
- ✅ Enroll person (full workflow)
- ✅ Enroll with quality filtering
- ✅ Verify embeddings stored correctly
- ✅ Verify event published

**`tests/test_attendance_stage.py`**:
- ✅ Recognize enrolled person
- ✅ Unknown face handling
- ✅ Cooldown prevention
- ✅ Event publishing

---

## 📊 Performance Considerations

### Similarity Search Optimization

**Current Implementation**: Linear search O(N)
- Acceptable for: <1000 people
- Time complexity: ~50ms for 1000 embeddings

**Future Optimizations** (if >1000 people):

**FAISS** (Facebook AI Similarity Search):
```python
import faiss

# Build index
index = faiss.IndexFlatIP(512)  # Inner product (cosine similarity)
index.add(embeddings)  # Add all embeddings

# Search
D, I = index.search(query_embedding, k=1)  # Find top-1 match
similarity = D[0][0]
person_id = I[0][0]
```
- Time complexity: O(log N)
- Memory: ~2KB per person (512 floats)

**Annoy** (Spotify):
```python
from annoy import AnnoyIndex

# Build index
index = AnnoyIndex(512, 'angular')  # Angular = cosine
for i, emb in enumerate(embeddings):
    index.add_item(i, emb)
index.build(10)  # 10 trees

# Search
matches = index.get_nns_by_vector(query_embedding, 1, include_distances=True)
```
- Time complexity: O(log N)
- Disk-based (low memory)

### Database Indexing

Already included in schema:
- `idx_embeddings_person_id` - Fast embedding retrieval
- `idx_attendance_person_id` - Fast attendance queries
- `idx_attendance_timestamp` - Date range queries
- `idx_sessions_check_in` - Session tracking

---

## 📝 Implementation Checklist

### Week 1: Database Foundation

- [ ] Create `database/schema.sql` with all tables
- [ ] Implement `database/base_database.py` (dataclasses)
- [ ] Implement `database/person_database.py` (CRUD)
- [ ] Implement `database/embedding_database.py` (storage + search)
- [ ] Implement `database/attendance_database.py` (logging + cooldown)
- [ ] Implement `database/database_manager.py` (Singleton + Facade)
- [ ] Write unit tests for each database class (6 test files)
- [ ] Verify database creation and schema initialization

### Week 2: Pipeline Integration

- [ ] Implement `pipeline/enrollment_stage.py`
- [ ] Implement `pipeline/attendance_stage.py`
- [ ] Update `pipeline/orchestrator.py` to integrate attendance
- [ ] Update `common/event_system.py` with new event types
- [ ] Update `config.yaml` with database/attendance sections
- [ ] Update `database/__init__.py` to export new classes
- [ ] Write integration tests (enrollment + attendance workflows)

### Week 3: Testing & Refinement

- [ ] End-to-end enrollment test (full workflow)
- [ ] End-to-end attendance test (multi-person)
- [ ] Benchmark similarity search performance
- [ ] Test cooldown logic (edge cases)
- [ ] Test with 10+ enrolled persons
- [ ] Create enrollment CLI tool (optional)
- [ ] Update documentation (README, module READMEs)
- [ ] Create demo script showing enrollment + attendance

---

## 🎯 Success Criteria

Phase 4 is complete when:

- ✅ Person enrollment works (5 embeddings per person)
- ✅ Similarity matching works (>90% accuracy with threshold=0.6)
- ✅ Attendance logged with cooldown (no duplicates within 60min)
- ✅ Unknown faces detected and logged
- ✅ All unit tests passing (database layer)
- ✅ All integration tests passing (enrollment + attendance)
- ✅ Events published correctly (Observer pattern)
- ✅ Config-driven (no hardcoded values)
- ✅ Follows existing design patterns (Singleton, Facade, Observer)
- ✅ Documentation complete

---

## 🚨 Known Challenges & Solutions

### Challenge 1: Similarity Threshold Tuning

**Problem**: What similarity threshold ensures good accuracy without false positives?

**Solutions**:
1. Start with 0.6 (standard for face recognition)
2. Collect real-world data during testing
3. Analyze false positive/negative rates
4. Make threshold configurable per deployment
5. Consider adaptive threshold based on confidence distribution

### Challenge 2: Multiple Embeddings Per Person

**Problem**: How to match when person has 5 different embeddings?

**Solutions**:
1. **Average voting**: Compare query with all 5, take best match
2. **Primary embedding**: Use only best quality embedding (faster)
3. **Threshold lowering**: Lower threshold if matching primary only
4. **Hybrid**: Try primary first, fallback to all if no match

**Recommendation**: Average voting for accuracy, primary embedding for speed.

### Challenge 3: Handling Lighting Changes

**Problem**: Person enrolled in good lighting, but recognized in poor lighting.

**Solutions**:
1. Enroll with varied lighting conditions (if possible)
2. Store multiple embeddings from different sessions
3. Use quality scorer to reject poor-quality matches
4. Consider re-enrollment if consistent mismatch

### Challenge 4: Database Performance at Scale

**Problem**: Linear search slows down with >1000 people.

**Solutions**:
1. Start with linear search (acceptable for MVP)
2. Add FAISS if >1000 people
3. Cache frequently accessed embeddings
4. Partition database by department/group
5. Consider vector database (Qdrant, Milvus) for large-scale

---

## 📚 References

### Face Recognition Papers
- **ArcFace**: [Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
- **FaceNet**: [A Unified Embedding for Face Recognition](https://arxiv.org/abs/1503.03832)

### Similarity Search
- **FAISS**: https://github.com/facebookresearch/faiss
- **Annoy**: https://github.com/spotify/annoy
- **Qdrant**: https://qdrant.tech/ (vector database)

### Related Documentation
- **Phase 3A**: `docs/PHASE_3A_WEEK1_COMPLETE.md` (Recognition baseline)
- **Architecture**: `ARCHITECTURE.md` (V3 HYBRID design)
- **Developer Guide**: `docs/DEVELOPER_GUIDE.md` (Contribution guidelines)

---

**Status**: 📋 Planning Document Complete

**Next Step**: Begin Phase 4 implementation (Week 1: Database Foundation)

**Last Updated**: November 20, 2025
