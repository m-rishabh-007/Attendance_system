# Database Component - Embeddings & Attendance Storage

**Status**: 🚧 Not Yet Implemented (Future Phase)

**Purpose**: Store face embeddings, person data, and attendance records.

---

## 📋 Planned Implementation

Database will store:
- Face embeddings for known individuals
- Person metadata (name, ID, enrollment date)
- Attendance records (timestamp, confidence)
- System logs and analytics

---

## 🎯 Planned Files

| File | Purpose | Status |
|------|---------|--------|
| `embeddings_db.py` | Face embeddings storage | 🚧 Planned |
| `attendance_db.py` | Attendance records | 🚧 Planned |
| `person_db.py` | Person metadata | 🚧 Planned |
| `schema.sql` | Database schema | 🚧 Planned |
| `migrations/` | Database migrations | 🚧 Planned |

---

## 🔄 Planned Integration

### Pipeline Flow (Future)

```
recognition_stage.process(aligned_faces)
    ↓ (identities)
attendance_stage.process(identities)  ← NEW STAGE
    ↓
database.record_attendance(identity, timestamp)
```

### Usage Example (Future)

```python
# pipeline/attendance_stage.py
from database.attendance_db import AttendanceDB

class AttendanceStage:
    def __init__(self, config):
        self.db = AttendanceDB(config['database']['path'])
    
    def process(self, identities):
        for identity in identities:
            self.db.record_attendance(
                person_id=identity.person_id,
                timestamp=time.time(),
                confidence=identity.confidence
            )
```

---

## 🗄️ Planned Schema

### Persons Table
```sql
CREATE TABLE persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    enrollment_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'active'
);
```

### Embeddings Table
```sql
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    embedding BLOB NOT NULL,  -- 512-dim float32 vector
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES persons(id)
);
```

### Attendance Table
```sql
CREATE TABLE attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    timestamp DATETIME NOT NULL,
    confidence FLOAT NOT NULL,
    track_id INTEGER,
    FOREIGN KEY (person_id) REFERENCES persons(id)
);
```

---

## 🔧 Planned Configuration

```yaml
# config.yaml (future)
database:
  type: sqlite  # 'sqlite', 'postgresql', etc.
  path: database/attendance.db
  embeddings_path: database/embeddings.db
  backup_enabled: true
  backup_interval: 86400  # Daily backup
```

---

## 🎯 Key Operations

### Embedding Storage

```python
# Add new person with embedding
db.add_person(name="John Doe", embedding=embedding_vector)

# Update embedding
db.update_embedding(person_id=1, embedding=new_embedding)

# Search similar embeddings
matches = db.search_embeddings(query_embedding, threshold=0.6)
```

### Attendance Recording

```python
# Record attendance
db.record_attendance(
    person_id=1,
    timestamp=datetime.now(),
    confidence=0.85
)

# Query attendance
records = db.get_attendance(
    person_id=1,
    start_date=datetime(2025, 11, 1),
    end_date=datetime(2025, 11, 30)
)
```

---

## 📚 Related Documentation

- **Parent Guide**: `docs/DEVELOPER_GUIDE.md`
- **Architecture**: `docs/ARCHITECTURE_V3_HYBRID.md#future-phases`
- **Pipeline**: `pipeline/README.md#adding-new-stages`
- **Recognizers**: `recognizers/README.md` (embedding source)

---

## 🚀 Implementation Checklist

When implementing this component:

- [ ] Design database schema
- [ ] Create `embeddings_db.py` with CRUD operations
- [ ] Create `attendance_db.py` with logging operations
- [ ] Create `person_db.py` with person management
- [ ] Implement efficient embedding search (FAISS/Annoy)
- [ ] Create `pipeline/attendance_stage.py`
- [ ] Update `pipeline/orchestrator.py` to use attendance
- [ ] Add database config to `config.yaml`
- [ ] Create backup/restore utilities
- [ ] Create unit tests
- [ ] Update this README with implementation details

---

**Status**: Waiting for recognition phase completion.

**Last Updated**: November 5, 2025
