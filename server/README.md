# Server Component - API & WebSocket Deployment

**Status**: 🚧 Not Yet Implemented (Future Phase)

**Purpose**: Provide REST API and WebSocket interface for remote access.

---

## 📋 Planned Implementation

Server will provide:
- REST API for system control and queries
- WebSocket for real-time updates
- Authentication and authorization
- Web dashboard for monitoring

---

## 🎯 Planned Files

| File | Purpose | Status |
|------|---------|--------|
| `main.py` | FastAPI application entry point | 🚧 Planned |
| `api.py` | REST API endpoints | 🚧 Planned |
| `websocket.py` | WebSocket connections | 🚧 Planned |
| `auth.py` | JWT authentication | 🚧 Planned |
| `models.py` | Pydantic request/response models | 🚧 Planned |
| `static/` | Web dashboard files | 🚧 Planned |

---

## 🔄 Planned Integration

### Architecture (Future)

```
┌─────────────────────────────────────────────┐
│           Web Dashboard (React)             │
└─────────────────────────────────────────────┘
        ↓ HTTP/WebSocket
┌─────────────────────────────────────────────┐
│         FastAPI Server (server/)            │
│  - REST API endpoints                       │
│  - WebSocket for real-time updates          │
│  - JWT authentication                       │
└─────────────────────────────────────────────┘
        ↓ Calls
┌─────────────────────────────────────────────┐
│    Pipeline Orchestrator (pipeline/)        │
│  - Detection + Tracking                     │
│  - Recognition + Attendance                 │
└─────────────────────────────────────────────┘
```

---

## 🌐 Planned REST API Endpoints

### System Control

```
POST   /api/start         Start attendance system
POST   /api/stop          Stop attendance system
GET    /api/status        Get system status
```

### Attendance Management

```
GET    /api/attendance             List attendance records
GET    /api/attendance/{person_id} Get person's attendance
POST   /api/attendance/export      Export attendance data
```

### Person Management

```
GET    /api/persons               List all persons
POST   /api/persons               Add new person
GET    /api/persons/{id}          Get person details
PUT    /api/persons/{id}          Update person
DELETE /api/persons/{id}          Delete person
```

### Face Enrollment

```
POST   /api/enroll                Enroll new face
GET    /api/enroll/{person_id}    Get enrollment status
```

### Analytics

```
GET    /api/analytics/today       Today's statistics
GET    /api/analytics/summary     Overall summary
GET    /api/analytics/trends      Attendance trends
```

---

## 🔌 Planned WebSocket Events

### Server → Client

```javascript
// Real-time face detection
{
  "event": "face_detected",
  "data": {
    "track_id": 1,
    "bbox": [100, 100, 200, 200],
    "confidence": 0.85
  }
}

// Face recognition
{
  "event": "face_recognized",
  "data": {
    "track_id": 1,
    "person_id": 42,
    "name": "John Doe",
    "confidence": 0.92
  }
}

// Attendance marked
{
  "event": "attendance_marked",
  "data": {
    "person_id": 42,
    "name": "John Doe",
    "timestamp": "2025-11-05T10:30:00Z"
  }
}

// System status update
{
  "event": "status_update",
  "data": {
    "fps": 25,
    "active_tracks": 3,
    "faces_recognized": 5
  }
}
```

### Client → Server

```javascript
// Subscribe to events
{
  "action": "subscribe",
  "events": ["face_detected", "attendance_marked"]
}

// Unsubscribe from events
{
  "action": "unsubscribe",
  "events": ["face_detected"]
}
```

---

## 🔧 Planned Configuration

```yaml
# config.yaml (future)
server:
  enabled: true
  host: 0.0.0.0
  port: 8000
  cors_origins:
    - http://localhost:3000
    - https://dashboard.example.com
  
  auth:
    enabled: true
    jwt_secret: ${JWT_SECRET}  # From environment
    token_expiry: 3600  # 1 hour
  
  websocket:
    max_connections: 100
    heartbeat_interval: 30
```

---

## 💻 Usage Examples (Future)

### Starting Server

```bash
# Run server
python -m server.main

# Or with uvicorn directly
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

### REST API Client

```python
import requests

# Start system
response = requests.post('http://localhost:8000/api/start')

# Get today's attendance
response = requests.get('http://localhost:8000/api/attendance/today')
records = response.json()

# Enroll new person
response = requests.post('http://localhost:8000/api/persons', json={
    'name': 'John Doe',
    'email': 'john@example.com'
})
person_id = response.json()['id']
```

### WebSocket Client

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws');

// Subscribe to events
ws.send(JSON.stringify({
  action: 'subscribe',
  events: ['face_detected', 'attendance_marked']
}));

// Handle events
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`Event: ${data.event}`, data.data);
};
```

---

## 🔐 Authentication (Planned)

### JWT Token Flow

```
1. Client: POST /api/auth/login {username, password}
2. Server: Return JWT token
3. Client: Include token in headers: Authorization: Bearer <token>
4. Server: Validate token on each request
```

### Example

```python
# Login
response = requests.post('http://localhost:8000/api/auth/login', json={
    'username': 'admin',
    'password': 'secure_password'
})
token = response.json()['access_token']

# Use token
headers = {'Authorization': f'Bearer {token}'}
response = requests.get('http://localhost:8000/api/attendance', headers=headers)
```

---

## 📚 Related Documentation

- **Parent Guide**: `docs/DEVELOPER_GUIDE.md`
- **Architecture**: `docs/ARCHITECTURE_V3_HYBRID.md#future-phases`
- **Pipeline**: `pipeline/README.md` (system control)
- **Database**: `database/README.md` (data source)

---

## 🚀 Implementation Checklist

When implementing this component:

- [ ] Create FastAPI application (`main.py`)
- [ ] Implement REST API endpoints (`api.py`)
- [ ] Implement WebSocket handler (`websocket.py`)
- [ ] Implement JWT authentication (`auth.py`)
- [ ] Create Pydantic models (`models.py`)
- [ ] Build web dashboard (React/Vue)
- [ ] Add CORS configuration
- [ ] Create API documentation (OpenAPI/Swagger)
- [ ] Add rate limiting
- [ ] Implement logging and monitoring
- [ ] Create deployment guide (Docker, systemd)
- [ ] Create unit tests
- [ ] Update this README with implementation details

---

**Status**: Waiting for complete pipeline implementation (detection → tracking → alignment → recognition → attendance).

**Last Updated**: November 5, 2025
