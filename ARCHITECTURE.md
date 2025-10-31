attendance_system/
│
├── models/                        # All models used in production
│   ├── detection/                 # Face detection models (e.g., YOLOv8n TFLite)
│   │   └── yolov8n_face_int8.tflite
│   ├── alignment/                 # Face alignment models (future)
│   ├── embedding/                 # Face embedding models (future)
│   └── ... (other model types)
│
├── pipeline/                      # Main pipeline code, modular by stage
│   ├── detection.py               # Face detection logic
│   ├── tracking.py                # ByteTrack or other tracking logic
│   ├── alignment.py               # Face alignment logic (future)
│   ├── embedding.py               # Face embedding logic (future)
│   ├── recognition.py             # Face recognition logic (future)
│   ├── attendance.py              # Attendance marking logic (future)
│   ├── config.yaml                # Pipeline configuration
│   └── main.py                    # Orchestrates all stages
│
├── server/                        # Server/API deployment code (future)
│   ├── app.py
│   └── ... (API, web, deployment scripts)
│
├── requirements.txt               # Production dependencies
└── README.md
