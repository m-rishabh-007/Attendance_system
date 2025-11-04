#!/usr/bin/env python3
"""
Face Attendance System v3.0 - V3 HYBRID Architecture

This is now a THIN WRAPPER (50 lines!) that uses the pipeline orchestrator.

Architecture V3 HYBRID features:
- Pipeline orchestration layer (pipeline/) - "what to do"
- Component implementations (detectors/, tracking/) - "how to do it"
- Clear separation of concerns
- Easy to add future phases (alignment, recognition, database, API)

All 4 design patterns preserved:
- Singleton: ConfigManager for centralized configuration
- Factory: DetectorFactory, TrackerFactory for component creation
- Observer: EventSystem for event-driven architecture
- Strategy: Interchangeable algorithms

Author: Rishabh Mishra
Date: November 4, 2025
Version: 3.0.0
"""

import cv2
import numpy as np
import logging
import sys
from pathlib import Path
from typing import Optional
import time

# Add project root to path (attendance_system.py is now at root)
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import our modules
from common.config_manager import ConfigManager
from common.event_system import EventSystem, EventType, Event
from pipeline.orchestrator import PipelineOrchestrator


class FaceAttendanceSystem:
    """
    Main application class - THIN WRAPPER for V3 HYBRID architecture.
    
    This is now just 50 lines! All logic moved to PipelineOrchestrator.
    
    Responsibilities:
    - Load configuration
    - Setup event system
    - Create pipeline orchestrator
    - Open camera
    - Display loop (capture → process → show → repeat)
    
    The orchestrator handles ALL pipeline logic (detection, tracking, etc.)
    
    Example:
        >>> system = FaceAttendanceSystem('config.yaml')
        >>> system.initialize()
        >>> system.run()
    """
    
    def __init__(self, config_path: Path | str):
        """
        Initialize Face Attendance System.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        
        # Setup logging first
        self._setup_logging()
        self.logger = logging.getLogger('FaceAttendanceSystem')
        self.logger.info("="*60)
        self.logger.info("Face Attendance System v3.0 (V3 HYBRID) Starting")
        self.logger.info("="*60)
        
        # Core components (initialized in initialize())
        self.config: Optional[ConfigManager] = None
        self.events: Optional[EventSystem] = None
        self.pipeline: Optional[PipelineOrchestrator] = None
        self.camera = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = 0
        self.last_stats_frame = 0
        
        # State
        self._is_initialized = False
        self._is_running = False
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def initialize(self) -> None:
        """
        Initialize all components.
        
        This method:
        1. Loads configuration (Singleton)
        2. Sets up event system (Observer)
        3. Creates pipeline orchestrator
        4. Opens camera
        5. Subscribes to events
        
        Raises:
            Exception: If initialization fails
        """
        if self._is_initialized:
            self.logger.warning("System already initialized")
            return
        
        try:
            # 1. Load configuration (Singleton pattern)
            self.logger.info("Loading configuration...")
            self.config = ConfigManager()
            self.config.load(str(self.config_path))
            self.logger.info(f"✅ Configuration loaded from {self.config_path}")
            
            # 2. Initialize event system (Observer pattern)
            self.logger.info("Setting up event system...")
            self.events = EventSystem()
            self.logger.info("✅ Event system ready")
            
            # 3. Create pipeline orchestrator (coordinates all stages)
            self.logger.info("Creating pipeline orchestrator...")
            self.pipeline = PipelineOrchestrator(self.config)
            self.logger.info("✅ Pipeline orchestrator ready")
            
            # 4. Open camera
            self.logger.info("Opening camera...")
            camera_config = self.config.get_section('camera')
            camera_id = camera_config.get('device_id', 0)  # Fixed: use device_id from config
            self.camera = cv2.VideoCapture(camera_id)
            
            if not self.camera.isOpened():
                raise RuntimeError(f"Failed to open camera {camera_id}")
            
            # Set camera properties (only if specified in config)
            resolution = camera_config.get('resolution', {})
            width = resolution.get('width') if resolution else None
            height = resolution.get('height') if resolution else None
            fps = camera_config.get('fps')
            
            if width:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps:
                self.camera.set(cv2.CAP_PROP_FPS, fps)
            
            # Get actual camera properties
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.camera.get(cv2.CAP_PROP_FPS))
            
            self.logger.info(f"✅ Camera {camera_id} opened: {actual_width}x{actual_height}@{actual_fps}fps")
            
            # 5. Subscribe to events
            self._subscribe_to_events()
            
            self._is_initialized = True
            self.logger.info("="*60)
            self.logger.info("✅ System initialization complete!")
            self.logger.info("="*60)
            
        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}")
            raise
    
    def _subscribe_to_events(self) -> None:
        """
        Subscribe to system events (Observer Pattern).
        
        This demonstrates loose coupling - event handlers can be
        added/removed without modifying the detection/tracking code.
        """
        if not self.config.get('events.enabled', True):
            return
        
        # Subscribe to face detection events
        if self.config.get('events.subscriptions.face_detected', True):
            self.events.subscribe(EventType.FACE_DETECTED, self._on_face_detected)
        
        if self.config.get('events.subscriptions.face_lost', True):
            self.events.subscribe(EventType.FACE_LOST, self._on_face_lost)
        
        # Future: Add more event handlers here
        # self.events.subscribe(EventType.FACE_RECOGNIZED, self._on_face_recognized)
        # self.events.subscribe(EventType.ATTENDANCE_MARKED, self._on_attendance_marked)
    
    def _on_face_detected(self, event: Event) -> None:
        """Event handler for face detection."""
        self.logger.debug(f"🆕 Face detected: ID={event.data.get('track_id')}")
    
    def _on_face_lost(self, event: Event) -> None:
        """Event handler for lost tracks."""
        self.logger.info(f"❌ Track lost: ID={event.data.get('track_id')}")
    
    def run(self) -> None:
        """
        Main execution loop.
        
        Processes frames continuously until user quits or error occurs.
        """
        if not self._is_initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        self._is_running = True
        self.start_time = time.time()
        
        self.logger.info("Starting main loop...")
        self.logger.info("Press 'q' to quit")
        self.logger.info("")
        
        try:
            while self._is_running:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    self.logger.warning("Failed to grab frame")
                    break
                
                self.frame_count += 1
                
                # Process frame through pipeline orchestrator
                result = self.pipeline.process_frame(frame)
                frame = result['annotated_frame']
                
                # Display frame
                if self.config.get('display.show_window', True):
                    cv2.imshow(
                        self.config.get('display.window_name', 'Face Attendance'),
                        frame
                    )
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.logger.info("\n🛑 User requested quit")
                    break
                
                # Print stats
                stats_interval = self.config.get('logging.stats_interval', 50)
                if self.frame_count % stats_interval == 0:
                    self._print_stats()
        
        except KeyboardInterrupt:
            self.logger.info("\n🛑 Interrupted by user (Ctrl+C)")
        except Exception as e:
            self.logger.error(f"\n❌ Error during execution: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    # NOTE: _process_frame() and _draw_annotations() removed in V3 HYBRID
    # All processing logic now handled by pipeline/orchestrator.py
    # This keeps attendance_system.py thin (just UI/camera handling)
    
    def _print_stats(self) -> None:
        """Print performance statistics."""
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        self.logger.info(
            f"[Frame {self.frame_count:4d}] "
            f"FPS: {avg_fps:5.1f} | "
            f"Elapsed: {elapsed:.1f}s"
        )
    
    def cleanup(self) -> None:
        """
        Clean up resources.
        
        Releases camera, closes windows, prints final statistics.
        """
        self._is_running = False
        
        # Print final stats
        if self.frame_count > 0:
            elapsed = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
            
            self.logger.info("")
            self.logger.info("="*60)
            self.logger.info("📊 Session Summary")
            self.logger.info("="*60)
            self.logger.info(f"   Total Frames:      {self.frame_count}")
            self.logger.info(f"   Duration:          {elapsed:.1f}s")
            self.logger.info(f"   Average FPS:       {avg_fps:.1f}")
            self.logger.info("="*60)
        
        # Release resources
        if self.camera is not None:
            self.camera.release()
        cv2.destroyAllWindows()
        
        # Publish stop event
        if self.events is not None:
            self.events.publish(
                EventType.PIPELINE_STOPPED,
                {'frame_count': self.frame_count},
                source='FaceAttendanceSystem'
            )
        
        self.logger.info("✅ Resources released. Shutdown complete.")
        self.logger.info("")


def main():
    """Main entry point."""
    # Configuration file path
    config_path = PROJECT_ROOT / 'config.yaml'
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        print("Please create config.yaml in project root.")
        sys.exit(1)
    
    # Create and run system
    system = FaceAttendanceSystem(config_path)
    system.initialize()
    system.run()


if __name__ == "__main__":
    main()
