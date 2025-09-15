"""
CCTV Analysis and Notification System

Main application entry point that orchestrates all system components.
"""

import asyncio
import logging
import sys
import signal
import yaml
from pathlib import Path
from typing import Dict, Any

from video_ingestion import VideoProcessor, FrameExtractor, Preprocessor
from video_ingestion.video_processor import VideoSource
from video_ingestion.preprocessor import PreprocessingConfig
try:
    from ai_engine import PersonIdentifier
except ImportError as e:
    logging.warning(f"Full AI engine not available ({e}), using basic face detection only")
    from ai_engine.basic_face_detector import BasicPersonIdentifier as PersonIdentifier
from database import PersonnelManager, EventLogger
from notification_service import NotificationManager
from utils.logging_config import setup_logging
from web_interface import start_web_server, update_system_status, add_detection_event


class CCTVAnalysisSystem:
    """
    Main CCTV Analysis and Notification System orchestrator.
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        self.config_path = config_path
        self.config = {}
        self.running = False
        
        # System components
        self.video_processor = None
        self.frame_extractor = None
        self.preprocessor = None
        self.person_identifier = None
        self.personnel_manager = None
        self.event_logger = None
        self.notification_manager = None
        self.web_server_task = None
        
        # Statistics tracking
        self.total_detections = 0
        self.recognized_persons = 0
        
        # Load configuration
        self._load_config()
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self):
        """Load system configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Configuration file not found: {self.config_path}")
            print("Please copy config.example.yaml to config.yaml and customize it.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            sys.exit(1)
    
    def _setup_logging(self):
        """Setup system logging."""
        logging_config = self.config.get('logging', {})
        setup_logging(logging_config)
    
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            self.logger.info("Initializing CCTV Analysis System components...")
            
            # Video Processing Components
            video_config = self.config.get('video_ingestion', {})
            self.video_processor = VideoProcessor(video_config)
            self.frame_extractor = FrameExtractor(video_config)
            
            # Preprocessing
            preprocessing_config = PreprocessingConfig(**self.config.get('preprocessing', {}))
            self.preprocessor = Preprocessor(preprocessing_config)
            
            # AI Engine
            ai_config = self.config.get('ai_engine', {})
            self.person_identifier = PersonIdentifier(ai_config)
            
            # Database Components
            db_config = self.config.get('database', {})
            self.personnel_manager = PersonnelManager(db_config.get('personnel', {}))
            self.event_logger = EventLogger(db_config.get('events', {}))
            
            # Notification Service
            notification_config = self.config.get('notifications', {})
            self.notification_manager = NotificationManager(notification_config)
            
            # Setup video sources
            self._setup_video_sources()
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _setup_video_sources(self):
        """Setup video sources from configuration."""
        video_sources = self.config.get('video_sources', [])
        
        for source_config in video_sources:
            if source_config.get('enabled', True):
                source = VideoSource(
                    source_id=source_config['source_id'],
                    url=source_config['url'],
                    fps=source_config.get('fps', 30),
                    resolution=tuple(source_config.get('resolution', [1920, 1080])),
                    enabled=source_config.get('enabled', True)
                )
                
                self.video_processor.add_video_source(source)
                self.logger.info(f"Added video source: {source.source_id}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.running = False
    
    async def process_video_stream(self, source_id: str):
        """
        Process video stream from a specific source.
        
        Args:
            source_id: ID of the video source to process
        """
        self.logger.info(f"Starting video processing for source: {source_id}")
        
        # Start video capture
        if not self.video_processor.start_capture(source_id):
            self.logger.error(f"Failed to start capture for source: {source_id}")
            return
        
        try:
            while self.running:
                # Get frame from video source
                ret, frame = self.video_processor.get_frame(source_id)
                
                if not ret or frame is None:
                    await asyncio.sleep(0.1)  # Longer delay when no frame available
                    continue
                
                # Extract frame with metadata
                extraction_result = self.frame_extractor.extract_frame(source_id, frame)
                
                if extraction_result is None:
                    continue
                
                extracted_frame, frame_metadata = extraction_result
                
                # Preprocess frame
                preprocessing_result = self.preprocessor.preprocess_frame(extracted_frame)
                
                if preprocessing_result is None:
                    continue
                
                processed_frame, preprocessing_metadata = preprocessing_result
                
                # Identify persons in frame
                identification_result = self.person_identifier.identify_persons_in_frame(
                    processed_frame, frame_metadata
                )
                
                # Update statistics and log events
                if identification_result.success and identification_result.identified_persons:
                    self.total_detections += len(identification_result.identified_persons)
                    self.recognized_persons += sum(1 for p in identification_result.identified_persons if p.is_recognized)
                    
                    # Log events to web interface
                    add_detection_event(source_id, identification_result.identified_persons)
                    
                    # Update web interface status
                    active_cameras = list(self.video_processor.active_captures.keys())
                    update_system_status(
                        status="running",
                        active_cameras=active_cameras,
                        total_detections=self.total_detections,
                        recognized_persons=self.recognized_persons
                    )
                    
                    # TODO: Convert identification results to database events and log them
                    pass
                
                # Process notifications
                if identification_result.success:
                    await self.notification_manager.process_identification_result(identification_result)
                
                # Brief pause to prevent overwhelming the system
                await asyncio.sleep(0.01)
                
        except Exception as e:
            self.logger.error(f"Error processing video stream {source_id}: {e}")
        finally:
            self.video_processor.stop_capture(source_id)
            self.logger.info(f"Stopped video processing for source: {source_id}")
    
    async def run(self):
        """
        Main application loop.
        """
        self.logger.info("Starting CCTV Analysis System...")
        self.running = True
        
        try:
            # Start web server if enabled
            web_config = self.config.get('web_interface', {})
            if web_config.get('enabled', True):  # Default to enabled
                self.web_server_task = asyncio.create_task(start_web_server(self.config))
                self.logger.info(f"Web interface starting on http://{web_config.get('host', '127.0.0.1')}:{web_config.get('port', 8080)}")
            
            # Get enabled video sources
            enabled_sources = []
            for source_id in self.video_processor.video_sources:
                source = self.video_processor.video_sources[source_id]
                if source.enabled:
                    enabled_sources.append(source_id)
            
            if not enabled_sources:
                self.logger.warning("No enabled video sources found")
                return
            
            # Initialize web interface status
            update_system_status(
                status="starting",
                active_cameras=enabled_sources,
                total_detections=0,
                recognized_persons=0
            )
            
            # Create tasks for each video source
            tasks = []
            for source_id in enabled_sources:
                task = asyncio.create_task(self.process_video_stream(source_id))
                tasks.append(task)
            
            # Add web server task if it exists
            if self.web_server_task:
                tasks.append(self.web_server_task)
            
            # Update status to running
            update_system_status(
                status="running",
                active_cameras=enabled_sources
            )
            
            # Wait for all tasks to complete or for shutdown signal
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"Error in main application loop: {e}")
        finally:
            await self._cleanup()
    
    async def _cleanup(self):
        """Clean up system resources."""
        self.logger.info("Cleaning up system resources...")
        
        # Update web interface status
        update_system_status(status="shutting_down")
        
        try:
            # Cancel web server task
            if self.web_server_task and not self.web_server_task.done():
                self.web_server_task.cancel()
                try:
                    await self.web_server_task
                except asyncio.CancelledError:
                    pass
            
            # Clean up video processing
            if self.video_processor:
                self.video_processor.cleanup()
            
            # Close database connections
            if self.personnel_manager:
                self.personnel_manager.close()
            
            if self.event_logger:
                self.event_logger.close()
            
            # Final status update
            update_system_status(status="stopped", active_cameras=[])
            
            self.logger.info("System cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


async def main():
    """Application entry point."""
    # Check for configuration file
    config_path = 'config/config.yaml'
    
    if not Path(config_path).exists():
        print(f"Configuration file not found: {config_path}")
        print("Please copy config.example.yaml to config.yaml and customize it.")
        return
    
    # Create and run the system
    system = CCTVAnalysisSystem(config_path)
    
    try:
        await system.run()
    except KeyboardInterrupt:
        print("\\nShutdown requested by user")
    except Exception as e:
        print(f"System error: {e}")
        logging.error(f"System error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
