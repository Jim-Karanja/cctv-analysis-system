#!/usr/bin/env python3
"""
Advanced Security Video Analysis - Uses Threaded Video Capture
for high-performance real-time security monitoring.

This system provides:
- Threaded video capture for non-blocking frame acquisition
- Advanced security analysis with face detection and recognition
- Real-time security event logging
- Performance monitoring and statistics
- Web-compatible frame output
"""

import cv2
import time
import os
import signal
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Import our custom modules
from threaded_video_capture import ThreadedVideoCapture, ThreadedVideoManager
from security_engine import SecurityEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedSecurityAnalysis:
    """Advanced security analysis system with threaded video capture."""
    
    def __init__(self, camera_source=0, buffer_size=5):
        self.camera_source = camera_source
        self.running = False
        
        # Initialize threaded video capture
        self.video_manager = ThreadedVideoManager()
        self.camera_id = "main_camera"
        
        # Initialize security engine
        self.security_engine = SecurityEngine()
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'detections_made': 0,
            'unauthorized_alerts': 0,
            'start_time': None,
            'last_frame_time': 0,
            'processing_fps': 0,
            'capture_fps': 0
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Ensure directories exist
        Path("data").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n🛑 Received signal {signum}, stopping advanced security analysis...")
        self.running = False
    
    def initialize_system(self) -> bool:
        """Initialize the video capture and security systems."""
        print("🔧 Initializing Advanced Security Analysis System...")
        
        # Test basic camera access first
        print(f"🎥 Testing basic camera access on index {self.camera_source}...")
        test_cap = cv2.VideoCapture(self.camera_source)
        if not test_cap.isOpened():
            logger.error(f"Basic camera test failed: Cannot open camera {self.camera_source}")
            return False
        
        ret, frame = test_cap.read()
        if not ret or frame is None:
            logger.error(f"Basic camera test failed: Cannot read frames from camera {self.camera_source}")
            test_cap.release()
            return False
        
        print(f"✅ Basic camera test passed: {frame.shape}")
        test_cap.release()
        time.sleep(0.5)  # Let camera settle
        
        # Add camera source to threaded video manager
        print(f"🧵 Setting up threaded video capture...")
        success = self.video_manager.add_source(
            self.camera_id, 
            self.camera_source, 
            buffer_size=5
        )
        
        if not success:
            logger.error(f"Failed to initialize threaded camera source: {self.camera_source}")
            return False
        
        # Wait for threaded capture to stabilize
        print("⏳ Waiting for threaded capture to stabilize...")
        time.sleep(2)
        
        # Check if threaded camera is providing frames
        attempts = 0
        max_attempts = 10
        while attempts < max_attempts:
            ret, frame, timestamp = self.video_manager.read_frame(self.camera_id)
            if ret and frame is not None:
                print(f"✅ Threaded capture working: {frame.shape}")
                break
            
            attempts += 1
            print(f"⏳ Attempt {attempts}/{max_attempts}: Waiting for frames...")
            time.sleep(0.5)
        
        if attempts >= max_attempts:
            logger.error("Threaded camera is not providing frames after multiple attempts")
            return False
        
        logger.info("✅ Advanced security system initialized successfully")
        return True
    
    def save_status_file(self):
        """Save system status to file for web interface."""
        try:
            status = {
                'system_status': 'running' if self.running else 'stopped',
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                'stats': self.stats.copy(),
                'personnel_count': len(self.security_engine.authorized_personnel),
                'camera_stats': self.video_manager.get_all_stats(),
                'recent_events_count': len(self.security_engine.security_events),
                'unauthorized_alerts': self.security_engine.get_unauthorized_alerts()
            }
            
            with open("data/system_status.json", 'w') as f:
                json.dump(status, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving status file: {e}")
    
    def process_frame(self, frame, timestamp) -> Optional[List]:
        """Process a single frame for security analysis."""
        try:
            # Perform security analysis
            events = self.security_engine.analyze_frame(frame)
            
            if events:
                self.stats['detections_made'] += len(events)
                unauthorized_count = sum(1 for event in events if not event.authorized)
                self.stats['unauthorized_alerts'] += unauthorized_count
                
                # Draw detection boxes on frame
                processed_frame = self.security_engine.draw_detections(frame, events)
            else:
                processed_frame = frame
            
            return processed_frame, events
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame, []
    
    def save_frame_for_web(self, frame):
        """Save current frame for web interface display."""
        try:
            # Resize frame for web display
            height, width = frame.shape[:2]
            if width > 640:
                new_width = 640
                new_height = int((new_width / width) * height)
                web_frame = cv2.resize(frame, (new_width, new_height))
            else:
                web_frame = frame
            
            # Save frame with high quality
            frame_path = "data/current_frame.jpg"
            success = cv2.imwrite(frame_path, web_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            if success:
                self.stats['last_frame_time'] = time.time()
            else:
                logger.warning("Failed to save web frame")
                
        except Exception as e:
            logger.error(f"Error saving web frame: {e}")
    
    def update_statistics(self):
        """Update system statistics."""
        current_time = time.time()
        if self.stats['start_time']:
            elapsed = current_time - self.stats['start_time']
            if elapsed > 0:
                self.stats['processing_fps'] = self.stats['frames_processed'] / elapsed
        
        # Get camera FPS from video manager
        camera_stats = self.video_manager.get_all_stats()
        if self.camera_id in camera_stats:
            self.stats['capture_fps'] = camera_stats[self.camera_id]['fps']
    
    def print_status(self):
        """Print current system status."""
        self.update_statistics()
        
        print(f"📊 Status: Frames: {self.stats['frames_processed']:,}, "
              f"Process FPS: {self.stats['processing_fps']:.1f}, "
              f"Capture FPS: {self.stats['capture_fps']:.1f}, "
              f"Detections: {self.stats['detections_made']}, "
              f"Alerts: {self.stats['unauthorized_alerts']}, "
              f"Time: {datetime.now().strftime('%H:%M:%S')}")
    
    def run(self):
        """Main analysis loop."""
        print("🔒 Advanced Security Video Analysis with Threaded Capture")
        print("🧵 Using non-blocking threaded video capture for optimal performance")
        print("🤖 Advanced face detection and recognition enabled")
        print("🌐 Real-time web interface frame updates")
        print("⏹️  Press Ctrl+C to stop")
        print("=" * 80)
        
        personnel_count = len(self.security_engine.authorized_personnel)
        print(f"👥 {personnel_count} authorized personnel loaded")
        
        if personnel_count == 0:
            print("⚠️  No authorized personnel photos found in data/personnel/")
            print("   Add photos using: python setup_personnel.py")
        
        # Initialize system
        if not self.initialize_system():
            print("❌ Failed to initialize system")
            return False
        
        self.running = True
        self.stats['start_time'] = time.time()
        last_status_time = self.stats['start_time']
        last_save_time = self.stats['start_time']
        
        print("✅ System initialized, starting analysis...")
        print()
        
        try:
            while self.running:
                # Get latest frame from threaded capture
                ret, frame, timestamp = self.video_manager.read_frame(self.camera_id)
                
                if not ret or frame is None:
                    time.sleep(0.01)  # Brief pause if no frame available
                    continue
                
                self.stats['frames_processed'] += 1
                
                # Process frame for security analysis
                processed_frame, events = self.process_frame(frame, timestamp)
                
                # Save frame for web interface (every 200ms for 5 FPS web update)
                current_time = time.time()
                if current_time - last_save_time >= 0.2:
                    self.save_frame_for_web(processed_frame)
                    last_save_time = current_time
                
                # Print status every 10 seconds
                if current_time - last_status_time >= 10:
                    self.print_status()
                    self.save_status_file()
                    last_status_time = current_time
                
                # Brief pause to prevent excessive CPU usage
                time.sleep(0.01)
        
        except Exception as e:
            logger.error(f"Analysis error: {e}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("\n🧹 Cleaning up advanced security analysis system...")
        
        # Stop video capture
        self.video_manager.cleanup()
        
        # Save final events and statistics
        self.security_engine.save_events()
        self.save_status_file()
        
        # Final statistics
        self.update_statistics()
        elapsed = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        
        print(f"\n📊 Final Statistics:")
        print(f"   Total frames processed: {self.stats['frames_processed']:,}")
        print(f"   Average processing FPS: {self.stats['processing_fps']:.1f}")
        print(f"   Average capture FPS: {self.stats['capture_fps']:.1f}")
        print(f"   Total detections: {self.stats['detections_made']}")
        print(f"   Security alerts: {self.stats['unauthorized_alerts']}")
        print(f"   Total runtime: {elapsed:.1f} seconds")
        
        # Keep only recent events
        if len(self.security_engine.security_events) > 100:
            self.security_engine.security_events = self.security_engine.security_events[-100:]
            self.security_engine.save_events()
        
        print("✅ Advanced security analysis completed")

def main():
    """Main entry point."""
    print("🚀 Starting Advanced CCTV Security Analysis System")
    print("=" * 60)
    
    try:
        # Create and run advanced analysis system
        analyzer = AdvancedSecurityAnalysis(camera_source=0, buffer_size=5)
        success = analyzer.run()
        
        if not success:
            print("❌ System failed to start properly")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        logger.error(f"System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
