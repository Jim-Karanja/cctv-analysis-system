#!/usr/bin/env python3
"""
Security-enabled video capture for CCTV system.

This module captures video frames, analyzes them for security threats,
and provides visual feedback on authorized vs unauthorized personnel.
"""

import cv2
import time
import os
import signal
import sys
import json
import logging
from pathlib import Path
from security_engine import SecurityEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecurityVideoCapture:
    """Video capture system with integrated security analysis."""
    
    def __init__(self):
        self.running = False
        self.security_engine = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Ensure data directory exists
        Path("data").mkdir(exist_ok=True)
        
        # Initialize security engine
        self._init_security()
    
    def _signal_handler(self, signum, frame):
        print(f"\n🛑 Received signal {signum}, stopping security video capture...")
        self.running = False
    
    def _init_security(self):
        """Initialize the security engine."""
        try:
            self.security_engine = SecurityEngine()
            logger.info("Security engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize security engine: {e}")
            self.security_engine = None
    
    def run(self):
        """Main capture and analysis loop."""
        print("🔒 Security-Enabled Video Capture")
        print("📁 Frames saved to data/current_frame.jpg with security analysis")
        print("👥 Add authorized personnel photos to data/personnel/")
        print("⏹️  Press Ctrl+C to stop")
        print("=" * 60)
        
        if self.security_engine is None:
            print("❌ Security engine not available, running without analysis")
        else:
            personnel_count = len(self.security_engine.authorized_personnel)
            print(f"👥 {personnel_count} authorized personnel loaded")
            
            if personnel_count == 0:
                print("⚠️  No authorized personnel found in data/personnel/")
                print("   All detected faces will be marked as UNAUTHORIZED")
        
        # Try to open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not open camera (index 0)")
            print("💡 Make sure your camera is connected and not used by another application")
            return False
        
        print("✅ Camera opened successfully")
        
        # Configure camera for better performance
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
        cap.set(cv2.CAP_PROP_FPS, 10)        # Set to 10 FPS
        
        self.running = True
        frame_count = 0
        detection_count = 0
        unauthorized_count = 0
        start_time = time.time()
        last_status_time = start_time
        last_save_time = start_time
        
        try:
            while self.running:
                # Capture frame
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    print("⚠️  Failed to capture frame, retrying...")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                processed_frame = frame.copy()
                
                # Security analysis
                if self.security_engine is not None:
                    try:
                        # Analyze frame for faces and security events
                        security_events = self.security_engine.analyze_frame(frame)
                        
                        if security_events:
                            detection_count += len(security_events)
                            
                            # Count unauthorized detections
                            unauthorized_in_frame = sum(1 for event in security_events if not event.authorized)
                            unauthorized_count += unauthorized_in_frame
                            
                            # Draw detection boxes and labels
                            processed_frame = self.security_engine.draw_detections(frame, security_events)
                            
                            # Log significant events
                            for event in security_events:
                                if not event.authorized:
                                    logger.warning(f"🚨 SECURITY ALERT: Unauthorized person detected - {event.person_name}")
                                else:
                                    logger.info(f"✅ Authorized person: {event.person_name}")
                    
                    except Exception as e:
                        logger.error(f"Security analysis error: {e}")
                
                # Resize frame for web display (640px width max)
                height, width = processed_frame.shape[:2]
                if width > 640:
                    new_width = 640
                    new_height = int((new_width / width) * height)
                    processed_frame = cv2.resize(processed_frame, (new_width, new_height))
                
                # Save frame every 0.2 seconds (5 FPS)
                current_time = time.time()
                if current_time - last_save_time >= 0.2:
                    frame_path = "data/current_frame.jpg"
                    success = cv2.imwrite(frame_path, processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    
                    if success:
                        last_save_time = current_time
                    else:
                        logger.error("Failed to save frame")
                
                # Status update every 15 seconds
                if current_time - last_status_time >= 15:
                    elapsed = current_time - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    
                    print(f"📊 Frames: {frame_count}, FPS: {fps:.1f}, "
                          f"Detections: {detection_count}, Unauthorized: {unauthorized_count}, "
                          f"Time: {time.strftime('%H:%M:%S')}")
                    
                    # Save security events
                    if self.security_engine is not None:
                        self.security_engine.save_events()
                    
                    last_status_time = current_time
                
                # Brief pause to control frame rate
                time.sleep(0.03)  # ~30 FPS processing
        
        except Exception as e:
            logger.error(f"Error during capture: {e}")
        
        finally:
            cap.release()
            
            # Final statistics
            elapsed = time.time() - start_time
            avg_fps = frame_count / elapsed if elapsed > 0 else 0
            
            print(f"\n📊 Final Statistics:")
            print(f"   Frames processed: {frame_count}")
            print(f"   Total time: {elapsed:.1f} seconds")
            print(f"   Average FPS: {avg_fps:.1f}")
            print(f"   Total detections: {detection_count}")
            print(f"   Unauthorized alerts: {unauthorized_count}")
            
            # Save final events
            if self.security_engine is not None:
                self.security_engine.save_events()
                print(f"   Security events saved to data/security_events.json")
            
            print("✅ Security video capture completed")
        
        return True

def main():
    """Main function."""
    capturer = SecurityVideoCapture()
    
    try:
        success = capturer.run()
        if not success:
            print("❌ Security video capture failed to start")
            sys.exit(1)
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
