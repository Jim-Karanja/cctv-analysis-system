#!/usr/bin/env python3
"""
Threaded Security Video Capture - Uses threaded video capture with OpenCV face detection
for improved performance and non-blocking video processing.
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
from threaded_video_capture import ThreadedVideoCapture

# Import face recognition system
try:
    from face_recognition_system import FaceRecognitionSystem
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("⚠️  Advanced face recognition not available - using basic template matching")

# Import notification system and data cleanup
try:
    from notification_service import NotificationManager
    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    NOTIFICATIONS_AVAILABLE = False
    print("⚠️  Notification service not available - alerts will be logged only")

try:
    from data_cleanup import DataCleanupManager, AutoCleanupService
    DATA_CLEANUP_AVAILABLE = True
except ImportError:
    DATA_CLEANUP_AVAILABLE = False
    print("⚠️  Data cleanup service not available")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThreadedSecurityCapture:
    """Threaded security video capture with face detection."""
    
    def __init__(self):
        self.running = False
        self.face_cascade = None
        self.face_recognition_system = None
        self.security_events = []
        self.threaded_capture = None
        self.notification_manager = None
        self.cleanup_manager = None
        self.auto_cleanup_service = None
        
        # Legacy face recognition data (fallback)
        self.known_faces = []
        self.known_names = []
        self.face_templates = []
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Ensure directories exist
        Path("data").mkdir(exist_ok=True)
        Path("data/personnel").mkdir(exist_ok=True)
        
        # Initialize advanced face recognition system
        self._init_face_recognition()
        
        # Initialize basic face detection (fallback)
        self._init_face_detection()
        
        # Initialize notification system
        self._init_notifications()
        
        # Initialize data cleanup system
        self._init_data_cleanup()
        
        # Load personnel faces for recognition
        self._load_personnel_faces()
    
    def _signal_handler(self, signum, frame):
        print(f"\n🛑 Received signal {signum}, stopping threaded security video capture...")
        self.running = False
    
    def _init_face_recognition(self):
        """Initialize advanced face recognition system."""
        if FACE_RECOGNITION_AVAILABLE:
            try:
                self.face_recognition_system = FaceRecognitionSystem()
                logger.info(f"Advanced face recognition initialized with {len(self.face_recognition_system.known_face_names)} known faces")
                print(f"🎯 Advanced face recognition loaded: {', '.join(self.face_recognition_system.known_face_names)}")
            except Exception as e:
                logger.error(f"Face recognition init error: {e}")
                self.face_recognition_system = None
                print("⚠️  Falling back to basic template matching")
        else:
            logger.info("Advanced face recognition not available")
            self.face_recognition_system = None
    
    def _init_face_detection(self):
        """Initialize face detection."""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                logger.error("Failed to load face cascade")
                self.face_cascade = None
            else:
                logger.info("Face detection initialized")
        except Exception as e:
            logger.error(f"Face detection init error: {e}")
            self.face_cascade = None
    
    def _init_notifications(self):
        """Initialize mobile notification system."""
        if NOTIFICATIONS_AVAILABLE:
            try:
                self.notification_manager = NotificationManager()
                logger.info("Mobile notification system initialized")
                print("📱 Mobile alerts enabled - configure with setup script")
            except Exception as e:
                logger.error(f"Notification system init error: {e}")
                self.notification_manager = None
        else:
            logger.info("Mobile notification system not available")
    
    def _init_data_cleanup(self):
        """Initialize automatic data cleanup system."""
        if DATA_CLEANUP_AVAILABLE:
            try:
                self.cleanup_manager = DataCleanupManager()
                self.auto_cleanup_service = AutoCleanupService(self.cleanup_manager)
                logger.info("Data cleanup system initialized")
                print("🧹 Automatic data cleanup enabled - will run every 24 hours")
            except Exception as e:
                logger.error(f"Data cleanup system init error: {e}")
                self.cleanup_manager = None
                self.auto_cleanup_service = None
        else:
            logger.info("Data cleanup system not available")
    
    def _load_personnel_faces(self):
        """Load authorized personnel faces for recognition."""
        personnel_dir = Path("data/personnel")
        
        if not personnel_dir.exists():
            logger.info("Personnel directory does not exist")
            return
        
        self.known_faces = []
        self.known_names = []
        self.face_templates = []
        
        # Load all personnel images
        image_files = list(personnel_dir.glob("*.jpg")) + list(personnel_dir.glob("*.png"))
        
        for image_file in image_files:
            try:
                # Load image
                img = cv2.imread(str(image_file))
                if img is None:
                    continue
                
                # Get person name from filename
                person_name = image_file.stem.replace('_', ' ').title()
                
                # Convert to grayscale and detect faces
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
                
                if len(faces) > 0:
                    # Use the largest face
                    if len(faces) > 1:
                        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                    
                    x, y, w, h = faces[0]
                    
                    # Extract face region and normalize size
                    face_roi = gray[y:y+h, x:x+w]
                    face_template = cv2.resize(face_roi, (100, 100))
                    
                    # Store the template and info
                    self.face_templates.append(face_template)
                    self.known_names.append(person_name)
                    
                    logger.info(f"Loaded personnel face: {person_name}")
                else:
                    logger.warning(f"No face found in {image_file.name}")
                    
            except Exception as e:
                logger.error(f"Error loading {image_file.name}: {e}")
        
        logger.info(f"Loaded {len(self.face_templates)} personnel face templates")
    
    def get_authorized_personnel_count(self):
        """Count authorized personnel photos."""
        personnel_dir = Path("data/personnel")
        if not personnel_dir.exists():
            return 0
        
        image_files = list(personnel_dir.glob("*.jpg")) + list(personnel_dir.glob("*.png"))
        return len(image_files)
    
    def detect_faces(self, frame):
        """Detect faces in frame and create security events using advanced face recognition."""
        try:
            events = []
            
            # Use advanced face recognition if available
            if self.face_recognition_system is not None:
                # Use the advanced face recognition system
                recognition_results = self.face_recognition_system.recognize_faces(frame)
                
                for result in recognition_results:
                    # Convert face_recognition_system results to our event format
                    event = {
                        'person_name': result.get('person_name', 'Unknown Person'),
                        'authorized': result.get('authorized', False),
                        'confidence': float(result.get('confidence', 0.0)),
                        'timestamp': float(time.time()),
                        'datetime': datetime.now().isoformat(),
                        'bbox': result.get('bbox', [0, 0, 0, 0]),
                        'alert_level': 'LOW' if result.get('authorized', False) else 'HIGH'
                    }
                    
                    logger.info(f"👤 Advanced recognition: {event['person_name']} - {'AUTHORIZED' if event['authorized'] else 'UNAUTHORIZED'} (confidence: {event['confidence']:.3f})")
                    events.append(event)
            
            # Fallback to basic face detection if advanced system not available
            elif self.face_cascade is not None:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                for (x, y, w, h) in faces:
                    # Extract face region for matching
                    face_roi = gray[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_roi, (100, 100))
                    
                    # Match against known personnel faces
                    best_match_score = 0
                    best_match_name = "Unknown Person"
                    authorized = False
                    
                    if len(self.face_templates) > 0:
                        # Compare with each known face template
                        for i, template in enumerate(self.face_templates):
                            try:
                                # Use template matching
                                result = cv2.matchTemplate(face_resized, template, cv2.TM_CCOEFF_NORMED)
                                _, max_val, _, _ = cv2.minMaxLoc(result)
                                
                                # Check if this is the best match
                                if max_val > best_match_score:
                                    best_match_score = max_val
                                    best_match_name = self.known_names[i]
                            
                            except Exception as e:
                                logger.debug(f"Template matching error: {e}")
                        
                        # Determine if it's authorized (threshold for recognition)
                        recognition_threshold = 0.25  # Adjusted for real-world lighting conditions
                        
                        if best_match_score > recognition_threshold:
                            authorized = True
                            person_name = best_match_name
                            confidence = float(best_match_score)
                        else:
                            authorized = False
                            person_name = "Unknown Person"
                            confidence = float(best_match_score)
                    else:
                        # No personnel templates = all unknown
                        authorized = False
                        person_name = "Unknown Person"
                        confidence = 0.0
                    
                    event = {
                        'person_name': person_name,
                        'authorized': bool(authorized),
                        'confidence': float(confidence),
                        'timestamp': float(time.time()),
                        'datetime': datetime.now().isoformat(),
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'alert_level': 'LOW' if authorized else 'HIGH'
                    }
                    
                    logger.info(f"👤 Basic recognition: {person_name} - {'AUTHORIZED' if authorized else 'UNAUTHORIZED'} (confidence: {confidence:.3f})")
                    events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return []
    
    def draw_detections(self, frame, events):
        """Draw detection boxes on frame."""
        result_frame = frame.copy()
        
        for event in events:
            if 'bbox' not in event:
                continue
                
            x, y, w, h = event['bbox']
            
            # Color based on authorization
            if event['authorized']:
                color = (0, 255, 0)  # Green for authorized
                status = "AUTHORIZED"
            else:
                color = (0, 0, 255)  # Red for unauthorized
                status = "UNAUTHORIZED"
            
            # Draw rectangle
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{event['person_name']} - {status}"
            if event['confidence'] > 0:
                label += f" ({event['confidence']:.2f})"
            
            # Text background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            cv2.rectangle(result_frame, 
                         (x, y - text_height - 10),
                         (x + text_width, y),
                         color, -1)
            
            cv2.putText(result_frame, label, (x, y - 5), 
                       font, font_scale, (255, 255, 255), thickness)
        
        return result_frame
    
    def save_events(self):
        """Save security events to file."""
        try:
            events_file = "data/security_events.json"
            with open(events_file, 'w') as f:
                json.dump(self.security_events, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving events: {e}")
    
    def run(self):
        """Main capture loop using threaded video capture."""
        print("🔒 Threaded Security Video Capture")
        print("🧵 Using threaded video capture for improved performance")
        if self.face_recognition_system is not None:
            print("🎯 Using ADVANCED face recognition with dlib encodings")
            recognition_count = len(self.face_recognition_system.known_face_names)
            print(f"👥 {recognition_count} personnel loaded for advanced recognition: {', '.join(self.face_recognition_system.known_face_names)}")
        else:
            print("🔍 Using basic OpenCV face detection with template matching")
            template_count = len(self.face_templates)
            print(f"👥 {template_count} face templates loaded for basic recognition")
        print("📊 Faces will be compared against authorized personnel photos")
        print("⏹️  Press Ctrl+C to stop")
        print("=" * 60)
        
        personnel_count = self.get_authorized_personnel_count()
        
        if personnel_count == 0:
            print("⚠️  No authorized personnel photos in data/personnel/")
            print("   Add photos using: python setup_personnel.py")
        
        # Initialize threaded video capture
        self.threaded_capture = ThreadedVideoCapture(0, buffer_size=5)
        
        if not self.threaded_capture.start():
            print("❌ Could not start threaded video capture")
            return False
        
        print("✅ Threaded video capture started successfully")
        
        # Start notification system
        if self.notification_manager:
            self.notification_manager.start()
        
        # Start automatic data cleanup service
        if self.auto_cleanup_service:
            self.auto_cleanup_service.start()
        
        self.running = True
        frame_count = 0
        detection_count = 0
        unauthorized_count = 0
        start_time = time.time()
        last_status_time = start_time
        last_save_time = start_time
        
        try:
            while self.running:
                # Read frame from threaded capture
                ret, frame, timestamp = self.threaded_capture.read()
                
                if not ret or frame is None:
                    # No frame available, continue
                    time.sleep(0.01)
                    continue
                
                frame_count += 1
                processed_frame = frame.copy()
                
                # Security analysis
                try:
                    events = self.detect_faces(frame)
                    
                    if events:
                        detection_count += len(events)
                        
                        # Count unauthorized
                        unauthorized_in_frame = sum(1 for event in events if not event['authorized'])
                        unauthorized_count += unauthorized_in_frame
                        
                        # Draw detection boxes
                        processed_frame = self.draw_detections(frame, events)
                        
                        # Log events and send mobile alerts
                        for event in events:
                            self.security_events.append(event)
                            
                            if event['authorized']:
                                logger.info(f"✅ Authorized: {event['person_name']} (confidence: {event['confidence']:.3f})")
                            else:
                                logger.warning(f"🚨 UNAUTHORIZED: {event['person_name']} (confidence: {event['confidence']:.3f})")
                                
                                # Send mobile alert for unauthorized access
                                if self.notification_manager:
                                    try:
                                        alert_result = self.notification_manager.send_security_alert(event)
                                        if alert_result and any(alert_result.values()):
                                            logger.info(f"📱 Mobile alert sent for unauthorized access: {event['person_name']}")
                                        elif not alert_result.get('blocked', False):
                                            logger.warning("Failed to send mobile alert")
                                    except Exception as e:
                                        logger.error(f"Error sending mobile alert: {e}")
                
                except Exception as e:
                    logger.error(f"Analysis error: {e}")
                
                # Resize frame for display
                height, width = processed_frame.shape[:2]
                if width > 640:
                    new_width = 640
                    new_height = int((new_width / width) * height)
                    processed_frame = cv2.resize(processed_frame, (new_width, new_height))
                
                # Save frame more frequently for better streaming consistency
                current_time = time.time()
                if current_time - last_save_time >= 0.067:  # ~15 FPS frame saving
                    frame_path = "data/current_frame.jpg"
                    # Use atomic write to prevent corrupted frames during web server read
                    temp_path = "data/current_frame_temp.jpg"
                    success = cv2.imwrite(temp_path, processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    
                    if success:
                        try:
                            # Atomic rename to prevent reading partial files
                            if os.path.exists(temp_path):
                                if os.path.exists(frame_path):
                                    os.replace(temp_path, frame_path)
                                else:
                                    os.rename(temp_path, frame_path)
                                last_save_time = current_time
                        except Exception as e:
                            logger.error(f"Failed to update frame file: {e}")
                    else:
                        logger.error("Failed to save frame")
                
                # Status update every 15 seconds
                if current_time - last_status_time >= 15:
                    elapsed = current_time - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    capture_fps = self.threaded_capture.get_fps()
                    
                    print(f"📊 Frames: {frame_count}, Processing FPS: {fps:.1f}, "
                          f"Capture FPS: {capture_fps:.1f}, "
                          f"Detections: {detection_count}, Alerts: {unauthorized_count}, "
                          f"Time: {time.strftime('%H:%M:%S')}")
                    
                    # Print capture stats
                    stats = self.threaded_capture.get_stats()
                    if stats['frames_dropped'] > 0:
                        print(f"   📉 Queue: {stats['queue_size']}, Dropped: {stats['frames_dropped']}")
                    
                    self.save_events()
                    
                    # Memory optimization - keep events list reasonable size during runtime
                    if len(self.security_events) > 200:  # Trim if too many events in memory
                        self.security_events = self.security_events[-100:]  # Keep last 100
                        logger.info("Trimmed security events list for memory optimization")
                    
                    last_status_time = current_time
                
                # Minimal sleep to prevent excessive CPU usage
                time.sleep(0.01)
        
        except Exception as e:
            logger.error(f"Capture error: {e}")
        
        finally:
            # Cleanup threaded capture
            if self.threaded_capture:
                self.threaded_capture.stop()
            
            # Cleanup notification system
            if self.notification_manager:
                self.notification_manager.stop()
            
            # Stop automatic data cleanup service
            if self.auto_cleanup_service:
                self.auto_cleanup_service.stop()
            
            # Final statistics
            elapsed = time.time() - start_time
            avg_fps = frame_count / elapsed if elapsed > 0 else 0
            
            print(f"\n📊 Final Statistics:")
            print(f"   Frames processed: {frame_count}")
            print(f"   Average processing FPS: {avg_fps:.1f}")
            print(f"   Total detections: {detection_count}")
            print(f"   Security alerts: {unauthorized_count}")
            
            # Save final events
            self.save_events()
            
            # Memory optimization - keep only recent events in memory
            if len(self.security_events) > 50:  # Reduced from 100 to 50 for better memory usage
                self.security_events = self.security_events[-50:]
                self.save_events()
            
            # Run final cleanup if cleanup manager is available
            if self.cleanup_manager:
                try:
                    cleanup_results = self.cleanup_manager.run_full_cleanup()
                    if cleanup_results.get('total_freed_mb', 0) > 0:
                        print(f"🧹 Final cleanup freed {cleanup_results['total_freed_mb']} MB of storage")
                except Exception as e:
                    logger.error(f"Error during final cleanup: {e}")
            
            print("✅ Threaded security video capture completed")
        
        return True

def main():
    capturer = ThreadedSecurityCapture()
    
    try:
        success = capturer.run()
        if not success:
            print("❌ Failed to start threaded security capture")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
