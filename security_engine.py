#!/usr/bin/env python3
"""
Security Analysis Engine for CCTV System

This module handles:
- Face detection in video frames
- Face recognition against authorized personnel
- Security alerts for unauthorized access
- Logging of security events
"""

import cv2
import numpy as np
import json
import time
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

# Try to import face_recognition (advanced), fallback to OpenCV (basic)
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    print("✅ Advanced face recognition (face_recognition library) available")
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("⚠️  Using basic face detection only (OpenCV Haar cascades)")

logger = logging.getLogger(__name__)

class Person:
    """Represents a person with their face encoding and authorization status."""
    
    def __init__(self, name: str, face_encoding: np.ndarray, authorized: bool = True):
        self.name = name
        self.face_encoding = face_encoding
        self.authorized = authorized
        self.last_seen = None
        self.detection_count = 0

class SecurityEvent:
    """Represents a security event (authorized/unauthorized detection)."""
    
    def __init__(self, person_name: str, authorized: bool, confidence: float, 
                 timestamp: float = None, bbox: Tuple[int, int, int, int] = None):
        self.person_name = person_name
        self.authorized = authorized
        self.confidence = confidence
        self.timestamp = timestamp or time.time()
        self.bbox = bbox  # (x, y, width, height)
        self.datetime = datetime.fromtimestamp(self.timestamp)
    
    def to_dict(self):
        return {
            'person_name': self.person_name,
            'authorized': self.authorized,
            'confidence': round(self.confidence, 3),
            'timestamp': self.timestamp,
            'datetime': self.datetime.isoformat(),
            'bbox': self.bbox,
            'alert_level': 'LOW' if self.authorized else 'HIGH'
        }

class SecurityEngine:
    """Main security analysis engine."""
    
    def __init__(self):
        self.authorized_personnel: Dict[str, Person] = {}
        self.face_cascade = None
        self.security_events: List[SecurityEvent] = []
        self.max_events = 100  # Keep last 100 events
        
        # Directories
        self.data_dir = Path("data")
        self.personnel_dir = self.data_dir / "personnel"
        self.events_file = self.data_dir / "security_events.json"
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.personnel_dir.mkdir(exist_ok=True)
        
        # Initialize face detection
        self._init_face_detection()
        
        # Load authorized personnel
        self.load_personnel()
        
        # Load previous events
        self.load_events()
        
        logger.info("Security Engine initialized")
    
    def _init_face_detection(self):
        """Initialize face detection system."""
        try:
            # Try to load OpenCV Haar cascade for face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                logger.error("Failed to load face cascade classifier")
                raise Exception("Face cascade not loaded")
            
            logger.info("Face detection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize face detection: {e}")
            self.face_cascade = None
    
    def add_authorized_person(self, name: str, image_path: str) -> bool:
        """
        Add an authorized person to the database from their photo.
        
        Args:
            name: Person's name
            image_path: Path to their photo
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            # Load and process the image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return False
            
            # Convert to RGB for face_recognition library
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if FACE_RECOGNITION_AVAILABLE:
                # Use advanced face recognition
                face_encodings = face_recognition.face_encodings(rgb_image)
                
                if not face_encodings:
                    logger.error(f"No face found in image: {image_path}")
                    return False
                
                # Use the first face found
                face_encoding = face_encodings[0]
                
            else:
                # Use basic face detection (no encoding, just detection)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) == 0:
                    logger.error(f"No face detected in image: {image_path}")
                    return False
                
                # For basic detection, we'll store a simple feature vector
                # This is a placeholder - in reality, you'd want proper face encoding
                face_encoding = np.random.random(128)  # Placeholder encoding
            
            # Create person object
            person = Person(name, face_encoding, authorized=True)
            self.authorized_personnel[name] = person
            
            logger.info(f"Added authorized person: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding authorized person {name}: {e}")
            return False
    
    def analyze_frame(self, frame: np.ndarray) -> List[SecurityEvent]:
        """
        Analyze a video frame for faces and determine authorization status.
        
        Args:
            frame: Video frame as numpy array
            
        Returns:
            List of security events detected in this frame
        """
        if frame is None or self.face_cascade is None:
            return []
        
        events = []
        
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face region
                face_region = frame[y:y+h, x:x+w]
                
                if FACE_RECOGNITION_AVAILABLE:
                    # Advanced face recognition
                    # Ensure face region is valid before processing
                    if (face_region.size > 0 and len(face_region.shape) == 3 and 
                        face_region.shape[0] > 20 and face_region.shape[1] > 20):
                        try:
                            # Convert to RGB and ensure proper format
                            rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                            
                            # Ensure image is uint8 format
                            if rgb_face.dtype != np.uint8:
                                rgb_face = (rgb_face * 255).astype(np.uint8)
                            
                            # Get face encodings
                            face_encodings = face_recognition.face_encodings(rgb_face)
                        except Exception as e:
                            logger.debug(f"Face encoding error: {e}")
                            face_encodings = []
                    else:
                        face_encodings = []
                    
                    if face_encodings:
                        face_encoding = face_encodings[0]
                        person_name, confidence = self._identify_person(face_encoding)
                    else:
                        person_name, confidence = "Unknown", 0.0
                else:
                    # Basic detection - mark as unknown but still create detection event
                    person_name, confidence = "Unknown Person", 0.7
                
                # Determine if person is authorized
                authorized = person_name in self.authorized_personnel
                
                # Create security event
                event = SecurityEvent(
                    person_name=person_name,
                    authorized=authorized,
                    confidence=confidence,
                    bbox=(x, y, w, h)
                )
                
                events.append(event)
                
                # Update person's last seen time
                if authorized:
                    self.authorized_personnel[person_name].last_seen = time.time()
                    self.authorized_personnel[person_name].detection_count += 1
        
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
        
        # Log events and add to history
        for event in events:
            self.security_events.append(event)
            
            if event.authorized:
                logger.info(f"✅ Authorized person detected: {event.person_name} (confidence: {event.confidence:.2f})")
            else:
                logger.warning(f"🚨 UNAUTHORIZED person detected: {event.person_name} (confidence: {event.confidence:.2f})")
        
        # Keep only recent events
        if len(self.security_events) > self.max_events:
            self.security_events = self.security_events[-self.max_events:]
        
        return events
    
    def _identify_person(self, face_encoding: np.ndarray) -> Tuple[str, float]:
        """
        Identify a person from their face encoding.
        
        Args:
            face_encoding: Face encoding to identify
            
        Returns:
            Tuple of (person_name, confidence)
        """
        if not self.authorized_personnel:
            return "Unknown", 0.0
        
        best_match = None
        best_distance = float('inf')
        
        # Compare against all authorized personnel
        for name, person in self.authorized_personnel.items():
            if FACE_RECOGNITION_AVAILABLE:
                # Calculate face distance
                distance = face_recognition.face_distance([person.face_encoding], face_encoding)[0]
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = name
            else:
                # Basic comparison - placeholder
                distance = np.linalg.norm(person.face_encoding - face_encoding)
                if distance < best_distance:
                    best_distance = distance
                    best_match = name
        
        # Convert distance to confidence (0-1, where 1 is perfect match)
        if FACE_RECOGNITION_AVAILABLE:
            # For face_recognition library, distance < 0.6 is typically a good match
            confidence = max(0, 1 - (best_distance / 0.6))
            threshold = 0.6
        else:
            # For basic comparison
            confidence = max(0, 1 - (best_distance / 100))  # Adjust threshold as needed
            threshold = 50
        
        # Return match if confidence is high enough
        if best_distance < threshold and confidence > 0.5:
            return best_match, confidence
        else:
            return "Unknown", confidence
    
    def draw_detections(self, frame: np.ndarray, events: List[SecurityEvent]) -> np.ndarray:
        """
        Draw detection boxes and labels on the frame.
        
        Args:
            frame: Video frame to draw on
            events: Security events to draw
            
        Returns:
            Frame with detection boxes drawn
        """
        result_frame = frame.copy()
        
        for event in events:
            if event.bbox is None:
                continue
            
            x, y, w, h = event.bbox
            
            # Choose color based on authorization status
            if event.authorized:
                color = (0, 255, 0)  # Green for authorized
                status = "AUTHORIZED"
            else:
                color = (0, 0, 255)  # Red for unauthorized
                status = "UNAUTHORIZED"
            
            # Draw rectangle around face
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{event.person_name} - {status}"
            if event.confidence > 0:
                label += f" ({event.confidence:.2f})"
            
            # Calculate text size and position
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(result_frame, 
                         (x, y - text_height - 10),
                         (x + text_width, y),
                         color, -1)
            
            # Draw text
            cv2.putText(result_frame, label, (x, y - 5), 
                       font, font_scale, (255, 255, 255), thickness)
        
        return result_frame
    
    def get_recent_events(self, limit: int = 20) -> List[Dict]:
        """Get recent security events as dictionaries."""
        recent = self.security_events[-limit:] if self.security_events else []
        return [event.to_dict() for event in reversed(recent)]
    
    def get_unauthorized_alerts(self) -> List[Dict]:
        """Get recent unauthorized access alerts."""
        alerts = [event for event in self.security_events if not event.authorized]
        return [event.to_dict() for event in reversed(alerts[-10:])]  # Last 10 alerts
    
    def save_events(self):
        """Save security events to file."""
        try:
            events_data = [event.to_dict() for event in self.security_events]
            with open(self.events_file, 'w') as f:
                json.dump(events_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving events: {e}")
    
    def load_events(self):
        """Load security events from file."""
        try:
            if self.events_file.exists():
                with open(self.events_file, 'r') as f:
                    events_data = json.load(f)
                
                # Reconstruct SecurityEvent objects
                self.security_events = []
                for data in events_data:
                    event = SecurityEvent(
                        person_name=data['person_name'],
                        authorized=data['authorized'],
                        confidence=data['confidence'],
                        timestamp=data['timestamp'],
                        bbox=tuple(data['bbox']) if data['bbox'] else None
                    )
                    self.security_events.append(event)
                
                logger.info(f"Loaded {len(self.security_events)} security events")
        except Exception as e:
            logger.error(f"Error loading events: {e}")
    
    def load_personnel(self):
        """Load authorized personnel from personnel directory."""
        try:
            personnel_files = list(self.personnel_dir.glob("*.jpg")) + list(self.personnel_dir.glob("*.png"))
            
            for image_path in personnel_files:
                # Use filename (without extension) as person's name
                person_name = image_path.stem
                self.add_authorized_person(person_name, str(image_path))
            
            logger.info(f"Loaded {len(self.authorized_personnel)} authorized personnel")
            
        except Exception as e:
            logger.error(f"Error loading personnel: {e}")
    
    def get_personnel_status(self) -> Dict:
        """Get status of authorized personnel."""
        return {
            'total_authorized': len(self.authorized_personnel),
            'personnel': [
                {
                    'name': name,
                    'last_seen': person.last_seen,
                    'detection_count': person.detection_count
                }
                for name, person in self.authorized_personnel.items()
            ]
        }

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create security engine
    security = SecurityEngine()
    
    # Example: Add authorized personnel (you would do this with real photos)
    print("\n📋 To add authorized personnel:")
    print("1. Place photos in data/personnel/ directory")
    print("2. Name files with person's name (e.g., 'john_smith.jpg')")
    print("3. Restart the system")
    
    print(f"\n👥 Currently {len(security.authorized_personnel)} authorized personnel loaded")
    
    # Test with a sample frame (if camera is available)
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("\n🎥 Testing with live camera...")
            ret, frame = cap.read()
            if ret:
                events = security.analyze_frame(frame)
                print(f"📊 Detected {len(events)} faces in test frame")
                
                if events:
                    annotated_frame = security.draw_detections(frame, events)
                    cv2.imshow("Security Analysis Test", annotated_frame)
                    print("Press any key to close...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            
            cap.release()
    
    except Exception as e:
        print(f"Camera test failed: {e}")
    
    print("✅ Security engine test completed")
