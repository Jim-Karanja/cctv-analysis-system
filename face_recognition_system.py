#!/usr/bin/env python3
"""
Real Face Recognition System for CCTV Security

Uses the face_recognition library to properly identify authorized personnel
by comparing detected faces against stored personnel photos.
"""

import face_recognition
import cv2
import numpy as np
import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pickle
import time

logger = logging.getLogger(__name__)

class FaceRecognitionSystem:
    """Real face recognition system using face_recognition library."""
    
    def __init__(self, personnel_dir: str = "data/personnel", encodings_file: str = "data/face_encodings.pkl"):
        self.personnel_dir = Path(personnel_dir)
        self.encodings_file = encodings_file
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_encodings = []
        
        # Recognition settings
        self.face_recognition_tolerance = 0.6  # Lower = more strict, Higher = more lenient
        self.model = 'hog'  # 'hog' is faster, 'cnn' is more accurate but slower
        
        # Load or create face encodings
        self._load_or_create_encodings()
    
    def _load_or_create_encodings(self):
        """Load existing face encodings or create new ones from personnel photos."""
        # Check if encodings file exists and is recent
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                
                logger.info(f"Loaded {len(self.known_face_names)} face encodings from file")
                
                # Check if we need to update encodings (if personnel photos changed)
                if self._should_update_encodings():
                    logger.info("Personnel photos have changed, updating encodings...")
                    self._create_face_encodings()
                else:
                    logger.info("Face encodings are up to date")
                
            except Exception as e:
                logger.error(f"Error loading face encodings: {e}")
                self._create_face_encodings()
        else:
            logger.info("No existing face encodings found, creating new ones...")
            self._create_face_encodings()
    
    def _should_update_encodings(self) -> bool:
        """Check if face encodings need to be updated."""
        if not os.path.exists(self.encodings_file):
            return True
        
        # Get modification time of encodings file
        encodings_mtime = os.path.getmtime(self.encodings_file)
        
        # Check if any personnel photo is newer than encodings file
        if self.personnel_dir.exists():
            for photo_file in self.personnel_dir.glob("*.jpg"):
                if photo_file.stat().st_mtime > encodings_mtime:
                    return True
            for photo_file in self.personnel_dir.glob("*.png"):
                if photo_file.stat().st_mtime > encodings_mtime:
                    return True
        
        return False
    
    def _create_face_encodings(self):
        """Create face encodings from personnel photos."""
        self.known_face_encodings = []
        self.known_face_names = []
        
        if not self.personnel_dir.exists():
            logger.warning(f"Personnel directory {self.personnel_dir} does not exist")
            return
        
        # Process all image files in personnel directory
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        personnel_files = []
        
        for ext in image_extensions:
            personnel_files.extend(self.personnel_dir.glob(ext))
        
        if not personnel_files:
            logger.warning("No personnel photos found")
            return
        
        logger.info(f"Processing {len(personnel_files)} personnel photos...")
        
        for photo_file in personnel_files:
            try:
                # Get person name from filename (remove extension)
                person_name = photo_file.stem.replace('_', ' ').title()
                
                # Load image with OpenCV first to ensure proper format
                cv_image = cv2.imread(str(photo_file))
                if cv_image is None:
                    logger.error(f"Could not load image: {photo_file.name}")
                    continue
                
                logger.debug(f"Loaded image {photo_file.name}: shape={cv_image.shape}, dtype={cv_image.dtype}")
                
                # Convert BGR to RGB for face_recognition library
                image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                
                # Ensure image is in the right format
                if image.dtype != np.uint8:
                    image = image.astype(np.uint8)
                
                logger.debug(f"Converted image: shape={image.shape}, dtype={image.dtype}")
                
                # Get face encodings
                face_encodings = face_recognition.face_encodings(image, model=self.model)
                
                if len(face_encodings) > 0:
                    # Use the first face found in the image
                    face_encoding = face_encodings[0]
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(person_name)
                    logger.info(f"Added face encoding for: {person_name}")
                    
                    if len(face_encodings) > 1:
                        logger.warning(f"Multiple faces found in {photo_file.name}, using the first one")
                else:
                    logger.warning(f"No face found in {photo_file.name}")
            
            except Exception as e:
                logger.error(f"Error processing {photo_file.name}: {e}")
        
        # Save encodings to file
        if self.known_face_encodings:
            try:
                # Ensure data directory exists
                os.makedirs(os.path.dirname(self.encodings_file), exist_ok=True)
                
                data = {
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names,
                    'created': time.time()
                }
                
                with open(self.encodings_file, 'wb') as f:
                    pickle.dump(data, f)
                
                logger.info(f"Saved {len(self.known_face_names)} face encodings to {self.encodings_file}")
                
            except Exception as e:
                logger.error(f"Error saving face encodings: {e}")
    
    def recognize_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Recognize faces in a frame and return recognition results.
        
        Args:
            frame: Input image frame (BGR format from OpenCV)
        
        Returns:
            List of dictionaries containing recognition results
        """
        if not self.known_face_encodings:
            logger.warning("No face encodings available for recognition")
            return []
        
        try:
            # Convert BGR to RGB (face_recognition uses RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame for faster processing (optional)
            small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
            
            # Find face locations and encodings
            self.face_locations = face_recognition.face_locations(small_frame, model=self.model)
            self.face_encodings = face_recognition.face_encodings(small_frame, self.face_locations, model=self.model)
            
            recognition_results = []
            
            for (top, right, bottom, left), face_encoding in zip(self.face_locations, self.face_encodings):
                # Scale back up face locations since frame was scaled down by 0.5
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2
                
                # Compare face against known faces
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, 
                    face_encoding,
                    tolerance=self.face_recognition_tolerance
                )
                
                # Calculate face distances for confidence scores
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                
                person_name = "Unknown Person"
                authorized = False
                confidence = 0.0
                
                # Find best match
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        person_name = self.known_face_names[best_match_index]
                        authorized = True
                        # Convert distance to confidence score (lower distance = higher confidence)
                        confidence = max(0.0, 1.0 - face_distances[best_match_index])
                    else:
                        # Not a match, but calculate confidence anyway
                        confidence = max(0.0, 1.0 - min(face_distances))
                
                # Create recognition result
                result = {
                    'person_name': person_name,
                    'authorized': authorized,
                    'confidence': round(confidence, 3),
                    'bbox': [left, top, right - left, bottom - top],  # [x, y, width, height]
                    'face_distance': round(min(face_distances), 3) if len(face_distances) > 0 else 1.0,
                    'alert_level': 'LOW' if authorized else 'HIGH'
                }
                
                recognition_results.append(result)
            
            return recognition_results
            
        except Exception as e:
            logger.error(f"Error during face recognition: {e}")
            return []
    
    def get_system_info(self) -> Dict:
        """Get information about the face recognition system."""
        return {
            'known_personnel': len(self.known_face_names),
            'personnel_names': self.known_face_names.copy(),
            'recognition_tolerance': self.face_recognition_tolerance,
            'model': self.model,
            'encodings_file': self.encodings_file,
            'personnel_directory': str(self.personnel_dir)
        }
    
    def update_recognition_tolerance(self, tolerance: float):
        """Update face recognition tolerance (0.0 = strict, 1.0 = lenient)."""
        self.face_recognition_tolerance = max(0.0, min(1.0, tolerance))
        logger.info(f"Updated recognition tolerance to {self.face_recognition_tolerance}")
    
    def add_personnel(self, image_path: str, person_name: str) -> bool:
        """Add a new person to the recognition system."""
        try:
            # Copy image to personnel directory
            personnel_file = self.personnel_dir / f"{person_name.lower().replace(' ', '_')}.jpg"
            
            import shutil
            shutil.copy2(image_path, personnel_file)
            
            # Recreate encodings
            self._create_face_encodings()
            
            logger.info(f"Added new person: {person_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding personnel {person_name}: {e}")
            return False
    
    def remove_personnel(self, person_name: str) -> bool:
        """Remove a person from the recognition system."""
        try:
            # Find and remove the person's photo
            photo_name = person_name.lower().replace(' ', '_')
            
            for ext in ['jpg', 'jpeg', 'png', 'bmp']:
                photo_file = self.personnel_dir / f"{photo_name}.{ext}"
                if photo_file.exists():
                    photo_file.unlink()
                    break
            
            # Recreate encodings
            self._create_face_encodings()
            
            logger.info(f"Removed person: {person_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing personnel {person_name}: {e}")
            return False

def test_face_recognition():
    """Test the face recognition system."""
    fr_system = FaceRecognitionSystem()
    
    print("🔍 Face Recognition System Test")
    print("=" * 40)
    
    info = fr_system.get_system_info()
    print(f"Known personnel: {info['known_personnel']}")
    print(f"Names: {info['personnel_names']}")
    print(f"Recognition tolerance: {info['recognition_tolerance']}")
    print(f"Model: {info['model']}")
    
    if info['known_personnel'] == 0:
        print("\n⚠️  No personnel photos found!")
        print("Add photos to data/personnel/ directory to enable face recognition.")
        return
    
    # Test with camera if available
    print("\n📹 Testing with camera (press 'q' to quit)...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Recognize faces
            results = fr_system.recognize_faces(frame)
            
            # Draw results
            for result in results:
                x, y, w, h = result['bbox']
                color = (0, 255, 0) if result['authorized'] else (0, 0, 255)
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                label = f"{result['person_name']} ({result['confidence']:.2f})"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.imshow('Face Recognition Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_face_recognition()
