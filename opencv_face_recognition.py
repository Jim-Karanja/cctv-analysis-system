#!/usr/bin/env python3
"""
OpenCV-based Face Recognition System

A more robust face recognition system using OpenCV's built-in face recognition capabilities
as a fallback when the face_recognition library has issues.
"""

import cv2
import numpy as np
import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pickle
import time

logger = logging.getLogger(__name__)

class OpenCVFaceRecognition:
    """Face recognition using OpenCV's built-in algorithms."""
    
    def __init__(self, personnel_dir: str = "data/personnel", encodings_file: str = "data/opencv_face_data.pkl"):
        self.personnel_dir = Path(personnel_dir)
        self.encodings_file = encodings_file
        
        # Initialize face detector and recognizer
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Use LBPH (Local Binary Patterns Histograms) face recognizer
        # This is more robust for varying lighting conditions
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Storage for training data
        self.known_face_labels = []
        self.known_face_names = []
        self.is_trained = False
        
        # Recognition parameters
        self.confidence_threshold = 80  # Lower = more strict, Higher = more lenient
        
        # Load or train the model
        self._load_or_train_model()
    
    def _load_or_train_model(self):
        """Load existing model or train a new one."""
        # Check if model file exists and is recent
        model_file = self.encodings_file.replace('.pkl', '_model.yml')
        
        if os.path.exists(model_file) and os.path.exists(self.encodings_file):
            try:
                # Load the trained model
                self.face_recognizer.read(model_file)
                
                # Load the name mappings
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_names = data['names']
                
                self.is_trained = True
                logger.info(f"Loaded trained model with {len(self.known_face_names)} people")
                
                # Check if we need to retrain (if personnel photos changed)
                if self._should_retrain():
                    logger.info("Personnel photos have changed, retraining model...")
                    self._train_model()
                
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                self._train_model()
        else:
            logger.info("No existing model found, training new model...")
            self._train_model()
    
    def _should_retrain(self) -> bool:
        """Check if model needs retraining based on file timestamps."""
        if not os.path.exists(self.encodings_file):
            return True
        
        model_mtime = os.path.getmtime(self.encodings_file)
        
        # Check if any personnel photo is newer than model
        if self.personnel_dir.exists():
            for photo_file in self.personnel_dir.glob("*.jpg"):
                if photo_file.stat().st_mtime > model_mtime:
                    return True
            for photo_file in self.personnel_dir.glob("*.png"):
                if photo_file.stat().st_mtime > model_mtime:
                    return True
        
        return False
    
    def _train_model(self):
        """Train the face recognition model with personnel photos."""
        if not self.personnel_dir.exists():
            logger.warning(f"Personnel directory {self.personnel_dir} does not exist")
            return
        
        # Collect training data
        faces = []
        labels = []
        names = []
        
        # Process all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        personnel_files = []
        
        for ext in image_extensions:
            personnel_files.extend(self.personnel_dir.glob(ext))
        
        if not personnel_files:
            logger.warning("No personnel photos found for training")
            return
        
        logger.info(f"Training model with {len(personnel_files)} personnel photos...")
        
        label_id = 0
        
        for photo_file in personnel_files:
            try:
                # Get person name from filename
                person_name = photo_file.stem.replace('_', ' ').title()
                
                # Load image
                image = cv2.imread(str(photo_file))
                if image is None:
                    logger.warning(f"Could not load image: {photo_file.name}")
                    continue
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                face_locations = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(50, 50)
                )
                
                if len(face_locations) > 0:
                    # Use the largest face (presumably the main subject)
                    if len(face_locations) > 1:
                        # Sort by area and take the largest
                        face_locations = sorted(face_locations, key=lambda x: x[2]*x[3], reverse=True)
                    
                    x, y, w, h = face_locations[0]
                    
                    # Extract face region
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Resize to standard size for consistency
                    face_roi = cv2.resize(face_roi, (100, 100))
                    
                    # Add to training data
                    faces.append(face_roi)
                    labels.append(label_id)
                    
                    # Map label to name
                    if label_id >= len(names):
                        names.append(person_name)
                    
                    logger.info(f"Added training data for: {person_name} (label {label_id})")
                    label_id += 1
                else:
                    logger.warning(f"No face found in {photo_file.name}")
            
            except Exception as e:
                logger.error(f"Error processing {photo_file.name}: {e}")
        
        # Train the model if we have training data
        if len(faces) > 0:
            try:
                self.face_recognizer.train(faces, np.array(labels))
                self.known_face_names = names
                self.is_trained = True
                
                # Save the trained model
                model_file = self.encodings_file.replace('.pkl', '_model.yml')
                self.face_recognizer.save(model_file)
                
                # Save the name mappings
                data = {
                    'names': names,
                    'trained': time.time()
                }
                
                os.makedirs(os.path.dirname(self.encodings_file), exist_ok=True)
                with open(self.encodings_file, 'wb') as f:
                    pickle.dump(data, f)
                
                logger.info(f"Successfully trained model with {len(names)} people")
                
            except Exception as e:
                logger.error(f"Error training model: {e}")
                self.is_trained = False
        else:
            logger.error("No training data available")
            self.is_trained = False
    
    def recognize_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Recognize faces in a frame.
        
        Args:
            frame: Input image frame (BGR format from OpenCV)
        
        Returns:
            List of recognition results
        """
        if not self.is_trained:
            logger.warning("Model not trained, cannot recognize faces")
            return []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            face_locations = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            recognition_results = []
            
            for (x, y, w, h) in face_locations:
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (100, 100))
                
                # Recognize face
                label, confidence = self.face_recognizer.predict(face_roi)
                
                # Determine if this is a match
                if confidence < self.confidence_threshold:
                    # It's a match
                    person_name = self.known_face_names[label] if label < len(self.known_face_names) else "Unknown"
                    authorized = True
                    # Convert confidence to a 0-1 scale (lower confidence value = higher certainty)
                    confidence_score = max(0.0, 1.0 - (confidence / 100.0))
                else:
                    # Not a match
                    person_name = "Unknown Person"
                    authorized = False
                    confidence_score = max(0.0, 1.0 - (confidence / 100.0))
                
                # Create result
                result = {
                    'person_name': person_name,
                    'authorized': authorized,
                    'confidence': round(confidence_score, 3),
                    'bbox': [x, y, w, h],  # [x, y, width, height]
                    'raw_confidence': round(confidence, 1),
                    'alert_level': 'LOW' if authorized else 'HIGH'
                }
                
                recognition_results.append(result)
            
            return recognition_results
            
        except Exception as e:
            logger.error(f"Error during face recognition: {e}")
            return []
    
    def get_system_info(self) -> Dict:
        """Get information about the recognition system."""
        return {
            'known_personnel': len(self.known_face_names),
            'personnel_names': self.known_face_names.copy(),
            'is_trained': self.is_trained,
            'confidence_threshold': self.confidence_threshold,
            'model_type': 'OpenCV LBPH',
            'personnel_directory': str(self.personnel_dir)
        }
    
    def update_confidence_threshold(self, threshold: float):
        """Update confidence threshold (0-100, lower = more strict)."""
        self.confidence_threshold = max(0.0, min(100.0, threshold))
        logger.info(f"Updated confidence threshold to {self.confidence_threshold}")

def test_opencv_recognition():
    """Test the OpenCV face recognition system."""
    fr_system = OpenCVFaceRecognition()
    
    print("🔍 OpenCV Face Recognition System Test")
    print("=" * 50)
    
    info = fr_system.get_system_info()
    print(f"Known personnel: {info['known_personnel']}")
    print(f"Names: {info['personnel_names']}")
    print(f"Model trained: {'✅ Yes' if info['is_trained'] else '❌ No'}")
    print(f"Confidence threshold: {info['confidence_threshold']}")
    
    if not info['is_trained']:
        print("\n⚠️ Model not trained!")
        print("Make sure you have personnel photos in data/personnel/ directory.")
        return
    
    # Test with camera
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
            
            cv2.imshow('OpenCV Face Recognition Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_opencv_recognition()
