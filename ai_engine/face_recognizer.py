"""
Face Recognizer

Implements face recognition by generating and comparing facial embeddings/encodings.
"""

import logging
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("Warning: face_recognition not available. Recognition features disabled.")
import numpy as np
import pickle
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from .face_detector import DetectedFace


@dataclass
class FaceEncoding:
    """Represents a face encoding with associated metadata."""
    person_id: str
    person_name: str
    encoding: np.ndarray
    confidence_threshold: float = 0.6
    created_at: float = None


@dataclass
class RecognitionResult:
    """Result of face recognition attempt."""
    person_id: Optional[str]
    person_name: Optional[str]
    confidence: float
    is_match: bool
    distance: float


class FaceRecognizer:
    """
    Face recognition engine using facial embeddings.
    
    Uses the face_recognition library which implements a deep neural network
    trained on face recognition tasks.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Storage for known face encodings
        self.known_encodings: Dict[str, FaceEncoding] = {}
        
        # Recognition parameters
        self.default_tolerance = config.get('recognition_tolerance', 0.6)
        self.num_jitters = config.get('num_jitters', 1)
        self.model = config.get('model', 'small')  # 'small' or 'large'
        
        # Load known faces if encodings file exists
        self.encodings_file = config.get('encodings_file', 'data/face_encodings.pkl')
        self.load_known_encodings()
    
    def extract_face_encoding(self, face_image: np.ndarray, landmarks: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Extract face encoding from a face image.
        
        Args:
            face_image: Cropped face image
            landmarks: Optional facial landmarks for better encoding
            
        Returns:
            Face encoding as numpy array, or None if extraction fails
        """
        try:
            # Ensure image is in RGB format
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                # Convert BGR to RGB if needed
                rgb_image = face_image[:, :, ::-1]
            else:
                rgb_image = face_image
            
            # Use landmarks if provided for better accuracy
            if landmarks is not None:
                # Convert landmarks to the format expected by face_recognition
                known_face_locations = [(
                    int(np.min(landmarks[:, 1])),  # top
                    int(np.max(landmarks[:, 0])),  # right
                    int(np.max(landmarks[:, 1])),  # bottom
                    int(np.min(landmarks[:, 0]))   # left
                )]
                
                encodings = face_recognition.face_encodings(
                    rgb_image,
                    known_face_locations=known_face_locations,
                    num_jitters=self.num_jitters,
                    model=self.model
                )
            else:
                # Let face_recognition detect the face location
                encodings = face_recognition.face_encodings(
                    rgb_image,
                    num_jitters=self.num_jitters,
                    model=self.model
                )
            
            if len(encodings) > 0:
                return encodings[0]
            else:
                self.logger.warning("No face encoding could be extracted from the image")
                return None
                
        except Exception as e:
            self.logger.error(f"Error extracting face encoding: {e}")
            return None
    
    def add_known_person(self, person_id: str, person_name: str, face_image: np.ndarray, landmarks: Optional[np.ndarray] = None) -> bool:
        """
        Add a known person to the recognition database.
        
        Args:
            person_id: Unique identifier for the person
            person_name: Human-readable name
            face_image: Reference face image
            landmarks: Optional facial landmarks
            
        Returns:
            True if person was added successfully
        """
        encoding = self.extract_face_encoding(face_image, landmarks)
        
        if encoding is None:
            self.logger.error(f"Failed to extract encoding for person {person_name}")
            return False
        
        face_encoding = FaceEncoding(
            person_id=person_id,
            person_name=person_name,
            encoding=encoding,
            confidence_threshold=self.default_tolerance
        )
        
        self.known_encodings[person_id] = face_encoding
        
        self.logger.info(f"Added person {person_name} (ID: {person_id}) to recognition database")
        
        # Save encodings to file
        self.save_known_encodings()
        
        return True
    
    def remove_known_person(self, person_id: str) -> bool:
        """
        Remove a person from the recognition database.
        
        Args:
            person_id: ID of person to remove
            
        Returns:
            True if person was removed successfully
        """
        if person_id in self.known_encodings:
            person_name = self.known_encodings[person_id].person_name
            del self.known_encodings[person_id]
            
            self.logger.info(f"Removed person {person_name} (ID: {person_id}) from recognition database")
            
            # Save updated encodings
            self.save_known_encodings()
            
            return True
        else:
            self.logger.warning(f"Person with ID {person_id} not found in database")
            return False
    
    def recognize_face(self, face_image: np.ndarray, landmarks: Optional[np.ndarray] = None, tolerance: Optional[float] = None) -> RecognitionResult:
        """
        Recognize a face by comparing it to known encodings.
        
        Args:
            face_image: Face image to recognize
            landmarks: Optional facial landmarks
            tolerance: Recognition tolerance (lower = more strict)
            
        Returns:
            Recognition result
        """
        if not self.known_encodings:
            return RecognitionResult(
                person_id=None,
                person_name=None,
                confidence=0.0,
                is_match=False,
                distance=float('inf')
            )
        
        # Extract encoding from input face
        unknown_encoding = self.extract_face_encoding(face_image, landmarks)
        
        if unknown_encoding is None:
            return RecognitionResult(
                person_id=None,
                person_name=None,
                confidence=0.0,
                is_match=False,
                distance=float('inf')
            )
        
        # Use provided tolerance or default
        if tolerance is None:
            tolerance = self.default_tolerance
        
        # Compare with all known encodings
        best_match_id = None
        best_match_name = None
        best_distance = float('inf')
        
        known_encodings_list = []
        person_ids = []
        
        for person_id, face_enc in self.known_encodings.items():
            known_encodings_list.append(face_enc.encoding)
            person_ids.append(person_id)
        
        # Calculate distances to all known faces
        distances = face_recognition.face_distance(known_encodings_list, unknown_encoding)
        
        # Find best match
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        
        if min_distance <= tolerance:
            best_match_id = person_ids[min_distance_idx]
            best_match_name = self.known_encodings[best_match_id].person_name
            best_distance = min_distance
        
        # Convert distance to confidence score (inverse relationship)
        confidence = max(0.0, 1.0 - min_distance)
        
        return RecognitionResult(
            person_id=best_match_id,
            person_name=best_match_name,
            confidence=confidence,
            is_match=best_match_id is not None,
            distance=best_distance
        )
    
    def recognize_faces_batch(self, detected_faces: List[DetectedFace], tolerance: Optional[float] = None) -> List[RecognitionResult]:
        """
        Recognize multiple faces in batch.
        
        Args:
            detected_faces: List of detected faces
            tolerance: Recognition tolerance
            
        Returns:
            List of recognition results
        """
        results = []
        
        for face in detected_faces:
            if face.face_image is not None:
                result = self.recognize_face(
                    face.face_image,
                    face.landmarks,
                    tolerance
                )
                results.append(result)
            else:
                # No face image available
                results.append(RecognitionResult(
                    person_id=None,
                    person_name=None,
                    confidence=0.0,
                    is_match=False,
                    distance=float('inf')
                ))
        
        return results
    
    def save_known_encodings(self) -> bool:
        """
        Save known face encodings to file.
        
        Returns:
            True if saved successfully
        """
        try:
            # Create directory if it doesn't exist
            encodings_path = Path(self.encodings_file)
            encodings_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for serialization
            encodings_data = {}
            for person_id, face_enc in self.known_encodings.items():
                encodings_data[person_id] = {
                    'person_name': face_enc.person_name,
                    'encoding': face_enc.encoding,
                    'confidence_threshold': face_enc.confidence_threshold,
                    'created_at': face_enc.created_at
                }
            
            # Save to pickle file
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(encodings_data, f)
            
            self.logger.info(f"Saved {len(encodings_data)} face encodings to {self.encodings_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving encodings to {self.encodings_file}: {e}")
            return False
    
    def load_known_encodings(self) -> bool:
        """
        Load known face encodings from file.
        
        Returns:
            True if loaded successfully
        """
        try:
            if not Path(self.encodings_file).exists():
                self.logger.info(f"No existing encodings file found at {self.encodings_file}")
                return False
            
            with open(self.encodings_file, 'rb') as f:
                encodings_data = pickle.load(f)
            
            # Reconstruct FaceEncoding objects
            self.known_encodings = {}
            for person_id, data in encodings_data.items():
                face_enc = FaceEncoding(
                    person_id=person_id,
                    person_name=data['person_name'],
                    encoding=data['encoding'],
                    confidence_threshold=data.get('confidence_threshold', self.default_tolerance),
                    created_at=data.get('created_at')
                )
                self.known_encodings[person_id] = face_enc
            
            self.logger.info(f"Loaded {len(self.known_encodings)} face encodings from {self.encodings_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading encodings from {self.encodings_file}: {e}")
            return False
    
    def get_known_persons(self) -> List[Dict[str, Any]]:
        """
        Get list of all known persons in the database.
        
        Returns:
            List of person information dictionaries
        """
        persons = []
        for person_id, face_enc in self.known_encodings.items():
            persons.append({
                'person_id': person_id,
                'person_name': face_enc.person_name,
                'confidence_threshold': face_enc.confidence_threshold,
                'created_at': face_enc.created_at
            })
        
        return persons
    
    def update_person_threshold(self, person_id: str, new_threshold: float) -> bool:
        """
        Update confidence threshold for a specific person.
        
        Args:
            person_id: ID of person to update
            new_threshold: New confidence threshold
            
        Returns:
            True if updated successfully
        """
        if person_id in self.known_encodings:
            self.known_encodings[person_id].confidence_threshold = new_threshold
            self.save_known_encodings()
            
            self.logger.info(f"Updated threshold for person {person_id} to {new_threshold}")
            return True
        else:
            self.logger.warning(f"Person with ID {person_id} not found")
            return False
