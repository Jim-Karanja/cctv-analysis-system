"""
Basic Face Detection Module

Fallback implementation using only OpenCV when face_recognition library is not available.
Provides face detection but no recognition capabilities.
"""

import cv2
import logging
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BasicDetectedFace:
    """Basic face detection result."""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float = 1.0


@dataclass
class BasicRecognitionResult:
    """Basic recognition result (always unknown)."""
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    confidence: float = 0.0
    is_match: bool = False


class BasicFaceDetector:
    """Basic face detector using OpenCV Haar cascades."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Load OpenCV face cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise ValueError("Failed to load face cascade")
        
        # Detection parameters
        self.scale_factor = self.config.get('scale_factor', 1.1)
        self.min_neighbors = self.config.get('min_neighbors', 5)
        self.min_size = tuple(self.config.get('min_face_size', [30, 30]))
        self.max_faces = self.config.get('max_faces_per_frame', 10)
        
        self.logger.info("Basic face detector initialized (OpenCV only)")
    
    def detect_faces(self, frame) -> List[BasicDetectedFace]:
        """
        Detect faces in a frame using OpenCV.
        
        Args:
            frame: Input image as numpy array
            
        Returns:
            List of detected faces
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size
            )
            
            # Convert to our format and limit results
            detected_faces = []
            for i, (x, y, w, h) in enumerate(faces[:self.max_faces]):
                detected_faces.append(BasicDetectedFace(
                    bbox=(x, y, w, h),
                    confidence=1.0  # OpenCV doesn't provide confidence scores
                ))
            
            return detected_faces
            
        except Exception as e:
            self.logger.error(f"Error detecting faces: {e}")
            return []


class BasicFaceRecognizer:
    """Basic face recognizer that always returns unknown."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.logger.warning("Using basic face recognizer - no recognition capabilities")
    
    def recognize_faces_batch(self, faces: List[BasicDetectedFace]) -> List[BasicRecognitionResult]:
        """
        Mock recognition that always returns unknown persons.
        
        Args:
            faces: List of detected faces
            
        Returns:
            List of recognition results (all unknown)
        """
        return [BasicRecognitionResult() for _ in faces]


class BasicPersonIdentifier:
    """Basic person identifier using only face detection."""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize basic detector
        detection_config = config.get('detection', {})
        self.face_detector = BasicFaceDetector(detection_config)
        self.face_recognizer = BasicFaceRecognizer(config.get('recognition', {}))
        
        # Performance settings
        self.min_detection_confidence = config.get('min_detection_confidence', 0.5)
        
        self.logger.info("Basic person identifier initialized (detection only)")
    
    def identify_persons_in_frame(self, frame, frame_metadata):
        """
        Identify persons in frame (basic detection only).
        
        Args:
            frame: Video frame
            frame_metadata: Frame metadata
            
        Returns:
            IdentificationResult with basic face detection
        """
        from .person_identifier import IdentificationResult, PersonDetection
        
        start_time = time.time()
        
        try:
            # Detect faces
            detected_faces = self.face_detector.detect_faces(frame)
            
            if not detected_faces:
                return IdentificationResult(
                    source_id=frame_metadata.source_id,
                    frame_number=frame_metadata.frame_number,
                    timestamp=frame_metadata.timestamp,
                    total_faces_detected=0,
                    identified_persons=[],
                    processing_time=time.time() - start_time,
                    success=True
                )
            
            # Convert to PersonDetection format
            identified_persons = []
            for face in detected_faces:
                person_detection = PersonDetection(
                    bbox=face.bbox,
                    detection_confidence=face.confidence,
                    person_id=None,
                    person_name=None,
                    recognition_confidence=0.0,
                    is_recognized=False,
                    timestamp=frame_metadata.timestamp,
                    source_id=frame_metadata.source_id,
                    frame_number=frame_metadata.frame_number
                )
                identified_persons.append(person_detection)
            
            processing_time = time.time() - start_time
            
            result = IdentificationResult(
                source_id=frame_metadata.source_id,
                frame_number=frame_metadata.frame_number,
                timestamp=frame_metadata.timestamp,
                total_faces_detected=len(detected_faces),
                identified_persons=identified_persons,
                processing_time=processing_time,
                success=True
            )
            
            self.logger.debug(
                f"Frame {frame_metadata.frame_number}: {len(detected_faces)} faces detected "
                f"(processed in {processing_time:.3f}s)"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Error in basic person identification: {e}"
            self.logger.error(error_msg)
            
            return IdentificationResult(
                source_id=frame_metadata.source_id,
                frame_number=frame_metadata.frame_number,
                timestamp=frame_metadata.timestamp,
                total_faces_detected=0,
                identified_persons=[],
                processing_time=time.time() - start_time,
                success=False,
                error_message=error_msg
            )
