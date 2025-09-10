"""
Face Detector

Implements face detection using various methods including Haar cascades,
HOG + Linear SVM, and deep learning models.
"""

import cv2
import logging
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("Warning: dlib not available. Only OpenCV Haar cascades will work.")
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DetectedFace:
    """Represents a detected face with bounding box and confidence."""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    landmarks: Optional[np.ndarray] = None
    face_image: Optional[np.ndarray] = None


@dataclass
class DetectionConfig:
    """Configuration for face detection."""
    method: str = 'dlib_hog'  # 'opencv_haar', 'dlib_hog', 'dlib_cnn'
    min_face_size: Tuple[int, int] = (30, 30)
    scale_factor: float = 1.1
    min_neighbors: int = 5
    confidence_threshold: float = 0.5
    max_faces_per_frame: int = 10


class FaceDetector:
    """
    Face detection engine supporting multiple detection methods.
    
    Supports:
    - OpenCV Haar Cascades (fast, less accurate)
    - Dlib HOG + Linear SVM (balanced speed/accuracy)
    - Dlib CNN (slow, most accurate)
    """
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize detectors
        self._opencv_detector = None
        self._dlib_hog_detector = None
        self._dlib_cnn_detector = None
        self._face_predictor = None
        
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Initialize the specified detection method."""
        try:
            if self.config.method == 'opencv_haar':
                self._initialize_opencv_detector()
            elif self.config.method == 'dlib_hog':
                self._initialize_dlib_hog_detector()
            elif self.config.method == 'dlib_cnn':
                self._initialize_dlib_cnn_detector()
            else:
                raise ValueError(f"Unknown detection method: {self.config.method}")
                
            # Initialize facial landmark predictor (optional)
            self._initialize_landmark_predictor()
            
            self.logger.info(f"Face detector initialized with method: {self.config.method}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize face detector: {e}")
            raise
    
    def _initialize_opencv_detector(self):
        """Initialize OpenCV Haar cascade detector."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self._opencv_detector = cv2.CascadeClassifier(cascade_path)
        
        if self._opencv_detector.empty():
            raise RuntimeError("Failed to load OpenCV Haar cascade")
    
    def _initialize_dlib_hog_detector(self):
        """Initialize Dlib HOG detector."""
        if not DLIB_AVAILABLE:
            self.logger.warning("Dlib not available, falling back to OpenCV")
            self._initialize_opencv_detector()
            self.config.method = 'opencv_haar'
            return
        self._dlib_hog_detector = dlib.get_frontal_face_detector()
    
    def _initialize_dlib_cnn_detector(self):
        """Initialize Dlib CNN detector."""
        if not DLIB_AVAILABLE:
            self.logger.warning("Dlib not available, falling back to HOG")
            self._initialize_dlib_hog_detector()
            return
        
        # Note: This requires the dlib CNN face detection model file
        model_path = "models/mmod_human_face_detector.dat"
        
        if not Path(model_path).exists():
            self.logger.warning(f"CNN model not found at {model_path}. Using HOG detector instead.")
            self._initialize_dlib_hog_detector()
            self.config.method = 'dlib_hog'
            return
        
        self._dlib_cnn_detector = dlib.cnn_face_detection_model_v1(model_path)
    
    def _initialize_landmark_predictor(self):
        """Initialize facial landmark predictor."""
        predictor_path = "models/shape_predictor_68_face_landmarks.dat"
        
        if Path(predictor_path).exists():
            self._face_predictor = dlib.shape_predictor(predictor_path)
            self.logger.info("Facial landmark predictor loaded")
        else:
            self.logger.warning(f"Landmark predictor not found at {predictor_path}")
    
    def detect_faces_opencv(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect faces using OpenCV Haar cascades."""
        if self._opencv_detector is None:
            return []
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = self._opencv_detector.detectMultiScale(
            gray,
            scaleFactor=self.config.scale_factor,
            minNeighbors=self.config.min_neighbors,
            minSize=self.config.min_face_size
        )
        
        detected_faces = []
        for (x, y, w, h) in faces[:self.config.max_faces_per_frame]:
            # Extract confidence (OpenCV doesn't provide confidence, so we use a default)
            confidence = 0.8  # Default confidence for Haar cascades
            
            # Extract face region
            face_image = image[y:y+h, x:x+w]
            
            detected_face = DetectedFace(
                bbox=(x, y, w, h),
                confidence=confidence,
                face_image=face_image
            )
            
            detected_faces.append(detected_face)
        
        return detected_faces
    
    def detect_faces_dlib_hog(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect faces using Dlib HOG detector."""
        if self._dlib_hog_detector is None:
            return []
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = self._dlib_hog_detector(gray)
        
        detected_faces = []
        for face in faces[:self.config.max_faces_per_frame]:
            # Extract bounding box
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            
            # Default confidence for HOG detector
            confidence = 0.9
            
            # Extract face region
            face_image = image[y:y+h, x:x+w]
            
            # Get facial landmarks if predictor is available
            landmarks = None
            if self._face_predictor is not None:
                landmarks_shape = self._face_predictor(gray, face)
                landmarks = np.array([[p.x, p.y] for p in landmarks_shape.parts()])
            
            detected_face = DetectedFace(
                bbox=(x, y, w, h),
                confidence=confidence,
                face_image=face_image,
                landmarks=landmarks
            )
            
            detected_faces.append(detected_face)
        
        return detected_faces
    
    def detect_faces_dlib_cnn(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect faces using Dlib CNN detector."""
        if self._dlib_cnn_detector is None:
            return self.detect_faces_dlib_hog(image)  # Fallback to HOG
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = self._dlib_cnn_detector(gray)
        
        detected_faces = []
        for face in faces[:self.config.max_faces_per_frame]:
            # Extract confidence and bounding box
            confidence = face.confidence
            
            if confidence < self.config.confidence_threshold:
                continue
            
            rect = face.rect
            x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
            
            # Extract face region
            face_image = image[y:y+h, x:x+w]
            
            # Get facial landmarks if predictor is available
            landmarks = None
            if self._face_predictor is not None:
                landmarks_shape = self._face_predictor(gray, rect)
                landmarks = np.array([[p.x, p.y] for p in landmarks_shape.parts()])
            
            detected_face = DetectedFace(
                bbox=(x, y, w, h),
                confidence=confidence,
                face_image=face_image,
                landmarks=landmarks
            )
            
            detected_faces.append(detected_face)
        
        return detected_faces
    
    def detect_faces(self, image: np.ndarray) -> List[DetectedFace]:
        """
        Main face detection method.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected faces
        """
        if image is None or image.size == 0:
            return []
        
        try:
            if self.config.method == 'opencv_haar':
                return self.detect_faces_opencv(image)
            elif self.config.method == 'dlib_hog':
                return self.detect_faces_dlib_hog(image)
            elif self.config.method == 'dlib_cnn':
                return self.detect_faces_dlib_cnn(image)
            else:
                self.logger.error(f"Unknown detection method: {self.config.method}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error during face detection: {e}")
            return []
    
    def annotate_faces(self, image: np.ndarray, faces: List[DetectedFace]) -> np.ndarray:
        """
        Annotate detected faces on the image.
        
        Args:
            image: Input image
            faces: List of detected faces
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        for face in faces:
            x, y, w, h = face.bbox
            confidence = face.confidence
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw confidence score
            label = f"Face: {confidence:.2f}"
            cv2.putText(annotated, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw landmarks if available
            if face.landmarks is not None:
                for point in face.landmarks:
                    cv2.circle(annotated, tuple(point), 1, (0, 0, 255), -1)
        
        return annotated
