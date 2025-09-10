"""
AI Recognition Engine

Core AI module for face detection, recognition, and person identification.
"""

from .face_detector import FaceDetector
from .face_recognizer import FaceRecognizer
from .person_identifier import PersonIdentifier

__all__ = ['FaceDetector', 'FaceRecognizer', 'PersonIdentifier']
