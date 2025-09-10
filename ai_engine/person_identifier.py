"""
Person Identifier

High-level module that combines face detection and recognition to identify people
in video frames and manage the overall AI pipeline.
"""

import asyncio
import logging
import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

from .face_detector import FaceDetector, DetectedFace, DetectionConfig
from .face_recognizer import FaceRecognizer, RecognitionResult
from video_ingestion.frame_extractor import FrameMetadata


@dataclass
class PersonDetection:
    """Complete person detection result combining detection and recognition."""
    # Detection info
    bbox: Tuple[int, int, int, int]
    detection_confidence: float
    
    # Recognition info
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    recognition_confidence: float = 0.0
    is_recognized: bool = False
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    source_id: str = ""
    frame_number: int = 0


@dataclass
class IdentificationResult:
    """Result of the complete identification process."""
    source_id: str
    frame_number: int
    timestamp: float
    total_faces_detected: int
    identified_persons: List[PersonDetection]
    processing_time: float
    success: bool = True
    error_message: str = ""


class PersonIdentifier:
    """
    High-level person identification system that orchestrates face detection
    and recognition to identify people in video frames.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize detection and recognition components
        detection_config = DetectionConfig(**config.get('detection', {}))
        self.face_detector = FaceDetector(detection_config)
        
        recognition_config = config.get('recognition', {})
        self.face_recognizer = FaceRecognizer(recognition_config)
        
        # Performance settings
        self.min_detection_confidence = config.get('min_detection_confidence', 0.5)
        self.min_recognition_confidence = config.get('min_recognition_confidence', 0.6)
        
        # Tracking settings
        self.enable_tracking = config.get('enable_tracking', False)
        self.tracking_history = {} if self.enable_tracking else None
        
        self.logger.info("Person identification system initialized")
    
    def identify_persons_in_frame(self, frame, frame_metadata: FrameMetadata) -> IdentificationResult:
        """
        Identify all persons in a single frame.
        
        Args:
            frame: Video frame as numpy array
            frame_metadata: Metadata about the frame
            
        Returns:
            Complete identification result
        """
        start_time = time.time()
        
        try:
            # Step 1: Detect faces
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
            
            # Step 2: Filter faces by detection confidence
            valid_faces = [
                face for face in detected_faces 
                if face.confidence >= self.min_detection_confidence
            ]
            
            # Step 3: Recognize faces
            recognition_results = self.face_recognizer.recognize_faces_batch(valid_faces)
            
            # Step 4: Combine detection and recognition results
            identified_persons = []
            
            for i, (face, recognition) in enumerate(zip(valid_faces, recognition_results)):
                # Determine if person is recognized based on confidence threshold
                is_recognized = (
                    recognition.is_match and 
                    recognition.confidence >= self.min_recognition_confidence
                )
                
                person_detection = PersonDetection(
                    bbox=face.bbox,
                    detection_confidence=face.confidence,
                    person_id=recognition.person_id if is_recognized else None,
                    person_name=recognition.person_name if is_recognized else None,
                    recognition_confidence=recognition.confidence,
                    is_recognized=is_recognized,
                    timestamp=frame_metadata.timestamp,
                    source_id=frame_metadata.source_id,
                    frame_number=frame_metadata.frame_number
                )
                
                identified_persons.append(person_detection)
            
            # Step 5: Update tracking if enabled
            if self.enable_tracking:
                self._update_tracking(identified_persons, frame_metadata)
            
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
            
            # Log results
            recognized_count = sum(1 for p in identified_persons if p.is_recognized)
            unknown_count = len(identified_persons) - recognized_count
            
            self.logger.debug(
                f"Frame {frame_metadata.frame_number}: {len(detected_faces)} faces detected, "
                f"{recognized_count} recognized, {unknown_count} unknown "
                f"(processed in {processing_time:.3f}s)"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Error identifying persons in frame: {e}"
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
    
    async def identify_persons_in_frames_async(self, frame_stream) -> List[IdentificationResult]:
        """
        Asynchronously process multiple frames for person identification.
        
        Args:
            frame_stream: Async generator yielding (frame, metadata) tuples
            
        Returns:
            List of identification results
        """
        results = []
        
        async for frame, metadata in frame_stream:
            result = await asyncio.create_task(
                asyncio.to_thread(self.identify_persons_in_frame, frame, metadata)
            )
            results.append(result)
        
        return results
    
    def _update_tracking(self, identified_persons: List[PersonDetection], frame_metadata: FrameMetadata):
        """
        Update person tracking information (placeholder for future tracking implementation).
        
        Args:
            identified_persons: List of identified persons in current frame
            frame_metadata: Current frame metadata
        """
        # This is a placeholder for implementing person tracking across frames
        # Features could include:
        # - Person trajectory tracking
        # - Identity consistency across frames
        # - Dwell time calculation
        # - Entry/exit detection
        
        source_id = frame_metadata.source_id
        
        if source_id not in self.tracking_history:
            self.tracking_history[source_id] = {
                'last_seen': {},
                'trajectories': {},
                'entry_times': {}
            }
        
        # Update last seen times for recognized persons
        for person in identified_persons:
            if person.is_recognized and person.person_id:
                self.tracking_history[source_id]['last_seen'][person.person_id] = frame_metadata.timestamp
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about detection performance.
        
        Returns:
            Dictionary containing detection statistics
        """
        stats = {
            'known_persons_count': len(self.face_recognizer.get_known_persons()),
            'detection_method': self.face_detector.config.method,
            'min_detection_confidence': self.min_detection_confidence,
            'min_recognition_confidence': self.min_recognition_confidence,
            'tracking_enabled': self.enable_tracking
        }
        
        if self.enable_tracking and self.tracking_history:
            stats['tracked_sources'] = len(self.tracking_history)
            total_tracked_persons = sum(
                len(history['last_seen']) 
                for history in self.tracking_history.values()
            )
            stats['total_tracked_persons'] = total_tracked_persons
        
        return stats
    
    def add_known_person(self, person_id: str, person_name: str, face_image, landmarks=None) -> bool:
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
        return self.face_recognizer.add_known_person(person_id, person_name, face_image, landmarks)
    
    def remove_known_person(self, person_id: str) -> bool:
        """
        Remove a person from the recognition database.
        
        Args:
            person_id: ID of person to remove
            
        Returns:
            True if person was removed successfully
        """
        return self.face_recognizer.remove_known_person(person_id)
    
    def get_known_persons(self) -> List[Dict[str, Any]]:
        """
        Get list of all known persons in the database.
        
        Returns:
            List of person information dictionaries
        """
        return self.face_recognizer.get_known_persons()
    
    def annotate_frame_with_detections(self, frame, identification_result: IdentificationResult):
        """
        Annotate a frame with detection and recognition results.
        
        Args:
            frame: Input frame
            identification_result: Results to annotate
            
        Returns:
            Annotated frame
        """
        import cv2
        
        annotated = frame.copy()
        
        for person in identification_result.identified_persons:
            x, y, w, h = person.bbox
            
            # Choose color based on recognition status
            if person.is_recognized:
                color = (0, 255, 0)  # Green for recognized
                label = f"{person.person_name} ({person.recognition_confidence:.2f})"
            else:
                color = (0, 0, 255)  # Red for unknown
                label = f"Unknown ({person.detection_confidence:.2f})"
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(
                annotated, 
                (x, y - label_size[1] - 10), 
                (x + label_size[0], y), 
                color, 
                -1
            )
            cv2.putText(
                annotated, 
                label, 
                (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
        
        # Add frame info
        info_text = f"Frame: {identification_result.frame_number} | Faces: {identification_result.total_faces_detected}"
        cv2.putText(
            annotated, 
            info_text, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        return annotated
