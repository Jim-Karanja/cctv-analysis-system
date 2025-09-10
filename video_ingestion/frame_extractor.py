"""
Frame Extractor

Handles extraction and buffering of frames from video streams.
"""

import asyncio
import logging
import time
from collections import deque
from typing import Optional, Tuple, List
from dataclasses import dataclass
import numpy as np


@dataclass
class FrameMetadata:
    """Metadata associated with a video frame."""
    source_id: str
    timestamp: float
    frame_number: int
    resolution: Tuple[int, int]
    confidence_score: float = 0.0


class FrameExtractor:
    """
    Extracts and manages frames from video streams with intelligent buffering
    and frame rate management.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.frame_buffer = deque(maxlen=config.get('buffer_size', 100))
        self.target_fps = config.get('target_fps', 10)
        self.frame_interval = 1.0 / self.target_fps
        self.last_extraction_time = {}
        self.frame_counters = {}
        self.logger = logging.getLogger(__name__)
    
    def should_extract_frame(self, source_id: str) -> bool:
        """
        Determine if a frame should be extracted based on target FPS.
        
        This helps reduce processing load by only extracting frames at the
        desired rate rather than processing every frame from the video source.
        """
        current_time = time.time()
        
        if source_id not in self.last_extraction_time:
            self.last_extraction_time[source_id] = current_time
            return True
        
        time_since_last = current_time - self.last_extraction_time[source_id]
        
        if time_since_last >= self.frame_interval:
            self.last_extraction_time[source_id] = current_time
            return True
        
        return False
    
    def extract_frame(self, source_id: str, frame: np.ndarray) -> Optional[Tuple[np.ndarray, FrameMetadata]]:
        """
        Extract and prepare a frame for processing.
        
        Args:
            source_id: Identifier for the video source
            frame: Raw frame data from video capture
            
        Returns:
            Tuple of processed frame and metadata, or None if frame should be skipped
        """
        if not self.should_extract_frame(source_id):
            return None
        
        try:
            # Initialize frame counter for new sources
            if source_id not in self.frame_counters:
                self.frame_counters[source_id] = 0
            
            self.frame_counters[source_id] += 1
            
            # Create metadata for the frame
            metadata = FrameMetadata(
                source_id=source_id,
                timestamp=time.time(),
                frame_number=self.frame_counters[source_id],
                resolution=(frame.shape[1], frame.shape[0])
            )
            
            # Add to buffer
            self.frame_buffer.append((frame.copy(), metadata))
            
            self.logger.debug(f"Extracted frame {metadata.frame_number} from {source_id}")
            
            return frame, metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting frame from {source_id}: {e}")
            return None
    
    def get_buffered_frames(self, max_frames: int = None) -> List[Tuple[np.ndarray, FrameMetadata]]:
        """
        Retrieve buffered frames for batch processing.
        
        Args:
            max_frames: Maximum number of frames to return
            
        Returns:
            List of frame and metadata tuples
        """
        if max_frames is None:
            max_frames = len(self.frame_buffer)
        
        frames = []
        for _ in range(min(max_frames, len(self.frame_buffer))):
            if self.frame_buffer:
                frames.append(self.frame_buffer.popleft())
        
        return frames
    
    def get_latest_frame(self, source_id: str) -> Optional[Tuple[np.ndarray, FrameMetadata]]:
        """
        Get the most recent frame from a specific source.
        
        Args:
            source_id: Identifier for the video source
            
        Returns:
            Latest frame and metadata, or None if no frames available
        """
        # Search buffer from most recent to oldest
        for frame, metadata in reversed(self.frame_buffer):
            if metadata.source_id == source_id:
                return frame, metadata
        
        return None
    
    def clear_buffer(self):
        """Clear the frame buffer."""
        self.frame_buffer.clear()
        self.logger.info("Frame buffer cleared")
    
    def get_buffer_info(self) -> dict:
        """Get information about the current buffer state."""
        source_counts = {}
        total_frames = len(self.frame_buffer)
        
        for _, metadata in self.frame_buffer:
            source_id = metadata.source_id
            source_counts[source_id] = source_counts.get(source_id, 0) + 1
        
        return {
            'total_frames': total_frames,
            'source_counts': source_counts,
            'buffer_capacity': self.frame_buffer.maxlen,
            'buffer_utilization': total_frames / self.frame_buffer.maxlen if self.frame_buffer.maxlen else 0
        }
