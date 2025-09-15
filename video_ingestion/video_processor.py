"""
Video Processor

Handles acquisition and management of CCTV video streams from various sources.
"""

import cv2
import asyncio
import logging
from typing import Optional, Generator, Tuple
from dataclasses import dataclass


@dataclass
class VideoSource:
    """Configuration for a video source."""
    source_id: str
    url: str
    fps: int = 30
    resolution: Tuple[int, int] = (1920, 1080)
    enabled: bool = True


class VideoProcessor:
    """
    Main video processor for handling CCTV camera feeds.
    
    Supports multiple video sources including IP cameras, USB cameras,
    and video files for testing purposes.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.video_sources = {}
        self.active_captures = {}
        self.logger = logging.getLogger(__name__)
        
    def add_video_source(self, source: VideoSource) -> bool:
        """Add a new video source to the processor."""
        try:
            self.video_sources[source.source_id] = source
            self.logger.info(f"Added video source: {source.source_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add video source {source.source_id}: {e}")
            return False
    
    def start_capture(self, source_id: str) -> bool:
        """Start capturing from a specific video source."""
        if source_id not in self.video_sources:
            self.logger.error(f"Video source {source_id} not found")
            return False
            
        source = self.video_sources[source_id]
        
        try:
            cap = cv2.VideoCapture(source.url)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video source: {source.url}")
                
            # Configure capture properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, source.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, source.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, source.fps)
            
            self.active_captures[source_id] = cap
            self.logger.info(f"Started capture for source: {source_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start capture for {source_id}: {e}")
            return False
    
    def stop_capture(self, source_id: str) -> bool:
        """Stop capturing from a specific video source."""
        if source_id in self.active_captures:
            self.active_captures[source_id].release()
            del self.active_captures[source_id]
            self.logger.info(f"Stopped capture for source: {source_id}")
            return True
        return False
    
    def get_frame(self, source_id: str) -> Optional[Tuple[bool, any]]:
        """Get a single frame from the specified video source."""
        if source_id not in self.active_captures:
            return None
            
        cap = self.active_captures[source_id]
        ret, frame = cap.read()
        
        # If we can't read a frame (likely end of video file), try to loop back to start
        if not ret:
            source = self.video_sources[source_id]
            # Check if this is a video file (not a camera/stream)
            if isinstance(source.url, str) and (source.url.endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')) or '/' in source.url):
                self.logger.info(f"End of video reached for {source_id}, looping back to start")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                ret, frame = cap.read()  # Try reading again
                if not ret:
                    self.logger.error(f"Failed to read frame even after reset for source: {source_id}")
            else:
                self.logger.warning(f"Failed to read frame from source: {source_id}")
            
        return ret, frame
    
    async def stream_frames(self, source_id: str) -> Generator[Tuple[str, any], None, None]:
        """Async generator that yields frames from the specified video source."""
        if source_id not in self.active_captures:
            self.logger.error(f"No active capture for source: {source_id}")
            return
            
        while True:
            ret, frame = self.get_frame(source_id)
            if ret:
                yield source_id, frame
            else:
                await asyncio.sleep(0.1)  # Brief pause if no frame available
    
    def cleanup(self):
        """Clean up all video captures and resources."""
        for source_id in list(self.active_captures.keys()):
            self.stop_capture(source_id)
        self.logger.info("Video processor cleanup completed")
