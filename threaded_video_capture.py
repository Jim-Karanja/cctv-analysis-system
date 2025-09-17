#!/usr/bin/env python3
"""
Threaded video capture to prevent blocking in async video processor.
"""

import cv2
import threading
import time
import queue
import logging
from typing import Optional, Tuple, Any

logger = logging.getLogger(__name__)

class ThreadedVideoCapture:
    """
    Video capture that runs in a background thread to prevent blocking.
    Continuously captures frames and puts them in a thread-safe queue.
    """
    
    def __init__(self, source: str, buffer_size: int = 10):
        """
        Initialize threaded video capture.
        
        Args:
            source: Video source (camera index, file path, or URL)
            buffer_size: Maximum number of frames to buffer
        """
        self.source = source
        self.buffer_size = buffer_size
        
        # Thread-safe frame queue
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        
        # Control variables
        self.capture_thread = None
        self.running = False
        self.cap = None
        
        # Stats
        self.frames_captured = 0
        self.frames_dropped = 0
        self.last_fps_time = time.time()
        self.fps = 0
        
    def start(self) -> bool:
        """
        Start the capture thread.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.running:
            logger.warning("Capture already running")
            return True
            
        # Initialize video capture
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            return False
        
        # Configure capture
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce latency
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        logger.info(f"Started threaded capture for source: {self.source}")
        return True
    
    def _capture_loop(self):
        """
        Main capture loop running in background thread.
        """
        logger.info(f"Capture thread started for source: {self.source}")
        
        while self.running:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    
                    if ret and frame is not None:
                        self.frames_captured += 1
                        
                        # Try to put frame in queue (non-blocking)
                        try:
                            self.frame_queue.put((ret, frame, time.time()), block=False)
                        except queue.Full:
                            # Queue is full, drop oldest frame
                            try:
                                self.frame_queue.get(block=False)
                                self.frame_queue.put((ret, frame, time.time()), block=False)
                                self.frames_dropped += 1
                            except queue.Empty:
                                pass
                        
                        # Calculate FPS
                        current_time = time.time()
                        if current_time - self.last_fps_time >= 1.0:
                            self.fps = self.frames_captured / (current_time - self.last_fps_time)
                            self.frames_captured = 0
                            self.last_fps_time = current_time
                            
                            if self.frames_dropped > 0:
                                logger.debug(f"Source {self.source}: {self.fps:.1f} FPS, dropped {self.frames_dropped} frames")
                                self.frames_dropped = 0
                    
                    else:
                        # Failed to read frame, wait a bit
                        time.sleep(0.1)
                        logger.warning(f"Failed to read frame from source: {self.source}")
                
                else:
                    logger.error(f"Video capture not available for source: {self.source}")
                    break
                    
            except Exception as e:
                logger.error(f"Error in capture loop for {self.source}: {e}")
                time.sleep(1)  # Wait before retrying
        
        logger.info(f"Capture thread stopped for source: {self.source}")
    
    def read(self) -> Tuple[bool, Optional[Any], float]:
        """
        Get the most recent frame from the queue.
        
        Returns:
            (success, frame, timestamp) tuple
        """
        try:
            # Get the most recent frame (discard older ones)
            ret, frame, timestamp = None, None, time.time()
            
            # Empty the queue to get only the latest frame
            while not self.frame_queue.empty():
                try:
                    ret, frame, timestamp = self.frame_queue.get(block=False)
                except queue.Empty:
                    break
            
            if ret is not None:
                return ret, frame, timestamp
            else:
                return False, None, timestamp
                
        except Exception as e:
            logger.error(f"Error reading frame from source {self.source}: {e}")
            return False, None, time.time()
    
    def stop(self):
        """
        Stop the capture thread and cleanup resources.
        """
        if not self.running:
            return
            
        logger.info(f"Stopping capture for source: {self.source}")
        self.running = False
        
        # Wait for thread to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        # Cleanup capture
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get(block=False)
            except queue.Empty:
                break
        
        logger.info(f"Stopped capture for source: {self.source}")
    
    def is_opened(self) -> bool:
        """
        Check if video capture is open and running.
        
        Returns:
            True if capture is running, False otherwise
        """
        return self.running and self.cap and self.cap.isOpened()
    
    def get_fps(self) -> float:
        """
        Get current capture FPS.
        
        Returns:
            Current FPS
        """
        return self.fps

    def get_stats(self) -> dict:
        """
        Get capture statistics.
        
        Returns:
            Dictionary with capture stats
        """
        return {
            'source': self.source,
            'running': self.running,
            'fps': self.fps,
            'queue_size': self.frame_queue.qsize(),
            'frames_dropped': self.frames_dropped
        }


class ThreadedVideoManager:
    """
    Manages multiple threaded video captures.
    """
    
    def __init__(self):
        self.captures = {}
    
    def add_source(self, source_id: str, source: str, buffer_size: int = 10) -> bool:
        """
        Add a video source.
        
        Args:
            source_id: Unique identifier for the source
            source: Video source (camera index, file path, or URL)
            buffer_size: Maximum number of frames to buffer
            
        Returns:
            True if added successfully, False otherwise
        """
        if source_id in self.captures:
            logger.warning(f"Source {source_id} already exists")
            return False
        
        capture = ThreadedVideoCapture(source, buffer_size)
        if capture.start():
            self.captures[source_id] = capture
            logger.info(f"Added video source: {source_id} -> {source}")
            return True
        else:
            logger.error(f"Failed to add video source: {source_id}")
            return False
    
    def remove_source(self, source_id: str) -> bool:
        """
        Remove a video source.
        
        Args:
            source_id: Source identifier to remove
            
        Returns:
            True if removed successfully, False otherwise
        """
        if source_id not in self.captures:
            logger.warning(f"Source {source_id} not found")
            return False
        
        self.captures[source_id].stop()
        del self.captures[source_id]
        logger.info(f"Removed video source: {source_id}")
        return True
    
    def read_frame(self, source_id: str) -> Tuple[bool, Optional[Any], float]:
        """
        Read frame from a specific source.
        
        Args:
            source_id: Source identifier
            
        Returns:
            (success, frame, timestamp) tuple
        """
        if source_id not in self.captures:
            logger.warning(f"Source {source_id} not found")
            return False, None, time.time()
        
        return self.captures[source_id].read()
    
    def get_active_sources(self) -> list:
        """
        Get list of active source IDs.
        
        Returns:
            List of active source IDs
        """
        return [source_id for source_id, capture in self.captures.items() if capture.is_opened()]
    
    def get_all_stats(self) -> dict:
        """
        Get statistics for all sources.
        
        Returns:
            Dictionary with stats for each source
        """
        return {source_id: capture.get_stats() for source_id, capture in self.captures.items()}
    
    def cleanup(self):
        """
        Stop all captures and cleanup.
        """
        logger.info("Cleaning up all video sources...")
        for source_id in list(self.captures.keys()):
            self.remove_source(source_id)
        logger.info("All video sources cleaned up")


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with default camera
    manager = ThreadedVideoManager()
    
    if manager.add_source("camera_0", 0):
        print("Testing threaded video capture...")
        
        # Read frames for 10 seconds
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < 10:
                ret, frame, timestamp = manager.read_frame("camera_0")
                
                if ret and frame is not None:
                    frame_count += 1
                    # Display frame
                    cv2.imshow("Threaded Video Capture", frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                time.sleep(0.03)  # ~30 FPS display
        
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        finally:
            cv2.destroyAllWindows()
            manager.cleanup()
            
            print(f"Captured {frame_count} frames in {time.time() - start_time:.1f} seconds")
            print("Stats:", manager.get_all_stats())
    
    else:
        print("Failed to initialize camera")
