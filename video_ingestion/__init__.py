"""
Video Ingestion Module

This module handles video stream acquisition, frame extraction,
and preprocessing for the CCTV analysis system.
"""

from .video_processor import VideoProcessor
from .frame_extractor import FrameExtractor
from .preprocessor import Preprocessor

__all__ = ['VideoProcessor', 'FrameExtractor', 'Preprocessor']
