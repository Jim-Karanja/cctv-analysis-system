"""
Preprocessor

Handles preprocessing of video frames to prepare them for AI analysis.
"""

import cv2
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    """Configuration for frame preprocessing."""
    target_size: Tuple[int, int] = (640, 480)
    normalize: bool = True
    enhance_contrast: bool = False
    denoise: bool = False
    color_space: str = 'BGR'  # BGR, RGB, GRAY
    quality_threshold: float = 0.5


class Preprocessor:
    """
    Preprocesses video frames to optimize them for face detection and recognition.
    
    Performs operations like resizing, normalization, noise reduction,
    and color space conversions.
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def resize_frame(self, frame: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Resize frame to target dimensions while maintaining aspect ratio.
        
        Args:
            frame: Input frame
            target_size: Target (width, height), uses config default if None
            
        Returns:
            Resized frame
        """
        if target_size is None:
            target_size = self.config.target_size
        
        height, width = frame.shape[:2]
        target_width, target_height = target_size
        
        # Calculate scaling factors
        scale_x = target_width / width
        scale_y = target_height / height
        scale = min(scale_x, scale_y)  # Maintain aspect ratio
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize frame
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Pad to exact target size if needed
        if new_width != target_width or new_height != target_height:
            # Create padded frame
            padded = np.zeros((target_height, target_width, frame.shape[2]), dtype=frame.dtype)
            
            # Calculate padding offsets
            y_offset = (target_height - new_height) // 2
            x_offset = (target_width - new_width) // 2
            
            # Place resized frame in center
            padded[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
            
            return padded
        
        return resized
    
    def convert_color_space(self, frame: np.ndarray, target_space: str = None) -> np.ndarray:
        """
        Convert frame to specified color space.
        
        Args:
            frame: Input frame
            target_space: Target color space (BGR, RGB, GRAY)
            
        Returns:
            Converted frame
        """
        if target_space is None:
            target_space = self.config.color_space
        
        if target_space == 'RGB' and len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif target_space == 'GRAY':
            if len(frame.shape) == 3:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                return frame  # Already grayscale
        else:
            return frame  # Keep as BGR or return as-is
    
    def enhance_contrast(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance frame contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            frame: Input frame
            
        Returns:
            Contrast-enhanced frame
        """
        if len(frame.shape) == 3:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(frame)
        
        return enhanced
    
    def denoise_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply denoising to the frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Denoised frame
        """
        if len(frame.shape) == 3:
            # Color image denoising
            denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        else:
            # Grayscale image denoising
            denoised = cv2.fastNlMeansDenoising(frame, None, 10, 7, 21)
        
        return denoised
    
    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize frame values to range [0, 1].
        
        Args:
            frame: Input frame
            
        Returns:
            Normalized frame
        """
        return frame.astype(np.float32) / 255.0
    
    def assess_quality(self, frame: np.ndarray) -> float:
        """
        Assess the quality of a frame using variance of Laplacian.
        
        Higher values indicate sharper images.
        
        Args:
            frame: Input frame
            
        Returns:
            Quality score (higher is better)
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Calculate variance of Laplacian
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 range (rough approximation)
        quality_score = min(laplacian_var / 1000.0, 1.0)
        
        return quality_score
    
    def preprocess_frame(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Apply full preprocessing pipeline to a frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (processed_frame, metadata) or None if frame quality is too low
        """
        try:
            # Assess initial quality
            quality_score = self.assess_quality(frame)
            
            if quality_score < self.config.quality_threshold:
                self.logger.debug(f"Frame quality too low: {quality_score:.3f} < {self.config.quality_threshold}")
                return None
            
            processed_frame = frame.copy()
            
            # Apply denoising if enabled
            if self.config.denoise:
                processed_frame = self.denoise_frame(processed_frame)
            
            # Apply contrast enhancement if enabled
            if self.config.enhance_contrast:
                processed_frame = self.enhance_contrast(processed_frame)
            
            # Resize frame
            processed_frame = self.resize_frame(processed_frame)
            
            # Convert color space
            processed_frame = self.convert_color_space(processed_frame)
            
            # Normalize if enabled
            if self.config.normalize:
                processed_frame = self.normalize_frame(processed_frame)
            
            # Prepare metadata
            metadata = {
                'original_shape': frame.shape,
                'processed_shape': processed_frame.shape,
                'quality_score': quality_score,
                'preprocessing_applied': {
                    'resize': True,
                    'color_conversion': self.config.color_space != 'BGR',
                    'contrast_enhancement': self.config.enhance_contrast,
                    'denoising': self.config.denoise,
                    'normalization': self.config.normalize
                }
            }
            
            self.logger.debug(f"Successfully preprocessed frame with quality score: {quality_score:.3f}")
            
            return processed_frame, metadata
            
        except Exception as e:
            self.logger.error(f"Error preprocessing frame: {e}")
            return None
