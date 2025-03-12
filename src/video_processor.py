import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple
import logging
import mimetypes
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import FRAME_SAMPLE_RATE, RESIZE_WIDTH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    A class for processing video files and extracting frames for OCR analysis.
    
    This class handles video file operations, frame extraction, and preprocessing
    for optimal OCR performance. It implements the context manager protocol for
    safe resource handling.
    
    Attributes:
        video_path (Path): Path to the video file to be processed
        cap (cv2.VideoCapture): OpenCV video capture object
        fps (float): Frames per second of the video
        frame_count (int): Total number of frames in the video
        duration (float): Total duration of the video in seconds
    """
    
    # Common video file extensions
    SUPPORTED_EXTENSIONS = {
        '.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm',
        '.m4v', '.3gp', '.mpeg', '.mpg', '.ts'
    }
    
    def __init__(self, video_path: str):
        """
        Initialize the VideoProcessor with a video file path.
        
        Args:
            video_path (str): Path to the video file to be processed
            
        Raises:
            ValueError: If the file extension is not recognized as a video format
            
        Note:
            The video file is not opened until the context manager is entered
        """
        self.video_path = Path(video_path)
        
        # Check if file exists
        if not self.video_path.exists():
            raise ValueError(f"Video file not found: {video_path}")
            
        # Check file extension
        extension = self.video_path.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported video format: {extension}\n"
                f"Supported formats: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )
            
        # Try to determine mime type
        mime_type = mimetypes.guess_type(video_path)[0]
        if mime_type and not mime_type.startswith('video/'):
            raise ValueError(f"File does not appear to be a video: {video_path}")
            
        self.cap = None
        self.fps = None
        self.frame_count = None
        self.duration = None
        
    def __enter__(self):
        """
        Context manager entry point that initializes video capture.
        
        This method opens the video file and extracts basic metadata like FPS
        and frame count. It's automatically called when using the 'with' statement.
        
        Returns:
            VideoProcessor: The initialized processor instance
            
        Raises:
            ValueError: If the video file cannot be opened or is corrupted
        """
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(
                f"Could not open video file: {self.video_path}\n"
                "This might be due to:\n"
                "1. Corrupted video file\n"
                "2. Missing video codec\n"
                "3. Unsupported video format\n"
                "Please ensure the video file is valid and not corrupted."
            )
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Validate video metadata
        if self.fps <= 0 or self.frame_count <= 0:
            raise ValueError(
                f"Invalid video metadata: FPS={self.fps}, frames={self.frame_count}\n"
                "The video file might be corrupted or in an unsupported format."
            )
            
        self.duration = self.frame_count / self.fps
        logger.info(f"Video loaded: {self.duration:.2f} seconds, {self.fps} FPS")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit point that ensures proper resource cleanup.
        
        This method releases the video capture object and associated resources.
        It's automatically called when exiting the 'with' statement block.
        
        Args:
            exc_type: The type of the exception that occurred, if any
            exc_val: The instance of the exception that occurred, if any
            exc_tb: The traceback of the exception that occurred, if any
        """
        if self.cap:
            self.cap.release()
            
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a video frame for optimal OCR performance.
        
        This method applies several preprocessing steps to improve text detection:
        1. Resizes the frame while maintaining aspect ratio
        2. Converts the frame to grayscale
        3. Applies image enhancement techniques
        4. Applies adaptive thresholding for better text contrast
        
        Args:
            frame (np.ndarray): Input frame in BGR format from OpenCV
            
        Returns:
            np.ndarray: Preprocessed frame optimized for OCR
            
        Note:
            The preprocessing steps are configured to optimize Google Cloud Vision's
            text detection performance while reducing noise and enhancing text clarity
        """
        try:
            logger.debug(f"Preprocessing frame with initial shape: {frame.shape}")
            
            # Resize while maintaining aspect ratio
            height, width = frame.shape[:2]
            scale = RESIZE_WIDTH / width
            new_height = int(height * scale)
            frame = cv2.resize(frame, (RESIZE_WIDTH, new_height))
            logger.debug(f"Resized frame to: {frame.shape}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            logger.debug("Converted frame to grayscale")
            
            # Apply image enhancement techniques
            # 1. Denoise
            denoised = cv2.fastNlMeansDenoising(gray)
            logger.debug("Applied denoising")
            
            # 2. Increase contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            logger.debug("Applied CLAHE for contrast enhancement")
            
            # 3. Thresholding
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            logger.debug("Applied Otsu's thresholding")
            
            return thresh
            
        except Exception as e:
            logger.error(f"Error in frame preprocessing: {str(e)}")
            # Return original frame if preprocessing fails
            return frame
        
    def extract_frames(self) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Extract frames from the video at regular intervals.
        
        This method implements a generator that yields preprocessed frames along
        with their timestamps. It samples frames based on the FRAME_SAMPLE_RATE
        configuration to optimize processing time while maintaining analysis quality.
        
        Yields:
            Tuple[np.ndarray, float]: A tuple containing:
                - preprocessed frame (np.ndarray)
                - timestamp in seconds (float)
                
        Note:
            The sampling rate can be adjusted in the config file to balance
            between processing speed and analysis granularity
        """
        frame_interval = int(self.fps * FRAME_SAMPLE_RATE)
        current_frame = 0
        total_frames_processed = 0
        total_frames_yielded = 0
        
        logger.info(f"Starting frame extraction with interval: {frame_interval} frames")
        
        while True:
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.info(f"Finished frame extraction. Processed: {total_frames_processed}, Yielded: {total_frames_yielded}")
                    break
                    
                total_frames_processed += 1
                timestamp = current_frame / self.fps
                
                logger.debug(f"Processing frame at timestamp: {timestamp:.2f}s")
                processed_frame = self.preprocess_frame(frame)
                
                if processed_frame is not None:
                    total_frames_yielded += 1
                    logger.debug(f"Yielding frame {total_frames_yielded} at timestamp {timestamp:.2f}s")
                    yield processed_frame, timestamp
                
                current_frame += frame_interval
                
            except Exception as e:
                logger.error(f"Error processing frame at position {current_frame}: {str(e)}")
                current_frame += frame_interval
                continue
            
    def get_video_info(self) -> dict:
        """
        Get basic information about the loaded video.
        
        Returns:
            dict: A dictionary containing video metadata:
                - duration: Total video duration in seconds
                - fps: Frames per second
                - frame_count: Total number of frames
                - frame_sample_rate: Current sampling rate for frame extraction
                
        Note:
            This information is useful for progress tracking and report generation
        """
        return {
            "duration": self.duration,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "frame_sample_rate": FRAME_SAMPLE_RATE
        } 