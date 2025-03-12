from google.cloud import vision
import numpy as np
import io
import os
from PIL import Image
from typing import List, Dict, Tuple
import logging
import sys
import cv2

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import MIN_TEXT_CONFIDENCE, OCR_BATCH_SIZE

logger = logging.getLogger(__name__)

class OCRProcessor:
    """
    A class for performing OCR on video frames using Google Cloud Vision API.
    
    This class handles the text detection process, including image preparation,
    API communication, and result filtering. It's optimized for batch processing
    of video frames and includes confidence-based filtering.
    
    Attributes:
        client (vision.ImageAnnotatorClient): Google Cloud Vision API client
    """
    
    def __init__(self):
        """
        Initialize the OCR processor with Google Cloud Vision client.
        
        Raises:
            ValueError: If GOOGLE_APPLICATION_CREDENTIALS environment variable is not set
            
        Note:
            Requires proper Google Cloud credentials to be set in the environment.
            The credentials file path should be set in GOOGLE_APPLICATION_CREDENTIALS
            environment variable. See README.md for setup instructions.
        """
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if not credentials_path:
            raise ValueError(
                "GOOGLE_APPLICATION_CREDENTIALS environment variable is not set. "
                "Please set it to the path of your Google Cloud credentials JSON file."
            )
        
        if not os.path.exists(credentials_path):
            raise ValueError(
                f"Google Cloud credentials file not found at: {credentials_path}"
            )
            
        try:
            self.client = vision.ImageAnnotatorClient()
            logger.info("Successfully initialized Google Cloud Vision client")
            logger.debug(f"Using credentials from: {credentials_path}")
        except Exception as e:
            raise ValueError(f"Failed to initialize Google Cloud Vision client: {str(e)}")
        
    def _prepare_image(self, frame: np.ndarray) -> vision.Image:
        """
        Convert an OpenCV frame to a format suitable for Google Cloud Vision API.
        
        This method handles the conversion of numpy array-based frames to the
        binary format required by the Vision API, including JPEG compression.
        
        Args:
            frame (np.ndarray): Input frame in numpy array format
            
        Returns:
            vision.Image: Prepared image object ready for API submission
            
        Raises:
            ValueError: If image encoding fails
        """
        try:
            # Log frame information
            logger.debug(f"Preparing frame with shape: {frame.shape}, dtype: {frame.dtype}")
            
            # Check if frame is valid
            if frame is None or frame.size == 0:
                raise ValueError("Invalid frame: empty or None")
            
            success, encoded = cv2.imencode('.jpg', frame)
            if not success:
                raise ValueError("Failed to encode image")
            
            image_content = encoded.tobytes()
            logger.debug(f"Successfully encoded image, size: {len(image_content)} bytes")
            return vision.Image(content=image_content)
        except Exception as e:
            logger.error(f"Error preparing image: {str(e)}")
            raise
        
    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Process a single frame to extract text using OCR.
        
        This method handles the complete OCR pipeline for a single frame:
        1. Prepares the image for API submission
        2. Calls the Google Cloud Vision API
        3. Filters and processes the results
        4. Applies confidence thresholding based on text characteristics
        
        Args:
            frame (np.ndarray): Input frame to process
            
        Returns:
            List[Dict]: List of detected text regions, each containing:
                - text: The detected text string
                - confidence: Detection confidence score
                - bounds: Coordinates of the text bounding box
                
        Raises:
            Exception: If the API returns an error
            
        Note:
            Confidence scores are calculated based on text characteristics:
            - Text length (longer text gets higher confidence)
            - Position in frame (text closer to center gets higher confidence)
            - Text content (more readable text gets higher confidence)
        """
        try:
            logger.debug("Starting frame processing")
            image = self._prepare_image(frame)
            
            logger.debug("Sending request to Google Cloud Vision API")
            response = self.client.text_detection(image=image)
            
            if response.error.message:
                logger.error(f"API Error: {response.error.message}")
                raise Exception(f"Error detecting text: {response.error.message}")
            
            # Log the raw response for debugging
            logger.debug(f"Received response with {len(response.text_annotations)} annotations")
            
            texts = []
            # Process the first annotation which contains all text
            if response.text_annotations:
                full_text = response.text_annotations[0]
                logger.debug(f"Full text detected: {full_text.description[:100]}...")
                
                # Get frame dimensions for relative position calculation
                frame_height, frame_width = frame.shape[:2]
                frame_center_x = frame_width / 2
                frame_center_y = frame_height / 2
                
                # Process individual text blocks (skip first one as it contains all text)
                for text in response.text_annotations[1:]:
                    # Calculate text characteristics for confidence scoring
                    vertices = text.bounding_poly.vertices
                    
                    # Calculate center of text bounding box
                    center_x = sum(v.x for v in vertices) / 4
                    center_y = sum(v.y for v in vertices) / 4
                    
                    # Calculate distance from center (normalized)
                    dx = (center_x - frame_center_x) / frame_width
                    dy = (center_y - frame_center_y) / frame_height
                    distance_from_center = (dx * dx + dy * dy) ** 0.5
                    
                    # Calculate text box dimensions
                    width = max(v.x for v in vertices) - min(v.x for v in vertices)
                    height = max(v.y for v in vertices) - min(v.y for v in vertices)
                    area = width * height
                    
                    # Text content characteristics
                    text_length = len(text.description)
                    word_count = len(text.description.split())
                    avg_word_length = text_length / max(1, word_count)
                    
                    # Calculate confidence score based on multiple factors
                    position_score = max(0, 1 - distance_from_center)  # Higher score for central text
                    size_score = min(1, area / (frame_width * frame_height) * 100)  # Normalize by frame size
                    length_score = min(1, text_length / 20)  # Cap at 20 characters
                    word_score = min(1, word_count / 5)  # Cap at 5 words
                    
                    # Combine scores with weights
                    confidence = (
                        0.3 * position_score +
                        0.3 * size_score +
                        0.2 * length_score +
                        0.2 * word_score
                    )
                    
                    logger.debug(
                        f"Text: '{text.description}' - Scores: "
                        f"position={position_score:.2f}, "
                        f"size={size_score:.2f}, "
                        f"length={length_score:.2f}, "
                        f"word={word_score:.2f}, "
                        f"final={confidence:.2f}"
                    )
                    
                    if confidence >= MIN_TEXT_CONFIDENCE:
                        text_info = {
                            'text': text.description,
                            'confidence': confidence,
                            'bounds': [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
                        }
                        texts.append(text_info)
                        logger.debug(f"Added text block: '{text.description}' with confidence: {confidence:.2f}")
                    else:
                        logger.debug(f"Filtered out text block due to low confidence: '{text.description}' ({confidence:.2f} < {MIN_TEXT_CONFIDENCE})")
                
            logger.debug(f"Processed frame, found {len(texts)} text regions above confidence threshold")
            return texts
            
        except Exception as e:
            logger.error(f"Error in process_frame: {str(e)}")
            raise
        
    def process_batch(self, frames: List[Tuple[np.ndarray, float]]) -> List[Dict]:
        """
        Process a batch of frames with their timestamps.
        
        This method handles batch processing of frames to optimize API usage and
        processing time. It includes error handling for individual frames to ensure
        the entire batch isn't lost if one frame fails.
        
        Args:
            frames (List[Tuple[np.ndarray, float]]): List of tuples containing:
                - frame: The image data as numpy array
                - timestamp: The frame's timestamp in seconds
                
        Returns:
            List[Dict]: List of results for each frame, containing:
                - timestamp: When the text was detected
                - texts: List of text detection results
                
        Note:
            Errors in individual frame processing are logged but don't stop the batch
        """
        logger.info(f"Processing batch of {len(frames)} frames")
        results = []
        
        for i, (frame, timestamp) in enumerate(frames):
            try:
                logger.debug(f"Processing frame {i+1}/{len(frames)} at timestamp {timestamp:.2f}s")
                texts = self.process_frame(frame)
                if texts:
                    result = {
                        'timestamp': timestamp,
                        'texts': texts
                    }
                    results.append(result)
                    logger.debug(f"Frame {i+1}: Found {len(texts)} text regions")
                else:
                    logger.debug(f"Frame {i+1}: No text detected")
                    
            except Exception as e:
                logger.error(f"Error processing frame at {timestamp}: {str(e)}")
                continue
                
        logger.info(f"Batch processing complete. Found text in {len(results)}/{len(frames)} frames")
        return results 