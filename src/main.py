import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.video_processor import VideoProcessor
from src.ocr_processor import OCRProcessor
from src.analyzer import ScreenTimeAnalyzer
from config.config import OCR_BATCH_SIZE, OUTPUT_DIR

# Set up logging configuration for the entire application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Update logging level for more detailed output
logging.getLogger('src').setLevel(logging.DEBUG)

def check_credentials():
    """
    Verify that all required API credentials are properly configured.
    
    Raises:
        ValueError: If any required credentials are missing
    """
    # Check OpenAI API key
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
        
    # Check Google Cloud credentials
    google_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not google_creds:
        raise ValueError(
            "GOOGLE_APPLICATION_CREDENTIALS environment variable is not set. "
            "Please set it to the path of your Google Cloud credentials JSON file."
        )
    
    if not os.path.exists(google_creds):
        raise ValueError(
            f"Google Cloud credentials file not found at: {google_creds}"
        )

def process_video(video_path: str) -> dict:
    """
    Process a video file to analyze screen time and generate a report.
    
    This function orchestrates the complete analysis pipeline:
    1. Loads environment variables and initializes components
    2. Processes the video frame by frame
    3. Performs OCR on extracted frames
    4. Analyzes the results and generates insights
    5. Saves a detailed report
    
    Args:
        video_path (str): Path to the video file to analyze
        
    Returns:
        dict: Analysis results containing:
            - time_distribution: Time spent in different categories
            - risk_instances: Number of detected risks
            - risk_timestamps: When risks were detected
            - analysis_summary: Detailed AI-generated report
            
    Raises:
        ValueError: If required API credentials are not set
        Exception: For video processing or analysis errors
        
    Note:
        This function requires both OpenAI and Google Cloud Vision
        credentials to be properly configured
    """
    # Load and verify API credentials
    load_dotenv()
    check_credentials()
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize processing components
    ocr_processor = OCRProcessor()
    analyzer = ScreenTimeAnalyzer(os.getenv('OPENAI_API_KEY'))
    
    logger.info(f"Starting analysis of video: {video_path}")
    
    # Process video frames using context manager for proper resource handling
    with VideoProcessor(video_path) as video_proc:
        video_info = video_proc.get_video_info()
        logger.info(f"Video info: {video_info}")
        
        # Process frames in batches for efficiency
        current_batch = []
        all_results = []
        
        for frame, timestamp in video_proc.extract_frames():
            current_batch.append((frame, timestamp))
            
            # Process batch when it reaches the configured size
            if len(current_batch) >= OCR_BATCH_SIZE:
                batch_results = ocr_processor.process_batch(current_batch)
                all_results.extend(batch_results)
                current_batch = []
                
        # Process any remaining frames
        if current_batch:
            batch_results = ocr_processor.process_batch(current_batch)
            all_results.extend(batch_results)
            
    # Generate analysis and insights
    logger.info("Generating analysis report...")
    analysis = analyzer.analyze_session(all_results)
    
    # Save detailed report to file
    output_path = OUTPUT_DIR / f"analysis_{Path(video_path).stem}.txt"
    with open(output_path, 'w') as f:
        f.write("Screen Time Analysis Report\n")
        f.write("=========================\n\n")
        f.write(f"Video Duration: {video_info['duration']:.2f} seconds\n\n")
        
        # Write time distribution statistics
        f.write("Time Distribution:\n")
        for category, time in analysis['time_distribution'].items():
            f.write(f"- {category}: {time:.2f} seconds\n")
            
        # Write risk assessment information
        f.write("\nRisk Assessment:\n")
        f.write(f"- Number of risk instances: {analysis['risk_instances']}\n")
        if analysis['risk_timestamps']:
            f.write("- Risk timestamps (seconds):\n")
            for timestamp in analysis['risk_timestamps']:
                f.write(f"  * {timestamp:.2f}\n")
                
        # Write the AI-generated analysis
        f.write("\nDetailed Analysis:\n")
        f.write(analysis['analysis_summary'])
        
    logger.info(f"Analysis complete. Report saved to {output_path}")
    return analysis

if __name__ == "__main__":
    # Set up command-line argument parsing
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze children's screen time from video recordings")
    parser.add_argument("video_path", help="Path to the video file to analyze")
    args = parser.parse_args()
    
    try:
        process_video(args.video_path)
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise 