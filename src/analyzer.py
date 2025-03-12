from openai import OpenAI
import pandas as pd
from typing import List, Dict
import logging
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import OPENAI_MODEL, MAX_TOKENS, APP_CATEGORIES, RISK_KEYWORDS, FRAME_SAMPLE_RATE

logger = logging.getLogger(__name__)

class ScreenTimeAnalyzer:
    """
    A class for analyzing screen time data using OpenAI's language models.
    
    This class processes OCR results to categorize activities, detect risks,
    and generate insights about a child's screen time session. It uses
    predefined categories and risk keywords to classify content and
    leverages OpenAI's API for generating parent-friendly reports.
    
    Attributes:
        client (OpenAI): OpenAI API client instance
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the analyzer with OpenAI API credentials.
        
        Args:
            api_key (str): OpenAI API key for authentication
            
        Note:
            The API key should be kept secure and not exposed in the code
        """
        self.client = OpenAI(api_key=api_key)
        
    def _categorize_app(self, text_data: List[Dict]) -> str:
        """
        Categorize detected text into predefined app categories.
        
        This method analyzes the text content from OCR results and
        classifies it into categories defined in APP_CATEGORIES config.
        It uses keyword matching to determine the most appropriate category.
        
        Args:
            text_data (List[Dict]): List of text detection results, each containing:
                - text: The detected text string
                - confidence: Detection confidence score
                - bounds: Coordinates of the text bounding box
                
        Returns:
            str: The determined category name ('educational', 'entertainment',
                'social', or 'other')
                
        Note:
            Categories are determined by the presence of specific keywords
            defined in the configuration
        """
        text_content = " ".join([t['text'].lower() for t in text_data])
        
        for category, keywords in APP_CATEGORIES.items():
            if any(keyword in text_content for keyword in keywords):
                return category
        return "other"
        
    def _detect_risks(self, text_data: List[Dict]) -> List[str]:
        """
        Detect potential risks in the detected text.
        
        This method scans the text content for predefined risk keywords
        that might indicate inappropriate content or behavior.
        
        Args:
            text_data (List[Dict]): List of text detection results, each containing:
                - text: The detected text string
                - confidence: Detection confidence score
                - bounds: Coordinates of the text bounding box
                
        Returns:
            List[str]: List of detected risk keywords found in the text
            
        Note:
            Risk keywords are defined in the RISK_KEYWORDS configuration
            and can be customized based on parents' concerns
        """
        text_content = " ".join([t['text'].lower() for t in text_data])
        return [keyword for keyword in RISK_KEYWORDS if keyword in text_content]
        
    def analyze_session(self, ocr_results: List[Dict]) -> Dict:
        """
        Analyze a complete screen time session and generate insights.
        
        This method processes all OCR results from a session to:
        1. Track time spent in different app categories
        2. Identify potential risks
        3. Generate a comprehensive analysis using OpenAI
        4. Provide actionable insights for parents
        
        Args:
            ocr_results (List[Dict]): List of OCR results for each processed frame,
                containing timestamps and detected text
                
        Returns:
            Dict: Analysis results containing:
                - time_distribution: Time spent in each category
                - risk_instances: Number of detected risk instances
                - risk_timestamps: Timestamps of risk detections
                - analysis_summary: Detailed AI-generated report
                
        Note:
            The analysis uses OpenAI's language model to generate
            human-readable insights and recommendations
        """
        # Add debug logging
        logger.info(f"Analyzing session with {len(ocr_results)} frames")
        if len(ocr_results) == 0:
            logger.warning("No OCR results to analyze")
            return {
                'time_distribution': {'other': 0},
                'risk_instances': 0,
                'risk_timestamps': [],
                'analysis_summary': "No content detected in the video."
            }

        # Debug first result
        if ocr_results:
            logger.debug(f"Sample OCR result: {ocr_results[0]}")

        # Prepare data for analysis with error handling
        timeline = []
        for result in ocr_results:
            try:
                timestamp = result['timestamp']
                texts = result.get('texts', [])  # Use get() with default empty list
                
                # Debug text detection
                logger.debug(f"Processing frame at {timestamp} with {len(texts)} text detections")
                
                category = self._categorize_app(texts) if texts else 'other'
                risks = self._detect_risks(texts) if texts else []
                
                timeline.append({
                    'timestamp': timestamp,
                    'category': category,
                    'risks': risks,
                    'text_content': " ".join([t['text'] for t in texts]) if texts else ""
                })
            except Exception as e:
                logger.error(f"Error processing frame at {timestamp}: {str(e)}")
                continue

        # Create DataFrame with error handling
        try:
            df = pd.DataFrame(timeline)
            logger.debug(f"Created DataFrame with columns: {df.columns.tolist()}")
            logger.debug(f"DataFrame head: {df.head()}")
        except Exception as e:
            logger.error(f"Error creating DataFrame: {str(e)}")
            return {
                'time_distribution': {'other': 0},
                'risk_instances': 0,
                'risk_timestamps': [],
                'analysis_summary': f"Error analyzing video content: {str(e)}"
            }

        # Calculate time spent in each category
        try:
            time_per_category = df.groupby('category').size() * FRAME_SAMPLE_RATE
            logger.info(f"Time distribution: {time_per_category.to_dict()}")
        except Exception as e:
            logger.error(f"Error calculating time distribution: {str(e)}")
            time_per_category = pd.Series({'other': len(df) * FRAME_SAMPLE_RATE})

        # Analyze risks
        try:
            risk_instances = df[df['risks'].apply(lambda x: len(x) > 0)]
            logger.info(f"Found {len(risk_instances)} risk instances")
        except Exception as e:
            logger.error(f"Error analyzing risks: {str(e)}")
            risk_instances = pd.DataFrame()

        # Generate summary using OpenAI
        try:
            summary_prompt = f"""
            Analyze this screen time session and generate a parent-friendly report. Key points:

            Time spent per category:
            {time_per_category.to_dict()}

            Number of risk instances detected: {len(risk_instances)}

            Sample of detected text content:
            {df['text_content'].sample(min(5, len(df))).to_list() if not df.empty else ["No text detected"]}

            Please provide:
            1. Overview of the session
            2. Time distribution analysis
            3. Risk assessment
            4. Recommendations for parents
            """

            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert in analyzing children's screen time and providing actionable insights for parents."},
                    {"role": "user", "content": summary_prompt}
                ],
                max_tokens=MAX_TOKENS
            )
            analysis_summary = response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating analysis summary: {str(e)}")
            analysis_summary = "Error generating analysis summary."

        return {
            'time_distribution': time_per_category.to_dict(),
            'risk_instances': len(risk_instances),
            'risk_timestamps': risk_instances['timestamp'].to_list() if not risk_instances.empty else [],
            'analysis_summary': analysis_summary
        } 