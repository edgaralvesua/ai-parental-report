from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Video processing settings
FRAME_SAMPLE_RATE = 1  # Extract 1 frame per second
MIN_TEXT_CONFIDENCE = 0.3  # Lowered from 0.7 to allow more detections
RESIZE_WIDTH = 1280  # Resize video frames to this width while maintaining aspect ratio

# OCR settings
OCR_BATCH_SIZE = 10  # Number of frames to process in one batch

# OpenAI settings
OPENAI_MODEL = "gpt-4-turbo-preview"
MAX_TOKENS = 4000

# Analysis settings
RISK_KEYWORDS = [
    "inappropriate",
    "violent",
    "adult",
    "gambling",
    "chat",
    "messaging"
]

# Time tracking categories
APP_CATEGORIES = {
    "educational": ["learning", "education", "school", "math", "science"],
    "entertainment": ["game", "video", "music", "play"],
    "social": ["chat", "message", "social"],
    "other": []
} 