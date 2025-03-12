# Screen Time Analysis Tool

This tool analyzes screen recordings of children's tablet usage to generate insights and reports for parents. It uses Google Cloud Vision OCR for text detection and OpenAI for analysis.

## Features

- Extracts text from screen recordings using Google Cloud Vision OCR
- Analyzes app usage patterns and time distribution
- Detects potential risk behaviors
- Generates parent-friendly reports with insights and recommendations
- Optimizes video processing for token usage

## Supported Video Formats

The tool supports various video formats through OpenCV, including:

- MP4 (H.264, MPEG-4)
- AVI
- MOV
- WMV
- FLV
- MKV
- WEBM

Note: The quality of analysis depends on the video quality and compression. For best results:

- Use screen recordings with clear, readable text
- Avoid highly compressed videos that might blur text
- Ensure stable frame rate and resolution
- If possible, use lossless or high-bitrate compression

## Prerequisites

- Python 3.8 or higher
- Google Cloud Vision API credentials
- OpenAI API key
- Required Python packages (see `requirements.txt`)

## Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd screen-time-analysis
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up environment variables:
   Create a `.env` file in the project root with the following:

```
OPENAI_API_KEY=your_openai_api_key
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/your/google-credentials.json
```

4. Set up Google Cloud Vision:
   1. Create a project in Google Cloud Console (https://console.cloud.google.com)
   2. Enable the Cloud Vision API for your project
   3. Create a service account:
      - Go to IAM & Admin > Service Accounts
      - Click "Create Service Account"
      - Give it a name and description
      - Grant it the "Cloud Vision API User" role
   4. Create and download credentials:
      - Select your service account
      - Go to the "Keys" tab
      - Click "Add Key" > "Create New Key"
      - Choose JSON format
      - Save the downloaded JSON file securely
   5. Set the GOOGLE_APPLICATION_CREDENTIALS environment variable in your .env file to point to this JSON file

## Usage

Run the analysis on a video file:

```bash
python src/main.py path/to/your/video.mp4
```

The tool will:

1. Process the video frame by frame
2. Extract text using OCR
3. Analyze the content
4. Generate a report in the `output` directory

## Configuration

You can modify the following settings in `config/config.py`:

- Frame sample rate
- OCR confidence threshold
- Video processing parameters
- Risk keywords
- App categories

## Output

The tool generates a detailed report containing:

- Session overview
- Time distribution across different app categories
- Risk assessment
- Recommendations for parents

## Optimization

The tool includes several optimizations:

- Frame sampling to reduce processing time
- Batch processing of OCR requests
- Image preprocessing for better OCR results
- Token optimization for OpenAI API usage

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
