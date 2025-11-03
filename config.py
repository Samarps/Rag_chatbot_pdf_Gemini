import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get the Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check if the key exists
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in your .env file")