"""
Config management. Reads the common keys from environment and shared across the project.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

class Config:
    # Load environment variables FIRST, before anything else
    load_dotenv()

    """Application Configuration."""
    # API Keys
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY","")
    XAI_API_KEY = os.environ.get("XAI_API_KEY","")

    # Base URLs
    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    XAI_BASE_URL = "https://api.x.ai/v1"

    # Model Names
    GEMINI_MODEL = "gemini-3-flash-preview"
    XAI_MODEL = "grok-4-1-fast-reasoning"

    # Cache settings
    CACHE_DIR = Path("./data/llm_cache")

    # Model Settings
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 1000