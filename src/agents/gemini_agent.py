"""
Gemini AI agent implementation.
"""
from openai import OpenAI
from typing import Dict, Any
from src.utils.config import Config


class GeminiAgent:
    """Agent for interacting with Google Gemini API."""

    def __init__(self, api_key: str = None, model_name: str = None):
        # Use provided values or fall back to Config
        self.api_key = api_key or Config.GEMINI_API_KEY
        self.model_name = model_name or Config.GEMINI_MODEL

        # Validate API key
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set. Please check your .env file.")

        # Initialize OpenAI client with Gemini endpoint
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=Config.GEMINI_BASE_URL
        )

    def generate(self, query: str, **kwargs) -> Dict[str, Any]:
        """Generate response from Gemini using OpenAI-compatible API."""
        try:
            # Use OpenAI SDK's chat completion method
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                temperature=kwargs.get("temperature", Config.DEFAULT_TEMPERATURE),
                max_tokens=kwargs.get("max_tokens", Config.DEFAULT_MAX_TOKENS)
            )

            return {
                "text": response.choices[0].message.content,
                "model": self.model_name,
                "metadata": kwargs
            }
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            raise