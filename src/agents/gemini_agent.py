"""
Gemini AI agent implementation.
"""
from openai import OpenAI
from typing import Dict, Any, List, Union, Type, Optional
from src.utils.config import Config
from pydantic import BaseModel


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

    def generate(
        self,
        query: Union[str, List[Dict[str, str]]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response from Gemini using OpenAI-compatible API.

        Args:
            query: Either a simple string or a list of message dictionaries.
                   - String: "What is quantum computing?"
                   - List: [
                       {"role": "system", "content": "You are a helpful assistant"},
                       {"role": "user", "content": "What is quantum computing?"}
                     ]
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Dict containing the response text, model name, and metadata
        """
        # Convert string to messages format if needed
        if isinstance(query, str):
            messages = [{"role": "user", "content": query}]
        else:
            messages = query

        try:
            # Use OpenAI SDK's chat completion method
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=kwargs.get("temperature", Config.DEFAULT_TEMPERATURE),
                max_tokens=kwargs.get("max_tokens", Config.DEFAULT_MAX_TOKENS)
            )

            return {
                "text": response.choices[0].message.content,
                "model": self.model_name,
                "metadata": {
                    "usage": response.usage.model_dump() if response.usage else None,
                    "finish_reason": response.choices[0].finish_reason,
                    **kwargs
                }
            }
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            raise

    def generate_structured(
            self,
            query: Union[str, List[Dict[str, str]]],
            response_format: Type[BaseModel],
            **kwargs
    ) -> BaseModel:
        """
        Generate a structured response matching the Pydantic model.
        Works with OpenAI SDK 1.x using json_schema response format.
        """
        messages = [{"role": "user", "content": query}] if isinstance(query, str) else query

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "schema": response_format.model_json_schema()
                    }
                },
                temperature=kwargs.get("temperature", Config.DEFAULT_TEMPERATURE),
                max_tokens=kwargs.get("max_tokens", Config.DEFAULT_MAX_TOKENS)
            )

            raw_text = response.choices[0].message.content
            print(f"Response: {raw_text}")
            return response_format.parse_raw(raw_text)
            return response_format.model_validate_json(raw_text)

        except Exception as e:
            print(f"Error calling Gemini structured API: {e}")
            raise