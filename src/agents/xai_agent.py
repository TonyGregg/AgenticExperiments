"""
xAI (Grok) agent implementation.
"""
from openai import OpenAI
from typing import Dict, Any
from src.utils.config import Config


class XAIAgent:
    """Agent for interacting with xAI (Grok) API."""

    def __init__(self, api_key: str = None, model_name: str = None):
        self.api_key = api_key or Config.XAI_API_KEY
        self.model_name = model_name or Config.XAI_MODEL

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=Config.XAI_BASE_URL
        )

    def generate(self, query: str, **kwargs) -> Dict[str, Any]:
        """Generate response from xAI."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=kwargs.get("temperature", Config.DEFAULT_TEMPERATURE),
            max_tokens=kwargs.get("max_tokens", Config.DEFAULT_MAX_TOKENS),
            **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
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