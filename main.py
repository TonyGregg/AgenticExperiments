"""
Main entry point for the agentic AI application.
"""
from dotenv import load_dotenv

from src.utils.cache import LLMCache
from src.utils.config import Config
from src.agents.gemini_agent import GeminiAgent
from src.agents.xai_agent import XAIAgent

# Load environment variables FIRST, before anything else
load_dotenv()

def main():
    # Initialize cache
    cache = LLMCache(cache_dir=str(Config.CACHE_DIR))

    # Initialize agents (no need to pass api_key, uses Config defaults)
    gemini = GeminiAgent()
    xai = XAIAgent()

    # Example query
    query = "Explain quantum computing in simple terms"

    # Query both models with caching
    print("Querying Gemini...")
    gemini_response = cache.cached_api_call(
        model_name=gemini.model_name,
        query=query,
        api_function=gemini.generate
    )

    print("\nQuerying xAI...")
    xai_response = cache.cached_api_call(
        model_name=xai.model_name,
        query=query,
        api_function=xai.generate
    )

    print("\n--- Gemini Response ---")
    print(gemini_response["text"])

    print("\n--- xAI Response ---")
    print(xai_response["text"])


if __name__ == "__main__":
    main()