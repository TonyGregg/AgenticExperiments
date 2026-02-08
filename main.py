"""
Main entry point for the agentic AI application.
"""
from src.utils.cache import LLMCache
from src.utils.config import Config
from src.agents.gemini_agent import GeminiAgent
from src.agents.xai_agent import XAIAgent


def main():
    # Initialize cache
    cache = LLMCache(cache_dir=str(Config.CACHE_DIR))

    # Initialize agents
    gemini = GeminiAgent()
    xai = XAIAgent()

    # Example 1: Simple string query (backward compatible)
    print("=== Example 1: Simple Query ===")
    simple_query = "What is fast API?"

    gemini_response = cache.cached_api_call(
        model_name=gemini.model_name,
        query=simple_query,
        api_function=gemini.generate
    )
    print(f"Gemini: {gemini_response['text'][:100]}...\n")

    # Example 2: Multi-role conversation
    print("=== Example 2: Multi-role with System Prompt ===")
    multi_role_query = [
        {
            "role": "system",
            "content": "You are Grok, a highly intelligent, helpful AI assistant with a witty personality."
        },
        {
            "role": "user",
            "content": "Why is India struggling with its economy improvements despite having a very high talent pool?"
        }
    ]

    xai_response = cache.cached_api_call(
        model_name=xai.model_name,
        query=multi_role_query,
        api_function=xai.generate
    )
    print(f"xAI: {xai_response['text']}\n")

    # Example 3: Conversation history
    print("=== Example 3: Conversation with History ===")
    conversation = [
        {
            "role": "system",
            "content": "You are a physics teacher explaining concepts to high school students."
        },
        {
            "role": "user",
            "content": "What is quantum entanglement?"
        },
        {
            "role": "assistant",
            "content": "Quantum entanglement is a phenomenon where two particles become connected..."
        },
        {
            "role": "user",
            "content": "Can you give me a real-world example?"
        }
    ]

    gemini_response = cache.cached_api_call(
        model_name=gemini.model_name,
        query=conversation,
        api_function=gemini.generate
    )
    print(f"Gemini: {gemini_response['text']}\n")

    # Example 4: Different parameters
    print("=== Example 4: Custom Parameters ===")
    creative_query = "Write a haiku about artificial intelligence"

    creative_response = cache.cached_api_call(
        model_name=gemini.model_name,
        query=creative_query,
        api_function=gemini.generate,
        temperature=1.2,  # More creative
        max_tokens=100
    )
    print(f"Creative Gemini: {creative_response['text']}\n")


if __name__ == "__main__":
    main()