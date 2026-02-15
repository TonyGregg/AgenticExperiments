"""
Gradio chatbot for Tony Gregg's personal website.
"""
from dotenv import load_dotenv
import gradio as gr

from src.agents.gemini_agent import GeminiAgent
from src.utils.cache import LLMCache
from src.utils.config import Config
from src.agents.me.profile_loader import profile   # ← Import shared profile

load_dotenv()

# -----------------------------------------------------------------------------
# Initialize agent & cache (once)
# -----------------------------------------------------------------------------
cache = LLMCache(cache_dir=str(Config.CACHE_DIR))
agent = GeminiAgent()


# -----------------------------------------------------------------------------
# Chat function
# -----------------------------------------------------------------------------
def chat_with_tony(message: str, history: list):
    messages = [{"role": "system", "content": profile.system_prompt}]  # ← Use profile

    for entry in history:
        if entry.get("role") and entry.get("content"):
            messages.append({"role": entry["role"], "content": entry["content"]})

    messages.append({"role": "user", "content": message})

    try:
        response = cache.cached_api_call(
            model_name=agent.model_name,
            query=messages,
            api_function=agent.generate
        )
        return response["text"]
    except Exception as e:
        return f"I'm sorry, something went wrong: {str(e)}"


# -----------------------------------------------------------------------------
# Launch Gradio
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Starting {profile.name}'s chatbot...")

    demo = gr.ChatInterface(
        fn=chat_with_tony,
        title=f"Chat with {profile.name}",
        description="Ask me anything about my career, experience, or skills!",
        examples=[
            "Can you tell me about your experience with cloud architecture?",
            "What kind of projects have you worked on recently?",
            "Are you open to new opportunities in 2025?",
        ],
        cache_examples=False,
        submit_btn="Send",
        stop_btn="Stop",
    )

    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
    )