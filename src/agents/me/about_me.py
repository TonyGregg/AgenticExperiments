from pathlib import Path
import os

from dotenv import load_dotenv
import gradio as gr

# Your own modules
from src.agents.gemini_agent import GeminiAgent
from src.utils.cache import LLMCache
from src.utils.config import Config
from pypdf import PdfReader

load_dotenv(verbose=True)


# -----------------------------------------------------------------------------
# 1. Load resume content once at startup
# -----------------------------------------------------------------------------
def load_resume_text():
    current_dir = Path(__file__).parent
    pdf_path = current_dir / "resume.pdf"

    if not pdf_path.exists():
        raise FileNotFoundError(f"Resume not found at {pdf_path}")

    try:
        reader = PdfReader(pdf_path)
        resume_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                resume_text += text + "\n\n"
        return resume_text.strip()
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {e}")


# -----------------------------------------------------------------------------
# 2. Prepare system prompt (only once)
# -----------------------------------------------------------------------------
resume_content = load_resume_text()

# Optional: if you have a separate summary file
summary_path = Path(__file__).parent / "summary.txt"
if summary_path.exists():
    with open(summary_path, encoding="utf-8") as f:
        summary = f.read().strip()
else:
    summary = resume_content[:2000]  # fallback: truncate resume

# You can add LinkedIn manually or load from file/env
linkedin = os.getenv("LINKEDIN_PROFILE", "https://linkedin.com/in/tony-gregg-example")

name = "Tony Gregg"

system_prompt = f"""You are acting as {name}. 
You are answering questions on {name}'s website, particularly questions related to {name}'s career, background, skills and experience. 
Your responsibility is to represent {name} for interactions on the website as faithfully as possible. 
You are given a summary of {name}'s background and LinkedIn profile which you can use to answer questions. 
Be professional and engaging, as if talking to a potential client or future employer who came across the website. 
If you don't know the answer, say so honestly — do not make things up.

## Resume / Background Summary:
{summary}

## Full Resume Text (reference when needed):
{resume_content[:8000]}  # truncate if very long to avoid token limits

## LinkedIn Profile:
{linkedin}

With this context, please chat with the user, always staying in character as {name}.
Never break character. Answer in first person as if you are {name}.
"""

# -----------------------------------------------------------------------------
# 3. Initialize your agent & cache (once)
# -----------------------------------------------------------------------------
cache = LLMCache(cache_dir=str(Config.CACHE_DIR))
agent = GeminiAgent()  # assuming this is correctly implemented


# -----------------------------------------------------------------------------
# 4. The chat function — this is what Gradio calls
# -----------------------------------------------------------------------------
def chat_with_tony(message: str, history: list):
    """
    Gradio ChatInterface expects:
    - message: the latest user input
    - history: list of [user_msg, assistant_msg] pairs (older → newer)
    """
    # Build the full conversation for the agent
    # Many agents expect: system prompt + list of messages
    messages = [{"role": "system", "content": system_prompt}]

    # Add history (Gradio gives [[user, assistant], [user, assistant], ...])
    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})

    # Add current user message
    messages.append({"role": "user", "content": message})
    gemini = GeminiAgent()

    # Call your Gemini agent (adapt this call to match your GeminiAgent API)
    try:
        response = cache.cached_api_call(
        model_name = gemini.model_name,
        query = messages,
        api_function = gemini.generate
    )  # ← replace with your actual method
        # If your agent returns an object, extract the text, e.g.:
        # response = agent.generate(messages).text
    except Exception as e:
        response = f"I'm sorry, something went wrong on my end: {str(e)}"

    # Return only the assistant's reply (Gradio appends it automatically)
    return response


# -----------------------------------------------------------------------------
# 5. Launch Gradio Chat Interface
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting Tony Gregg's personal website chatbot...")
    print("Resume loaded successfully." if resume_content else "Warning: no resume content loaded.")

    # The simplest & cleanest way — gr.ChatInterface
    demo = gr.ChatInterface(
        fn=chat_with_tony,
        title="Chat with Tony Gregg",
        description=(
            "Ask me anything about my career, experience, skills, projects, "
            "or how I can help your team/company!"
        ),
        examples=[
            "Can you tell me about your experience with cloud architecture?",
            "What kind of projects have you worked on recently?",
            "Are you open to new opportunities in 2025?",
        ],
        cache_examples=False,  # can keep this
        # Remove these lines:
        # retry_btn="Retry",
        # undo_btn="Undo",
        # clear_btn="Clear Chat",
        # You can still customize submit & stop if desired:
        submit_btn="Send",  # optional: str, bool, or None
        stop_btn="Stop",  # optional: shown during generation/streaming
    )

    demo.launch(
        server_name="0.0.0.0",  # optional — makes it accessible on LAN
        server_port=7860,
        share=False,  # set True for public temporary link (good for testing)
    )