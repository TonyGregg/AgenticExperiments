from pathlib import Path
import os

from dotenv import load_dotenv
import gradio as gr

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

summary_path = Path(__file__).parent / "summary.txt"
if summary_path.exists():
    with open(summary_path, encoding="utf-8") as f:
        summary = f.read().strip()
else:
    print("Summary not found, using resume as fallback...")
    summary = resume_content[:2000]

linkedin = os.getenv("LINKEDIN_PROFILE", "https://linkedin.com/in/tony-gregg-example")
name = "Tony Gregg"

# ✅ Fix: Truncate BEFORE the f-string, no comments inside
truncated_resume = resume_content[:8000]

system_prompt = f"""You are acting as {name}. 
You are answering questions on {name}'s website, particularly questions related to {name}'s career, background, skills and experience. 
Your responsibility is to represent {name} for interactions on the website as faithfully as possible. 
You are given a summary of {name}'s background and LinkedIn profile which you can use to answer questions. 
Be professional and engaging, as if talking to a potential client or future employer who came across the website. 
If you don't know the answer, say so honestly — do not make things up.

## Resume / Background Summary:
{summary}

## Full Resume Text (reference when needed):
{truncated_resume}

## LinkedIn Profile:
{linkedin}

With this context, please chat with the user, always staying in character as {name}.
Never break character. Answer in first person as if you are {name}.
"""

# -----------------------------------------------------------------------------
# 3. Initialize agent & cache ONCE at startup
# -----------------------------------------------------------------------------
cache = LLMCache(cache_dir=str(Config.CACHE_DIR))
agent = GeminiAgent()  # ✅ Only initialize once!


# -----------------------------------------------------------------------------
# 4. The chat function
# -----------------------------------------------------------------------------
def chat_with_tony(message: str, history: list):
    """
    In Gradio 6.x history is always:
    [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]
    """
    # Build messages for the agent
    messages = [{"role": "system", "content": system_prompt}]

    # In Gradio 6.x history is always dicts - no need to check format
    for entry in history:
        if entry.get("role") and entry.get("content"):
            messages.append({
                "role": entry["role"],
                "content": entry["content"]
            })

    # Add current user message
    messages.append({"role": "user", "content": message})

    try:
        response = cache.cached_api_call(
            model_name=agent.model_name,
            query=messages,
            api_function=agent.generate,
            use_full_context=False  # default - caches by last user message
        )
        return response["text"]
    except Exception as e:
        return f"I'm sorry, something went wrong: {str(e)}"


# -----------------------------------------------------------------------------
# 5. Launch Gradio
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting Tony Gregg's personal website chatbot...")
    print("Resume loaded successfully." if resume_content else "Warning: no resume content.")

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
        cache_examples=False,
        submit_btn="Send",
        stop_btn="Stop",
    )

    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
    )