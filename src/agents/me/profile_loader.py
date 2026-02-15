"""
Profile loader - loads resume and summary content.
Shared between about_me.py and evaluator.py.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PROFILE_DIR = Path(__file__).parent  # Points to src/agents/me/
NAME = "Tony Gregg"
LINKEDIN = os.getenv("LINKEDIN_PROFILE", "https://linkedin.com/in/tony-gregg-example")


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------
def load_resume_text(pdf_path: Path = None) -> str:
    """Load and extract text from resume PDF."""
    if pdf_path is None:
        pdf_path = PROFILE_DIR / "resume.pdf"

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


def load_summary(summary_path: Path = None, fallback_text: str = "") -> str:
    """Load summary from file, or fall back to provided text."""
    if summary_path is None:
        summary_path = PROFILE_DIR / "summary.txt"

    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            return f.read().strip()
    else:
        print("Summary not found, using fallback...")
        return fallback_text[:2000]


def build_system_prompt(
    name: str,
    summary: str,
    resume_content: str,
    linkedin: str,
    max_resume_chars: int = 8000
) -> str:
    """Build the system prompt for the AI agent."""
    truncated_resume = resume_content[:max_resume_chars]

    return f"""You are acting as {name}. 
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
# Convenience: Load everything at once
# -----------------------------------------------------------------------------
class ProfileData:
    """Holds all profile data. Load once and share across files."""

    def __init__(self):
        self.name = NAME
        self.linkedin = LINKEDIN
        self.resume_content = load_resume_text()
        self.summary = load_summary(fallback_text=self.resume_content)
        self.system_prompt = build_system_prompt(
            name=self.name,
            summary=self.summary,
            resume_content=self.resume_content,
            linkedin=self.linkedin
        )
        print(f"✓ Profile loaded for: {self.name}")


# Singleton: load once, reuse everywhere
# Import this in any file that needs profile data
profile = ProfileData()