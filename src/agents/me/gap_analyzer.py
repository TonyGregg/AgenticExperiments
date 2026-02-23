"""
Gap Analyzer - Identifies questions that cannot be answered from resume content.
Helps identify what information is missing from your profile.
"""
from typing import List
from src.agents.gemini_agent import GeminiAgent
from src.utils.cache import LLMCache
from src.utils.config import Config
from src.agents.me.profile_loader import profile
from src.models.response_log import ResponseLog

# -----------------------------------------------------------------------------
# Initialize
# -----------------------------------------------------------------------------
cache = LLMCache(cache_dir=str(Config.CACHE_DIR))
gemini = GeminiAgent()

# -----------------------------------------------------------------------------
# Test Questions (mix of answerable and unanswerable)
# -----------------------------------------------------------------------------
TEST_QUESTIONS = [
    # ✅ Should be answerable from resume
    "What programming languages do you know?",
    "What is your experience with AKS?",

    # ❌ Likely NOT answerable from resume
    "What is your favorite food?",
    "Do you have any pets?",
    # "What are your hobbies outside of work?",
    # "What is your Myers-Briggs personality type?",
    # "Where do you see yourself in 10 years?",
    # "What is your favorite programming language and why?",
    # "What time do you usually wake up?",
    # "Do you prefer working from home or office?",
    # "What motivates you every day?",
    # "What books are you currently reading?",

    # ⚠️ Edge cases - might be partially answerable
    "What is your management philosophy?",
    # "How do you handle conflict in teams?",
    # "What are your salary expectations?",
    # "Are you willing to relocate?",
    # "What is your notice period?",
]


# -----------------------------------------------------------------------------
# Analyzer Functions
# -----------------------------------------------------------------------------
def analyze_question(question: str) -> ResponseLog:
    """
    Ask the chatbot a question and have the LLM self-evaluate if it could answer.
    """
    # Get the chatbot's response
    chat_messages = [
        {"role": "system", "content": profile.system_prompt},
        {"role": "user", "content": question}
    ]

    chat_response = cache.cached_api_call(
        model_name=gemini.model_name,
        query=chat_messages,
        api_function=gemini.generate
    )

    # Have the LLM evaluate if it could answer based on the resume
    eval_messages = [
        {
            "role": "system",
            "content": """You are evaluating whether an AI assistant could 
answer a question based solely on the provided resume/profile information.
Determine if the response was based on actual resume content or if the 
assistant had to make assumptions, deflect, or say it didn't know."""
        },
        {
            "role": "user",
            "content": f"""
## Resume Content Available:
{profile.resume_content}

## Summary Available:
{profile.summary}

## Question Asked:
{question}

## Response Given:
{chat_response["text"]}

## Your Task:
Evaluate if the response could be confidently answered from the resume/summary content.

could_answer should be:
- true: if the resume contains the information needed to answer
- false: if the assistant had to say "I don't know", make assumptions, or the resume lacks this info

If could_answer is false, briefly explain what information is missing.
"""
        }
    ]

    # Get structured evaluation
    log = gemini.generate_structured(
        query=eval_messages,
        response_format=ResponseLog
    )

    return log


def run_gap_analysis():
    """
    Run through all test questions and identify gaps.
    """
    print(f"\n{'=' * 80}")
    print(f"Gap Analysis for: {profile.name}")
    print(f"Analyzing {len(TEST_QUESTIONS)} questions...")
    print(f"{'=' * 80}\n")

    answerable = []
    unanswerable = []

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i}/{len(TEST_QUESTIONS)}] Analyzing: {question}")

        log = analyze_question(question)

        if log.could_answer:
            answerable.append(log)
            print(f"  ✅ Could answer\n")
        else:
            unanswerable.append(log)
            print(f"  ❌ Could NOT answer")
            print(f"     Reason: {log.reason}\n")

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 80}\n")
    print(f"✅ Could answer: {len(answerable)}/{len(TEST_QUESTIONS)}")
    print(f"❌ Could NOT answer: {len(unanswerable)}/{len(TEST_QUESTIONS)}")

    # Print detailed gaps
    if unanswerable:
        print(f"\n{'=' * 80}")
        print(f"QUESTIONS THAT COULD NOT BE ANSWERED")
        print(f"{'=' * 80}\n")

        for log in unanswerable:
            print(f"Question: {log.query}")
            print(f"Response: {log.response[:150]}...")
            print(f"Missing:  {log.reason}")
            print(f"{'-' * 80}\n")

    # Save to file
    save_gap_report(answerable, unanswerable)

    return answerable, unanswerable


def save_gap_report(answerable: List[ResponseLog], unanswerable: List[ResponseLog]):
    """Save gap analysis report to a text file."""
    from pathlib import Path
    from datetime import datetime

    output_dir = Path(__file__).parent / "reports"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"gap_analysis_{timestamp}.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Gap Analysis Report for {profile.name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 80}\n\n")

        f.write(f"SUMMARY\n")
        f.write(f"{'-' * 80}\n")
        f.write(f"Total questions: {len(answerable) + len(unanswerable)}\n")
        f.write(f"Could answer: {len(answerable)}\n")
        f.write(f"Could NOT answer: {len(unanswerable)}\n\n")

        if unanswerable:
            f.write(f"QUESTIONS THAT COULD NOT BE ANSWERED\n")
            f.write(f"{'=' * 80}\n\n")

            for i, log in enumerate(unanswerable, 1):
                f.write(f"{i}. {log.query}\n")
                f.write(f"   Response: {log.response}\n")
                f.write(f"   Missing info: {log.reason}\n\n")

        if answerable:
            f.write(f"\n\nQUESTIONS THAT COULD BE ANSWERED\n")
            f.write(f"{'=' * 80}\n\n")

            for i, log in enumerate(answerable, 1):
                f.write(f"{i}. {log.query}\n")
                f.write(f"   Response: {log.response[:200]}...\n\n")

    print(f"\n📄 Report saved to: {output_file}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    answerable, unanswerable = run_gap_analysis()

    print(f"\n{'=' * 80}")
    print(f"ACTIONABLE INSIGHTS")
    print(f"{'=' * 80}\n")

    if unanswerable:
        print("Consider adding the following to your resume or summary:")
        for log in unanswerable:
            if log.reason:
                print(f"  • {log.reason}")
    else:
        print("✨ Great! Your resume covers all common questions.")