"""
Evaluator - evaluates AI responses against Tony's profile.
"""
import json
from src.agents.gemini_agent import GeminiAgent
from src.agents.xai_agent import XAIAgent
from src.utils.cache import LLMCache
from src.utils.config import Config
from src.agents.me.profile_loader import profile
from src.models.evaluation import Evaluation  # ← Import Pydantic model


# -----------------------------------------------------------------------------
# Initialize
# -----------------------------------------------------------------------------
cache = LLMCache(cache_dir=str(Config.CACHE_DIR))
gemini = GeminiAgent()
xai = XAIAgent()


# -----------------------------------------------------------------------------
# Evaluator
# -----------------------------------------------------------------------------
def evaluate_response(question: str, response: str) -> Evaluation:
    """
    Evaluate if an AI response accurately represents Tony's profile.
    Returns a structured Evaluation object.
    """
    eval_prompt = [
        {
            "role": "system",
            "content": """You are an evaluator checking if an AI response 
accurately represents a person's background.

You MUST respond with ONLY a valid JSON object in this exact format:
{
    "is_accepted": true or false,
    "feedback": "your detailed feedback here"
}

Rules:
- is_accepted: true if the response is accurate and professional, false otherwise
- feedback: brief explanation of your evaluation
- Do NOT include any text outside the JSON object
- Do NOT use markdown code blocks
- Return ONLY the raw JSON"""
        },
        {
            "role": "user",
            "content": f"""
## Actual Resume:
{profile.resume_content}

## Summary:
{profile.summary}

## Question Asked:
{question}

## AI Response to Evaluate:
{response}

Evaluate if the response is accurate, professional, and consistent 
with the actual resume. Return ONLY the JSON object.
"""
        }
    ]

    try:
        result = cache.cached_api_call(
            model_name=xai.model_name,
            query=eval_prompt,
            api_function=xai.generate,
            use_full_context=True
        )

        # Parse the JSON response into Evaluation model
        raw_text = result["text"].strip()

        # Clean up if model returns markdown code blocks despite instructions
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]         # Remove opening ```
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]                  # Remove 'json' label
        raw_text = raw_text.strip()

        # Parse JSON and validate with Pydantic
        json_data = json.loads(raw_text)
        evaluation = Evaluation(**json_data)             # ← Pydantic validates!

        return evaluation

    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Raw response was: {result['text']}")
        # Return a default failed evaluation
        return Evaluation(
            is_accepted=False,
            feedback=f"Evaluator failed to parse response: {str(e)}"
        )
    except Exception as e:
        print(f"Evaluation error: {e}")
        return Evaluation(
            is_accepted=False,
            feedback=f"Evaluation error: {str(e)}"
        )


# -----------------------------------------------------------------------------
# Evaluation Suite
# -----------------------------------------------------------------------------
def run_evaluation_suite():
    """Run a set of test questions and evaluate responses."""
    test_questions = [
        "What is your experience with Python?",
        "Have you worked in fintech?",
        "What is your leadership experience?",
        "Are you open to remote work?",
    ]

    print(f"\n{'='*60}")
    print(f"Evaluating AI responses for: {profile.name}")
    print(f"{'='*60}\n")

    passed = 0
    failed = 0

    for question in test_questions:
        # Get chatbot response using Gemini
        messages = [
            {"role": "system", "content": profile.system_prompt},
            {"role": "user",   "content": question}
        ]

        response = cache.cached_api_call(
            model_name=gemini.model_name,
            query=messages,
            api_function=gemini.generate
        )

        # Evaluate using xAI
        evaluation = evaluate_response(question, response["text"])

        # evaluation is now a proper Pydantic object!
        status = "✅ ACCEPTED" if evaluation.is_accepted else "❌ REJECTED"

        if evaluation.is_accepted:
            passed += 1
        else:
            failed += 1

        print(f"Question:    {question}")
        print(f"Response:    {response['text'][:200]}...")
        print(f"Status:      {status}")
        print(f"Feedback:    {evaluation.feedback}")
        print(f"{'-'*60}\n")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(test_questions)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_evaluation_suite()