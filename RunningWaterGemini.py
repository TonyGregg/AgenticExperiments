# In your terminal, first run:

# pip3 install openai

import os
import httpx
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
google_api_key = os.getenv("GOOGLE_API_KEY")
google_model = os.getenv("GEMINI_MODEL")
gemini = OpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
response = gemini.chat.completions.create(model=google_model, messages=[
        {
            "role": "system",
            "content": "You are Grok, a highly intelligent, helpful AI assistant."
        },
        {
            "role": "user",
            "content": "Why is India struggling with it's economy improvements despite having a very high talent pool?"
        },
    ])

print(response.choices[0].message.content)