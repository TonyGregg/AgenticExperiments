# In your terminal, first run:

# pip3 install openai

import os
import httpx
from openai import OpenAI
from dotenv import load_dotenv
from src.utils.config import Config


load_dotenv()


# google_model = os.getenv("GEMINI_MODEL")
gemini = OpenAI(base_url=Config.GEMINI_BASE_URL, api_key=Config.GEMINI_API_KEY)

response = gemini.chat.completions.create(model= Config.GEMINI_MODEL, messages=[
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