# In your terminal, first run:

# pip3 install openai

import os
import httpx
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY")

client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
    timeout=httpx.Timeout(600.0), # Override default timeout with longer timeout for reasoning models
)

completion = client.responses.create(
    model="grok-4-1-fast-reasoning",
    input=[
        {
            "role": "system",
            "content": "You are Grok, a highly intelligent, helpful AI assistant."
        },
        {
            "role": "user",
            "content": "Why is India struggling with it's economy improvements despite having a very high talent pool?"
        },
    ],
)

print(completion.output[0].content)