# In your terminal, first run:

# pip3 install openai

import os
import json
import httpx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 1. Google
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
API_KEY = os.getenv("GOOGLE_API_KEY")
model_name = os.getenv("GEMINI_MODEL")
gemini = OpenAI(base_url=BASE_URL, api_key=API_KEY)
competitors = []
answers = []
question = "Why is India struggling with it's economy improvements despite having a very high talent pool?"
customMessages = [{"role": "user", "content": question}]

response = gemini.chat.completions.create(model=model_name, messages = customMessages)
answer = response.choices[0].message.content

# print(answer)
competitors.append(model_name)
answers.append(answer)

# 2. Grok
BASE_URL = "https://api.x.ai/v1"
API_KEY = os.getenv("XAI_API_KEY")
model_name = "grok-4-1-fast-reasoning"
grok = OpenAI(base_url=BASE_URL, api_key=API_KEY)
response = grok.chat.completions.create(model=model_name, messages = customMessages)
answer = response.choices[0].message.content

# print(answer)
competitors.append(model_name)
answers.append(answer)

print(competitors)
# print(answers)

for competitor, answer in zip(competitors, answers):
    print(f"Competitor: {competitor}\n\n{answer}")
together = ""
for index, answer in enumerate(answers):
    together += f"# Response from competitor {index+1}\n\n"
    together += answer + "\n\n"

judge = f"""You are judging a competition between {len(competitors)} competitors.
Each model has been given this question:

{question}

Your job is to evaluate each response for clarity and strength of argument, and rank them in order of best to worst.
Respond with JSON, and only JSON, with the following format:
{{"results": ["best competitor number", "second best competitor number", "third best competitor number", ...]}}

Here are the responses from each competitor:

{together}

Now respond with the JSON with the ranked order of the competitors, nothing else. Do not include markdown formatting or code blocks."""

judge_messages = [{"role": "user", "content": judge}]

response = grok.chat.completions.create(model=model_name, messages = judge_messages)

results = response.choices[0].message.content
print(results)


results_dict = json.loads(results)
ranks = results_dict["results"]
for index, result in enumerate(ranks):
    competitor = competitors[int(result)-1]
    print(f"Rank {index+1}: {competitor}")

#Print a lengthy line about 40 dashes.
print("-" * 40)
#Check Gemini the ranking and print.
model_name = os.getenv("GEMINI_MODEL")
response = gemini.chat.completions.create(model=model_name, messages = judge_messages)

results = response.choices[0].message.content
print(results)

results_dict = json.loads(results)
ranks = results_dict["results"]
for index, result in enumerate(ranks):
    competitor = competitors[int(result) - 1]
    print(f"Rank {index + 1}: {competitor}")



'''
Results ::
Both Grok and Gemini concluding, Grok is the # 1. 

{"results": ["2", "1"]}
Rank 1: grok-4-1-fast-reasoning
Rank 2: gemini-3-flash-preview
----------------------------------------
{"results": ["2", "1"]}
Rank 1: grok-4-1-fast-reasoning
Rank 2: gemini-3-flash-preview
'''