from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You rewrite user questions to improve document retrieval.

Rules:
- Rewrite the question into a clear, explicit search query
- Expand abbreviations
- Add key technical terms
- DO NOT answer the question
- DO NOT add extra explanation
- Output ONE concise rewritten query
"""

def rewrite_query(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )

    return response.choices[0].message.content.strip()
