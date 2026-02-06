import os
import re
from openai import OpenAI

# Create client once
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Phase C: Strict grounded answering
SYSTEM_PROMPT = """You are a technical assistant for a Retrieval-Augmented Generation (RAG) system.

You MUST answer using ONLY the provided context.
Rules:
- Do not use prior knowledge.
- Do not infer missing details.
- If the answer is not explicitly stated in the context, respond EXACTLY:
  The provided context does not specify this.
- Keep the answer short (2-5 sentences).
- Do not include apologies, meta commentary, or speculation.
- Do not mention "context" in the answer.
"""

def _clean(text: str) -> str:
    """Light cleanup to reduce noisy tokens and keep answers concise."""
    text = text.strip()

    # Remove common model verbosity patterns if they slip in
    text = re.sub(r"(?i)\b(as an ai language model|i can't|i cannot|i'm unable)\b.*", "", text).strip()

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def generate_answer(question: str, context: str) -> str:
    """
    Generates a grounded answer using ONLY the given context.
    Designed to improve RAGAS faithfulness + correctness by reducing hallucinations.
    """

    # Hard guard: if context is empty, we must not hallucinate
    if not context or not context.strip():
        return "The provided context does not specify this."

    # Smaller max_tokens to control verbosity + cost
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=220,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Context:\n{context}\n\n"
                    "Answer (follow the rules exactly):"
                ),
            },
        ],
    )

    answer = response.choices[0].message.content or ""
    answer = _clean(answer)

    # Hard guard: enforce the exact fallback sentence if model gets fancy
    if not answer:
        return "The provided context does not specify this."

    # If model still tries to reference context, remove those lines
    if "provided context" in answer.lower() and answer.strip() != "The provided context does not specify this.":
        # If it deviates, force the safe fallback
        return "The provided context does not specify this."

    # Keep it short: truncate long answers (RAGAS prefers concise grounded output)
    # (simple sentence-based truncation)
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    if len(sentences) > 5:
        answer = " ".join(sentences[:5]).strip()

    return answer
