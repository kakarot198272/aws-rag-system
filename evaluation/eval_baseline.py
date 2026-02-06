import json
import sys
import os
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from datasets import Dataset
from ragas import evaluate
from ragas.llms import llm_factory
from openai import OpenAI

from ragas.metrics import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
)

from run_rag import load_and_index
from generation.generator import build_context
from generation.llm import generate_answer

QUESTIONS_PATH = Path("evaluation/questions.json")


def run_baseline_rag(question: str, retriever):
    docs = retriever.invoke(question)  # FAISS only
    context = build_context(docs)
    answer = generate_answer(question, context)

    return {
        "question": question,
        "answer": answer,
        "contexts": [d.page_content for d in docs],
    }


def main():
    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions")

    print("Indexing documents and building BASELINE retriever...")
    vectorstore, _ = load_and_index()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    results = []
    for q in questions:
        out = run_baseline_rag(q["question"], retriever)
        out["ground_truth"] = q["ground_truth"]
        results.append(out)

    dataset = Dataset.from_list(results)

    # Judge LLM
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    judge_llm = llm_factory("gpt-4o-mini", client=client)

    metrics = [
        Faithfulness(llm=judge_llm),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
        AnswerCorrectness(llm=judge_llm, weights=[1.0, 0.0]),
    ]

    print("\nStarting BASELINE RAGAS evaluation...")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    scores = evaluate(dataset, metrics=metrics)

    print("\n=== BASELINE RAGAS RESULTS ===")
    print(scores)


if __name__ == "__main__":
    main()
