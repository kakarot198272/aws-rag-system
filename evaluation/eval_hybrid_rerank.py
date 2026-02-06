import sys
import json
import os
from pathlib import Path

# --------------------------------------------------
# ✅ ADD PROJECT ROOT TO PYTHON PATH (CRITICAL FIX)
# --------------------------------------------------
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
from retrieval.hybrid import build_hybrid_retriever
from retrieval.rerank import rerank
from generation.generator import build_context
from generation.llm import generate_answer

# --------------------------------------------------
QUESTIONS_PATH = Path("evaluation/questions.json")
# --------------------------------------------------


def run_hybrid_rag(question: str, retriever):
    # ✅ Phase B: NO query rewrite
    docs = retriever.invoke(question)

    # ✅ Rerank and KEEP all reranked docs
    docs = rerank(question, docs, top_k=12)
    docs = docs[:3]   # 🔥 THIS IS THE FIX


    context = build_context(docs)
    answer = generate_answer(question, context)

    return {
        "question": question,
        "answer": answer,
        "contexts": [d.page_content for d in docs],
    }


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return

    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions")

    print("Indexing documents and building HYBRID retriever...")
    vectorstore, chunks = load_and_index()

    # ✅ Phase B retrieval width
    retriever = build_hybrid_retriever(
        chunks,
        vectorstore,
        k=8
    )

    results = []
    print("Running HYBRID + RERANK pipeline...")
    for q in questions:
        out = run_hybrid_rag(q["question"], retriever)
        out["ground_truth"] = q["ground_truth"]
        results.append(out)

    dataset = Dataset.from_list(results)

    client = OpenAI(api_key=api_key)
    judge_llm = llm_factory("gpt-4o-mini", client=client)

    metrics = [
        Faithfulness(llm=judge_llm),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
        AnswerCorrectness(llm=judge_llm, weights=[1.0, 0.0]),
    ]

    print("\nStarting HYBRID + RERANK RAGAS evaluation...")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    scores = evaluate(dataset, metrics=metrics)

    print("\n=== HYBRID + RERANK RAGAS RESULTS ===")
    print(scores)


if __name__ == "__main__":
    main()
