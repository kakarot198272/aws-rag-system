import json
import sys
import os
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from datasets import Dataset
from ragas import evaluate
from ragas.llms import llm_factory
from openai import OpenAI

# ✅ Updated Imports: Use these specific paths for version 0.4.x
from ragas.metrics import Faithfulness
from ragas.metrics import ContextPrecision
from ragas.metrics import ContextRecall
from ragas.metrics import AnswerCorrectness

from run_rag import load_and_index
from retrieval.hybrid import build_hybrid_retriever
from retrieval.rerank import rerank
from generation.generator import build_context
from generation.llm import generate_answer

QUESTIONS_PATH = Path("evaluation/questions.json")

def run_rag_for_question(question: str, retriever):
    docs = retriever.invoke(question)
    docs = rerank(question, docs, top_k=5)
    context = build_context(docs)
    answer = generate_answer(question, context)
    return {
        "question": question,
        "answer": answer,
        "contexts": [d.page_content for d in docs],
    }

def main():
    # 1. Load Data
    if not QUESTIONS_PATH.exists():
        print(f"Error: {QUESTIONS_PATH} not found.")
        return

    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} evaluation questions")

    # 2. Setup RAG
    print("Indexing documents and building retriever...")
    vectorstore, chunks = load_and_index()
    retriever = build_hybrid_retriever(chunks, vectorstore, k=8)

    results = []
    print("Running RAG pipeline...")
    for q in questions:
        out = run_rag_for_question(q["question"], retriever)
        out["ground_truth"] = q["ground_truth"]
        results.append(out)

    dataset = Dataset.from_list(results)

    # 3. Setup Judge
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    judge_llm = llm_factory("gpt-4o-mini", client=client)

    # 4. ✅ The Fix: Explicitly instantiate metrics
    # We pass the judge_llm directly into the constructor
    metrics = [
        Faithfulness(llm=judge_llm),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
        AnswerCorrectness(llm=judge_llm, weights=[1.0, 0.0])
    ]

    # 5. Run Evaluation
    print("Starting RAGAS evaluation...")
    # Prevent tokenizer deadlocks
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    try:
        scores = evaluate(dataset, metrics=metrics)
        print("\n=== RAGAS EVALUATION RESULTS ===")
        print(scores)
    except Exception as e:
        print(f"\nEvaluation failed with error: {e}")
        # Debugging step: check if metrics were initialized correctly
        for i, m in enumerate(metrics):
            print(f"Metric {i} type: {type(m)}")

if __name__ == "__main__":
    main()