# AWS RAG System with Quantitative Evaluation (RAGAS)

## 🎯 Overview
This project implements and rigorously evaluates a **Retrieval-Augmented Generation (RAG)** system over technical documentation using multiple retrieval strategies and LLM-based evaluation metrics (**RAGAS**). 

Unlike typical RAG demos, this system emphasizes **measurement, failure analysis, and engineering trade-offs**, aligning closely with real-world production-grade ML system design.

---

## Problem Statement
Large Language Models (LLMs) often hallucinate when responses are not strongly grounded in retrieved context. The objective of this project was to:
* **Build** an end-to-end RAG pipeline.
* **Compare** baseline dense retrieval vs. hybrid + reranked retrieval.
* **Quantitatively evaluate** answer quality using RAGAS.
* **Identify** why more complex retrieval pipelines can sometimes degrade performance.

---

## System Architecture



The system follows a modular flow to allow for A/B testing of different retrieval and reranking strategies:

1. **User Query**: Input question.
2. **Retriever**: Choice between Baseline (Dense) or Hybrid (Dense + BM25).
3. **Reranker**: Optional Cross-Encoder to re-score top-k documents.
4. **Context Builder**: Formats retrieved snippets for the LLM.
5. **Generation**: LLM generates the final answer based strictly on context.
6. **RAGAS Evaluation**: Automated grading of the output.

---

## Retrieval Strategies Implemented

### Baseline RAG
* **Dense vector retrieval** using semantic embeddings.
* **Deterministic prompting** to minimize variance and hallucinations.
* **Compact context**: High-signal context filtering for grounding.

### Hybrid + Rerank RAG
* **Hybrid retrieval**: Combines semantic (Vector) and lexical (Keyword/BM25) signals.
* **Cross-Encoder Reranking**: Re-evaluates document relevance for better precision.
* **Increased Width**: Retrieves 20 chunks before pruning to the top 8.

---

## Evaluation Methodology

### Dataset
* **40 curated question–answer pairs.**
* Ground truths derived directly from source AWS documentation to ensure factual accuracy.

### Metrics (RAGAS)


* **Faithfulness**: Measures if the answer is derived solely from the retrieved context.
* **Context Precision**: Measures the signal-to-noise ratio of the retrieved documents.
* **Context Recall**: Measures if all the information required to answer the question was actually retrieved.
* **Answer Correctness**: Measures semantic and factual alignment with the ground truth.

> **Evaluation Judge**: LLM-based evaluation performed using `gpt-4o-mini`.

---

## Results

| Metric | Baseline RAG | Hybrid + Rerank RAG |
| :--- | :--- | :--- |
| **Faithfulness** | **0.98** 🟢 | 0.76 🟡 |
| **Context Precision** | **0.62** | 0.55 |
| **Context Recall** | 0.70 | 0.70 |
| **Answer Correctness** | **0.49** 🟢 | 0.27 🔴 |

---

## Key Findings

### What Worked
* **Simpler retrieval pipelines** produced significantly higher answer correctness.
* **Deterministic prompts** effectively forced the model to admit when information was missing.
* **Smaller contexts** prevented "lost in the middle" phenomena during generation.

### Why Hybrid + Rerank Underperformed
* **Context Dilution**: Larger retrieval sets introduced irrelevant data that distracted the LLM.
* **Reranker misalignment**: The reranker optimized for document relevance, but not necessarily for "answerability."
* **Conflicting Contexts**: Increased noise led to a drop in Faithfulness as the LLM tried to reconcile irrelevant snippets.

> **Core Insight**: Improving retrieval **recall** does not guarantee better **answer quality**. In production RAG, noise is often more damaging than a slight lack of recall.

---

## How to Run

### Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set API Key (Required for RAGAS judge)
export OPENAI_API_KEY='your_key_here'
export TOKENIZERS_PARALLELISM=false
