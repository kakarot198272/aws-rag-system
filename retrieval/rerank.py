from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# Cross-encoder for reranking
_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, docs: List[Document], top_k=None) -> List[Document]:
    """
    Rerank documents by relevance score.
    IMPORTANT:
    - Does NOT cut documents
    - Only sorts by score
    - Context truncation is handled later
    """
    if not docs:
        return []

    pairs = [(query, d.page_content) for d in docs]
    scores = _model.predict(pairs)

    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Return all docs, sorted
    return [d for d, _ in scored_docs]
