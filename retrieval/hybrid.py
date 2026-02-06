from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS


class HybridRetriever(BaseRetriever):
    """
    Custom hybrid retriever that combines:
    - BM25 (lexical retrieval)
    - FAISS (dense semantic retrieval)

    We merge results explicitly instead of using deprecated EnsembleRetriever.
    """

    bm25: BM25Retriever
    vectorstore: FAISS
    k: int = 8

    def _get_relevant_documents(self, query: str) -> List[Document]:
        bm25_docs = self.bm25.invoke(query)
        dense_docs = self.vectorstore.similarity_search(query, k=self.k)


        # Deduplicate by page content
        seen = set()
        merged = []

        for d in bm25_docs + dense_docs:
            if d.page_content not in seen:
                seen.add(d.page_content)
                merged.append(d)

        return merged[: self.k]


def build_hybrid_retriever(
    chunks: List[Document],
    vectorstore: FAISS,
    k: int = 8,
) -> HybridRetriever:
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = k

    return HybridRetriever(
        bm25=bm25,
        vectorstore=vectorstore,
        k=k,
    )
