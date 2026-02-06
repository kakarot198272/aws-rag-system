from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from generation.llm import generate_answer
from generation.generator import build_context

DATA_DIR = Path("data/raw_docs")


def dedupe_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    unique_docs = []

    for d in docs:
        key = d.page_content[:200]
        if key not in seen:
            seen.add(key)
            unique_docs.append(d)

    return unique_docs


def load_and_index():
    docs = []

    for pdf in DATA_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf))
        loaded = loader.load()
        for d in loaded:
            d.metadata["doc_name"] = pdf.name
        docs.extend(loaded)

    # ✅ Improved chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=650,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, chunks


def main():
    vectorstore, _ = load_and_index()

    # ✅ Increased retrieval depth
    retriever = vectorstore.as_retriever(search_kwargs={"k": 12})

    while True:
        q = input("\nAsk a question (or type 'exit'): ").strip()
        if q.lower() == "exit":
            break

        docs = retriever.invoke(q)

        # ✅ Deduplicate
        docs = dedupe_docs(docs)

        print("\n=== RETRIEVED CONTEXT ===")
        context = build_context(docs)
        print(context)

        print("\n=== MODEL ANSWER (BASELINE RAG) ===")
        answer = generate_answer(q, context)
        print(answer)


if __name__ == "__main__":
    main()
