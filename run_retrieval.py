from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = Path("data/raw_docs")

def load_pdfs():
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {DATA_DIR.resolve()}")

    docs = []
    for pdf in pdfs:
        loader = PyPDFLoader(str(pdf))
        loaded = loader.load()
        for d in loaded:
            d.metadata["doc_name"] = pdf.name
        docs.extend(loaded)
    return docs


def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,      # safe starting point
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vs = FAISS.from_documents(chunks, embeddings)
    return vs

def main():
    docs = load_pdfs()
    vectorstore = build_vectorstore(docs)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    while True:
        q = input("\nAsk a question (or type 'exit'): ").strip()
        if q.lower() == "exit":
            break

        hits = retriever.get_relevant_documents(q)
        print("\nTop retrieved chunks:\n" + "-"*60)
        for i, d in enumerate(hits, 1):
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", "?")
            text = d.page_content.replace("\n", " ")
            print(f"\n[{i}] source={src} | page={page}\n{text[:450]}...")

if __name__ == "__main__":
    main()
