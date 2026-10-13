"""
Version 2 - Basic RAG with FAISS
Documents -> Loader -> Splitter -> Embeddings -> FAISS -> Retriever
-> Prompt + Context -> LLM -> Answer
Run: python v2_rag_faiss.py
"""
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from common import (load_documents, split_documents, get_embeddings,
                      get_llm, format_docs, RAG_PROMPT)

INDEX_DIR = "faiss_index"


def get_vectorstore():
    embeddings = get_embeddings()
    if Path(INDEX_DIR).exists():                       # reuse saved index
        return FAISS.load_local(INDEX_DIR, embeddings,
                                allow_dangerous_deserialization=True)
    docs = load_documents()
    chunks = split_documents(docs)
    print(f"Loaded {len(docs)} documents -> {len(chunks)} chunks")
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(INDEX_DIR)
    return store


def build_chain(store, k: int = 4):
    retriever = store.as_retriever(search_kwargs={"k": k})
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT | get_llm() | StrOutputParser()
    )


if __name__ == "__main__":
    store = get_vectorstore()

    # Learn similarity search directly (lower score = closer in FAISS L2)
    q = "Which Article protects life and personal liberty?"
    print("=== Similarity search with scores ===")
    for doc, score in store.similarity_search_with_score(q, k=3):
        print(f"{score:.3f} | {doc.metadata['source']} | {doc.page_content[:80]!r}")

    print("\n=== RAG answer ===")
    print(build_chain(store).invoke(q))