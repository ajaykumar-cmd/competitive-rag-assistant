"""
version_3_chroma_DB.py - RAG over local files using ChromaDB.

Folder layout:
    RAG project/
        data/                      <- put .md / .txt / .pdf files here
            upsc_polity_questions_1_20.md
        version_3_chroma_DB.py
        .env

.env:
    PROVIDER=gemini
    GOOGLE_API_KEY=...

Install:
    pip install langchain langchain-core langchain-community langchain-text-splitters
    pip install langchain-chroma chromadb langchain-huggingface sentence-transformers
    pip install langchain-google-genai python-dotenv pypdf
    pip install "posthog<6"     # optional: stops the Chroma telemetry warnings
"""

import os

# Must be set before chromadb is imported (silences telemetry errors)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION = "upsc_polity"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PROVIDER = os.getenv("PROVIDER", "gemini").lower()
REBUILD = os.getenv("REBUILD_DB", "0") == "1"  # set REBUILD_DB=1 to re-index

MODELS = {
    "gemini": ("gemini-3.8-flash", ["gemini-3.7-flash", "gemini-3.5-flash-lite"]),
    "openai": ("gpt-4o-mini", ["gpt-4o"]),  # check OpenAI's docs for current names
    "anthropic": ("claude-sonnet-5-5", ["claude-haiku-5-5"]),
}


# ---------------------------------------------------------------------------
# LLM with retries + fallbacks (handles 503 / 429)
# ---------------------------------------------------------------------------
def _build_model(provider: str, name: str):
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=name, temperature=0.2, max_retries=5, timeout=60
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=name, temperature=0.2, max_retries=5, timeout=60)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=name, temperature=0.2, max_retries=5, timeout=60)
    raise ValueError(f"Unknown provider: {provider}")


def get_llm():
    primary_name, fallback_names = MODELS[PROVIDER]
    primary_name = os.getenv(f"{PROVIDER.upper()}_MODEL", primary_name)
    models = [_build_model(PROVIDER, n) for n in [primary_name] + fallback_names]
    primary, *backups = models
    return primary.with_fallbacks(backups) if backups else primary


# ---------------------------------------------------------------------------
# Load and split documents
# ---------------------------------------------------------------------------
def load_documents(folder: Path) -> List[Document]:
    docs: List[Document] = []
    if not folder.exists():
        raise FileNotFoundError(f"Data folder not found: {folder}")

    for path in sorted(folder.iterdir()):
        ext = path.suffix.lower()
        if ext in {".md", ".txt"}:
            loaded = TextLoader(str(path), encoding="utf-8").load()
            for d in loaded:
                d.metadata["page"] = 1
        elif ext == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
            for d in loaded:
                d.metadata["page"] = int(d.metadata.get("page", 0)) + 1
        else:
            continue

        for d in loaded:
            d.metadata["source"] = path.name
            d.metadata["doc_type"] = ext.lstrip(".")
            d.metadata["folder"] = folder.name
        docs.extend(loaded)

    if not docs:
        raise ValueError(f"No .md, .txt or .pdf files found in {folder}")
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    # Splits on markdown "## " headings first, so each question stays together
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=60,
        separators=["\n## ", "\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
    )
    return splitter.split_documents(docs)


# ---------------------------------------------------------------------------
# Vector store (Chroma, persisted on disk)
# ---------------------------------------------------------------------------
def get_vector_store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if REBUILD and CHROMA_DIR.exists():
        import shutil

        shutil.rmtree(CHROMA_DIR)

    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    # Index only if the collection is empty (avoids duplicates on re-runs)
    if store._collection.count() == 0:
        docs = load_documents(DATA_DIR)
        chunks = split_documents(docs)
        store.add_documents(chunks)
        print(f"Indexed {len(chunks)} chunks from {len(docs)} document(s).")
    else:
        print(f"Using existing index ({store._collection.count()} chunks).")

    return store


# ---------------------------------------------------------------------------
# RAG chain
# ---------------------------------------------------------------------------
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful UPSC / State PSC tutor. Answer ONLY using the "
            "context below. If the answer is not in the context, say "
            "\"I couldn't find that in the provided material.\"\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def format_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('source')}, page {d.metadata.get('page')}]\n{d.page_content}"
        for d in docs
    )


def build_chain(store: Chroma, source: str | None = None, k: int = 3):
    """Build a RAG chain. Pass `source` to restrict search to one file."""
    search_kwargs = {"k": k}
    if source:
        search_kwargs["filter"] = {"source": source}

    retriever = store.as_retriever(search_kwargs=search_kwargs)
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | get_llm()
        | StrOutputParser()
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Provider: {PROVIDER}")
    store = get_vector_store()

    q = "How many Fundamental Rights are there at present, and which article gives President's Rule in a state?"

    print("\n=== Retrieved chunks with metadata ===")
    for d in store.as_retriever(search_kwargs={"k": 3}).invoke(q):
        print(d.metadata, "->", d.page_content[:70].replace("\n", " "))

    try:
        print("\n=== Answer (all files) ===")
        print(build_chain(store).invoke(q))

        print("\n=== Answer (only upsc_polity_questions_1_20.md) ===")
        print(build_chain(store, source="upsc_polity_questions_1_20.md").invoke(q))

    except Exception as e:
        print(f"\nGeneration failed after retries and fallbacks: {e}")
        print("Retrieval worked, so the index is fine. Try again shortly or change PROVIDER.")