"""
live_rag.py - Interactive RAG: local documents first, web search as fallback.

How it works for each question:
    1. Search ChromaDB (local files) and rerank with the cross-encoder.
    2. If the best rerank score >= threshold  -> answer from local files.
       Otherwise                              -> search the web and answer from results.
    3. Prefix a question with 'local:' or 'web:' to force a source.

Needs version_3_chroma_DB.py and version_4_Reranking.py in the same folder.

Install (extra package for web search):
    pip install -U ddgs

.env (optional):
    RERANK_THRESHOLD=0.0
    PROVIDER=gemini
"""

import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"  # before chromadb gets imported

from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from version_3_chroma_DB import CHROMA_DIR, PROMPT, format_docs, get_llm, get_vector_store
from version_4_Reranking import FETCH_K, TOP_K, fetch_candidates, rerank

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
THRESHOLD = float(os.getenv("RERANK_THRESHOLD", "0.0"))  # min score to trust local files
WEB_RESULTS = 5


def _drop_store(inputs: dict) -> dict:
    """Keep the (non-serializable) vector store out of LangSmith traces."""
    return {k: v for k, v in inputs.items() if k != "store"}

WEB_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the question using ONLY the web "
            "search results below. Be concise. If the results don't answer the "
            "question, say so. Mention the source titles you relied on.\n\n"
            "Web results:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


# ---------------------------------------------------------------------------
# Local source
# ---------------------------------------------------------------------------
@traceable(name="Local Search", run_type="chain", process_inputs=_drop_store)
def local_search(store, query: str) -> Tuple[List[Document], float]:
    """Return (top reranked chunks, best rerank score)."""
    candidates = fetch_candidates(store, query, k=FETCH_K)
    ranked = rerank(query, candidates, top_k=TOP_K)
    if not ranked:
        return [], float("-inf")
    docs = [d for d, _ in ranked]
    best_score = ranked[0][1]
    return docs, best_score


def answer_local(query: str, docs: List[Document]) -> str:
    chain = PROMPT | get_llm() | StrOutputParser()
    answer = chain.invoke({"context": format_docs(docs), "question": query})
    sources = sorted({f"{d.metadata.get('source')} (page {d.metadata.get('page')})" for d in docs})
    return f"{answer}\n\n[Source: local files -> {', '.join(sources)}]"


# ---------------------------------------------------------------------------
# Web source
# ---------------------------------------------------------------------------
@traceable(name="Web Search (DuckDuckGo)", run_type="retriever")
def web_search(query: str, n: int = WEB_RESULTS) -> List[dict]:
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # older package name
        except ImportError:
            raise RuntimeError("Web search needs a package. Run:  pip install -U ddgs")
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=n))


def answer_web(query: str) -> str:
    results = web_search(query)
    if not results:
        return "No web results found."
    context = "\n\n".join(
        f"[{i}] {r.get('title')}\n{r.get('body')}\nURL: {r.get('href')}"
        for i, r in enumerate(results, 1)
    )
    chain = WEB_PROMPT | get_llm() | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})
    links = "\n".join(f"  - {r.get('title')}: {r.get('href')}" for r in results)
    return f"{answer}\n\n[Source: web]\n{links}"


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def parse_prefix(text: str) -> Tuple[Optional[str], str]:
    lower = text.lower()
    for prefix in ("web:", "local:"):
        if lower.startswith(prefix):
            return prefix[:-1], text[len(prefix):].strip()
    return None, text


@traceable(name="Competitive RAG Query", run_type="chain", process_inputs=_drop_store)
def route_and_answer(store, text: str, mode: str = "auto", threshold: float = THRESHOLD) -> dict:
    """
    Full pipeline (one LangSmith trace per question):
        Query -> Local Search (Retriever -> Reranker) -> Prompt -> LLM -> Answer
        or, when local files don't match -> Web Search -> Prompt -> LLM -> Answer

    mode: "auto" | "local" | "web"   (a 'web:' / 'local:' prefix in the text overrides it)
    Returns: {"answer": str, "route": "local" | "web" | "web-fallback" | "none", "score": float | None}
    """
    forced, query = parse_prefix(text)
    if not query:
        return {"answer": "Please type a question after the prefix.", "route": "none", "score": None}

    effective = forced or mode

    if effective == "web":
        return {"answer": answer_web(query), "route": "web", "score": None}

    docs, best = local_search(store, query)
    if effective == "local" or best >= threshold:
        return {"answer": answer_local(query, docs), "route": "local", "score": best}

    return {"answer": answer_web(query), "route": "web-fallback", "score": best}


def answer_question(store, text: str) -> str:
    """Command-line wrapper around route_and_answer()."""
    result = route_and_answer(store, text)
    if result["score"] is not None:
        print(f"  [local] best rerank score: {result['score']:.2f} (needs >= {THRESHOLD})")
    if result["route"] == "web-fallback":
        print("  [web] local files don't cover this, searched the web instead.")
    return result["answer"]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    store = get_vector_store()
    print("Local files: ChromaDB found" if CHROMA_DIR.exists() else "Local files: none")
    print("Ask a question. Prefix with 'web:' or 'local:' to force a source. Type 'exit' to quit.")

    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            break

        try:
            print("\n" + answer_question(store, text))
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()