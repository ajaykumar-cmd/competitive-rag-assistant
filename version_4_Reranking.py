"""
version_4_Reranking.py - RAG with a cross-encoder reranking step.

Pipeline:
    question -> Chroma similarity search (fetch many, e.g. 10)
             -> cross-encoder reranker scores each (question, chunk) pair
             -> keep the best few (e.g. 3)
             -> LLM answers from those chunks

Needs version_3_chroma_DB.py in the same folder.

Install (sentence-transformers is already there from version 3):
    pip install -U sentence-transformers
"""

import os
import sys

os.environ["ANONYMIZED_TELEMETRY"] = "False"  # before chromadb gets imported

from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langsmith import traceable
from sentence_transformers import CrossEncoder

from version_3_chroma_DB import PROMPT, format_docs, get_llm, get_vector_store

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
FETCH_K = 10  # candidates pulled from the vector DB
TOP_K = 3  # chunks kept after reranking

_reranker: Optional[CrossEncoder] = None


def get_reranker() -> CrossEncoder:
    """Load the cross-encoder once and reuse it."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


# ---------------------------------------------------------------------------
# Retrieve + rerank
# ---------------------------------------------------------------------------
def _drop_store(inputs: dict) -> dict:
    """Keep the (non-serializable) vector store out of LangSmith traces."""
    return {k: v for k, v in inputs.items() if k != "store"}


def _rerank_inputs(inputs: dict) -> dict:
    return {"query": inputs.get("query"), "num_candidates": len(inputs.get("docs") or [])}


def _rerank_outputs(out) -> dict:
    items = out.get("output", out) if isinstance(out, dict) else out
    return {
        "ranked": [
            {
                "score": round(float(score), 3),
                "source": d.metadata.get("source"),
                "text": d.page_content[:200],
            }
            for d, score in items
        ]
    }


@traceable(run_type="retriever", name="Retriever (Chroma)", process_inputs=_drop_store)
def fetch_candidates(
    store, query: str, k: int = FETCH_K, source: Optional[str] = None
) -> List[Document]:
    kwargs = {"k": k}
    if source:
        kwargs["filter"] = {"source": source}
    return store.similarity_search(query, **kwargs)


@traceable(
    run_type="chain",
    name="Reranker (cross-encoder)",
    process_inputs=_rerank_inputs,
    process_outputs=_rerank_outputs,
)
def rerank(query: str, docs: List[Document], top_k: int = TOP_K) -> List[Tuple[Document, float]]:
    """Score every (query, chunk) pair and return the best top_k with scores."""
    if not docs:
        return []
    pairs = [(query, d.page_content) for d in docs]
    scores = get_reranker().predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [(d, float(s)) for d, s in ranked[:top_k]]


def build_rerank_chain(store, source: Optional[str] = None, fetch_k: int = FETCH_K, top_k: int = TOP_K):
    def retrieve_and_rerank(query: str) -> List[Document]:
        candidates = fetch_candidates(store, query, k=fetch_k, source=source)
        return [d for d, _ in rerank(query, candidates, top_k=top_k)]

    return (
        {
            "context": RunnableLambda(retrieve_and_rerank) | format_docs,
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | get_llm()
        | StrOutputParser()
    )


# ---------------------------------------------------------------------------
# Helpers for display
# ---------------------------------------------------------------------------
def preview(doc: Document, n: int = 70) -> str:
    return doc.page_content[:n].replace("\n", " ")


def show_comparison(store, query: str, source: Optional[str] = None) -> None:
    candidates = fetch_candidates(store, query, source=source)

    print(f"\n--- Before reranking (vector search order, top {TOP_K} of {len(candidates)}) ---")
    for i, d in enumerate(candidates[:TOP_K], 1):
        print(f"{i}. {d.metadata.get('source')} -> {preview(d)}")

    print(f"\n--- After reranking (cross-encoder, top {TOP_K}) ---")
    for i, (d, score) in enumerate(rerank(query, candidates), 1):
        print(f"{i}. score={score:7.3f} | {d.metadata.get('source')} -> {preview(d)}")


def ask_loop(store) -> None:
    """Type your own questions. Press Enter on an empty line (or type 'exit') to quit."""
    chain = build_rerank_chain(store)
    print("Ask a question about your documents (empty line or 'exit' to quit).")
    while True:
        q = input("\nYou: ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break
        try:
            print("\nAnswer:", chain.invoke(q))
        except Exception as e:
            print(f"Generation failed after retries and fallbacks: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    store = get_vector_store()

    # Interactive mode:  python version_4_Reranking.py --ask
    if "--ask" in sys.argv:
        ask_loop(store)
        sys.exit(0)

    questions = [
        "How many Fundamental Rights are there at present?",
        "Which article provides for President's Rule in a state?",
        "In which house is a Money Bill introduced?",
    ]

    for q in questions:
        print("\n" + "=" * 70)
        print("Q:", q)
        show_comparison(store, q)

        try:
            print("\n=== Answer (with reranking) ===")
            print(build_rerank_chain(store).invoke(q))
        except Exception as e:
            print(f"Generation failed after retries and fallbacks: {e}")
            print("Retrieval and reranking worked. Try again shortly or change PROVIDER.")