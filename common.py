"""
Shared helpers used by V2, V3 and V4 (loader, splitter, embeddings, LLM, prompt).
Folder layout:
    exam_data/            <- your files (.md and .pdf), subfolders allowed
"""
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # reads GOOGLE_API_KEY from the .env file

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

DATA_DIR = "data"
GEMINI_MODEL = "gemini-3.8-flash"  # check the current model name in Google AI Studio
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_llm():
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def load_documents(data_dir: str = DATA_DIR):
    """Load every .md and .pdf file in the folder and add metadata."""
    md_docs = DirectoryLoader(
        data_dir, glob="**/*.md", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    ).load()
    pdf_docs = DirectoryLoader(
        data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader,
    ).load()
    docs = md_docs + pdf_docs
    if not docs:
        raise FileNotFoundError(f"No .md or .pdf files found in '{data_dir}/'")

    for d in docs:
        p = Path(d.metadata["source"])
        d.metadata["source"] = p.name                      # Source
        d.metadata["doc_type"] = p.suffix.lstrip(".")      # Document Type
        d.metadata["folder"] = p.parent.name               # e.g. jee_main, questions
        # Page: PyPDFLoader gives 0-based page; markdown files count as page 1
        d.metadata["page"] = int(d.metadata.get("page", 0)) + 1
    return docs


def split_documents(docs, chunk_size: int = 800, chunk_overlap: int = 100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def format_docs(docs):
    return "\n\n".join(
        f"[{d.metadata.get('source')} | page {d.metadata.get('page')}]\n{d.page_content}"
        for d in docs
    )


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a tutor for Indian competitive exams. Answer ONLY from the "
     "context below. If the answer is not in the context, say you do not have "
     "it in the files. Mention the source file name."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])