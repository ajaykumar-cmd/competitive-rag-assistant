# 🎓 Competitive RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot that helps students prepare for competitive exams. It answers questions from your own study material first, and falls back to web search when the notes don't cover the question. Every request is traced in LangSmith.

**Live demo:** _add your Streamlit URL here, e.g. `https://competitive-rag-assistant.streamlit.app`_

---

## ✨ Features

- **Answers from your notes:** retrieves relevant passages from local study files (`.md`, `.txt`, `.pdf`) stored in ChromaDB.
- **Reranking:** a cross-encoder re-scores the retrieved chunks so the best context reaches the LLM.
- **Web fallback:** if the best rerank score is below the threshold, the app searches the web (DuckDuckGo) and answers from the results.
- **Grounded answers:** the model is told to answer only from the supplied context, and to say so when the answer isn't there.
- **Source control:** choose Auto, Study material only, or Web only in the sidebar, or start a question with `local:` / `web:`.
- **Streamlit web app:** chat interface with source badges, match scores and session stats.
- **LangSmith monitoring:** one trace per question showing retriever, reranker, prompt, LLM, latency, token usage and errors.
- **Resilient LLM calls:** automatic retries and fallback models when a model is overloaded or unavailable.

---

## 🧭 How it works

```
User question
      ↓
Retriever (ChromaDB similarity search, top 10)
      ↓
Reranker (cross-encoder, keep best 3)
      ↓
Best score ≥ threshold?
   ├── Yes → Prompt (with local context) → LLM → Answer + sources
   └── No  → Web search → Prompt (with web results) → LLM → Answer + links
```

---

## 🛠️ Project evolution (V1 → Final)

| Version | File | What it adds |
|---|---|---|
| **V1** | `version_1.py` | LangChain basics: Prompt → LLM → Parser → Runnable |
| **V2** | `version_2.py` | FAISS RAG: Documents → Embeddings → FAISS → Retrieval → LLM |
| **V3** | `version_3_chroma_DB.py` | ChromaDB RAG with metadata (source, page, doc type) and persistent storage |
| **V4** | `version_4_Reranking.py` | Cross-encoder reranking: Retrieval → Reranker → Best context → LLM |
| **Final** | `live_rag.py` + `streamlit_app.py` | Local + web routing, Streamlit app, LangSmith monitoring |

---

## 🧰 Tech stack

- **Framework:** LangChain
- **LLM:** Google Gemini (OpenAI and Anthropic also supported through the `PROVIDER` setting)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector database:** ChromaDB
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Web search:** DuckDuckGo (`ddgs`)
- **UI:** Streamlit
- **Monitoring:** LangSmith

---

## 📁 Project structure

```
.
├── data/                       # Your study files (.md, .txt, .pdf)
├── version_1.py                # V1: LangChain basics
├── version_2.py                # V2: FAISS RAG
├── version_3_chroma_DB.py      # V3: ChromaDB RAG (also the shared setup module)
├── version_4_Reranking.py      # V4: reranking
├── live_rag.py                 # Final pipeline: local + web routing, command-line chat
├── streamlit_app.py            # Streamlit web app
├── requirements.txt
├── .gitignore
└── README.md
```

`chroma_db/` (the vector index) is created automatically on first run.

---

## 🚀 Getting started

### 1. Clone and install

```bash
git clone https://github.com/ajaykumar-cmd/competitive-rag-assistant.git
cd competitive-rag-assistant

python -m venv env
env\Scripts\activate          # Windows
# source env/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### 2. Add your keys

Create a `.env` file in the project folder (never commit this file):

```
PROVIDER=gemini
GOOGLE_API_KEY=your-gemini-key
GEMINI_MODEL=gemini-3.7-flash

# Optional: LangSmith monitoring
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=competitive-rag-assistant
```

Get a Gemini key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and a LangSmith key at [smith.langchain.com](https://smith.langchain.com) (Settings → API Keys).

### 3. Add study material

Put your `.md`, `.txt` or `.pdf` files in the `data/` folder. If you change the files later, re-index once:

```powershell
$env:REBUILD_DB=1; streamlit run streamlit_app.py     # Windows PowerShell
```

```bash
REBUILD_DB=1 streamlit run streamlit_app.py           # macOS / Linux
```

### 4. Run

```bash
streamlit run streamlit_app.py     # web app at http://localhost:8501
python live_rag.py                 # command-line chat
```

---

## 💬 Example questions

**From the study material**
- How many Fundamental Rights are there at present?
- Which article provides for President's Rule in a state?
- In which house is a Money Bill introduced?

**Forcing a source**
- `web: latest news about ISRO`
- `local: Which article lists the Fundamental Duties?`

Questions that aren't in your files should return _"I couldn't find that in the provided material"_ in study-material mode, or switch to web search in Auto mode.

---

## 📊 LangSmith monitoring

With the LangSmith variables set, each question produces one trace named **Competitive RAG Query** containing:

1. **Local Search** → Retriever (Chroma) and Reranker (cross-encoder) with scores
2. **Prompt** and **LLM** call with token usage and latency
3. **Web Search** step when the question falls back to the web
4. The final answer, the route taken and any errors

Use the project's Monitoring tab for latency, token usage and error rates over time.

---

## ☁️ Deploying on Streamlit Community Cloud

1. Push this repo to GitHub (keep `.env` and `env/` out of it, as `.gitignore` does).
2. On [share.streamlit.io](https://share.streamlit.io), create an app from the repo: branch `main`, main file `streamlit_app.py`.
3. Under **Advanced settings → Secrets**, add your keys in TOML format:

```toml
GOOGLE_API_KEY = "your-gemini-key"
PROVIDER = "gemini"
GEMINI_MODEL = "gemini-3.7-flash"
LANGSMITH_TRACING = "true"
LANGSMITH_API_KEY = "your-langsmith-key"
LANGSMITH_PROJECT = "competitive-rag-assistant"
```

The index is rebuilt each time the app restarts, because the cloud filesystem is temporary.

---

## 📈 Evaluation

_Fill this in after running your 20+ evaluation questions on each version._

| Version | Retrieval | Answer accuracy | Avg. latency | Notes |
|---|---|---|---|---|
| V1 (no retrieval) | – | | | |
| V2 (FAISS) | | | | |
| V3 (ChromaDB) | | | | |
| V4 (reranking) | | | | |

---

## ⚠️ Limitations

- Answers are only as good as the files in `data/`. Topics not covered there fall back to web search, which can be less reliable than your own notes.
- The free Gemini tier has a small daily request limit per model. Set another model in `GEMINI_MODEL`, or switch `PROVIDER`, if you hit a quota error.
- The first start downloads the embedding and reranker models, so it is slow.

---

## 👤 Author

**M Ajay Kumar** — [@ajaykumar-cmd](https://github.com/ajaykumar-cmd)