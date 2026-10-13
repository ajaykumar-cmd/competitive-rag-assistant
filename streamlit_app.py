"""
streamlit_app.py - Competitive RAG Assistant (Streamlit UI)

Run locally:
    streamlit run streamlit_app.py

Needs in the same folder:
    version_3_chroma_DB.py, version_4_Reranking.py, live_rag.py, data/
"""

import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import streamlit as st

# On Streamlit Cloud, copy keys from st.secrets into environment variables
# (must happen BEFORE importing the project files, which read os.environ).
try:
    for _key, _val in st.secrets.items():
        if isinstance(_val, (str, int, float)):
            os.environ.setdefault(_key, str(_val))
except Exception:
    pass  # no secrets file locally; .env is used instead

from live_rag import route_and_answer
from version_3_chroma_DB import get_vector_store

APP_NAME = "Competitive RAG Assistant"

st.set_page_config(page_title=APP_NAME, page_icon="🎓", layout="centered")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 2rem; max-width: 820px;}

    .hero {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        padding: 2.2rem 1.5rem;
        border-radius: 22px;
        text-align: center;
        color: #ffffff;
        box-shadow: 0 12px 32px rgba(99, 102, 241, 0.35);
        margin-bottom: 1.4rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    .hero p {
        margin: 0.5rem 0 0 0;
        font-size: 1.02rem;
        opacity: 0.92;
        color: #ffffff;
    }

    .feature-card {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 16px;
        padding: 1rem 0.8rem;
        text-align: center;
        height: 100%;
    }
    .feature-card .icon {font-size: 1.7rem;}
    .feature-card .title {font-weight: 700; margin-top: 0.2rem;}
    .feature-card .desc {font-size: 0.82rem; opacity: 0.75; margin-top: 0.2rem;}

    .section-label {
        font-weight: 700;
        margin: 1.4rem 0 0.5rem 0;
        font-size: 1.02rem;
    }

    .badge {
        display: inline-block;
        padding: 2px 11px;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 700;
        margin-right: 6px;
    }
    .badge-local {background: rgba(16, 185, 129, 0.18); color: #10b981;}
    .badge-web {background: rgba(59, 130, 246, 0.18); color: #3b82f6;}
    .badge-score {background: rgba(128, 128, 128, 0.2);}

    .stButton > button {
        border-radius: 14px;
        border: 1px solid rgba(139, 92, 246, 0.45);
        padding: 0.55rem 0.9rem;
        text-align: left;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: #8b5cf6;
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(139, 92, 246, 0.25);
    }
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 0.7rem 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached resources (loaded once, not on every message)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading study material and models...")
def load_store():
    return get_vector_store()


store = load_store()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stats" not in st.session_state:
    st.session_state.stats = {"local": 0, "web": 0}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    mode = st.radio(
        "Answer source",
        ["Auto (study material, then web)", "Study material only", "Web only"],
        help="Auto uses your documents when they match the question, otherwise searches the web.",
    )
    threshold = st.slider(
        "Match strictness",
        min_value=-5.0,
        max_value=10.0,
        value=0.0,
        step=0.5,
        help="Minimum rerank score needed to trust your study material (Auto mode). Higher = stricter.",
    )

    st.markdown("---")
    st.markdown("### 📊 This session")
    c1, c2 = st.columns(2)
    c1.metric("From notes", st.session_state.stats["local"])
    c2.metric("From web", st.session_state.stats["web"])

    st.markdown("---")
    st.caption(f"Model provider: **{os.getenv('PROVIDER', 'gemini')}**")
    _tracing_on = os.getenv("LANGSMITH_TRACING", "").lower() == "true" and bool(os.getenv("LANGSMITH_API_KEY"))
    st.caption("LangSmith tracing: " + ("**ON** 🟢" if _tracing_on else "off ⚪"))
    st.caption("Tip: start a question with `web:` or `local:` to force a source.")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.stats = {"local": 0, "web": 0}
        st.rerun()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
MODE_MAP = {
    "Auto (study material, then web)": "auto",
    "Study material only": "local",
    "Web only": "web",
}


def respond(query_text: str):
    """Return (answer, route, score). One LangSmith trace per call."""
    result = route_and_answer(store, query_text, mode=MODE_MAP[mode], threshold=threshold)
    return result["answer"], result["route"], result["score"]


def badge_html(route: str, score) -> str:
    if route == "local":
        html = '<span class="badge badge-local">📘 Study material</span>'
    elif route == "web-fallback":
        html = '<span class="badge badge-web">🌐 Web (not found in notes)</span>'
    else:
        html = '<span class="badge badge-web">🌐 Web</span>'
    if score is not None:
        html += f'<span class="badge badge-score">match {score:.1f}</span>'
    return html


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1>🎓 {APP_NAME}</h1>
        <p>Your smart study companion for competitive exams.<br>
        Answers from your own study material first, with web search as backup.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Welcome screen (only before the first question)
# ---------------------------------------------------------------------------
EXAMPLES = [
    "How many Fundamental Rights are there at present?",
    "Which article provides for President's Rule in a state?",
    "In which house is a Money Bill introduced?",
    "Which article lists the Fundamental Duties?",
]

queued = None

if not st.session_state.messages:
    f1, f2, f3 = st.columns(3)
    f1.markdown(
        '<div class="feature-card"><div class="icon">📘</div><div class="title">Your notes first</div>'
        '<div class="desc">Answers grounded in your study files</div></div>',
        unsafe_allow_html=True,
    )
    f2.markdown(
        '<div class="feature-card"><div class="icon">🎯</div><div class="title">Smart ranking</div>'
        '<div class="desc">Picks the most relevant passages</div></div>',
        unsafe_allow_html=True,
    )
    f3.markdown(
        '<div class="feature-card"><div class="icon">🌐</div><div class="title">Web backup</div>'
        '<div class="desc">Searches online when notes fall short</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">✨ Try a question</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    for i, q in enumerate(EXAMPLES):
        with (col_a if i % 2 == 0 else col_b):
            if st.button(q, key=f"ex_{i}", use_container_width=True):
                queued = q

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🧑‍🎓" if m["role"] == "user" else "🎓"):
        st.markdown(m["content"])
        if m.get("meta"):
            st.markdown(m["meta"], unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# New question
# ---------------------------------------------------------------------------
typed = st.chat_input("Ask anything from your study material...")
prompt = typed or queued

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🎓"):
        try:
            with st.spinner("Searching and thinking..."):
                answer, route, score = respond(prompt)
            meta = badge_html(route, score)
            st.markdown(answer)
            st.markdown(meta, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
            if route == "local":
                st.session_state.stats["local"] += 1
            elif route.startswith("web"):
                st.session_state.stats["web"] += 1
        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                st.error(
                    "The model's daily quota is used up. Set another model (e.g. GEMINI_MODEL) "
                    "or switch PROVIDER, then try again."
                )
            else:
                st.error(f"Error: {msg}")