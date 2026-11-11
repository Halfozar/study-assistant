"""Streamlit UI:  streamlit run app.py"""
import tempfile
from pathlib import Path

import streamlit as st

from study_assistant import Config, StudyAssistant

st.set_page_config(page_title="AI Study Assistant", page_icon="📚", layout="wide")
st.title("📚 AI-Powered Personalized Study Assistant")

# ------------------------------------------------------------------ sidebar: parameters
with st.sidebar:
    st.header("Settings")
    backend = st.selectbox("Generation backend", ["extractive", "openai", "hf"],
                           help="extractive = offline, no API key. openai needs OPENAI_API_KEY.")
    embedder = st.selectbox("Embedder", ["tfidf", "minilm"])
    use_rag = st.toggle("Use RAG (retrieval)", value=True)
    chunk_size = st.slider("Chunk size (words)", 40, 400, 120, 10)
    overlap = st.slider("Chunk overlap (words)", 0, 100, 30, 5)
    top_k = st.slider("Top-k chunks", 1, 8, 4)
    min_score = st.slider("Decline threshold (min similarity)", 0.0, 0.8, 0.05 if embedder == "tfidf" else 0.30, 0.01)
    level = st.selectbox("Explanation level", ["beginner", "intermediate", "advanced"])
    n_sum = st.slider("Summary sentences", 2, 12, 5)
    n_quiz = st.slider("Quiz questions", 1, 10, 5)
    temperature = st.slider("Temperature (LLM backends)", 0.0, 1.0, 0.2, 0.1)

cfg = Config(chunk_size=chunk_size, chunk_overlap=min(overlap, chunk_size - 1), top_k=top_k,
             min_score=min_score, embedder=embedder, backend=backend, use_rag=use_rag,
             level=level, summary_sentences=n_sum, num_questions=n_quiz, temperature=temperature)

# ------------------------------------------------------------------ load material
up = st.file_uploader("Upload study material (PDF, DOCX, TXT, MD)", type=["pdf", "docx", "txt", "md"])
pasted = st.text_area("...or paste text", height=120)
use_sample = st.checkbox("Use the bundled sample notes (data/sample_notes.txt)")

text_key = (getattr(up, "name", None), pasted, use_sample, tuple(sorted(cfg.to_dict().items(), key=str)))
if st.session_state.get("key") != text_key:
    st.session_state.pop("bot", None)
    st.session_state["key"] = text_key
    st.session_state["chat"] = []

if "bot" not in st.session_state and (up or pasted.strip() or use_sample):
    try:
        bot = StudyAssistant(cfg)
        if up:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix) as f:
                f.write(up.read())
            n = bot.load_file(f.name)
        elif pasted.strip():
            n = bot.load_text(pasted)
        else:
            n = bot.load_file(Path(__file__).parent / "data" / "sample_notes.txt")
        st.session_state["bot"] = bot
        st.success(f"Indexed {n} chunks.")
    except Exception as e:  # show config / API-key problems nicely
        st.error(f"Could not start the assistant: {e}")

bot = st.session_state.get("bot")
if not bot:
    st.info("Upload or paste material (or tick the sample notes) to begin.")
    st.stop()


def show(res):
    st.write(res["answer"])
    if res.get("sources"):
        with st.expander("Retrieved context"):
            for i, s in enumerate(res["sources"]):
                score = f" (score {res['scores'][i]})" if res.get("scores") and i < len(res["scores"]) else ""
                st.markdown(f"**Chunk {i + 1}{score}**\n\n{s}")


tab_chat, tab_sum, tab_exp, tab_quiz = st.tabs(["💬 Ask", "📝 Summary", "💡 Explain", "❓ Quiz"])

with tab_chat:
    for role, msg in st.session_state["chat"]:
        st.chat_message(role).write(msg)
    q = st.chat_input("Ask a question about your material")
    if q:
        st.chat_message("user").write(q)
        res = bot.ask(q)
        with st.chat_message("assistant"):
            show(res)
        st.session_state["chat"] += [("user", q), ("assistant", res["answer"])]

with tab_sum:
    topic = st.text_input("Topic (leave empty to summarize everything)", key="sum_topic")
    if st.button("Summarize"):
        show(bot.summarize(topic or None))

with tab_exp:
    t = st.text_input("Concept to explain", key="exp_topic")
    if st.button("Explain") and t:
        show(bot.explain(t))

with tab_quiz:
    qt = st.text_input("Quiz topic (optional)", key="quiz_topic")
    if st.button("Generate quiz"):
        res = bot.quiz(qt or None, n_quiz)
        st.text(res["answer"])
