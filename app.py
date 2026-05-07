import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import streamlit as st
from dotenv import load_dotenv
from retriever import retrieve_chunks
from generator import ask_question

load_dotenv()

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="SEC Filing Intelligence Assistant",
    page_icon="📊",
    layout="wide"
)

# ── Header ─────────────────────────────────────────────────────
st.title("📊 SEC Filing Intelligence Assistant")
st.markdown("Ask questions about **Apple**, **Microsoft**, and **Tesla** annual reports (10-K filings)")
st.divider()

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Chunks to retrieve", min_value=2, max_value=10, value=5)
    company_filter = st.multiselect(
        "Filter by company",
        ["AAPL", "MSFT", "TSLA"],
        default=["AAPL", "MSFT", "TSLA"]
    )
    st.divider()
    st.header("💡 Sample Questions")
    sample_questions = [
        "What was Apple's total revenue in 2023?",
        "What are Microsoft's main business segments?",
        "How did Tesla's automotive revenue change?",
        "What risks does Apple identify in its filings?",
        "What is Microsoft's cloud strategy?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state.question = q

# ── Chat history ───────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📎 View Sources"):
                for i, src in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}** — `{src['company']}` | `{src['filing_id']}`")
                    st.caption(src["text"][:400] + "...")
                    st.divider()

# ── Question input ─────────────────────────────────────────────
question = st.chat_input("Ask a question about the SEC filings...")

if "question" in st.session_state:
    question = st.session_state.pop("question")

if question:
    # Show user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching filings and generating answer..."):
            chunks = retrieve_chunks(question, k=k)

            # Filter by selected companies
            chunks = [c for c in chunks if c.metadata.get("company") in company_filter]

            if not chunks:
                answer = "No relevant chunks found for the selected companies. Try adjusting your filters."
                sources = []
            else:
                answer = ask_question(question, k=k)
                sources = [
                    {
                        "company": c.metadata.get("company"),
                        "filing_id": c.metadata.get("filing_id"),
                        "text": c.page_content
                    }
                    for c in chunks
                ]

        st.markdown(answer)

        if sources:
            with st.expander("📎 View Sources"):
                for i, src in enumerate(sources):
                    st.markdown(f"**Source {i+1}** — `{src['company']}` | `{src['filing_id']}`")
                    st.caption(src["text"][:400] + "...")
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })