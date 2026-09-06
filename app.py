"""
LocalDoc AI
----------
A local, privacy-friendly AI assistant that summarizes documents, answers questions (RAG),
compares documents, and translates — all powered by Ollama running on your own machine.

Run with:
    streamlit run app.py
"""

import streamlit as st

from ingestion.pdf_loader import load_document
from ingestion.text_cleaner import clean_text
from ingestion.chunker import split_into_documents
from summarization.summarizer import summarize_document
from summarization.llm import get_llm
from summarization.prompts import COMPARISON_PROMPT
from translation.translator import translate_text
from rag.vectorstore import build_vectorstore
from rag.qa_chain import answer_question
from utils.file_utils import save_uploaded_file, compression_ratio
from utils.token_counter import count_tokens
from config.settings import SUPPORTED_LANGUAGES, SUMMARY_LENGTHS

st.set_page_config(page_title="LocalDoc AI", page_icon="📄", layout="wide")

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, answer, sources)

st.title("📄 LocalDoc AI — Your Private Document Assistant")
st.caption("Runs 100% locally on Ollama - no paid API calls.")

tab_summarize, tab_qa, tab_compare = st.tabs(
    ["📝 Summarize", "💬 Document Q&A (RAG)", "🔀 Compare Documents"]
)

# --------------------------------------------------------------------------
# TAB 1: Summarize
# --------------------------------------------------------------------------
with tab_summarize:
    st.subheader("Upload a document or paste text")

    col_input1, col_input2 = st.columns(2)
    with col_input1:
        uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    with col_input2:
        pasted_text = st.text_area("...or paste text here", height=150)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        length_choice = st.radio("Summary length", list(SUMMARY_LENGTHS.keys()), index=1, horizontal=True)
    with col_b:
        language_choice = st.selectbox("Output language", SUPPORTED_LANGUAGES)
    with col_c:
        method_choice = st.selectbox(
            "Summarization method",
            ["Auto", "Stuff", "Map-Reduce", "Refine"],
            help="Auto picks Stuff for short docs and Map-Reduce for long ones.",
        )

    if st.button("Generate Summary", type="primary"):
        text = ""
        if uploaded_file is not None:
            path = save_uploaded_file(uploaded_file)
            text = load_document(path, uploaded_file.name)
        elif pasted_text.strip():
            text = pasted_text

        if not text.strip():
            st.warning("Please upload a file or paste some text first.")
        else:
            text = clean_text(text)
            st.session_state.raw_text = text

            with st.spinner(f"Summarizing with local Ollama model ({method_choice})..."):
                method_arg = "auto" if method_choice == "Auto" else method_choice
                result = summarize_document(
                    text, method=method_arg, length=length_choice, language=language_choice
                )

            st.success(f"Done — method used: **{result['method_used']}**")
            st.markdown("### Summary")
            st.write(result["summary"])

            ratio = compression_ratio(text, result["summary"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Input tokens (approx.)", result["input_tokens"])
            m2.metric("Compression ratio", f"{ratio}%")
            m3.metric("Method used", result["method_used"])

            st.download_button(
                "⬇️ Download Summary",
                data=result["summary"],
                file_name="summary.txt",
                mime="text/plain",
            )

            # Build the vectorstore in the background so the user can immediately switch to the Q&A tab without re-uploading.
            with st.spinner("Indexing document for Q&A..."):
                docs = split_into_documents(text, metadata={"source": uploaded_file.name if uploaded_file else "pasted text"})
                st.session_state.vectorstore = build_vectorstore(docs)
            st.info("Document indexed — you can now ask questions about it in the **Document Q&A** tab.")

# --------------------------------------------------------------------------
# TAB 2: Document Q&A (RAG)
# --------------------------------------------------------------------------
with tab_qa:
    st.subheader("Ask questions about your document")

    if st.session_state.vectorstore is None:
        st.info("Generate a summary first (in the Summarize tab) to index a document, then come back here.")
    else:
        question = st.text_input("Your question")
        if st.button("Ask") and question.strip():
            with st.spinner("Retrieving relevant context and asking the local LLM..."):
                result = answer_question(st.session_state.vectorstore, question)
            st.session_state.chat_history.append((question, result["answer"], result["sources"]))

        for q, a, sources in reversed(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(a)
                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            page_info = f", page {s['page']}" if s.get("page") is not None else ""
                            st.markdown(f"**[{s['index']}] {s['source']}{page_info}**")
                            st.caption(s["excerpt"] + "...")

# --------------------------------------------------------------------------
# TAB 3: Compare Documents
# --------------------------------------------------------------------------
with tab_compare:
    st.subheader("Compare two documents (e.g. two versions of a report)")

    col1, col2 = st.columns(2)
    with col1:
        file_a = st.file_uploader("Document A", type=["pdf", "txt"], key="doc_a")
    with col2:
        file_b = st.file_uploader("Document B", type=["pdf", "txt"], key="doc_b")

    if st.button("Compare Documents", type="primary"):
        if not file_a or not file_b:
            st.warning("Please upload both documents.")
        else:
            path_a = save_uploaded_file(file_a)
            path_b = save_uploaded_file(file_b)
            text_a = clean_text(load_document(path_a, file_a.name))
            text_b = clean_text(load_document(path_b, file_b.name))

            with st.spinner("Comparing documents with local Ollama model..."):
                llm = get_llm()
                prompt = COMPARISON_PROMPT.format(doc_a=text_a, doc_b=text_b)
                response = llm.invoke(prompt)
                comparison = response.content if hasattr(response, "content") else str(response)

            st.markdown("### Key Changes")
            st.write(comparison)
            st.download_button(
                "⬇️ Download Comparison",
                data=comparison,
                file_name="comparison.txt",
                mime="text/plain",
            )

st.divider()
st.caption(
    "made by Harsh"
)
