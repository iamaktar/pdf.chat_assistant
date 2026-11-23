"""Streamlit web interface for the PDF Chat Assistant."""

from __future__ import annotations

import streamlit as st

import config
from rag_pipeline import (
    Document,
    RAGAssistant,
    clear_vector_store,
    create_or_update_vector_store,
    extract_documents_from_pdfs,
    load_vector_store,
    split_documents,
)

st.set_page_config(page_title="PDF Chat Assistant", page_icon="📄", layout="wide")


def _extract_unique_citations(docs: list[Document]) -> list[str]:
    """Return a deduplicated, ordered list of 'Filename — Page X' citations."""
    seen: set[tuple[str, str]] = set()
    citations: list[str] = []
    for doc in docs:
        source = str(doc.metadata.get("source", "Unknown"))
        page = str(doc.metadata.get("page", "?"))
        key = (source, page)
        if key not in seen:
            seen.add(key)
            citations.append(f"**{source}** — Page {page}")
    return citations


def _init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vector_store" not in st.session_state:
        try:
            st.session_state.vector_store = load_vector_store()
        except Exception:
            st.session_state.vector_store = None

    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []


def main() -> None:
    st.title("📄 PDF Chat Assistant")
    st.caption(
        f"Powered by **{config.LLM_PROVIDER.upper()}** + Local NumPy Vector Search"
    )

    # Validate API keys before rendering interactive controls
    is_valid, error_msg = config.validate_config()
    if not is_valid:
        st.error(f"Configuration Error: {error_msg}")
        st.info("Create a `.env` file based on `.env.example` and restart the app.")
        st.stop()

    _init_session_state()

    # --- Sidebar: Document & Session Management ---
    with st.sidebar:
        st.header("Knowledge Base")
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if st.button("Process PDFs", type="primary", use_container_width=True):
            if not uploaded_files:
                st.warning("Please upload at least one PDF file first.")
            else:
                with st.spinner("Extracting text and building vector index..."):
                    docs, warnings = extract_documents_from_pdfs(uploaded_files)

                    for warning in warnings:
                        st.warning(warning)

                    if docs:
                        chunks = split_documents(docs)
                        st.session_state.vector_store = create_or_update_vector_store(
                            chunks
                        )
                        new_names = [f.name for f in uploaded_files]
                        for name in new_names:
                            if name not in st.session_state.indexed_files:
                                st.session_state.indexed_files.append(name)

                        st.success(
                            f"Indexed {len(chunks)} chunks from {len(docs)} pages."
                        )
                    else:
                        st.error("No valid text could be extracted from the uploaded PDFs.")

        if st.session_state.indexed_files:
            st.subheader("Indexed Documents")
            for fname in st.session_state.indexed_files:
                st.markdown(f"- `{fname}`")
        elif st.session_state.vector_store is not None:
            st.info("Loaded existing local knowledge base from disk.")

        st.divider()
        st.header("Session Controls")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("Reset KB", use_container_width=True):
                clear_vector_store()
                st.session_state.vector_store = None
                st.session_state.indexed_files = []
                st.session_state.messages = []
                st.rerun()

    # --- Main Area: Chat Interface ---
    if st.session_state.vector_store is None:
        st.info("Upload and process at least one PDF in the sidebar to start chatting.")
        return

    # Render prior chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                with st.expander("Sources"):
                    for citation in message["citations"]:
                        st.markdown(f"- {citation}")

    # Handle new user prompt
    if user_prompt := st.chat_input("Ask a question about your PDFs..."):
        prior_history = list(st.session_state.messages)

        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            try:
                assistant = RAGAssistant(st.session_state.vector_store)
                token_stream, retrieved_docs = assistant.stream_answer(
                    question=user_prompt,
                    chat_history=prior_history,
                )
                full_response = st.write_stream(token_stream)
                citations = _extract_unique_citations(retrieved_docs)

                if citations:
                    with st.expander("Sources"):
                        for citation in citations:
                            st.markdown(f"- {citation}")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_response,
                        "citations": citations,
                    }
                )
            except Exception as exc:
                st.error(f"Error generating response: {exc}")


if __name__ == "__main__":
    main()