"""RAG pipeline: PDF ingestion, chunking, NumPy vector store, and streaming QA."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from pypdf import PdfReader
from pypdf.errors import PdfReadError

import config


@dataclass
class Document:
    """Represents a text chunk with source metadata."""

    page_content: str
    metadata: dict[str, Any]


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of strings using the configured provider."""
    if not texts:
        return []

    if config.LLM_PROVIDER == "gemini":
        from google import genai

        client = genai.Client(api_key=config.GOOGLE_API_KEY)
        embeddings: list[list[float]] = []
        # Batch in groups of 50 to stay within API limits
        for i in range(0, len(texts), 50):
            batch = texts[i : i + 50]
            response = client.models.embed_content(
                model=config.GEMINI_EMBEDDING_MODEL,
                contents=batch,
            )
            embeddings.extend([emb.values for emb in response.embeddings])
        return embeddings

    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.embeddings.create(
        model=config.OPENAI_EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


class LocalVectorStore:
    """Lightweight, pure-Python vector store backed by NumPy and local JSON."""

    def __init__(
        self,
        documents: list[Document] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> None:
        self.documents: list[Document] = documents or []
        self.embeddings: list[list[float]] = embeddings or []

    def add_documents(self, docs: list[Document]) -> None:
        """Embed and append new documents to the store."""
        texts = [doc.page_content for doc in docs]
        new_embeddings = _embed_texts(texts)
        self.documents.extend(docs)
        self.embeddings.extend(new_embeddings)

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Return the top-k most similar documents using cosine similarity."""
        if not self.documents or not self.embeddings:
            return []

        query_vec = np.array(_embed_texts([query])[0], dtype=np.float32)
        matrix = np.array(self.embeddings, dtype=np.float32)

        # Normalize vectors for cosine similarity
        query_norm = np.linalg.norm(query_vec)
        matrix_norms = np.linalg.norm(matrix, axis=1)
        denom = np.where(matrix_norms * query_norm == 0, 1e-10, matrix_norms * query_norm)
        scores = np.dot(matrix, query_vec) / denom

        top_indices = np.argsort(scores)[::-1][:k]
        return [self.documents[int(idx)] for idx in top_indices]

    def save(self, persist_file: Path) -> None:
        """Persist documents and embeddings to a local JSON file."""
        payload = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "embedding": emb,
            }
            for doc, emb in zip(self.documents, self.embeddings)
        ]
        persist_file.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, persist_file: Path) -> LocalVectorStore:
        """Load documents and embeddings from a local JSON file."""
        raw = json.loads(persist_file.read_text(encoding="utf-8"))
        docs = [
            Document(page_content=item["page_content"], metadata=item["metadata"])
            for item in raw
        ]
        embs = [item["embedding"] for item in raw]
        return cls(documents=docs, embeddings=embs)


def extract_documents_from_pdfs(
    uploaded_files: list[Any],
) -> tuple[list[Document], list[str]]:
    """Extract text page-by-page from uploaded PDF files with source metadata."""
    documents: list[Document] = []
    warnings: list[str] = []

    for uploaded_file in uploaded_files:
        filename = getattr(uploaded_file, "name", "unknown.pdf")
        try:
            reader = PdfReader(uploaded_file)

            if reader.is_encrypted:
                try:
                    if reader.decrypt("") == 0:
                        warnings.append(f"Skipped '{filename}': PDF is password-protected.")
                        continue
                except Exception:
                    warnings.append(f"Skipped '{filename}': Unable to decrypt PDF.")
                    continue

            extracted_pages = 0
            for page_idx, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": filename, "page": page_idx},
                        )
                    )
                    extracted_pages += 1

            if extracted_pages == 0:
                warnings.append(
                    f"Skipped '{filename}': No extractable text found (it may be a scanned image)."
                )

        except PdfReadError:
            warnings.append(f"Skipped '{filename}': File is corrupted or not a valid PDF.")
        except Exception as exc:
            warnings.append(f"Skipped '{filename}': Unexpected error ({exc}).")

    return documents, warnings


def split_documents(documents: list[Document]) -> list[Document]:
    """Split extracted PDF pages into overlapping chunks while preserving metadata."""
    chunks: list[Document] = []
    chunk_size = config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP

    for doc in documents:
        text = doc.page_content
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            # Try to break cleanly on a paragraph or sentence boundary
            if end < text_len:
                for sep in ("\n\n", "\n", ". ", " "):
                    split_pos = text.rfind(sep, start + overlap, end)
                    if split_pos != -1:
                        end = split_pos + len(sep)
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Document(page_content=chunk_text, metadata=dict(doc.metadata))
                )

            if end >= text_len:
                break
            start = max(end - overlap, start + 1)

    return chunks


def create_or_update_vector_store(
    chunks: list[Document],
    persist_file: Path = config.VECTOR_STORE_FILE,
) -> LocalVectorStore:
    """Create a new local vector store or merge new chunks into an existing one."""
    vector_store = load_vector_store(persist_file) or LocalVectorStore()
    vector_store.add_documents(chunks)
    vector_store.save(persist_file)
    return vector_store


def load_vector_store(
    persist_file: Path = config.VECTOR_STORE_FILE,
) -> LocalVectorStore | None:
    """Load an existing local vector store from disk if available."""
    if not persist_file.exists():
        return None
    return LocalVectorStore.load(persist_file)


def clear_vector_store(persist_file: Path = config.VECTOR_STORE_FILE) -> None:
    """Delete the persisted vector store file from disk."""
    if persist_file.exists():
        persist_file.unlink()


def _format_context(docs: list[Document]) -> str:
    """Format retrieved documents with explicit filename and page citations."""
    formatted_chunks: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        formatted_chunks.append(f"\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted_chunks)


def _format_history_text(chat_history: list[dict[str, Any]]) -> str:
    """Format recent chat history into a compact transcript."""
    lines: list[str] = []
    for msg in chat_history[-6:]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


class RAGAssistant:
    """History-aware RAG assistant using native Gemini or OpenAI SDKs."""

    def __init__(self, vector_store: LocalVectorStore) -> None:
        self.vector_store = vector_store

    def _reformulate_query(
        self, question: str, chat_history: list[dict[str, Any]]
    ) -> str:
        """Rewrite a follow-up question into a standalone search query."""
        if not chat_history:
            return question

        history_text = _format_history_text(chat_history)
        prompt = (
            "Given the chat history and the latest user question, rewrite the question "
            "into a standalone search query that can be understood without the chat history. "
            "Do NOT answer the question—return only the rewritten query.\n\n"
            f"Chat History:\n{history_text}\n\n"
            f"Latest Question: {question}"
        )

        if config.LLM_PROVIDER == "gemini":
            from google import genai

            client = genai.Client(api_key=config.GOOGLE_API_KEY)
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
            )
            return (response.text or question).strip()

        from openai import OpenAI

        client = OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return (response.choices[0].message.content or question).strip()

    def stream_answer(
        self, question: str, chat_history: list[dict[str, Any]]
    ) -> tuple[Iterator[str], list[Document]]:
        """Retrieve relevant chunks and return a token stream alongside source documents."""
        standalone_query = self._reformulate_query(question, chat_history)
        retrieved_docs = self.vector_store.similarity_search(
            standalone_query, k=config.TOP_K_RESULTS
        )
        context_text = _format_context(retrieved_docs)
        history_text = _format_history_text(chat_history)

        system_instruction = (
            "You are a precise PDF research assistant. Answer the user's question "
            "using ONLY the provided context below.\n\n"
            "Strict Rules:\n"
            "1. If the answer is not explicitly present in the context, respond with: "
            "'I could not find the answer to that in the uploaded PDFs.'\n"
            "2. Do not fabricate information or rely on outside knowledge.\n"
            "3. Cite your sources inline using the exact format (Filename, Page X) "
            "after every factual claim."
        )

        user_payload = (
            f"Context:\n{context_text}\n\n"
            + (f"Recent Conversation:\n{history_text}\n\n" if history_text else "")
            + f"Question: {question}"
        )

        if config.LLM_PROVIDER == "gemini":
            return self._stream_gemini(system_instruction, user_payload), retrieved_docs
        return self._stream_openai(system_instruction, user_payload), retrieved_docs

    def _stream_gemini(self, system_instruction: str, user_payload: str) -> Iterator[str]:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=config.GOOGLE_API_KEY)
        stream = client.models.generate_content_stream(
            model=config.GEMINI_MODEL,
            contents=user_payload,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=config.TEMPERATURE,
            ),
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text

    def _stream_openai(self, system_instruction: str, user_payload: str) -> Iterator[str]:
        from openai import OpenAI

        client = OpenAI(api_key=config.OPENAI_API_KEY)
        stream = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_payload},
            ],
            temperature=config.TEMPERATURE,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta