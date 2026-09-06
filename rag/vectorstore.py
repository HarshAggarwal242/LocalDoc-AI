"""
Build, save, and load a local FAISS vector store from document chunks.
Fully local: FAISS runs in-process, embeddings come from Ollama.
"""

import os
from langchain_community.vectorstores import FAISS

from rag.embeddings import get_embeddings
from config.settings import VECTORSTORE_DIR


def build_vectorstore(documents) -> FAISS:
    """documents: list[langchain.docstore.document.Document]"""
    embeddings = get_embeddings()
    return FAISS.from_documents(documents, embeddings)


def save_vectorstore(vectorstore: FAISS, path: str = VECTORSTORE_DIR) -> None:
    os.makedirs(path, exist_ok=True)
    vectorstore.save_local(path)


def load_vectorstore(path: str = VECTORSTORE_DIR) -> FAISS:
    embeddings = get_embeddings()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def vectorstore_exists(path: str = VECTORSTORE_DIR) -> bool:
    return os.path.exists(os.path.join(path, "index.faiss"))
