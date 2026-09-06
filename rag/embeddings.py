"""
Local embeddings via Ollama (e.g. `nomic-embed-text`).
No HuggingFace downloads or paid APIs required — everything runs
through the same local Ollama server as the LLM.
"""

from langchain_ollama import OllamaEmbeddings

from config.settings import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL


def get_embeddings() -> OllamaEmbeddings:
    """
    Make sure the embedding model is pulled first, e.g.:
        ollama pull nomic-embed-text
    """
    return OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=OLLAMA_EMBED_MODEL)
