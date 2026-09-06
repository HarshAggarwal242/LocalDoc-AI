"""
Splits long text into overlapping chunks that fit comfortably inside
the LLM's context window, and wraps them as LangChain Documents.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """Return a list of raw text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def split_into_documents(text: str, metadata: dict | None = None):
    """Return a list of LangChain `Document` objects, ready for chains."""
    metadata = metadata or {}
    chunks = split_text(text)
    return [Document(page_content=chunk, metadata=metadata) for chunk in chunks]
