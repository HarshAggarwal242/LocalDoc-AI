"""
Loads text out of uploaded files (PDF or plain text).
"""

from io import BytesIO
from pypdf import PdfReader


def load_pdf(file) -> str:
    """
    Extract text from a PDF.

    `file` can be a path (str) or a file-like object (e.g. what
    Streamlit's file_uploader gives you).
    """
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text.strip()


def load_txt(file) -> str:
    """Extract text from a plain .txt upload or path."""
    if hasattr(file, "read"):
        raw = file.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        return raw.strip()
    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def load_document(file, filename: str) -> str:
    """
    Dispatch based on file extension. `filename` is used only to
    detect the type (Streamlit UploadedFile objects carry `.name`).
    """
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return load_pdf(file)
    if lower.endswith(".txt"):
        return load_txt(file)
    raise ValueError(f"Unsupported file type: {filename}")
