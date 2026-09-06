"""
Small file/text helpers shared across the app.
"""

import os
import tempfile


def save_uploaded_file(uploaded_file) -> str:
    """
    Persist a Streamlit UploadedFile to a temp path on disk and
    return that path (needed because pypdf reads paths/file-handles).
    """
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def compression_ratio(original_text: str, summary_text: str) -> float:
    """Return summary_length / original_length as a percentage."""
    orig_len = len(original_text.split())
    summ_len = len(summary_text.split())
    if orig_len == 0:
        return 0.0
    return round((summ_len / orig_len) * 100, 2)


def make_download_bytes(text: str) -> bytes:
    return text.encode("utf-8")
