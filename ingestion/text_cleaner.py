"""
Basic text cleaning before chunking/summarization.
"""

import re


def clean_text(text: str) -> str:
    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse 3+ blank lines into a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse runs of spaces/tabs
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Remove common PDF artifacts: page-number-only lines, form feeds
    text = text.replace("\x0c", "\n")
    lines = [ln for ln in text.split("\n") if not re.fullmatch(r"\s*\d{1,4}\s*", ln)]
    text = "\n".join(lines)

    return text.strip()
