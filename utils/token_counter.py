"""
Approximate token counting. Uses tiktoken if available (good enough
proxy even for non-OpenAI models); falls back to a whitespace-based
estimate so the app never breaks if tiktoken/encodings aren't reachable.
"""

try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - fallback path
    _ENCODING = None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENCODING is not None:
        try:
            return len(_ENCODING.encode(text))
        except Exception:
            pass
    # Fallback: rough estimate, ~0.75 tokens per word
    return int(len(text.split()) / 0.75)
