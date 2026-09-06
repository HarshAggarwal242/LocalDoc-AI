"""
Top-level summarization orchestrator. Picks a strategy (stuff /
map-reduce / refine) based on the user's choice or document size,
then runs it and optionally translates the result.
"""

from ingestion.chunker import split_text
from summarization.stuff import summarize_stuff
from summarization.map_reduce import summarize_map_reduce
from summarization.refine import summarize_refine
from translation.translator import translate_text
from utils.token_counter import count_tokens
from config.settings import LARGE_DOC_TOKEN_THRESHOLD, SUMMARY_LENGTHS


def summarize_document(
    text: str,
    method: str = "auto",
    length: str = "Medium",
    language: str = "English",
) -> dict:
    """
    method: "auto" | "Stuff" | "Map-Reduce" | "Refine"
    length: one of config.settings.SUMMARY_LENGTHS keys
    language: target output language

    Returns a dict with the summary, the method actually used, and
    basic stats useful for the UI.
    """
    length_instruction = SUMMARY_LENGTHS.get(length, SUMMARY_LENGTHS["Medium"])
    token_count = count_tokens(text)

    if method == "auto":
        method = "Stuff" if token_count <= LARGE_DOC_TOKEN_THRESHOLD else "Map-Reduce"

    if method == "Stuff":
        summary = summarize_stuff(text, length_instruction, language)
    else:
        chunks = split_text(text)
        if method == "Map-Reduce":
            summary = summarize_map_reduce(chunks, length_instruction, language)
        elif method == "Refine":
            summary = summarize_refine(chunks)
            # Refine doesn't take a language directly, so translate after if needed
            if language != "English":
                summary = translate_text(summary, language)
        else:
            raise ValueError(f"Unknown summarization method: {method}")

    return {
        "summary": summary,
        "method_used": method,
        "input_tokens": token_count,
    }
