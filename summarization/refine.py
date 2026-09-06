"""
Refine summarization: summarize the first chunk, then iteratively
refine that summary with each subsequent chunk. Keeps a running
summary that gradually incorporates the whole document.
"""

from summarization.llm import get_llm
from summarization.prompts import REFINE_INITIAL_PROMPT, REFINE_PROMPT


def summarize_refine(chunks: list[str]) -> str:
    if not chunks:
        return ""

    llm = get_llm()

    # Initial summary from the first chunk
    prompt = REFINE_INITIAL_PROMPT.format(text=chunks[0])
    response = llm.invoke(prompt)
    summary = response.content if hasattr(response, "content") else str(response)

    # Refine using each subsequent chunk
    for chunk in chunks[1:]:
        prompt = REFINE_PROMPT.format(existing_summary=summary, text=chunk)
        response = llm.invoke(prompt)
        summary = response.content if hasattr(response, "content") else str(response)

    return summary
