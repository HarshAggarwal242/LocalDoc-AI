"""
Map-Reduce summarization: summarize each chunk independently (map),
then combine those partial summaries into one final summary (reduce).
Better than "stuff" for long documents that don't fit in one context window.
"""

from summarization.llm import get_llm
from summarization.prompts import MAP_PROMPT, REDUCE_PROMPT


def summarize_map_reduce(chunks: list[str], length_instruction: str, language: str) -> str:
    llm = get_llm()

    # --- Map step: summarize every chunk on its own ---
    partial_summaries = []
    for chunk in chunks:
        prompt = MAP_PROMPT.format(text=chunk)
        response = llm.invoke(prompt)
        partial_summaries.append(response.content if hasattr(response, "content") else str(response))

    combined = "\n\n".join(f"- {s}" for s in partial_summaries)

    # --- Reduce step: combine all partial summaries into one ---
    reduce_prompt = REDUCE_PROMPT.format(
        text=combined,
        length_instruction=length_instruction,
        language=language,
    )
    final_response = llm.invoke(reduce_prompt)
    return final_response.content if hasattr(final_response, "content") else str(final_response)
