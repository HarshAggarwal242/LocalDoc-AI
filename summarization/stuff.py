"""
"Stuff" summarization: put the whole text into one prompt.
Good for short/medium documents that fit in the model's context window.
"""

from summarization.llm import get_llm
from summarization.prompts import STUFF_SUMMARY_PROMPT


def summarize_stuff(text: str, length_instruction: str, language: str) -> str:
    llm = get_llm()
    prompt = STUFF_SUMMARY_PROMPT.format(
        text=text,
        length_instruction=length_instruction,
        language=language,
    )
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)
