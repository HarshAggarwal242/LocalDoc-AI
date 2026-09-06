"""
Translation via the local Ollama LLM.
"""

from summarization.llm import get_llm
from summarization.prompts import TRANSLATION_PROMPT


def translate_text(text: str, target_language: str) -> str:
    if target_language.lower() == "english" or not text.strip():
        return text

    llm = get_llm()
    prompt = TRANSLATION_PROMPT.format(text=text, language=target_language)
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)
