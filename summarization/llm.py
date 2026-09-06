"""
Single source of truth for creating the local Ollama LLM client.
No OpenAI / Gemini / any paid API is used anywhere in this project.
"""

from langchain_ollama import ChatOllama

from config.settings import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, LLM_TEMPERATURE


def get_llm(temperature: float = LLM_TEMPERATURE) -> ChatOllama:
    """
    Returns a LangChain-compatible chat model backed by a locally
    running Ollama server. Make sure Ollama is running and the model
    has been pulled, e.g.:

        ollama pull llama3.1
        ollama serve   # usually already running as a service
    """
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_LLM_MODEL,
        temperature=temperature,
    )
