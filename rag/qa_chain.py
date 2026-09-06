"""
Retrieval-Augmented Generation Q&A: retrieve relevant chunks from the
FAISS store, then ask the local Ollama LLM to answer using only that
context, citing the snippet numbers it used.
"""

from summarization.llm import get_llm
from summarization.prompts import QA_PROMPT
from rag.retriever import retrieve_context


def answer_question(vectorstore, question: str) -> dict:
    context, source_docs = retrieve_context(vectorstore, question)

    if not context.strip():
        return {
            "answer": "I couldn't find anything relevant in the document to answer that.",
            "sources": [],
        }

    llm = get_llm()
    prompt = QA_PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    sources = [
        {
            "index": i + 1,
            "source": doc.metadata.get("source", "document"),
            "page": doc.metadata.get("page"),
            "excerpt": doc.page_content[:250],
        }
        for i, doc in enumerate(source_docs)
    ]

    return {"answer": answer, "sources": sources}
