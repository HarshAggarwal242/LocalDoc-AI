"""
Thin wrapper around a FAISS vectorstore's retriever, plus a helper to
format retrieved chunks (with source numbering) for the QA prompt.
"""

from config.settings import RAG_TOP_K


def get_retriever(vectorstore, k: int = RAG_TOP_K):
    return vectorstore.as_retriever(search_kwargs={"k": k})


def retrieve_context(vectorstore, question: str, k: int = RAG_TOP_K):
    """
    Returns (formatted_context_string, list_of_source_documents)
    Each retrieved chunk is numbered [1], [2], ... so the LLM (and the
    UI) can cite exactly which snippet an answer came from.
    """
    retriever = get_retriever(vectorstore, k)
    docs = retriever.invoke(question)

    formatted_chunks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "document")
        page = doc.metadata.get("page")
        label = f"{source}" + (f", page {page}" if page is not None else "")
        formatted_chunks.append(f"[{i}] (from {label})\n{doc.page_content}")

    context = "\n\n".join(formatted_chunks)
    return context, docs
