"""
Prompt templates used across the summarization pipelines.
"""

from langchain_core.prompts import PromptTemplate

# ---- Stuff / single-pass summarization ---------------------------------
STUFF_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["text", "length_instruction", "language"],
    template=(
        "You are an expert document summarizer.\n"
        "Summarize the following text {length_instruction}.\n"
        "Write the summary in {language}.\n"
        "Do not add information that is not present in the text.\n\n"
        "TEXT:\n{text}\n\n"
        "SUMMARY:"
    ),
)

# ---- Map step (per chunk) ------------------------------------------------
MAP_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=(
        "Summarize the key points of the following text chunk in a few "
        "concise bullet points. Only use information present in the text.\n\n"
        "TEXT CHUNK:\n{text}\n\n"
        "KEY POINTS:"
    ),
)

# ---- Reduce / combine step -----------------------------------------------
REDUCE_PROMPT = PromptTemplate(
    input_variables=["text", "length_instruction", "language"],
    template=(
        "You are given a set of bullet-point summaries taken from different "
        "sections of the same document:\n\n{text}\n\n"
        "Combine them into a single, coherent final summary {length_instruction}. "
        "Remove redundancy. Write the summary in {language}.\n\n"
        "FINAL SUMMARY:"
    ),
)

# ---- Refine: initial summary ---------------------------------------------
REFINE_INITIAL_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=(
        "Write a concise summary of the following text.\n\n"
        "TEXT:\n{text}\n\n"
        "SUMMARY:"
    ),
)

# ---- Refine: refine step with new context --------------------------------
REFINE_PROMPT = PromptTemplate(
    input_variables=["existing_summary", "text"],
    template=(
        "Here is an existing summary so far:\n{existing_summary}\n\n"
        "Below is additional context from the next part of the document. "
        "Refine the existing summary using this new context if it adds "
        "useful information. If the new context is not useful, return the "
        "existing summary unchanged.\n\n"
        "NEW CONTEXT:\n{text}\n\n"
        "REFINED SUMMARY:"
    ),
)

# ---- Translation -----------------------------------------------------------
TRANSLATION_PROMPT = PromptTemplate(
    input_variables=["text", "language"],
    template=(
        "Translate the following text into {language}. "
        "Preserve the meaning and tone. Output only the translation, "
        "with no extra commentary.\n\nTEXT:\n{text}\n\nTRANSLATION:"
    ),
)

# ---- RAG Q&A -----------------------------------------------------------------
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant answering questions about a document. "
        "Use only the context below to answer. If the answer is not in the "
        "context, say you don't know — do not make anything up. "
        "Cite the source snippet number(s) you used, like [1], [2].\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION: {question}\n\n"
        "ANSWER:"
    ),
)

# ---- Document comparison ------------------------------------------------
COMPARISON_PROMPT = PromptTemplate(
    input_variables=["doc_a", "doc_b"],
    template=(
        "Compare the two documents below and describe what changed between "
        "them. Organize the answer under clear topic headings (e.g. "
        "Agriculture, Infrastructure, Employment) based on what's actually "
        "discussed. Be specific and only use information from the texts.\n\n"
        "DOCUMENT A:\n{doc_a}\n\n"
        "DOCUMENT B:\n{doc_b}\n\n"
        "KEY CHANGES:"
    ),
)
