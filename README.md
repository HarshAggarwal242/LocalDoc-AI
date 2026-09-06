# AI Document Summarizer & Multilingual Text Assistant

A local, privacy-friendly AI app that summarizes PDFs/text, answers
questions about them (RAG), translates summaries, and compares two
documents — all powered by **Ollama running on your own machine**.
No OpenAI, no Gemini, no paid API keys required.

## Features

- Upload a PDF or paste text
- Choose summary length (Short / Medium / Detailed) and output language
- Three summarization strategies: **Stuff**, **Map-Reduce**, **Refine**
  (auto-selected based on document size, or pick manually)
- Document Q&A via Retrieval-Augmented Generation (FAISS + local
  Ollama embeddings), with source citations
- Compare two documents and get a structured "what changed" report
- Download summaries / comparisons as `.txt`

## Project structure

```
localdoc-ai/
│
├── app.py                     # Streamlit UI, wires everything together
├── requirements.txt
├── README.md
├── .env                       # Ollama connection + model names
├── .gitignore
│
├── config/
│   └── settings.py            # All tunables (models, chunk size, etc.)
│
├── ingestion/
│   ├── pdf_loader.py          # PDF/TXT -> raw text
│   ├── text_cleaner.py        # Cleanup of extracted text
│   └── chunker.py             # RecursiveCharacterTextSplitter wrapper
│
├── summarization/
│   ├── llm.py                 # Local Ollama chat-model factory
│   ├── prompts.py             # All prompt templates
│   ├── summarizer.py          # Orchestrator: picks a strategy
│   ├── stuff.py               # Single-pass summarization
│   ├── map_reduce.py          # Map-Reduce summarization
│   └── refine.py              # Refine (iterative) summarization
│
├── rag/
│   ├── embeddings.py          # Local Ollama embeddings
│   ├── vectorstore.py         # FAISS build/save/load
│   ├── retriever.py           # Similarity search + context formatting
│   └── qa_chain.py            # RAG question answering with citations
│
├── translation/
│   └── translator.py          # Local LLM-based translation
│
├── utils/
│   ├── token_counter.py       # Approx. token counting (tiktoken or fallback)
│   └── file_utils.py          # Upload handling, compression ratio, etc.
│
├── data/
│   └── sample_documents/      # Put test PDFs/TXT files here
│
└── notebooks/
    └── experiments.ipynb      # Scratchpad for prototyping
```

## Prerequisites

1. **Install Ollama**: https://ollama.com/download
2. **Pull the models** used by this app (you can swap these — see
   `.env` — for any models you already have):

   ```bash
   ollama pull llama3.1
   ollama pull nomic-embed-text
   ```

3. Make sure Ollama is running (it usually runs as a background
   service after install; otherwise `ollama serve`).

## Setup

```bash
cd localdoc-ai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Adjust `.env` if your Ollama server or model names differ from the
defaults:

```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.1
OLLAMA_EMBED_MODEL=nomic-embed-text
```

## Run

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Notes on the architecture

- **Stuff** summarization sends the whole document in one prompt —
  fast, but limited by context window size.
- **Map-Reduce** splits the document into chunks, summarizes each
  chunk independently, then combines those partial summaries — scales
  to long documents.
- **Refine** summarizes the first chunk, then iteratively updates that
  summary as each subsequent chunk is read — keeps a running summary
  in context.
- **RAG Q&A**: document chunks are embedded locally via Ollama,
  stored in an in-memory FAISS index, and the most relevant chunks are
  retrieved and passed to the LLM for each question, with citations
  back to the source chunk.

## Swapping models

Any model you've pulled into Ollama works — just change
`OLLAMA_LLM_MODEL` / `OLLAMA_EMBED_MODEL` in `.env`. For example:

```
OLLAMA_LLM_MODEL=mistral
OLLAMA_EMBED_MODEL=mxbai-embed-large
```


