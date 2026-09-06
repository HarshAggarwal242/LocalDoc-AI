"""
Central configuration for the app.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Ollama connection
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Models 
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# LLM generation params 
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# Chunking 
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Rough word-count target above which we treat a document as "large"
# and switch from the simple "stuff" method to map-reduce/refine.
LARGE_DOC_TOKEN_THRESHOLD = int(os.getenv("LARGE_DOC_TOKEN_THRESHOLD", "3000"))

# RAG 
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "data/vectorstore")

# Supported output languages for summaries 
SUPPORTED_LANGUAGES = [
    "English",
    "Hindi",
    "Spanish",
    "French",
    "German",
    "Chinese",
    "Japanese",
    "Arabic",
]

SUMMARY_LENGTHS = {
    "Short": "in 3-4 concise sentences",
    "Medium": "in one well-structured paragraph (6-10 sentences)",
    "Detailed": "in multiple paragraphs covering all key points, with headings if useful",
}
