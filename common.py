"""Shared helpers: DB connection, embeddings, chunking."""
import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBED_MODEL = "all-MiniLM-L6-v2"  # 384-dim, small and fast, runs locally
_model = None


def embed(texts):
    """Return a list of embedding vectors for the given texts."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model.encode(list(texts), normalize_embeddings=True).tolist()


def connect():
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL not set — copy .env.example to .env")
    conn = psycopg.connect(url, autocommit=True)
    register_vector(conn)
    return conn


def chunk_text(text, size=600, overlap=100):
    """Split text into overlapping character windows on paragraph-ish breaks."""
    text = text.strip()
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        window = text[start:end]
        # prefer to cut on a blank line or sentence end, not mid-word
        cut = max(window.rfind("\n\n"), window.rfind(". "))
        if cut > size // 2 and end < len(text):
            end = start + cut + 1
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]
