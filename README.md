# rag-demo — a minimal, working RAG pipeline

A small, self-contained Retrieval-Augmented Generation pipeline you can clone and run in a few minutes. It's the same technique behind a production doc-grounded copilot I built ("ask a question, get an answer grounded in a knowledge base, with citations") — stripped down to its essentials on a generic sample dataset.

The point isn't the sample data. It's to show the moving parts of RAG working end-to-end: chunking, embeddings, a `pgvector` similarity index, a retrieval query, and a grounded LLM answer that refuses to make things up.

## How it works

```mermaid
flowchart LR
    A[docs/*.md] -->|chunk| B[text chunks]
    B -->|embed| C[(pgvector<br/>Postgres)]
    Q[user question] -->|embed| D[query vector]
    D -->|cosine top-k| C
    C -->|retrieved chunks| E[prompt + context]
    E --> F[LLM]
    F -->|grounded answer + sources| G[stdout]
```

- **Chunking** — docs are split into overlapping windows so retrieval lands on focused passages.
- **Embeddings** — a local `sentence-transformers` model (no API key needed to index), so ingestion is free and offline.
- **Vector store** — Postgres + `pgvector`, cosine distance, an IVFFlat index, and a `match_chunks()` SQL function (the same shape you'd deploy as a Supabase RPC).
- **Retrieval** — top-k nearest chunks to the question.
- **Generation** — the retrieved chunks become the *only* context the LLM is allowed to answer from; if the answer isn't in them, it says so. Citations point back to the source doc.

## Run it

```bash
# 1. Postgres + pgvector (Docker)
docker compose up -d
psql "$DATABASE_URL" -f schema.sql        # or: make schema

# 2. Python deps
pip install -r requirements.txt
cp .env.example .env                        # set LLM_API_KEY for the answer step

# 3. Index the sample knowledge base, then ask
python ingest.py
python ask.py "How do I export my notes?"
```

Example:

```
$ python ask.py "Can I use Nimbus offline?"
Yes. Nimbus keeps a local-first copy of every note, so you can read and edit
completely offline; changes sync automatically when you reconnect. [nimbus-faq.md]
```

Swap the files in `data/` for your own and re-run `ingest.py` — the pipeline is domain-agnostic.

## Design notes

- **Grounded, not generative-from-memory.** The system prompt hard-limits the model to the retrieved context and tells it to answer "I don't know" when the context doesn't cover the question. This is the single most important guard against a doc copilot hallucinating.
- **Local embeddings, hosted generation.** Embedding every chunk through a paid API gets expensive and slow; a local model keeps ingestion free and lets the demo run without a key until the final answer step.
- **`pgvector` over a bespoke vector DB.** One database for rows *and* vectors means one backup, one connection, one RLS policy — and it ports straight to Supabase.

## Stack

`Python` · `pgvector` / `Postgres` · `sentence-transformers` · `Gemini` (swappable) · `pytest`

MIT licensed. Sample data is fictional.
