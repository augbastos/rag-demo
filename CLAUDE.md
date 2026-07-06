# rag-demo — CLAUDE.md

Minimal, runnable RAG pipeline demo for Augusto's portfolio: chunk markdown →
local embeddings → pgvector → grounded, cited answers (Gemini). Public repo,
**MIT** (github.com/augbastos/rag-demo). The 4-file minimalism
(common.py / ingest.py / ask.py / schema.sql) is a deliberate design choice —
don't "improve" it into a framework.

## Verified commands
```bash
docker compose up -d && psql "$DATABASE_URL" -f schema.sql   # or: make up && make schema
pip install -r requirements.txt && cp .env.example .env
python ingest.py            # chunks+embeds data/*.md — TRUNCATES and reloads (not additive!)
python ask.py "question"    # no LLM_API_KEY → falls back to printing top chunk (intended)
pytest -q                   # offline; first run downloads ~90MB HF model
```

## Invariants
- Public repo: never commit real client/business data or `.env` (`.env.example`
  is the only tracked template).
- `data/*.md` describes a FICTIONAL app ("Nimbus") — demo corpus, not real docs.
- `schema.sql` hardcodes `vector(384)` for all-MiniLM-L6-v2 — changing
  `EMBED_MODEL` in common.py without migrating the schema breaks inserts.

## Gotchas
- CI (`.github/workflows/tests.yml`) runs offline unit tests only, with
  PYTHONPATH=repo root (tests have no packaging).
- `docs/competitive-analysis/benchmark.md` is an untracked self-review artifact,
  not a spec.

## State (2026-07-06)
Stable; docs-polish commits only. Local main may be ahead of origin — check
`git status` before assuming pushed.
