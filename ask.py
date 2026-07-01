"""Answer a question from the indexed knowledge base, grounded + cited."""
import os
import sys

from common import connect, embed

TOP_K = 4

SYSTEM = """You answer strictly from the CONTEXT below.
If the context does not contain the answer, say "I don't know from the docs."
Be concise. End with the source filename(s) you used in square brackets."""


def retrieve(question):
    qvec = embed([question])[0]
    with connect() as conn:
        return conn.execute(
            "select source, content, similarity from match_chunks(%s, %s)",
            (qvec, TOP_K),
        ).fetchall()


def answer(question, hits):
    context = "\n\n".join(f"[{src}]\n{content}" for src, content, _ in hits)
    prompt = f"{SYSTEM}\n\nCONTEXT:\n{context}\n\nQUESTION: {question}"

    key = os.environ.get("LLM_API_KEY")
    if not key:
        # No key: show what retrieval found so the pipeline is still demonstrable.
        top = hits[0]
        return f"(no LLM_API_KEY set — showing top retrieved chunk)\n\n{top[1]}\n\n[{top[0]}]"

    import google.generativeai as genai

    genai.configure(api_key=key)
    model = genai.GenerativeModel(os.environ.get("LLM_MODEL", "gemini-2.5-flash-lite"))
    return model.generate_content(prompt).text.strip()


def main():
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        raise SystemExit('Usage: python ask.py "your question"')
    hits = retrieve(question)
    if not hits:
        raise SystemExit("Nothing indexed — run `python ingest.py` first.")
    print(answer(question, hits))


if __name__ == "__main__":
    main()
