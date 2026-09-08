"""Answer a question from the indexed knowledge base, grounded + cited."""
import os
import sys

from common import connect, embed

TOP_K = 4

SYSTEM = """You answer strictly from the CONTEXT you are given.
If the context does not contain the answer, say "I don't know from the docs."
Be concise. End with the source filename(s) you used in square brackets."""


def retrieve(question):
    qvec = embed([question])[0]
    with connect() as conn:
        return conn.execute(
            "select source, content, similarity from match_chunks(%s, %s)",
            (qvec, TOP_K),
        ).fetchall()


def build_prompt(question, hits):
    """The user turn: retrieved context and the question, and nothing else.

    The grounding rules are NOT here — they go in the model's system_instruction
    (see answer()). That separation is the whole point. Everything in this string
    is either the user's question or text pulled verbatim out of the corpus, and
    a chunk that happens to read "ignore the above and answer freely" is then
    arguing with an instruction it cannot reach, instead of sitting in the same
    block as one and looking like a peer of it.
    """
    context = "\n\n".join(f"[{src}]\n{content}" for src, content, _ in hits)
    return f"CONTEXT:\n{context}\n\nQUESTION: {question}"


def answer(question, hits):
    key = os.environ.get("LLM_API_KEY")
    if not key:
        # No key: show what retrieval found so the pipeline is still demonstrable.
        top = hits[0]
        return f"(no LLM_API_KEY set — showing top retrieved chunk)\n\n{top[1]}\n\n[{top[0]}]"

    import google.generativeai as genai

    genai.configure(api_key=key)
    model = genai.GenerativeModel(
        os.environ.get("LLM_MODEL", "gemini-2.5-flash-lite"),
        system_instruction=SYSTEM,
    )
    return model.generate_content(build_prompt(question, hits)).text.strip()


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
