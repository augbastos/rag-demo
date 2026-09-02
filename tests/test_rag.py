"""Unit tests for the parts that don't need a database or an API key."""
from ask import SYSTEM, answer, build_prompt
from common import chunk_text, embed


def test_chunking_covers_full_text_with_overlap():
    text = "\n\n".join(f"Paragraph {i}. " * 20 for i in range(10))
    chunks = chunk_text(text, size=400, overlap=80)
    assert len(chunks) > 1
    # every chunk fits roughly within the window
    assert all(len(c) <= 500 for c in chunks)
    # overlap means the join re-covers the original content
    assert "Paragraph 0" in chunks[0]
    assert "Paragraph 9" in chunks[-1]


def test_short_text_is_one_chunk():
    assert chunk_text("just a line") == ["just a line"]


def test_embeddings_are_normalised_384d():
    vecs = embed(["hello world", "another sentence"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 384
    norm = sum(x * x for x in vecs[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-3  # normalize_embeddings=True


def test_retrieval_ranks_relevant_chunk_first():
    """Pure-vector sanity check: the closest chunk to a query is the relevant one."""
    docs = [
        "Nimbus works fully offline; notes are stored locally.",
        "Annual billing saves about two months on paid plans.",
        "Press Cmd+N to create a new note.",
    ]
    doc_vecs = embed(docs)
    qvec = embed(["can I use it without internet?"])[0]

    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))

    scores = [cosine(qvec, dv) for dv in doc_vecs]
    assert scores.index(max(scores)) == 0  # the offline sentence wins


# --- grounding rules belong to the system instruction, not to the user turn ---
#
# They used to be concatenated onto the front of the same string as the retrieved
# chunks. That put the rules and the corpus in one undifferentiated block, so a
# document saying "ignore the above" addressed them as a peer. These tests pin the
# separation: the rules must not appear in the user turn, and the model must be
# constructed with them.

HITS = [
    ("nimbus.md", "Nimbus works fully offline; notes are stored locally.", 0.91),
    ("billing.md", "Annual billing saves about two months.", 0.42),
]


def test_user_turn_carries_context_and_question_only():
    prompt = build_prompt("can I use it without internet?", HITS)

    assert "CONTEXT:" in prompt
    assert "QUESTION: can I use it without internet?" in prompt
    assert "Nimbus works fully offline" in prompt
    assert "[nimbus.md]" in prompt  # the citation handle survives

    # the rules live in system_instruction; none of them may leak into this string
    assert SYSTEM not in prompt
    assert "I don't know from the docs" not in prompt


def test_answer_puts_the_rules_in_system_instruction(monkeypatch):
    """The model is built with system_instruction, and sent only the user turn."""
    import sys
    import types

    seen = {}

    class FakeModel:
        def __init__(self, name, system_instruction=None):
            seen["model"] = name
            seen["system_instruction"] = system_instruction

        def generate_content(self, prompt):
            seen["prompt"] = prompt
            return types.SimpleNamespace(text="  offline, yes [nimbus.md]  ")

    fake = types.ModuleType("google.generativeai")
    fake.configure = lambda api_key=None: seen.__setitem__("key", api_key)
    fake.GenerativeModel = FakeModel
    google_pkg = types.ModuleType("google")
    google_pkg.generativeai = fake
    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake)
    monkeypatch.setenv("LLM_API_KEY", "not-a-real-key")

    out = answer("can I use it without internet?", HITS)

    assert seen["system_instruction"] == SYSTEM
    assert SYSTEM not in seen["prompt"]
    assert "QUESTION: can I use it without internet?" in seen["prompt"]
    assert out == "offline, yes [nimbus.md]"


def test_no_api_key_still_shows_retrieval(monkeypatch):
    """The keyless path is unchanged: it never builds a prompt at all."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    out = answer("anything", HITS)
    assert "no LLM_API_KEY set" in out
    assert "Nimbus works fully offline" in out
    assert "[nimbus.md]" in out
