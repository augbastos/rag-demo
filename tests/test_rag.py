"""Unit tests for the parts that don't need a database or an API key."""
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
