"""Read docs/*.md, chunk + embed them, and load them into pgvector."""
import pathlib

from common import chunk_text, connect, embed

DOCS = pathlib.Path(__file__).parent / "data"


def main():
    files = sorted(DOCS.glob("*.md"))
    if not files:
        raise SystemExit(f"No .md files in {DOCS}")

    rows = []
    for path in files:
        for chunk in chunk_text(path.read_text(encoding="utf-8")):
            rows.append((path.name, chunk))

    vectors = embed(c for _, c in rows)

    with connect() as conn:
        conn.execute("truncate chunks")
        with conn.cursor() as cur:
            cur.executemany(
                "insert into chunks (source, content, embedding) values (%s, %s, %s)",
                [(src, content, vec) for (src, content), vec in zip(rows, vectors)],
            )

    print(f"Indexed {len(rows)} chunks from {len(files)} docs.")


if __name__ == "__main__":
    main()
