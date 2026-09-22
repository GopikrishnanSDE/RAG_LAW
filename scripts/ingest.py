"""CLI: PDF -> section-aware chunks -> embeddings -> Postgres.

Usage:
    python scripts/ingest.py [--pdf path/to.pdf] [--reset]

--reset truncates the chunks table first (use when re-ingesting after a
chunking or embedding-model change, so stale vectors don't linger).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_core import db, embeddings
from rag_core.chunking import count_tokens
from rag_core.ingestion_pipeline import DEFAULT_PDF, chunk_data, load_documents

ACT_NAME = "The Income-tax Act, 2025 (as amended by FA Act 2026)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=str(DEFAULT_PDF))
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    documents = load_documents(args.pdf)
    chunks = chunk_data(documents, act_name=ACT_NAME, source_file=Path(args.pdf).name)

    token_counts = [count_tokens(c.text) for c in chunks]
    print(
        f"chunk token count: min={min(token_counts)} "
        f"max={max(token_counts)} avg={sum(token_counts) // len(token_counts)}"
    )

    print(f"Embedding {len(chunks)} chunks...")
    vectors = embeddings.embed_texts([c.text for c in chunks])

    with db.get_connection() as conn:
        db.init_schema(conn)
        if args.reset:
            print("Clearing existing chunks...")
            db.clear_chunks(conn)

        records = [
            {"text": c.text, "embedding": v, "metadata": c.metadata}
            for c, v in zip(chunks, vectors)
        ]
        print("Inserting into Postgres...")
        db.insert_chunks(conn, records)

        print("Building vector index...")
        db.create_vector_index(conn)

    print(f"Done. Ingested {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
