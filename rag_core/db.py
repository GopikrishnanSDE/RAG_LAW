"""Postgres storage: pgvector for embeddings, native tsvector for keyword search.

One table, two indexes — an IVFFlat index over `embedding` for ANN cosine
search, and a GIN index over the generated `tsv` column for BM25-ish keyword
search via ts_rank_cd. Kept in one instance instead of a separate vector DB +
search engine: the corpus is a few thousand chunks, not billions, and hybrid
retrieval only pays off if both signals stay trivially joinable.
"""

import json
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector

from rag_core import config

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    act_name TEXT NOT NULL,
    source_file TEXT NOT NULL,
    section_number TEXT,
    section_title TEXT,
    chapter TEXT,
    part TEXT,
    chunk_index INT,
    chunk_count INT,
    page_start INT,
    page_end INT,
    amendments JSONB,
    text TEXT NOT NULL,
    embedding VECTOR({dim}),
    tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv);
""".format(dim=config.EMBEDDING_DIM)

# IVFFlat needs rows in the table before it can pick cluster centroids, so the
# vector index is created separately, after the first bulk load.
VECTOR_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""


@contextmanager
def get_connection():
    conn = psycopg.connect(config.DATABASE_URL, autocommit=True)
    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()


def init_schema(conn) -> None:
    conn.execute(SCHEMA_SQL)


def create_vector_index(conn) -> None:
    conn.execute(VECTOR_INDEX_SQL)


def clear_chunks(conn) -> None:
    conn.execute("TRUNCATE TABLE chunks RESTART IDENTITY")


def insert_chunks(conn, records: list[dict]) -> None:
    """records: [{text, embedding, metadata: {...}}]"""
    with conn.cursor() as cur:
        for r in records:
            m = r["metadata"]
            cur.execute(
                """
                INSERT INTO chunks (
                    act_name, source_file, section_number, section_title,
                    chapter, part, chunk_index, chunk_count,
                    page_start, page_end, amendments, text, embedding
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    m.get("act_name"),
                    m.get("source_file"),
                    m.get("section_number"),
                    m.get("section_title"),
                    m.get("chapter"),
                    m.get("part"),
                    m.get("chunk_index"),
                    m.get("chunk_count"),
                    m.get("page_start"),
                    m.get("page_end"),
                    json.dumps(m.get("amendments")) if m.get("amendments") else None,
                    r["text"],
                    r["embedding"],
                ),
            )


def _row_to_dict(row, columns) -> dict:
    d = dict(zip(columns, row))
    metadata_keys = [
        "act_name", "source_file", "section_number", "section_title",
        "chapter", "part", "chunk_index", "chunk_count",
        "page_start", "page_end", "amendments",
    ]
    d["metadata"] = {k: d.pop(k) for k in metadata_keys}
    return d


def vector_search(conn, query_embedding: list[float], k: int) -> list[dict]:
    columns = [
        "id", "act_name", "source_file", "section_number", "section_title",
        "chapter", "part", "chunk_index", "chunk_count", "page_start",
        "page_end", "amendments", "text", "distance",
    ]
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, act_name, source_file, section_number, section_title,
                   chapter, part, chunk_index, chunk_count, page_start,
                   page_end, amendments, text,
                   embedding <=> %s AS distance
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (query_embedding, query_embedding, k),
        )
        rows = cur.fetchall()
    results = []
    for row in rows:
        d = _row_to_dict(row, columns)
        d["vector_distance"] = d.pop("distance")
        results.append(d)
    return results


def keyword_search(conn, query_text: str, k: int) -> list[dict]:
    columns = [
        "id", "act_name", "source_file", "section_number", "section_title",
        "chapter", "part", "chunk_index", "chunk_count", "page_start",
        "page_end", "amendments", "text", "rank",
    ]
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, act_name, source_file, section_number, section_title,
                   chapter, part, chunk_index, chunk_count, page_start,
                   page_end, amendments, text,
                   ts_rank_cd(tsv, plainto_tsquery('english', %s)) AS rank
            FROM chunks
            WHERE tsv @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
            """,
            (query_text, query_text, k),
        )
        rows = cur.fetchall()
    results = []
    for row in rows:
        d = _row_to_dict(row, columns)
        d["keyword_rank"] = d.pop("rank")
        results.append(d)
    return results
