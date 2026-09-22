"""
print_chunks.py — dump every chunk from the ingestion pipeline to the terminal.

Usage:
    python print_chunks.py [pdf_path]

Defaults to Income_Tax_Act_2025_as_amended_by_FA_Act_2026.pdf if no path is given.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_core.ingestion_pipeline import load_documents, chunk_data


def main():
    source_file = (
        sys.argv[1]
        if len(sys.argv) > 1
        else str(Path(__file__).resolve().parent.parent / "data" / "income_tax_act_2025.pdf")
    )

    documents = load_documents(source_file)
    chunks = chunk_data(
        documents,
        act_name="The Income-tax Act, 2025 (as amended by FA Act 2026)",
        source_file=source_file,
    )

    for i, chunk in enumerate(chunks, start=1):
        print("=" * 80)
        print(f"CHUNK {i}/{len(chunks)}")
        print(f"metadata: {chunk.metadata}")
        print("-" * 80)
        print(chunk.text)
        print()

    print("=" * 80)
    print(f"Total chunks: {len(chunks)}")


if __name__ == "__main__":
    main()
