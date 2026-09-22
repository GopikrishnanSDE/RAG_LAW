from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from rag_core.chunking import Chunk, chunk_documents as _chunk_documents, count_tokens

load_dotenv()

DEFAULT_PDF = Path(__file__).resolve().parent.parent / "data" / "income_tax_act_2025.pdf"


def load_documents(file_path: str) -> list:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif path.suffix == ".txt":
        loader = TextLoader(str(path))
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    docs = loader.load()
    for doc in docs:
        doc.metadata["source_file"] = path.name

    print(f"Loaded {len(docs)} page(s) from {path.name}")
    return docs


def chunk_data(
    docs: list,
    act_name: str,
    source_file: str,
    max_tokens: int = 700,
    hard_max_tokens: int = 1200,
) -> list[Chunk]:
    """Chunk loaded page-level Documents into section-aware chunks.

    Thin wrapper around chunking.chunk_documents() so the ingestion pipeline
    has a single call site for "raw docs -> chunks" — see chunking.py for why
    this is section-aware rather than a fixed-size splitter (a fixed window
    doesn't know where a section/sub-section ends and will cut a clause away
    from the number it modifies, which is the exact silent-failure mode this
    project is built to catch).
    """
    if not docs:
        raise ValueError("chunk_data() got no documents to chunk")

    chunks = _chunk_documents(
        docs,
        act_name=act_name,
        source_file=source_file,
        max_tokens=max_tokens,
        hard_max_tokens=hard_max_tokens,
    )

    if not chunks:
        raise ValueError(
            f"chunk_data() found 0 sections in {source_file} — "
            "check that the PDF text extracted cleanly and section markers matched"
        )

    print(f"Chunked {len(docs)} page(s) into {len(chunks)} section-aware chunks")
    return chunks


def main():
    documents = load_documents(str(DEFAULT_PDF))
    chunks = chunk_data(
        documents,
        act_name="The Income-tax Act, 2025 (as amended by FA Act 2026)",
        source_file=DEFAULT_PDF.name,
    )

    token_counts = [count_tokens(c.text) for c in chunks]
    print(
        f"chunk token count: min={min(token_counts)} "
        f"max={max(token_counts)} avg={sum(token_counts)//len(token_counts)}"
    )


if __name__ == "__main__":
    main()
