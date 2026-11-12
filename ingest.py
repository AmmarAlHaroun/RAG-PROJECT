"""Read PDFs and build the searchable Chroma index.

Example:
    python scripts/ingest.py --reset
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import chromadb

# Add the project folder to Python's import path when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chunker import chunk_text
from src.rag.embedder import embed_texts
from src.rag.loader import load_documents


DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_DATABASE_DIR = PROJECT_ROOT / "chroma_db"
DEFAULT_COLLECTION_NAME = "my_documents"


def build_index(
    data_dir: Path,
    database_dir: Path,
    collection_name: str,
    reset: bool = False,
) -> tuple[int, int]:
    """Load PDFs, create embeddings, and save them in Chroma.

    Returns:
        A tuple containing the number of documents and chunks indexed.
    """
    files = sorted(data_dir.rglob("*.pdf"))
    if not files:
        raise FileNotFoundError(f"No PDF files found under {data_dir}")

    documents = load_documents(files)
    chunks: list[str] = []
    metadatas: list[dict[str, str | int]] = []
    chunk_ids: list[str] = []

    for document in documents:
        document_chunks = chunk_text(document.text, chunk_size=180, overlap=30)
        for chunk_number, chunk in enumerate(document_chunks):
            chunks.append(chunk)
            metadatas.append(
                {
                    "source": document.source,
                    "file_name": Path(document.source).name,
                    "chunk": chunk_number,
                }
            )
            # Stable IDs let us safely run ingestion more than once.
            unique_id = f"{document.source}:{chunk_number}"
            chunk_ids.append(hashlib.sha256(unique_id.encode()).hexdigest())

    if not chunks:
        raise ValueError("The PDFs contained no extractable text.")

    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embed_texts(chunks)
    client = chromadb.PersistentClient(path=str(database_dir))

    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            # It is fine if the collection does not exist yet.
            pass

    collection = client.get_or_create_collection(name=collection_name)
    collection.upsert(
        ids=chunk_ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )
    return len(documents), len(chunks)


def main() -> None:
    """Read command-line arguments and build the index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Folder containing PDFs (default: data/raw).",
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=DEFAULT_DATABASE_DIR,
        help="Folder for the Chroma database (default: chroma_db).",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the old collection before indexing.",
    )
    args = parser.parse_args()

    document_count, chunk_count = build_index(
        data_dir=args.data_dir,
        database_dir=args.db_dir,
        collection_name=args.collection,
        reset=args.reset,
    )
    print(
        f"Indexed {document_count} documents and {chunk_count} chunks "
        f"in {args.db_dir}."
    )


if __name__ == "__main__":
    main()
