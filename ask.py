"""Ask questions about the PDF documents in the local vector database.

Example:
    python scripts/ask.py "What is a cappuccino?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chromadb

# Add the project folder to Python's import path when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.embedder import _get_model
from src.rag.generator import generate_answer


DEFAULT_DATABASE_DIR = PROJECT_ROOT / "chroma_db"
DEFAULT_COLLECTION_NAME = "my_documents"


def answer_question(
    question: str,
    database_dir: Path = DEFAULT_DATABASE_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    top_k: int = 5,
) -> str:
    """Find relevant document chunks and use them to answer a question."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    # Connect to the database created by scripts/ingest.py.
    client = chromadb.PersistentClient(path=str(database_dir))
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as exc:
        raise RuntimeError(
            "The document index was not found. Run 'python scripts/ingest.py' first."
        ) from exc

    # Convert the question into the same kind of vector used for the documents.
    question_vector = _get_model().encode(question, normalize_embeddings=True)
    results = collection.query(
        query_embeddings=[question_vector.tolist()],
        n_results=top_k,
        include=["documents", "metadatas"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    if not documents:
        return "I don't know based on the provided documents."

    # Give the language model both the text and the PDF filename for references.
    context_parts = []
    for document, metadata in zip(documents, metadatas):
        filename = metadata.get("file_name", "unknown file")
        context_parts.append(f"[{filename}]\n{document}")
    context = "\n\n".join(context_parts)

    return generate_answer(question, context)


def main() -> None:
    """Read command-line arguments and ask one question."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question",
        nargs="?",
        help="The question to ask. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of document chunks to retrieve (default: 5).",
    )
    args = parser.parse_args()

    question = args.question or input("Question: ").strip()
    if not question:
        parser.error("Please enter a question.")

    answer = answer_question(question, top_k=args.top_k)
    print(answer)


if __name__ == "__main__":
    main()
