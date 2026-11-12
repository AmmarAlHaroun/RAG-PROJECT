
import chromadb
from sentence_transformers import SentenceTransformer

# Same configuration used during ingestion
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DB_PATH = "storage/chroma"
COLLECTION_NAME = "university_rag_v1"

# Initialize once
model = SentenceTransformer(MODEL_NAME)
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)


def retrieve(question: str, k: int = 3, source: str | None = None):
    if not question.strip():
        return []

    if k < 1:
        raise ValueError("k must be at least 1")

    if collection.count() == 0:
        return []

    # 1. Embed ONLY the question
    question_vector = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    # 2. Prepare our search
    options = {
        "query_embeddings": [question_vector],
        "n_results": min(k, collection.count()),
        "include": ["documents", "metadatas", "distances"]
    }

    # Optional: search only one source document
    if source is not None:
        options["where"] = {"source": source}

    # 3. Search existing stored vectors
    results = collection.query(**options)

    # 4. Convert database output into useful records
    chunks = []

    for i, chunk_id in enumerate(results["ids"][0]):
        metadata = results["metadatas"][0][i] or {}

        chunks.append({
            "id": chunk_id,
            "text": results["documents"][0][i],
            "source": metadata.get("source", "unknown"),
            "distance": results["distances"][0][i]
        })

    return chunks
