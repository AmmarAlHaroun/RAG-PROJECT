
import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load saved data, not source documents
client = chromadb.PersistentClient(path="storage/chroma")

collection = client.get_collection(
    name="university_rag_v1"
)

# Load the same embedding model used for indexing
model = SentenceTransformer(MODEL_NAME)

print("Stored chunks:", collection.count())

while True:
    question = input("\nAsk a question (or 'exit'): ").strip()

    if question.lower() == "exit":
        break

    if not question:
        continue

    if collection.count() == 0:
        print("No documents indexed yet.")
        continue

    # Embed ONLY the user's question
    query_vector = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    # Search the saved embeddings
    TOP_K = 1

    results = collection.query(
    query_embeddings=[query_vector],
    n_results=min(TOP_K, collection.count()),
    include=["documents", "metadatas", "distances"]
    )

    MAX_DISTANCE = 1.7

    for text, metadata, distance in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0] ):
        if distance > MAX_DISTANCE:
            print("\nI could not find a relevant answer.")
            continue

        print(f"\nSource: {metadata['source']}")
        print(" ".join(text.split()))
   