
from pathlib import Path
from hashlib import sha256
import json

import chromadb
from sentence_transformers import SentenceTransformer
from chunker import chunk_text

# 1. Configuration
DOCS = Path("data/raw")
STORAGE = Path("storage")
STORAGE.mkdir(exist_ok=True)

MANIFEST_PATH = STORAGE / "manifest.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 100
OVERLAP = 20

CONFIG = {
    "model": MODEL_NAME,
    "chunk_size": CHUNK_SIZE,
    "overlap": OVERLAP
}

# 2. Open the persistent database
client = chromadb.PersistentClient(
    path=str(STORAGE / "chroma")
)

collection = client.get_or_create_collection(
    name="university_rag_v1"
)
if not DOCS.is_dir():
    raise FileNotFoundError(f"Missing documents directory: {DOCS}")

# 3. Read the processing history
if MANIFEST_PATH.exists():
    manifest = json.loads(MANIFEST_PATH.read_text())
else:
    manifest = {"config": CONFIG, "files": {}}

if manifest["config"] != CONFIG:
    raise RuntimeError(
        "Model or chunk settings changed. "
        "Rebuild the index with a new collection."
    )

model = None
current_files = set()

# 4. Process documents
for path in sorted(DOCS.rglob("*")):
    if not path.is_file() or path.suffix not in {".txt", ".md", ".pdf"}:
        continue

    source = path.relative_to(DOCS).as_posix()
    current_files.add(source)

    file_hash = sha256(path.read_bytes()).hexdigest()
    old = manifest["files"].get(source)

    # Key optimization: skip unchanged documents!
    if old and old["hash"] == file_hash:
        print("SKIPPED:", source)
        continue

    # Replace this read with your existing loader as needed
    if path.suffix == ".pdf":
        # Use a PDF loader here
        pass
    else:
        text = path.read_text(encoding="utf-8")

    chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)

    ids = [
        sha256(f"{source}:{i}".encode()).hexdigest()
        for i in range(len(chunks))
    ]

    if chunks:
        # Load the model only when embedding is needed
        if model is None:
            model = SentenceTransformer(MODEL_NAME)

        vectors = model.encode(
            chunks,
            normalize_embeddings=True
        ).tolist()

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=vectors,
            metadatas=[
                {"source": source, "chunk_index": i}
                for i in range(len(chunks))
            ]
        )

    # Delete obsolete chunks from the old version
    old_ids = set(old["ids"]) if old else set()
    stale_ids = list(old_ids - set(ids))

    if stale_ids:
        collection.delete(ids=stale_ids)

    manifest["files"][source] = {
        "hash": file_hash,
        "ids": ids
    }

    print("INDEXED:", source, len(chunks), "chunks")

# 5. Remove records for deleted documents
for source in list(manifest["files"]):
    if source not in current_files:
        old_ids = manifest["files"][source]["ids"]
        if old_ids:
            collection.delete(ids=old_ids)

        del manifest["files"][source]
        print("REMOVED:", source)

# 6. Save processing history
temp_path = MANIFEST_PATH.with_suffix(".tmp")
temp_path.write_text(json.dumps(manifest, indent=2))
temp_path.replace(MANIFEST_PATH)

print("Total stored chunks:", collection.count())
