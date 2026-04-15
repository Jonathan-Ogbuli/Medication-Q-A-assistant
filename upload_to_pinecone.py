import json
import os
import numpy as np
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_NAME = "medication-index"
BATCH_SIZE = 100


def load_existing_data(index_path="medication_faiss.index", metadata_path="metadata.json"):
    import faiss
    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    vectors = index.reconstruct_n(0, index.ntotal)
    return vectors, metadata


def upload_vectors(vectors, metadata, index_name=INDEX_NAME):
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(index_name)

    vectors_list = vectors.tolist()
    total = len(vectors_list)

    print(f"Uploading {total} vectors to Pinecone...")

    for i in range(0, total, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, total)
        batch_vectors = []
        batch_meta = []

        for j in range(i, batch_end):
            batch_vectors.append({
                "id": str(j),
                "values": vectors_list[j],
                "metadata": {
                    "title": metadata[j]["title"],
                    "section": metadata[j]["section"],
                    "content": metadata[j]["content"],
                    "intent": metadata[j]["intent"],
                    "url": metadata[j]["url"],
                    "tags": json.dumps(metadata[j]["tags"]),
                }
            })
            batch_meta.append(metadata[j])

        index.upsert(vectors=batch_vectors, namespace="")

        print(f"Uploaded {batch_end}/{total} vectors")

    print("Done!")


def main():
    print("Loading existing FAISS index...")
    vectors, metadata = load_existing_data()
    print(f"Loaded {len(vectors)} vectors")

    upload_vectors(vectors, metadata)


if __name__ == "__main__":
    main()