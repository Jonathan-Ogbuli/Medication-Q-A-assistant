import json
import os
import numpy as np
import unicodedata
from pinecone import Pinecone
from dotenv import load_dotenv


def ascii_safe(text):
    text = str(text)
    text = unicodedata.normalize('NFKD', text)
    return text.encode('ascii', errors='replace').decode('ascii')

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_NAME = "medication-index"
BATCH_SIZE = 100


import os
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def load_existing_data(index_path=None, metadata_path=None):
    if index_path is None:
        index_path = os.path.join(DATA_DIR, "medication_faiss.index")
    if metadata_path is None:
        metadata_path = os.path.join(DATA_DIR, "metadata.json")
    import faiss
    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    vectors = index.reconstruct_n(0, index.ntotal)
    return vectors, metadata


def upload_vectors(vectors, metadata, index_name=INDEX_NAME, clear=False):
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(index_name)

    if clear:
        print("Clearing existing vectors...")
        index.delete(delete_all=True)

    vectors_list = vectors.tolist()
    total = len(vectors_list)

    print(f"Uploading {total} vectors to Pinecone...")

    for i in range(0, total, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, total)
        batch_vectors = []
        batch_meta = []

        for j in range(i, batch_end):
            m = metadata[j]
            content = ascii_safe(m.get("content", ""))
            if len(content) > 10000:
                content = content[:10000]
            chunk_id = f"{m.get('title', '')}_{m.get('section', '')}"
            batch_vectors.append({
                "id": str(j),
                "values": vectors_list[j],
                "metadata": {
                    "chunk_id": ascii_safe(chunk_id)[:200],
                    "title": ascii_safe(m.get("title", ""))[:500],
                    "section": ascii_safe(m.get("section", ""))[:100],
                    "content": content,
                    "intent": ascii_safe(m.get("intent", ""))[:100],
                    "url": ascii_safe(m.get("url", ""))[:500],
                    "tags": ascii_safe(m.get("tags", "[]"))[:500],
                }
            })
            batch_meta.append(metadata[j])

        try:
            index.upsert(vectors=batch_vectors, namespace="")
            print(f"Uploaded {batch_end}/{total} vectors")
        except Exception as e:
            print(f"Error at batch {i}-{batch_end}: {e}")
            print(f"First vector metadata sample: {batch_vectors[0]['metadata']}")
            raise

    print("Done!")


def main():
    import sys
    clear = "--clear" in sys.argv
    
    print("Loading existing FAISS index...")
    vectors, metadata = load_existing_data()
    print(f"Loaded {len(vectors)} vectors")

    upload_vectors(vectors, metadata, clear=True)


if __name__ == "__main__":
    main()