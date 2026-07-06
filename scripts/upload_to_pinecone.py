import json
import os
import numpy as np
import unicodedata
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv


def ascii_safe(text):
    text = str(text)
    text = unicodedata.normalize('NFKD', text)
    return text.encode('ascii', errors='replace').decode('ascii')

load_dotenv()

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
INDEX_NAME = "medication-index"
BATCH_SIZE = 100


def ensure_index(pc, index_name=INDEX_NAME, dimension=EMBEDDING_DIM, metric="euclidean"):
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name in existing:
        info = pc.describe_index(index_name)
        if info.dimension == dimension and info.metric == metric:
            print(f"Index '{index_name}' already exists with dim={dimension}, metric='{metric}'")
            return
        print(f"Index '{index_name}' exists with dim={info.dimension}, metric='{info.metric}' "
              f"but we need dim={dimension}, metric='{metric}'. Deleting and recreating...")
        pc.delete_index(index_name)
    print(f"Creating index '{index_name}' with dim={dimension}, metric='{metric}'...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("Index ready.")

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


def upload_vectors(vectors, metadata, pc, index_name=INDEX_NAME, clear=False):
    index = pc.Index(index_name)

    if clear:
        print("Clearing existing vectors...")
        try:
            index.delete(delete_all=True)
        except Exception:
            print("  (nothing to clear — index is already empty)")

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
    create_index = "--create-index" in sys.argv

    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    if create_index:
        ensure_index(pc)

    print("Loading existing FAISS index...")
    vectors, metadata = load_existing_data()
    print(f"Loaded {len(vectors)} vectors with dimension {vectors.shape[1]}")

    upload_vectors(vectors, metadata, pc, clear=clear)


if __name__ == "__main__":
    main()