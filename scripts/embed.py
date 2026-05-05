import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = os.path.join(DATA_DIR, "medication_faiss.index")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")


import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def load_chunks(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "apotheek_dataset.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_embeddings(chunks):
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [chunk["search_text"] for chunk in chunks]
    embeddings = model.encode(texts)
    return np.array(embeddings).astype("float32")


def build_index(embeddings, chunks):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    metadata = [
        {
            "title": chunk["title"],
            "section": chunk["section"],
            "content": chunk["content"],
            "intent": chunk["intent"],
            "url": chunk["url"],
            "tags": chunk["tags"],
        }
        for chunk in chunks
    ]

    return index, metadata


def save_index(index, metadata, index_path=INDEX_PATH, metadata_path=METADATA_PATH):
    faiss.write_index(index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main():
    print("Loading chunks...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    print("Creating embeddings...")
    embeddings = create_embeddings(chunks)

    print("Building FAISS index...")
    index, metadata = build_index(embeddings, chunks)

    print(f"Saving index to {INDEX_PATH}...")
    save_index(index, metadata)
    print("Done")


if __name__ == "__main__":
    main()
