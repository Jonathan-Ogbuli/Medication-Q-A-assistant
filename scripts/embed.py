import json
import numpy as np
import faiss
import os
import gc
import torch
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

EMBEDDING_MODEL = "BAAI/bge-m3"
INDEX_PATH = os.path.join(DATA_DIR, "medication_faiss.index")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")
CHECKPOINT_PATH = os.path.join(DATA_DIR, "embed_checkpoint.json")

BATCH_SIZE = 8
CHECKPOINT_INTERVAL = 10


def load_chunks(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "combined_dataset.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            return json.load(f)
    return None


def save_checkpoint(state):
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, CHECKPOINT_PATH)


def build_metadata_for_chunks(chunks):
    return [
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


def save_index_and_metadata(index, metadata, index_path, metadata_path):
    tmp_idx = index_path + ".tmp"
    tmp_meta = metadata_path + ".tmp"
    faiss.write_index(index, tmp_idx)
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    os.replace(tmp_idx, index_path)
    os.replace(tmp_meta, metadata_path)


def main():
    # Limit CPU threads to avoid oversubscription
    torch.set_num_threads(min(8, os.cpu_count() or 4))

    print("Loading chunks...")
    chunks = load_chunks()
    total = len(chunks)
    print(f"Loaded {total} chunks")

    checkpoint = load_checkpoint()

    # Determine embedding dimension by briefly loading the model
    print(f"Loading model {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    model.eval()
    dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {dim}")

    # Resume from checkpoint or start fresh
    if checkpoint and os.path.exists(INDEX_PATH):
        print(f"Resuming from checkpoint (batch {checkpoint.get('batch_index', 0)}, "
              f"{checkpoint.get('processed_count', 0)} chunks done)")
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        start_idx = checkpoint["processed_count"]
        batch_idx = checkpoint["batch_index"] + 1
    else:
        # Clean up stale temp/checkpoint files
        for p in [CHECKPOINT_PATH, INDEX_PATH + ".tmp", METADATA_PATH + ".tmp",
                  INDEX_PATH, METADATA_PATH]:
            if os.path.exists(p):
                os.remove(p)
        start_idx = 0
        batch_idx = 0
        index = faiss.IndexFlatL2(dim)
        metadata = []

    texts = [chunk["search_text"] for chunk in chunks]

    # Process remaining batches
    for i in range(start_idx, total, BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]

        with torch.no_grad():
            batch_emb = model.encode(batch_texts, show_progress_bar=False)
        batch_emb = np.ascontiguousarray(batch_emb.astype("float32"))
        index.add(batch_emb)

        for chunk in chunks[i:i + BATCH_SIZE]:
            metadata.append({
                "title": chunk["title"],
                "section": chunk["section"],
                "content": chunk["content"],
                "intent": chunk["intent"],
                "url": chunk["url"],
                "tags": chunk["tags"],
            })

        processed = min(i + BATCH_SIZE, total)
        print(f"  Encoded {processed}/{total} chunks (batch {batch_idx})")

        # Save checkpoint periodically
        if (batch_idx + 1) % CHECKPOINT_INTERVAL == 0 or processed >= total:
            print(f"  Saving checkpoint ({processed} chunks)...")
            save_index_and_metadata(index, metadata, INDEX_PATH, METADATA_PATH)
            save_checkpoint({"processed_count": processed, "batch_index": batch_idx})

        batch_idx += 1

        # Free memory
        del batch_emb
        if batch_idx % 2 == 0:
            gc.collect()

    # Clean up checkpoint on success
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    # Final save (already done above on last batch, but ensure it's written)
    save_index_and_metadata(index, metadata, INDEX_PATH, METADATA_PATH)
    print(f"Done — saved {len(metadata)} vectors to {INDEX_PATH}")


if __name__ == "__main__":
    main()
