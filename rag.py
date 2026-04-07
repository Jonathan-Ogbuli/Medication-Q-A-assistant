import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "medication_faiss.index"
METADATA_PATH = "metadata.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "tinyllama"


def load_index(index_path=INDEX_PATH, metadata_path=METADATA_PATH):
    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def embed_query(query):
    model = SentenceTransformer(EMBEDDING_MODEL)
    embedding = model.encode([query]).astype("float32")
    return embedding


def retrieve(query, top_k=3, index=None, metadata=None):
    if index is None or metadata is None:
        index, metadata = load_index()
    
    query_embedding = embed_query(query)
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:
            results.append({
                "distance": float(dist),
                "title": metadata[idx]["title"],
                "section": metadata[idx]["section"],
                "content": metadata[idx]["content"],
                "intent": metadata[idx]["intent"],
                "url": metadata[idx]["url"],
                "tags": metadata[idx]["tags"],
            })
    
    return results


def build_prompt(query, retrieved_chunks):
    ctx = retrieved_chunks[0]["content"][:80]
    return f"{ctx} V: {query}"


def format_retrieved_results(results):
    output = "## Relevante informatie\n\n"
    for i, r in enumerate(results, 1):
        output += f"**{i}. {r['title']}** ({r['section']})\n"
        output += f"{r['content']}\n\n"
    output += "*Raadpleeg altijd een arts voor medisch advies.*"
    return output


def generate_response(query, retrieved_chunks, model=OLLAMA_MODEL, url=OLLAMA_URL):
    try:
        prompt = build_prompt(query, retrieved_chunks)
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.1,
                "num_ctx": 256,
                "repeat_penalty": 1.0,
                "num_predict": 60
            }
        }
        
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        return response.json()["response"]
    except requests.exceptions.ConnectionError:
        print("Error: Ollama is not running. Start it with: ollama serve")
        print("Then pull the model: ollama pull phi3")
        return None
    except Exception as e:
        print(f"Error generating response: {e}")
        return None


def rag_query(query, top_k=3, use_llm=True):
    print(f"Retrieving for query: {query}")
    results = retrieve(query, top_k=top_k)
    
    print(f"Found {len(results)} results:")
    for i, r in enumerate(results):
        print(f"  {i+1}. [{r['section']}] {r['title']} (distance: {r['distance']:.2f})")
    
    if use_llm:
        print("Generating response...")
        answer = generate_response(query, results)
        if answer:
            return {"results": results, "answer": answer}
    
    formatted = format_retrieved_results(results)
    return {"results": results, "answer": formatted}


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter query: ")
    
    use_llm = "--no-llm" not in sys.argv
    top_k = 3
    
    for arg in sys.argv:
        if arg.startswith("--top_k="):
            top_k = int(arg.split("=")[1])
    
    result = rag_query(query, top_k=top_k, use_llm=use_llm)
    
    if "answer" in result:
        print("\n" + "="*50)
        print("ANTWOORD:")
        print("="*50)
        print(result["answer"])