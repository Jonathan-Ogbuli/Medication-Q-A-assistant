import json
import os
import uuid
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import nltk
from collections import defaultdict

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
INDEX_NAME = "medication-index"

GROQ_MODEL = "llama-3.1-8b-instant"

pinecone_client = None
pinecone_index = None
bm25_index = None
dataset = None
reranker = None
embedding_model = None

conversations = {}

DISCLAIMER_TEXT = """**Belangrijk:**
- Deze informatie is alleen bedoeld voor educatieve of informatieve doeleinden en is geen vervanging voor professioneel medisch advies, diagnose of behandeling.
- De informatie kan onvolledig, verouderd of onjuist zijn.
- Bij een medische noodsituatie: bel direct 112 of ga naar de spoedeisende hulp.

"""


def get_session(session_id=None):
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    if session_id not in conversations:
        conversations[session_id] = []
    
    return session_id, conversations[session_id]


def create_session():
    session_id = str(uuid.uuid4())
    conversations[session_id] = []
    return session_id


def end_session(session_id):
    if session_id in conversations:
        del conversations[session_id]
    return True


def get_pinecone_index():
    global pinecone_client, pinecone_index
    if pinecone_client is None:
        pinecone_client = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        pinecone_index = pinecone_client.Index(INDEX_NAME)
    return pinecone_index


def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return embedding_model


def get_reranker():
    global reranker
    if reranker is None:
        reranker = CrossEncoder(RERANKER_MODEL)
    return reranker


def load_dataset():
    global dataset
    if dataset is None:
        with open("apotheek_dataset.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
    return dataset


def init_bm25():
    global bm25_index, dataset
    if bm25_index is not None:
        return bm25_index

    # Download NLTK data if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

    dataset = load_dataset()
    # Tokenize search_text for BM25
    tokenized_docs = []
    for chunk in dataset:
        text = chunk.get("search_text", "")
        tokens = nltk.word_tokenize(text.lower())
        tokenized_docs.append(tokens)

    bm25_index = BM25Okapi(tokenized_docs)
    return bm25_index


def embed_query(query):
    model = get_embedding_model()
    embedding = model.encode([query]).astype("float32")
    return embedding[0].tolist()


def retrieve(query, top_k=5, dense_k=20, bm25_k=20):
    index = get_pinecone_index()
    bm25 = init_bm25()
    dataset = load_dataset()

    query_embedding = embed_query(query)

    # Dense retrieval from Pinecone
    dense_results = index.query(
        vector=query_embedding,
        top_k=dense_k,
        include_metadata=True,
        include_values=False
    )

    # Collect dense candidates
    candidates = {}
    for match in dense_results.matches:
        meta = match.metadata
        chunk_id = meta.get("chunk_id", meta.get("title", "") + meta.get("section", ""))
        candidates[chunk_id] = {
            "distance": match.score,
            "title": meta.get("title", ""),
            "section": meta.get("section", ""),
            "content": meta.get("content", ""),
            "intent": meta.get("intent", ""),
            "url": meta.get("url", ""),
            "tags": meta.get("tags", "[]"),
            "source": "dense"
        }

    # BM25 sparse retrieval
    tokenized_query = nltk.word_tokenize(query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)

    # Get top BM25 results
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:bm25_k]

    for idx in top_bm25_indices:
        if bm25_scores[idx] > 0:
            chunk = dataset[idx]
            chunk_id = chunk.get("chunk_id", chunk.get("title", "") + chunk.get("section", ""))
            if chunk_id not in candidates:
                candidates[chunk_id] = {
                    "distance": float(bm25_scores[idx]),
                    "title": chunk.get("title", ""),
                    "section": chunk.get("section", ""),
                    "content": chunk.get("content", ""),
                    "intent": chunk.get("intent", ""),
                    "url": chunk.get("url", ""),
                    "tags": chunk.get("tags", "[]"),
                    "source": "bm25"
                }

    # Rerank candidates using cross-encoder
    reranker = get_reranker()
    candidate_list = list(candidates.values())

    if candidate_list:
        # Prepare pairs for reranking
        pairs = [[query, c["content"]] for c in candidate_list]
        scores = reranker.predict(pairs)

        # Sort by reranker scores
        for i, score in enumerate(scores):
            candidate_list[i]["rerank_score"] = float(score)

        candidate_list.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidate_list[:top_k]

    return []


def build_prompt(query, retrieved_chunks):
    ctx = retrieved_chunks[0]["content"][:80]
    return f"{ctx} V: {query}"


def format_retrieved_results(results):
    output = "## Relevante informatie\n\n"
    for i, r in enumerate(results, 1):
        source = r.get("source", "unknown")
        score = r.get("rerank_score", r.get("distance", 0))
        output += f"**{i}. {r['title']}** ({r['section']}) - {source} (score: {score:.3f})\n"
        output += f"{r['content']}\n\n"
    output += "*Raadpleeg altijd een arts voor medisch advies.*"
    return output


def generate_response(query, retrieved_chunks, conversation_history=None, model=GROQ_MODEL, num_context=3):
    try:
        from groq import Groq

        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        # Use top num_context chunks for context
        context_chunks = retrieved_chunks[:num_context] if retrieved_chunks else []
        context_text = "\n\n---\n\n".join([f"[{c['title']} - {c['section']}]\n{c['content']}" for c in context_chunks])

        history_text = ""
        if conversation_history:
            history_text = "\n\nEerdere conversatie:\n"
            for msg in conversation_history[-5:]:
                history_text += f"{msg['role']}: {msg['content']}\n"

        prompt = f"""Je hebt informatie over medicijnen. Gebruik de onderstaande context om de vraag te beantwoorden.

Context:
{context_text}
{history_text}

Instructies:
- Als de context informatie bevat over het medicijn dat in de vraag wordt genoemd, gebruik die informatie dan om de vraag te beantwoorden.
- Als de context GEEN informatie bevat over het medicijn uit de vraag, zeg dan: "Ik heb geen informatie hierover in mijn database. Raadpleeg een arts of apotheker voor medisch advies."
- Vermijd het gebruik van het woord "context" in je antwoorden.
- Houd de antwoorden kort en bondig.
- Als er gevraagd wordt naar bijwerkingen, noem deze op van meest voorkoment naar minst voorkomend.
- Als er gevraagd wordt naar medicatie die niet in de dataset of context voorkomt, zeg dan: "Ik heb geen informatie hierover in mijn database. Raadpleeg een arts of apotheker voor medisch advies."
- Verzin geen informatie.
- Als de vraag onduidelijk is of belangrijke details mist, stel dan een verduidelijkende vraag in plaats van te antwoorden.
- Als de gebruiker een eerdere vraag heeft gesteld en nu een vervolgvraag stelt (zoals "en hoe zit het met X?" of "en de bijwerkingen?"), ga dan uit van de eerdere vraag.
- Geef altijd duidelijk aan dat je een AI bent en geen arts, verpleegkundige of andere medische professional.

Vraag: {query}
Antwoord:"""
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            temperature=0.0,
            max_tokens=400,
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return None


def rag_query(query, top_k=5, use_llm=True, session_id=None, num_context=3):
    session_id, history = get_session(session_id)
    results = retrieve(query=query, top_k=top_k)

    if use_llm:
        print("Generating response...")
        conversation_history = [{"role": "user", "content": query}] if history else None
        answer = generate_response(query, results, conversation_history=history, num_context=num_context)
        if answer:
            # Prepend disclaimer only for the first message in session
            if len(history) == 0:
                answer = DISCLAIMER_TEXT + answer
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": answer})
            return {"results": results, "answer": answer, "session_id": session_id}

    formatted = format_retrieved_results(results)
    return {"results": results, "answer": formatted, "session_id": session_id}


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter query: ")
    
    use_llm = "--no-llm" not in sys.argv
    top_k = 10
    
    for arg in sys.argv:
        if arg.startswith("--top_k="):
            top_k = int(arg.split("=")[1])
    
    result = rag_query(query, top_k=top_k, use_llm=use_llm)
    
    if "answer" in result:
        print(result["answer"])