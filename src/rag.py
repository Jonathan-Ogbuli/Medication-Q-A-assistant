import json
import os
import uuid
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import nltk

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
INDEX_NAME = "medication-index"

USE_PINECONE = os.environ.get("USE_PINECONE", "false").strip().lower() in ("true", "1", "yes")

GROQ_MODEL = "llama-3.1-8b-instant" # faster
# GROQ_MODEL = "llama-3.3-70b-versatile" # more reasoning

# Module-level caches (lazy-loaded, auto-refreshed via mtime)
bm25_index = None
dataset = None
reranker = None
embedding_model = None
faiss_index_obj = None
faiss_metadata = None
pinecone_client = None
pinecone_index = None

# File mtime tracking for auto-reload
_file_mtimes = {
    "dataset": 0,
    "faiss_index": 0,
    "faiss_metadata": 0,
}
_bm25_dataset_len = 0

conversations = {}
session_languages = {}

DISCLAIMER_TEXT_NL = """**Belangrijk:**
- Dit is een AI, geen vervanging voor professioneel medisch advies, diagnose of behandeling.
- Deze informatie is alleen bedoeld voor educatieve of informatieve doeleinden.
- De informatie kan onvolledig, verouderd of onjuist zijn.
- Bij een medische noodsituatie: bel direct 112 of ga naar de spoedeisende hulp.

"""

DISCLAIMER_TEXT_EN = """**Important:**
- This is an AI, not a substitute for professional medical advice, diagnosis, or treatment.
- This information is for educational and informational purposes only.
- The information may be incomplete, outdated, or incorrect.
- In case of a medical emergency: call 112 immediately or go to the emergency room.

"""

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

DATASET_PATH = os.path.join(DATA_DIR, "combined_dataset.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "medication_faiss.index")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")


def get_session(session_id=None, language="nl"):
    if session_id is None:
        session_id = str(uuid.uuid4())

    if session_id not in conversations:
        conversations[session_id] = []
        session_languages[session_id] = language

    return session_id, conversations[session_id]


def create_session(language="nl"):
    session_id = str(uuid.uuid4())
    conversations[session_id] = []
    session_languages[session_id] = language
    return session_id


def end_session(session_id):
    if session_id in conversations:
        del conversations[session_id]
    if session_id in session_languages:
        del session_languages[session_id]
    return True


def get_pinecone_index():
    global pinecone_client, pinecone_index
    if pinecone_client is None:
        try:
            from pinecone import Pinecone
            pinecone_client = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
            pinecone_index = pinecone_client.Index(INDEX_NAME)
        except Exception:
            pinecone_client = None
            pinecone_index = None
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
    global dataset, _file_mtimes
    current_mtime = os.path.getmtime(DATASET_PATH) if os.path.exists(DATASET_PATH) else 0

    if dataset is None or current_mtime != _file_mtimes["dataset"]:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        _file_mtimes["dataset"] = current_mtime

    return dataset


def init_bm25():
    global bm25_index, _bm25_dataset_len

    dataset = load_dataset()

    if bm25_index is not None and len(dataset) == _bm25_dataset_len:
        return bm25_index

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

    tokenized_docs = []
    for chunk in dataset:
        text = chunk.get("search_text", "")
        tokens = nltk.word_tokenize(text.lower())
        tokenized_docs.append(tokens)

    bm25_index = BM25Okapi(tokenized_docs)
    _bm25_dataset_len = len(dataset)
    return bm25_index


def get_local_faiss():
    global faiss_index_obj, faiss_metadata, _file_mtimes

    index_mtime = os.path.getmtime(FAISS_INDEX_PATH) if os.path.exists(FAISS_INDEX_PATH) else 0
    meta_mtime = os.path.getmtime(METADATA_PATH) if os.path.exists(METADATA_PATH) else 0

    if (faiss_index_obj is None or faiss_metadata is None or
        index_mtime != _file_mtimes["faiss_index"] or
        meta_mtime != _file_mtimes["faiss_metadata"]):

        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
            faiss_index_obj = faiss.read_index(FAISS_INDEX_PATH)
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                faiss_metadata = json.load(f)
            _file_mtimes["faiss_index"] = index_mtime
            _file_mtimes["faiss_metadata"] = meta_mtime
        else:
            faiss_index_obj = None
            faiss_metadata = None

    return faiss_index_obj, faiss_metadata


def local_dense_search(query_embedding, top_k=50):
    index, metadata = get_local_faiss()
    if index is None or metadata is None:
        return []

    query_np = np.array([query_embedding], dtype="float32")
    distances, indices = index.search(query_np, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        meta = metadata[idx]
        results.append({
            "id": str(idx),
            "score": float(distances[0][i]),
            "title": meta.get("title", ""),
            "section": meta.get("section", ""),
            "content": meta.get("content", ""),
            "intent": meta.get("intent", ""),
            "url": meta.get("url", ""),
            "tags": meta.get("tags", "[]"),
        })
    return results


def embed_query(query):
    model = get_embedding_model()
    embedding = model.encode([query]).astype("float32")
    return embedding[0].tolist()


def extract_medication_and_intent(query, language="nl", conversation_history=None):
    """Extract medication name and intent from query"""
    query_lower = query.lower()

    en_intents = {
        'side effects': 'bijwerkingen',
        'adverse': 'bijwerkingen',
        'what are': 'bijwerkingen',
        'interactions': 'interacties',
        'interaction': 'interacties',
        'contraindication': 'interacties',
        'warning': 'interacties',
        'alcohol': 'interacties',
        'combine with': 'interacties',
        'dosage': 'vergeten',
        'dose': 'vergeten',
        'missed': 'vergeten',
        'overdose': 'vergeten',
        'what is': 'wat_is_het',
        'what does': 'wat_is_het',
        'how does': 'wat_is_het',
        'tell me about': 'wat_is_het',
        'describe': 'wat_is_het',
        'explain': 'wat_is_het',
        'information about': 'wat_is_het',
        'stop': 'stoppen',
        'stopping': 'stoppen',
        'withdrawal': 'stoppen',
        'discontinuation': 'stoppen',
        'discontinue': 'stoppen',
        'pregnancy': 'zwangerschap',
        'pregnant': 'zwangerschap',
        'breastfeeding': 'borstvoeding',
        'nursing': 'borstvoeding',
        'driving': 'rijvaardigheid',
        'drive': 'rijvaardigheid',
        'drowsy': 'rijvaardigheid',
    }

    nl_intents = {
        'bijwerkingen': 'bijwerkingen',
        'bijwerking': 'bijwerkingen',
        'interacties': 'interacties',
        'interactie': 'interacties',
        'wisselwerking': 'interacties',
        'dosering': 'vergeten',
        'inname': 'vergeten',
        'hoeveel': 'vergeten',
        'hoe werkt': 'wat_is_het',
        'wat is': 'wat_is_het',
        'werking': 'wat_is_het',
        'stoppen': 'stoppen',
        'stop': 'stoppen',
        'zwangerschap': 'zwangerschap',
        'borstvoeding': 'borstvoeding',
        'rijvaardigheid': 'rijvaardigheid',
        'rijden': 'rijvaardigheid',
        'autorijden': 'rijvaardigheid',
        'sap': 'interacties',
        'drank': 'interacties',
        'combinatie': 'interacties',
        'waarschuwing': 'interacties',
        'contra': 'interacties',
    }

    detected_intent = None
    intents = en_intents if language == "en" else nl_intents
    for key, section in intents.items():
        if key in query_lower:
            detected_intent = section
            break

    dataset = load_dataset()
    known_meds = {}
    for d in dataset:
        title = d.get('title', '').lower()
        if title:
            known_meds[title] = title
            first_word = title.split()[0]
            if first_word not in known_meds:
                known_meds[first_word] = title

    detected_med = None
    for med_key in sorted(known_meds.keys(), key=len, reverse=True):
        if med_key in query_lower:
            detected_med = known_meds[med_key]
            break

    if (detected_med is None or detected_intent is None) and conversation_history:
        for msg in reversed(conversation_history):
            if msg["role"] == "user":
                prev_query = msg["content"]
                for med_key in sorted(known_meds.keys(), key=len, reverse=True):
                    if med_key in prev_query.lower():
                        if detected_med is None:
                            detected_med = known_meds[med_key]
                        break
                for key, section in intents.items():
                    if key in prev_query.lower():
                        if detected_intent is None:
                            detected_intent = section
                        break
                if detected_med is not None:
                    break

    return detected_intent, detected_med


def retrieve(query, top_k=5, dense_k=50, bm25_k=50, language="nl", conversation_history=None):
    bm25 = init_bm25()
    dataset = load_dataset()

    intent, med_name = extract_medication_and_intent(query, language, conversation_history)

    exact_matches = []
    if med_name and intent:
        for i, chunk in enumerate(dataset):
            if (med_name.lower() in chunk.get('title', '').lower() and
                chunk.get('section', '').lower() == intent):
                exact_matches.append((i, chunk))

    query_embedding = embed_query(query)

    # --- Dense retrieval: toggle between Pinecone and local FAISS ---
    if USE_PINECONE:
        dense_results = []
        dense_source = "pinecone"
        pine_index = get_pinecone_index()
        if pine_index is not None:
            try:
                pine_results = pine_index.query(
                    vector=query_embedding,
                    top_k=dense_k,
                    include_metadata=True,
                    include_values=False
                )
                for match in pine_results.matches:
                    meta = match.metadata
                    dense_results.append({
                        "id": match.id,
                        "score": match.score,
                        "title": meta.get("title", ""),
                        "section": meta.get("section", ""),
                        "content": meta.get("content", ""),
                        "intent": meta.get("intent", ""),
                        "url": meta.get("url", ""),
                        "tags": meta.get("tags", "[]"),
                    })
            except Exception as e:
                print(f"Pinecone query failed: {e}")
                dense_results = []
    else:
        dense_results = local_dense_search(query_embedding, top_k=dense_k)
        dense_source = "local_faiss"

    # --- Collect candidates with dataset-index-based keys (dedup across sources) ---
    candidates = {}
    for i, r in enumerate(dense_results):
        chunk_idx = r.get("id", str(i))
        unique_key = f"chunk_{chunk_idx}"
        if unique_key not in candidates:
            candidates[unique_key] = {
                "distance": r.get("score", 0),
                "title": r.get("title", ""),
                "section": r.get("section", ""),
                "content": r.get("content", ""),
                "intent": r.get("intent", ""),
                "url": r.get("url", ""),
                "tags": r.get("tags", "[]"),
                "source": dense_source,
            }

    # --- BM25 sparse retrieval ---
    tokenized_query = nltk.word_tokenize(query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:bm25_k]

    for idx in top_bm25_indices:
        if bm25_scores[idx] > 0:
            chunk = dataset[idx]
            unique_key = f"chunk_{idx}"
            if unique_key not in candidates:
                candidates[unique_key] = {
                    "distance": float(bm25_scores[idx]),
                    "title": chunk.get("title", ""),
                    "section": chunk.get("section", ""),
                    "content": chunk.get("content", ""),
                    "intent": chunk.get("intent", ""),
                    "url": chunk.get("url", ""),
                    "tags": chunk.get("tags", "[]"),
                    "source": "bm25"
                }

    # --- Exact matches with high base score ---
    for idx, chunk in exact_matches:
        unique_key = f"chunk_{idx}"
        if unique_key not in candidates:
            candidates[unique_key] = {
                "distance": 100.0,
                "title": chunk.get("title", ""),
                "section": chunk.get("section", ""),
                "content": chunk.get("content", ""),
                "intent": chunk.get("intent", ""),
                "url": chunk.get("url", ""),
                "tags": chunk.get("tags", "[]"),
                "source": "exact"
            }

    # --- Rerank with cross-encoder ---
    reranker = get_reranker()
    candidate_list = list(candidates.values())

    if candidate_list:
        pairs = [[query, c["content"]] for c in candidate_list]
        scores = reranker.predict(pairs)

        for i, score in enumerate(scores):
            candidate_list[i]["rerank_score"] = float(score)

        if intent:
            for c in candidate_list:
                section_match = c.get("section", "").lower() == intent
                med_match = med_name and med_name in c.get("title", "").lower()

                exact_med_match = med_name and (c.get("title", "").lower().startswith(med_name.lower() + " ") or
                                                c.get("title", "").lower() == med_name.lower())

                if section_match and exact_med_match:
                    c["rerank_score"] += 20.0
                elif section_match and med_match:
                    c["rerank_score"] += 10.0
                elif section_match:
                    c["rerank_score"] += 5.0
                elif exact_med_match:
                    c["rerank_score"] += 3.0

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


def generate_response(query, retrieved_chunks, conversation_history=None, model=GROQ_MODEL, num_context=3, language="nl"):
    try:
        from groq import Groq

        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        # Use top num_context chunks for context
        context_chunks = retrieved_chunks[:num_context] if retrieved_chunks else []
        context_text = "\n\n---\n\n".join([f"[{c['title']} - {c['section']}]\n{c['content']}" for c in context_chunks])

        history_text = ""
        if conversation_history:
            history_text = "\n\nPrevious conversation:\n" if language == "en" else "\n\nEerdere conversatie:\n"
            for msg in conversation_history[-5:]:
                history_text += f"{msg['role']}: {msg['content']}\n"

        if language == "en":
            prompt = f"""You have information about medications. Use the context below to answer the question IN ENGLISH.

Context:
{context_text}

Instructions:
- If the context contains information about the medication mentioned in the question, use that information to answer the question IN ENGLISH.
- If the context does NOT contain information about the medication in the question, say IN ENGLISH: "I have no information about this in my database. Please consult a doctor or pharmacist for medical advice."
- Respond IN ENGLISH only, even if the context is in another language.
- Avoid using the word "context" in your answers.
- Keep answers short and concise, don't tell more than necessary.
- When asked about side effects, list them from most common to least common.
- When asked about medication not in the dataset or context, say: "I have no information about this in my database. Please consult a doctor or pharmacist for medical advice."
- Do not make up information.
- If the question is unclear or missing important details, ask a clarifying question instead of answering.
- If the user asked a previous question and now asks a follow-up question (like "what about X?" or "and the side effects?"), base your answer on the previous question.
- Use newlines to separate paragraphs for readability.

Question: {query}
Answer (in English):"""
        else:
            prompt = f"""Je hebt informatie over medicijnen. Gebruik de onderstaande context om de vraag te beantwoorden in het Nederlands.

Context:
{context_text}

Instructies:
- Als de context informatie bevat over het medicijn dat in de vraag wordt genoemd, gebruik die informatie dan om de vraag te beantwoorden in het Nederlands.
- Als de context GEEN informatie bevat over het medicijn uit de vraag, zeg dan in het Nederlands: "Ik heb geen informatie hierover in mijn database. Raadpleeg een arts of apotheker voor medisch advies."
- Antwoord alleen in het Nederlands, ook als de context in een andere taal is.
- Vermijd het gebruik van het woord "context" in je antwoorden.
- Houd de antwoorden kort en bondig, vertel niet meer dan nodig is.
- Als er gevraagd wordt naar bijwerkingen, noem deze op van meest voorkoment naar minst voorkomend.
- Als er gevraagd wordt naar medicatie die niet in de dataset of context voorkomt, zeg dan: "Ik heb geen informatie hierover in mijn database. Raadpleeg een arts of apotheker voor medisch advies."
- Verzin geen informatie.
- Als de vraag onduidelijk is of belangrijke details mist, stel dan een verduidelijkende vraag in plaats van te antwoorden.
- Als de gebruiker een eerdere vraag heeft gesteld en nu een vervolgvraag stelt (zoals "en hoe zit het met X?" of "en de bijwerkingen?"), ga dan uit van de eerdere vraag.
- Gebruik newlines om paragrafen te scheiden voor leesbaarheid.

Vraag: {query}
Antwoord (in het Nederlands):"""
        
        system_msg = "You are a medical information assistant. You ALWAYS respond in English." if language == "en" else "Je bent een medische informatie-assistent. Je antwoordt ALTIJD in het Nederlands."
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_msg},
            ] + conversation_history + [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            temperature=0.0,
            max_tokens=500,
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return None


def rag_query(query, top_k=5, use_llm=True, session_id=None, num_context=3, language="nl"):
    session_id, history = get_session(session_id, language)
    # Use session-stored language, overriding only if a non-default was explicitly passed
    session_language = session_languages.get(session_id, language)
    if language != "nl":
        session_language = language
    session_languages[session_id] = session_language
    results = retrieve(query=query, top_k=top_k, language=session_language, conversation_history=history)

    if use_llm:
        print("Generating response...")
        answer = generate_response(query, results, conversation_history=history, num_context=num_context, language=session_language)
        if answer:
            # Prepend disclaimer only for the first message in session
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": answer})
            # Prepend disclaimer only for the first assistant message in session
            if len(history) == 2:
                disclaimer = DISCLAIMER_TEXT_EN if session_language == "en" else DISCLAIMER_TEXT_NL
                answer = disclaimer + answer
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