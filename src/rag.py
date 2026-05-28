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

# GROQ_MODEL = "llama-3.1-8b-instant" # faster
GROQ_MODEL = "llama-3.3-70b-versatile" # more reasoning

pinecone_client = None
pinecone_index = None
bm25_index = None
dataset = None
reranker = None
embedding_model = None

conversations = {}

DISCLAIMER_TEXT_NL = """**Belangrijk:**
- Deze informatie is alleen bedoeld voor educatieve of informatieve doeleinden en is geen vervanging voor professioneel medisch advies, diagnose of behandeling.
- De informatie kan onvolledig, verouderd of onjuist zijn.
- Bij een medische noodsituatie: bel direct 112 of ga naar de spoedeisende hulp.

"""

DISCLAIMER_TEXT_EN = """**Important:**
- This information is for educational and informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.
- The information may be incomplete, outdated, or incorrect.
- In case of a medical emergency: call 112 immediately or go to the emergency room.

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


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def load_dataset():
    global dataset
    if dataset is None:
        with open(os.path.join(DATA_DIR, "apotheek_dataset.json"), "r", encoding="utf-8") as f:
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


def extract_medication_and_intent(query, language="nl"):
    """Extract medication name and intent (bijwerkingen, interacties, etc.) from query"""
    query_lower = query.lower()
    
    # Common intents/sections in Dutch medication leaflets
    intents = {
        'bijwerkingen': 'bijwerkingen',
        'bijwerking': 'bijwerkingen',
        'side effects': 'bijwerkingen',
        'interacties': 'interacties',
        'interactie': 'interacties',
        'wisselwerking': 'interacties',
        'interactions': 'interacties',
        'dosering': 'vergeten',  # Dataset doesn't have dosering section, using vergeten
        'dosage': 'vergeten',
        'inname': 'vergeten',
        'hoeveel': 'vergeten',  # "hoeveel moet ik innemen"
        'hoe werkt': 'wat_is_het',
        'wat is': 'wat_is_het',
        'werking': 'wat_is_het',
        'stoppen': 'stoppen',
        'stop': 'stoppen',
        'discontinuation': 'stoppen',
        'zwangerschap': 'zwangerschap',
        'pregnancy': 'zwangerschap',
        'borstvoeding': 'borstvoeding',
        'breastfeeding': 'borstvoeding',
        'rijvaardigheid': 'rijvaardigheid',
        'rijden': 'rijvaardigheid',
        'autorijden': 'rijvaardigheid',
        'driving': 'rijvaardigheid',
        'what is': 'wat_is_het',
        'how does': 'wat_is_het',
        'what are': 'bijwerkingen',
        'tell me about': 'wat_is_het',
    }
    
    detected_intent = None
    for key, section in intents.items():
        if key in query_lower:
            detected_intent = section
            break
    
    # Try to extract medication name from dataset
    dataset = load_dataset()
    known_meds = {}
    for d in dataset:
        title = d.get('title', '').lower()
        # Store full title and its variations
        if title:
            # Add full title
            known_meds[title] = title
            # Add first word as fallback
            first_word = title.split()[0]
            if first_word not in known_meds:
                known_meds[first_word] = title
    
    detected_med = None
    # Sort by length (longest first) to match more specific names first
    for med_key in sorted(known_meds.keys(), key=len, reverse=True):
        if med_key in query_lower:
            detected_med = known_meds[med_key]  # Return the canonical title
            break
    
    return detected_intent, detected_med


def retrieve(query, top_k=5, dense_k=50, bm25_k=50, language="nl"):
    index = get_pinecone_index()
    bm25 = init_bm25()
    dataset = load_dataset()

    # Extract intent and medication name to boost relevant sections
    intent, med_name = extract_medication_and_intent(query, language)
    
    # Pre-filter: if we have both medication and intent, try to get exact matches first
    exact_matches = []
    if med_name and intent:
        for i, chunk in enumerate(dataset):
            if (med_name.lower() in chunk.get('title', '').lower() and 
                chunk.get('section', '').lower() == intent):
                exact_matches.append((i, chunk))
    
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
    
    # Add exact matches to candidates with a high base score
    for idx, chunk in exact_matches:
        chunk_id = chunk.get("chunk_id", chunk.get("title", "") + chunk.get("section", ""))
        if chunk_id not in candidates:  # Only add if not already present from dense/BM25
            candidates[chunk_id] = {
                "distance": 100.0,  # High base score for exact matches
                "title": chunk.get("title", ""),
                "section": chunk.get("section", ""),
                "content": chunk.get("content", ""),
                "intent": chunk.get("intent", ""),
                "url": chunk.get("url", ""),
                "tags": chunk.get("tags", "[]"),
                "source": "exact"
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

        # Boost chunks that match the detected intent section and medication
        if intent:
            for c in candidate_list:
                section_match = c.get("section", "").lower() == intent
                med_match = med_name and med_name in c.get("title", "").lower()
                
                # Exact title match (title starts with medication name as whole word)
                exact_med_match = med_name and (c.get("title", "").lower().startswith(med_name.lower() + " ") or 
                                                c.get("title", "").lower() == med_name.lower())
                
                if section_match and exact_med_match:
                    c["rerank_score"] += 20.0  # Extremely strong boost for exact matches
                elif section_match and med_match:
                    c["rerank_score"] += 10.0  # Very strong boost for both matches
                elif section_match:
                    c["rerank_score"] += 5.0  # Strong boost for section match
                elif exact_med_match:
                    c["rerank_score"] += 3.0  # Medium boost for exact medication match

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
- Always clearly state that you are an AI and not a doctor, nurse, or other medical professional.
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
- Geef altijd duidelijk aan dat je een AI bent en geen arts, verpleegkundige of andere medische professional.
- Gebruik newlines om paragrafen te scheiden voor leesbaarheid.

Vraag: {query}
Antwoord (in het Nederlands):"""
        
        chat_completion = client.chat.completions.create(
            messages=conversation_history + [
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
    session_id, history = get_session(session_id)
    results = retrieve(query=query, top_k=top_k, language=language)

    if use_llm:
        print("Generating response...")
        answer = generate_response(query, results, conversation_history=history, num_context=num_context, language=language)
        if answer:
            # Prepend disclaimer only for the first message in session
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": answer})
            # Prepend disclaimer only for the first assistant message in session
            if len(history) == 2: # This will be true only for the very first assistant message
                disclaimer = DISCLAIMER_TEXT_EN if language == "en" else DISCLAIMER_TEXT_NL
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