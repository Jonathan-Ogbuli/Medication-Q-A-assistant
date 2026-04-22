import json
import os
import uuid
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_NAME = "medication-index"

GROQ_MODEL = "llama-3.1-8b-instant"

pinecone_client = None
pinecone_index = None

conversations = {}


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


def embed_query(query):
    model = SentenceTransformer(EMBEDDING_MODEL)
    embedding = model.encode([query]).astype("float32")
    return embedding[0].tolist()


def extract_medication_names(query):
    import re
    query_lower = query.lower()
    words = query_lower.split()
    
    found = []
    for word in words:
        cleaned = re.sub(r'[^a-z]', '', word)
        if len(cleaned) > 4:
            found.append(cleaned)
    
    return found


def retrieve(query, top_k=10):
    index = get_pinecone_index()
    query_embedding = embed_query(query)
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k * 2,
        include_metadata=True,
        include_values=False
    )
    
    medication_words = extract_medication_names(query)
    
    retrieved_results = []
    for match in results.matches:
        meta = match.metadata
        title_lower = meta.get("title", "").lower()
        content_lower = meta.get("content", "").lower()
        
        is_relevant = True
        if medication_words:
            found_in_result = False
            for med_word in medication_words:
                if med_word in title_lower or med_word in content_lower:
                    found_in_result = True
                    break
            is_relevant = found_in_result
        
        retrieved_results.append({
            "distance": match.score,
            "title": meta.get("title", ""),
            "section": meta.get("section", ""),
            "content": meta.get("content", ""),
            "intent": meta.get("intent", ""),
            "url": meta.get("url", ""),
            "tags": meta.get("tags", "[]"),
            "is_relevant": is_relevant,
        })
    
    relevant_results = [r for r in retrieved_results if r["is_relevant"]]
    
    if len(relevant_results) < 3:
        relevant_results = retrieved_results[:top_k]
    else:
        relevant_results = relevant_results[:top_k]
    
    if not relevant_results:
        relevant_results = retrieved_results[:top_k]
    
    return relevant_results


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


def generate_response(query, retrieved_chunks, conversation_history=None, model=GROQ_MODEL):
    try:
        from groq import Groq
        
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        
        context_parts = []
        for chunk in retrieved_chunks[:3]:
            source = f"Source: {chunk.get('title', 'Unknown')} - {chunk.get('section', 'Unknown')}"
            content = chunk["content"]
            context_parts.append(f"{source}\n{content}")
        
        context = "\n\n---\n\n".join(context_parts) if context_parts else ""
        
        history_text = ""
        if conversation_history:
            history_text = "\n\nEerdere conversatie:\n"
            for msg in conversation_history[-5:]:
                history_text += f"{msg['role']}: {msg['content']}\n"
        
        prompt = f"""Je beantwoordt vragen over medicijnen ALLEEN op basis van de onderstaande bronnen.

Context:
{context}
{history_text}

Instructies:
- ALS DE VRAAG GAAT OVER [medicijnnaam]: gebruik alleen informatie uit de bronnen die over dat specifieke medicijn gaan. NOOIT informatie van andere medicijnen gebruiken.
- Als geen van de bronnen over het medicijn uit de vraag gaat, zeg dan: "Ik heb geen informatie over [medicijnnaam] in mijn database. Raadpleeg een arts of apotheker voor medisch advies."
- NOOIT informatie bedenken of afleiden die niet expliciet in de bronnen staat. Geen vergelijkingen maken met andere medicijnen.
- NOOIT verwijzen naar andere medicijnen uit de bronnen als die niet relevant zijn voor de vraag.
- Als de bronnen wel over het medicijn gaan maar geen informatie over het specifieke onderwerp bevatten, zeg dan: "Ik heb geen specifieke informatie hierover in mijn database over [medicijnnaam]."
- Citeer de bron door de exacte sectienaam te noemen (bijv. "Volgens [sectienaam]...").
- Geef altijd duidelijk aan dat je een AI bent en geen arts, verpleegkundige of andere medische professional.
- Vermeld expliciet dat de gegeven informatie alleen bedoeld is voor educatieve of informatieve doeleinden en geen vervanging is voor professioneel medisch advies.
- Adviseer gebruikers om bij een medische noodsituatie onmiddellijk contact op te nemen met hulpdiensten.

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
            max_tokens=500,
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return None


def rag_query(query, top_k=5, use_llm=True, session_id=None):
    if top_k < 3:
        top_k = 3
    session_id, history = get_session(session_id)
    results = retrieve(query=query, top_k=top_k)
    
    if use_llm:
        print("Generating response...")
        conversation_history = [{"role": "user", "content": query}] if history else None
        answer = generate_response(query, results, conversation_history=history)
        if answer:
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
    top_k = 5
    
    for arg in sys.argv:
        if arg.startswith("--top_k="):
            top_k = int(arg.split("=")[1])
    
    result = rag_query(query, top_k=top_k, use_llm=use_llm)
    
    if "answer" in result:
        print(result["answer"])