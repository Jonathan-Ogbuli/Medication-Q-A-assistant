import json
import os
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


def retrieve(query, top_k=5):
    index = get_pinecone_index()
    
    query_embedding = embed_query(query)
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )
    
    retrieved_results = []
    for match in results.matches:
        meta = match.metadata
        retrieved_results.append({
            "distance": match.score,
            "title": meta.get("title", ""),
            "section": meta.get("section", ""),
            "content": meta.get("content", ""),
            "intent": meta.get("intent", ""),
            "url": meta.get("url", ""),
            "tags": json.loads(meta.get("tags", "[]")),
        })
    
    return retrieved_results


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


def generate_response(query, retrieved_chunks, model=GROQ_MODEL):
    try:
        from groq import Groq
        
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        
        context = "\n\n".join([r["content"] for r in retrieved_chunks])
        
        prompt = f"""Je hebt informatie over medicijnen. Gebruik de onderstaande context om de vraag te beantwoorden.

Context:
{context}

Instructies:
- Als de context informatie bevat over het medicijn dat in de vraag wordt genoemd, gebruik die informatie dan om de vraag te beantwoorden.
- Als de context GEEN informatie bevat over het medicijn uit de vraag, zeg dan: "Ik heb geen informatie hierover in mijn database. Raadpleeg een arts of apotheker voor medisch advies."
- Als er gevraagd wordt naar medicatie die niet in de dataset of context voorkomt, zeg dan: "Ik heb geen informatie hierover in mijn database. Raadpleeg een arts of apotheker voor medisch advies."
- Verzin geen informatie.
- Geef altijd duidelijk aan dat je een AI bent en geen arts, verpleegkundige of andere medische professional.
- Vermeld expliciet dat de gegeven informatie alleen bedoeld is voor educatieve of informatieve doeleinden en geen vervanging is voor professioneel medisch advies, diagnose of behandeling.
- Geef aan dat de informatie mogelijk onvolledig, verouderd of onjuist kan zijn.
- Adviseer gebruikers om bij een medische noodsituatie onmiddellijk contact op te nemen met hulpdiensten of naar de spoedeisende hulp te gaan.

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


def rag_query(query, top_k=5, use_llm=True):
    results = retrieve(query=query, top_k=top_k)
    # print(f"Retrieving for query: {query}")
    
    # print(f"Found {len(results)} results:")
    # for i, r in enumerate(results):
    #     print(f"  {i+1}. [{r['section']}] {r['title']} (distance: {r['distance']:.2f})")
    
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
    top_k = 5
    
    for arg in sys.argv:
        if arg.startswith("--top_k="):
            top_k = int(arg.split("=")[1])
    
    result = rag_query(query, top_k=top_k, use_llm=use_llm)
    
    if "answer" in result:
        # print("\n" + "="*50)
        # print("ANTWOORD:")
        # print("="*50)
        print(result["answer"])