# Medication Q&A Assistant

A RAG (Retrieval-Augmented Generation) assistant for answering questions about medications in Dutch. Built with FastAPI, Pinecone, and Groq LLM.

## Features

- Answers medication-related questions using Dutch medical data from [apotheek.nl](https://www.apotheek.nl)
- **Intent-aware retrieval** - detects questions about side effects, interactions, dosage, pregnancy, etc.
- **Medication-specific search** - matches medication names with relevant document sections
- Hybrid search (dense + sparse BM25 retrieval) with cross-encoder reranking
- Exact match pre-filtering for improved accuracy
- Conversational sessions with context memory
- Session-based disclaimer (shows only once per session)
- Simple web UI

## Tech Stack

- **Backend**: FastAPI
- **Vector DB**: Pinecone (with FAISS for local embeddings)
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **LLM**: Groq (Llama 3.1 8B)
- **Reranking**: CrossEncoder (ms-marco-MiniLM-L-6-v2)
- **Data Source**: Scraped from [apotheek.nl](https://www.apotheek.nl)

## Setup

1. Clone the repo and install dependencies:
```bash
pip install fastapi uvicorn sentence-transformers pinecone-client groq rank-bm25 nltk python-dotenv requests beautifulsoup4 faiss-cpu
```

2. Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

3. Scrape the dataset:
```bash
python dataset_scraper.py
```

4. Generate embeddings and upload to Pinecone:
```bash
python embed.py
python upload_to_pinecone.py
```

## Running

Start the API server:
```bash
uvicorn src.app:app --reload
```

Open `http://localhost:8000` in your browser.

## Usage

- **Web UI**: Visit the homepage and type medication questions
- **API**: 
  - `POST /session` - Create new session
  - `POST /answer` with `{"question": "...", "session_id": "..."}` 
  - `DELETE /session/{session_id}` - End session
- **CLI**: `python rag.py "your question here"`
