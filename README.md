# Medication Q&A Assistant

A RAG (Retrieval-Augmented Generation) assistant for answering questions about medications in Dutch and English. Built with FastAPI, FAISS/Pinecone, and Groq LLM.

## Features

- Answers medication-related questions using Dutch medical data from [apotheek.nl](https://www.apotheek.nl) and [farmacotherapeutischkompas.nl](https://www.farmacotherapeutischkompas.nl)
- **Bilingual support** - answers in both Dutch and English with intent detection for both languages
- **Intent-aware retrieval** - detects questions about side effects, interactions, dosage, pregnancy, etc.
- **Medication-specific search** - matches medication names with relevant document sections
- **Local or cloud vector DB** - run with local FAISS or Pinecone (toggle via `USE_PINECONE` env var)
- Hybrid search (dense + sparse BM25 retrieval) with cross-encoder reranking
- Exact match pre-filtering for improved accuracy
- Conversational sessions with context memory and follow-up question handling
- Session-based disclaimer (shows only once per session)
- Token usage tracking per response
- Retry logic for LLM API failures
- Simple web UI

## Tech Stack

- **Backend**: FastAPI
- **Vector DB**: Local FAISS or Pinecone (toggle via env var)
- **Embeddings**: SentenceTransformers (BAAI/bge-m3, 1024-dim)
- **LLM**: Groq (openai/gpt-oss-20b)
- **Reranking**: CrossEncoder (ms-marco-MiniLM-L-6-v2)
- **BM25**: rank-bm25 for sparse keyword retrieval
- **Data Sources**: Scraped from [apotheek.nl](https://www.apotheek.nl) and [farmacotherapeutischkompas.nl](https://www.farmacotherapeutischkompas.nl)

## Setup

1. Clone the repo and install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file (see `.env.example`):
```
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
USE_PINECONE=true
```
Set `USE_PINECONE=false` to use the local FAISS index instead of Pinecone.

3. Scrape the dataset:
```bash
python scripts/dataset_scraper.py
```

4. Generate embeddings:
```bash
python scripts/embed.py
```
The embedding script supports checkpointing - if interrupted, re-running will resume from where it left off.

5. (Optional) Upload to Pinecone:
```bash
python scripts/upload_to_pinecone.py --create-index
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
  - `POST /session` - Create new session (accepts `language` parameter: `"nl"` or `"en"`)
  - `POST /answer` with `{"question": "...", "session_id": "..."}` 
  - `DELETE /session/{session_id}` - End session
- **CLI**: `python src/rag.py "your question here"`
