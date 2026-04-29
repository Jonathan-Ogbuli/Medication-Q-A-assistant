# Medication Q&A Assistant (In Development)

A RAG (Retrieval-Augmented Generation) assistant for answering questions about medications in Dutch. Built with FastAPI, Pinecone, and Groq LLM.

## Features

- Answers medication-related questions using Dutch medical data
- Hybrid search (dense + sparse BM25 retrieval)
- Cross-encoder reranking for better results
- Conversational sessions with context memory
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
uvicorn app:app --reload
```

Open `http://localhost:8000` in your browser.

## Usage

- **Web UI**: Visit the homepage and type medication questions
- **API**: `POST /answer` with `{"question": "..."}`
- **CLI**: `python rag.py "your question here"`
