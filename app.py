import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rag import rag_query, create_session, end_session, get_session

app = FastAPI()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Static files not found"}


class Question(BaseModel):
    question: str
    top_k: int = 5
    use_llm: bool = True
    session_id: str | None = None


class Answer(BaseModel):
    answer: str | None
    results: list
    session_id: str


class SessionResponse(BaseModel):
    session_id: str


@app.post("/session", response_model=SessionResponse)
def create_new_session():
    session_id = create_session()
    return SessionResponse(session_id=session_id)


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    end_session(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.post("/answer", response_model=Answer)
def ask_question(q: Question):
    result = rag_query(q.question, top_k=q.top_k, use_llm=q.use_llm, session_id=q.session_id)
    return Answer(
        answer=result.get("answer"), 
        results=result.get("results"),
        session_id=result.get("session_id")
    )


@app.get("/health")
def health_check():
    return {"status": "ok"}