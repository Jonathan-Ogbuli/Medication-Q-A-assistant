from fastapi import FastAPI
from pydantic import BaseModel
from rag import rag_query

app = FastAPI()


class Question(BaseModel):
    question: str
    top_k: int = 5
    use_llm: bool = True


class Answer(BaseModel):
    answer: str | None
    results: list


@app.post("/answer", response_model=Answer)
def ask_question(q: Question):
    result = rag_query(q.question, top_k=q.top_k, use_llm=q.use_llm)
    return Answer(answer=result.get("answer"), results=result.get("results"))


@app.get("/health")
def health_check():
    return {"status": "ok"}