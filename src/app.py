from fastapi import FastAPI
from pydantic import BaseModel

from chat_request import get_response

app = FastAPI(title="rag-flow")


class ChatRequest(BaseModel):
    question: str
    chat_history: list = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(request: ChatRequest):
    return get_response(request.question, request.chat_history)
