"""RAG API service: ingest -> chunk -> embed -> retrieve -> answer.

Documents are chunked and embedded, then a question is answered using
the most relevant retrieved chunks as context for the LLM.
"""

import os

from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI(title="rag-api-service")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

CHUNKS = []


class Document(BaseModel):
    title: str
    text: str


class Question(BaseModel):
    question: str


def chunk_text(text, size=500):
    return [text[i:i + size] for i in range(0, len(text), size)]


def embed(text):
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


def retrieve(question, top_k=3):
    query = embed(question)
    scored = []
    for chunk in CHUNKS:
        score = sum(a * b for a, b in zip(query, chunk["embedding"]))
        scored.append((score, chunk["text"]))
    scored.sort(reverse=True)
    return [text for _, text in scored[:top_k]]


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(CHUNKS)}


@app.post("/ingest")
def ingest(doc: Document):
    for chunk in chunk_text(doc.text):
        CHUNKS.append({"title": doc.title, "text": chunk, "embedding": embed(chunk)})
    return {"title": doc.title, "chunks": len(CHUNKS)}


@app.post("/ask")
def ask(payload: Question):
    context = "\n\n".join(retrieve(payload.question))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer using only the context provided."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {payload.question}"},
        ],
    )
    return {"answer": response.choices[0].message.content, "context": context}
