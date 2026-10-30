# rag-api-service

Retrieval-augmented generation API — document ingestion, embeddings and a question-answering endpoint built with FastAPI and LangChain/LlamaIndex concepts.

## What it does

- Ingest documents through `POST /ingest`.
- Answer questions about the ingested documents through `POST /ask`.
- Retrieve the most relevant document chunks before generating an answer, so answers stay grounded in the uploaded content.

## Pipeline

Ingest → chunk → embed → retrieve → answer.

1. Ingest: a document is received and stored.
2. Chunk: the document is split into smaller passages.
3. Embed: each chunk is turned into a vector embedding.
4. Retrieve: a question is embedded and the closest chunks are found.
5. Answer: an LLM answers the question using the retrieved chunks as context.

## Stack

- Python
- FastAPI
- LangChain / LlamaIndex concepts
- OpenAI API

## How to run

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Set `OPENAI_API_KEY` before running, then ingest a document at `POST /ingest` and ask a question at `POST /ask`.
