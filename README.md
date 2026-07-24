# RAG Pipeline

Two parallel RAG (Retrieval-Augmented Generation) implementations over the same PDF, one built with LangChain, one with LlamaIndex, for comparison. Both use Google Gemini (`gemini-embedding-2-preview` for embeddings, `gemini-3.6-flash` for generation) and store vectors in Postgres with `pgvector`.

## Structure

- `langchain_pipeline.py`, loads, chunks, embeds, and stores the PDF using LangChain + a raw SQLAlchemy model for pgvector.
- `llamaindex_pipeline.py`, same task using LlamaIndex's `VectorStoreIndex`, with separate Postgres-backed stores for vectors, document content, and index metadata. Supports incremental re-indexing via `refresh_ref_docs`.

## Requirements

- Docker + Docker Compose
- A `GOOGLE_API_KEY` with access to Gemini
- A `DATABASE_URL` pointing at the Postgres/pgvector service

## Setup

```bash
cp .env.example .env   # fill in GOOGLE_API_KEY and DATABASE_URL
docker compose up -d pgvector
```

## Running

```bash
# LangChain pipeline
docker compose run --rm langchain_pipeline

# LlamaIndex pipeline
docker compose run --rm llamaindex_pipeline
```

Each run embeds `PDF_PATH` (see top of each script) into Postgres, then answers a sample question using the top-3 retrieved chunks.