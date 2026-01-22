# BookSearch - RAG Demo Project

A collection of Retrieval-Augmented Generation (RAG) implementations demonstrating different approaches to semantic search and question-answering over document collections.

## Overview

This project contains three progressive implementations of RAG systems:

1. **app_v1.py** - Basic RAG demo with in-memory documents
2. **app_v2.py** - File-based RAG with ingestion pipeline
3. **hybrid_rag.py** - Hybrid search combining BM25 and semantic vector search

All implementations use **Ollama** for LLM inference and embeddings, with **ChromaDB** as the vector store.

## Features

- 🔍 **Semantic Search**: Vector-based similarity search using embeddings
- 📊 **Hybrid Search**: Combines BM25 keyword matching with semantic search via Reciprocal Rank Fusion (RRF)
- 📚 **Document Ingestion**: Automatic chunking and indexing of `.txt` and `.md` files
- 💬 **Conversational Q&A**: Ask questions about ingested documents with source citations
- 🎯 **Grounded Responses**: Answers are constrained to the provided context

## Requirements

- Python 3.8+
- Ollama running locally
- Required models:
  - `llama3.3:latest` (or compatible LLM)
  - `nomic-embed-text` (for embeddings)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure Ollama is running with required models:
```bash
ollama pull llama3.3:latest
ollama pull nomic-embed-text
```

## Usage

### Version 1: Basic Demo (`app_v1.py`)

Minimal RAG example with a single in-memory document.

```bash
# Check environment setup
python app_v1.py init

# Run demo with sample document
python app_v1.py demo
```

### Version 2: File-Based RAG (`app_v2.py`)

Ingest and query local text files.

```bash
# Check environment
python app_v2.py init

# Ingest documents from a directory (default: ./data)
python app_v2.py ingest --dir ./books

# Ask questions
python app_v2.py ask "What is the SLO target?"

# Query with custom top-k
python app_v2.py ask "Who is Sherlock Holmes?" --k 10

# Check statistics
python app_v2.py stats

# Reset the index
python app_v2.py reset
```

### Version 3: Hybrid RAG (`hybrid_rag.py`)

Advanced implementation combining BM25 and vector search.

```bash
# Ingest documents
python hybrid_rag.py ingest --dir ./books

# Ask questions with hybrid search
python hybrid_rag.py ask --query "What happened to Frankenstein?"

# Customize search parameters
python hybrid_rag.py ask \
  --query "Your question here" \
  --llm llama3.3:latest \
  --embed-model nomic-embed-text \
  --k-each 6 \
  --final-k 5
```

## Project Structure

```
BookSearch/
├── app_v1.py           # Basic RAG demo
├── app_v2.py           # File-based RAG with ingestion
├── hybrid_rag.py       # Hybrid BM25 + vector search
├── requirements.txt    # Python dependencies
├── books/              # Sample document collection
├── data/               # Default ingestion directory
├── backup/             # Additional sample documents
├── index/              # BM25 index storage (created on ingest)
└── .chroma/            # ChromaDB vector store (created on ingest)
```

## How It Works

### Chunking Strategy

- **app_v2.py**: Paragraph-based chunking with configurable overlap (default: 800 chars, 150 char overlap)
- **hybrid_rag.py**: Fixed-size chunking with overlap (default: 1024 chars, 200 char overlap)

### Retrieval Methods

1. **Semantic Search** (v1, v2): Uses cosine similarity between query and document embeddings
2. **Hybrid Search** (v3): 
   - BM25 performs keyword-based retrieval
   - Vector search performs semantic retrieval
   - Results merged using Reciprocal Rank Fusion (RRF)

### Answer Generation

All versions use a grounded prompting strategy:
- Retrieve relevant document chunks
- Build context from top-k results
- Prompt LLM to answer using ONLY the provided context
- Include source citations

## Configuration

Key parameters you can modify:

- `LLM_MODEL`: Language model for generation (default: `llama3.3:latest`)
- `EMBED_MODEL`: Embedding model (default: `nomic-embed-text`)
- `TOP_K`: Number of chunks to retrieve (default: 5)
- `CHROMA_PATH`: ChromaDB storage location (default: `./.chroma`)
- `COLLECTION_NAME`: ChromaDB collection name

## Tips

- Start with `app_v1.py` to verify your setup
- Use `app_v2.py` for straightforward document Q&A
- Use `hybrid_rag.py` when you need both keyword and semantic matching
- Adjust chunk size and overlap based on your document structure
- Tune `k_each` and `final_k` in hybrid search for optimal retrieval

## Sample Documents

The `books/` directory contains classic literature texts for testing:
- Adventures of Sherlock Holmes
- Complete Works of William Shakespeare  
- Frankenstein

Additional samples in `backup/` include DevOps runbooks and SLO documents.

## Troubleshooting

**Issue**: "Ollama not responding"
- Ensure Ollama is running: `ollama serve`
- Verify models are downloaded: `ollama list`

**Issue**: "No results found"
- Run ingest command first to index documents
- Check that documents exist in the specified directory

**Issue**: ChromaDB errors
- Try resetting: `python app_v2.py reset`
- Delete `.chroma` directory manually if needed

## License

This is a demo project for educational purposes.
