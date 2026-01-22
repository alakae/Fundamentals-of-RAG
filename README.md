# Hybrid RAG - Advanced Retrieval System

A Retrieval-Augmented Generation (RAG) implementation combining BM25 keyword search with semantic vector search using Reciprocal Rank Fusion (RRF).

## Overview

This project implements a **hybrid search RAG system** that provides superior retrieval quality by combining:
- **BM25** for keyword-based matching
- **Vector Search** for semantic similarity
- **Reciprocal Rank Fusion** to merge results optimally

Uses **Ollama** for LLM inference and embeddings, with **ChromaDB** as the vector store.

## Features

- 🔍 **Hybrid Search**: Combines BM25 keyword matching with semantic vector search via RRF
- 📚 **Smart Chunking**: Paragraph-aware text splitting with configurable overlap
- 📄 **Multi-Format**: Supports `.txt` and `.md` files
- 💬 **Q&A with Citations**: Source-grounded answers with chunk references
- 🎯 **Deterministic IDs**: SHA1-based chunk identifiers for reproducibility
- ⚡ **Batch Processing**: Efficient ingestion with progress tracking
- 🛡️ **Duplicate Handling**: Graceful handling of re-ingestion

## Requirements

- [uv](https://docs.astral.sh/uv/) package manager
- Ollama running locally
- Required models:
  - `llama3.2:3b` (or compatible LLM)
  - `nomic-embed-text` (for embeddings)

## Installation

1. Install dependencies:
```bash
uv sync
```

This will create a virtual environment and install all dependencies from `uv.lock`.

2. Ensure Ollama is running with required models:
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

## Usage

### Quick Start

```bash
# Check environment setup
uv run hybrid_rag.py init

# Ingest documents from a directory
uv run hybrid_rag.py ingest --dir ./books

# Ask questions with hybrid search
uv run hybrid_rag.py ask --query "What happened to Frankenstein?"

# Check index statistics
uv run hybrid_rag.py stats

# Reset both indices
uv run hybrid_rag.py reset
```

### Advanced Usage

```bash
# Customize search parameters
uv run hybrid_rag.py ask \
  --query "Your question here" \
  --llm llama3.2:3b \
  --embed-model nomic-embed-text \
  --k-each 6 \
  --final-k 5

# Use different embedding model
uv run hybrid_rag.py ingest --dir ./books --embed-model mxbai-embed-large
```

## Project Structure

```
Fundamentals-of-RAG/
├── hybrid_rag.py       # Main hybrid RAG implementation
├── pyproject.toml      # Project dependencies (uv)
├── uv.lock             # Locked dependencies
├── books/              # Sample document collection
├── data/               # Alternative ingestion directory
├── backup/             # Additional sample documents
├── index/              # BM25 index storage (created on ingest)
└── .chroma/            # ChromaDB vector store (created on ingest)
```

## How It Works

### Chunking Strategy

Paragraph-aware text splitting with configurable overlap:
- Default: 800 characters per chunk
- 150 character overlap between chunks
- Preserves paragraph boundaries for better semantic coherence
- Uses deterministic SHA1-based IDs for reproducibility

### Retrieval Pipeline

1. **Dual Retrieval**:
   - **BM25**: Keyword-based ranking using tokenized text
   - **Vector Search**: Cosine similarity on embeddings

2. **Reciprocal Rank Fusion (RRF)**:
   - Merges results from both retrievers
   - Default: top-6 from each → fused to final top-5
   - Formula: `score = Σ 1/(k + rank)` where k=60

3. **Answer Generation**:
   - Build context from fused chunks
   - Prompt LLM with grounded instructions
   - Include source citations in output

## Configuration

Key parameters in `hybrid_rag.py`:

- `LLM_MODEL`: Language model for generation (default: `llama3.2:3b`)
- `EMBED_MODEL`: Embedding model (default: `nomic-embed-text`)
- `CHROMA_DIR`: Vector store location (default: `./.chroma`)
- `INDEX_DIR`: BM25 index location (default: `./index`)
- `COLLECTION_NAME`: ChromaDB collection name (default: `books`)

### Chunking Parameters

In `make_chunks()`:
- `max_chars=800`: Maximum characters per chunk
- `overlap=150`: Character overlap between chunks

### Retrieval Parameters

Via CLI:
- `--k-each`: Top-k from each retriever (default: 6)
- `--final-k`: Final top-k after RRF fusion (default: 5)

## Tips

- Run `init` first to verify your Ollama setup
- Use `stats` to monitor index sizes
- Adjust `max_chars` based on your document structure (smaller for dense content)
- Increase `k_each` if relevant results are being missed
- The hybrid approach excels when queries contain both specific terms and concepts

## Development

### Adding New Dependencies

```bash
uv add <package-name>
```

### Updating Dependencies

```bash
uv lock --upgrade
uv sync
```

### Exporting to requirements.txt

If you need a `requirements.txt` for compatibility:

```bash
uv export --format requirements-txt > requirements.txt
```

## Sample Documents

The `books/` directory contains classic literature texts for testing:
- Adventures of Sherlock Holmes
- Complete Works of William Shakespeare  
- Frankenstein

Additional samples in `backup/` include DevOps runbooks and SLO documents.

## Commands Reference

| Command | Description |
|---------|-------------|
| `init` | Check Ollama, Chroma, and BM25 setup |
| `ingest --dir <path>` | Index .txt/.md files from directory |
| `ask --query "<q>"` | Query with hybrid search |
| `stats` | Show chunk counts for both indices |
| `reset` | Delete Chroma and BM25 indices |

## Troubleshooting

**Issue**: "Ollama not responding"
- Ensure Ollama is running: `ollama serve`
- Verify models are downloaded: `ollama list`

**Issue**: "No results found"
- Run `ingest` command first to index documents
- Check that `.txt` or `.md` files exist in the specified directory
- Verify indices with `stats`

**Issue**: ChromaDB or BM25 errors
- Try resetting: `uv run hybrid_rag.py reset`
- Delete `.chroma` and `index/` directories manually if needed

**Issue**: "FileNotFoundError" for BM25 index
- Run `ingest` before `ask`
- The BM25 index is created during ingestion

## License

This is a demo project for educational purposes.
