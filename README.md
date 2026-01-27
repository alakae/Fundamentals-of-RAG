# Hybrid RAG - Advanced Retrieval System

A Retrieval-Augmented Generation (RAG) implementation combining BM25 keyword search with semantic vector search, featuring two reranking methods: Reciprocal Rank Fusion (RRF) and Neural Cross-Encoder reranking.

> [!IMPORTANT]
> ## Project notes (differences vs upstream)
> This repository is a **modified** version of the upstream course/demo material:
>
> - **Single “full” demo only**: the earlier incremental versions were removed (e.g., `app_v1.py`, `app_v2.py`) and the project is now centered around **`hybrid_rag.py`** as the main entry point.
> - **Hybrid RAG focus**: documentation and usage were updated to emphasize the **hybrid retrieval pipeline** (BM25 + vector search + Reciprocal Rank Fusion), with expanded CLI guidance (init/ingest/ask/stats/reset) and updated project structure.
> - **Add neural reranking option**
> - **Include detailed reranking info and complete prompt in verbose output**
> - **Dependency management migrated to `uv`**: added `pyproject.toml` and `uv.lock`, updated install/run instructions to use `uv sync` and `uv run ...`, and updated defaults (e.g., the example LLM model).
> - **Ignore local DB artifacts**: `.gitignore` was updated to avoid committing local persistence artifacts (e.g., `chroma.sqlite3`).
>
> ## License / educational use notice
> This codebase was originally provided by **KodeKloud** and is **licensed for educational purposes only**.  
> That educational-only licensing **still applies** to this repository and any modifications contained here.
>
> **Disclaimer:** No copyright infringement intended.


## Overview

This project implements a **hybrid search RAG system** that provides superior retrieval quality by combining:
- **BM25** for keyword-based matching
- **Vector Search** for semantic similarity
- **Two reranking options**:
  - **RRF (Reciprocal Rank Fusion)**: Fast mathematical fusion
  - **Neural Reranker**: AI-powered Cross-Encoder for +30% accuracy boost

Uses **Ollama** for LLM inference and embeddings, with **ChromaDB** as the vector store.

## Features

- 🔍 **Hybrid Search**: Combines BM25 keyword matching with semantic vector search
- 🧠 **Dual Reranking**: Choose between RRF (fast) or Neural Cross-Encoder (accurate)
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

# Ask questions with hybrid search (default: RRF reranker)
uv run hybrid_rag.py ask --query "What happened to Frankenstein?"

# Ask with Neural Reranker for better accuracy
uv run hybrid_rag.py ask --query "What happened to Frankenstein?" --reranker neural

# Check index statistics
uv run hybrid_rag.py stats

# Reset both indices
uv run hybrid_rag.py reset
```

### Advanced Usage

```bash
# Customize search parameters with neural reranker
uv run hybrid_rag.py ask \
  --query "Your question here" \
  --reranker neural \
  --llm llama3.2:3b \
  --embed-model nomic-embed-text \
  --k-each 6 \
  --final-k 5

# Show the complete prompt sent to the LLM (system and user)
uv run hybrid_rag.py ask \
  --query "Your question here" \
  --verbose

# Compare both rerankers on the same query
uv run hybrid_rag.py ask --query "Your question" --reranker rrf
uv run hybrid_rag.py ask --query "Your question" --reranker neural

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

## Reranking Methods

This implementation offers two reranking strategies to merge results from BM25 and Vector search:

### 1. RRF (Reciprocal Rank Fusion) - Default
**Mathematical fusion, fast and efficient**

- Combines rankings using the formula: `score = Σ 1/(k + rank)` where k=60
- No model loading, instant results
- Good baseline performance
- Best for: Speed-critical applications, real-time queries

### 2. Neural Reranker (Cross-Encoder)
**AI-powered contextual understanding, +30% accuracy boost**

- Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` model from sentence-transformers
- Reads and understands the actual content of query-document pairs
- Predicts true relevance scores (0.0 to 1.0) for each candidate
- Handles noise better - assigns low scores to irrelevant documents even if both retrievers ranked them high
- **First run**: Downloads model (~100MB) to `~/.cache/huggingface/`
- **Subsequent runs**: Loads from cache (takes a few seconds)

**When to use Neural Reranker:**
- Complex queries requiring deep understanding
- When accuracy is more important than speed
- Domain-specific or ambiguous questions
- Production systems where quality matters most

**Performance comparison:**
```bash
# Fast but may miss nuanced matches
--reranker rrf          # ~0.1s additional latency

# More accurate, understands context
--reranker neural       # ~1-2s additional latency (after model cache)
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

2. **Reranking** (choose one):
   - **RRF Mode**: Mathematical fusion using reciprocal ranks
     - Default: top-6 from each → fused to final top-5
     - Formula: `score = Σ 1/(k + rank)` where k=60
   - **Neural Mode**: Cross-Encoder deep learning reranking
     - Combines unique candidates from both retrievers
     - Creates query-document pairs
     - Neural network predicts relevance score for each pair
     - Sorts by AI scores and returns top-k

3. **Answer Generation**:
   - Build context from reranked chunks
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
- `--final-k`: Final top-k after reranking (default: 5)
- `--reranker`: Reranking method - "rrf" (default) or "neural"
- `--verbose`: Show complete prompt given to LLM (both system and user prompts)

## Tips

- Run `init` first to verify your Ollama setup
- Use `stats` to monitor index sizes
- Try both rerankers on your queries to see which performs better for your use case
- **Neural reranker first run**: Expect a one-time ~100MB download for the Cross-Encoder model
- Adjust `max_chars` based on your document structure (smaller for dense content)
- Increase `k_each` if relevant results are being missed
- Use `--reranker neural` for complex questions or when accuracy is critical
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
