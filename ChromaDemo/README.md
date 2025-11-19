# ChromaDemo - Vector Database Demo

A simple demonstration of ChromaDB for semantic search over text documents. This project shows how to ingest, chunk, and query text files using vector embeddings.

## Overview

ChromaDemo is a minimal implementation showcasing ChromaDB's core features:
- **Persistent storage** of document embeddings
- **Automatic chunking** of large documents with overlap
- **Semantic search** using cosine similarity
- **Metadata filtering** capabilities

This demo uses ChromaDB's built-in default embedding function (all-MiniLM-L6-v2 via sentence-transformers).

## Features

- 📚 **Automatic Ingestion**: Reads all `.txt` files from the `data/` directory
- ✂️ **Smart Chunking**: Splits long documents into overlapping chunks (1500 chars with 200 char overlap)
- 💾 **Persistent Storage**: Vector database persists in `chroma_db/` directory
- 🔍 **Semantic Search**: Query documents using natural language
- 🏷️ **Metadata Tracking**: Each chunk includes source file and chunk index
- 🔄 **Idempotent Ingestion**: Re-running won't create duplicates

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
ChromaDemo/
├── ingest_and_query.py    # Main script for ingestion and querying
├── requirements.txt        # Python dependencies
├── data/                   # Text files to ingest
│   ├── adventuresofhuckleberryfinn.txt
│   ├── adventuresofsherlockholmes.txt
│   ├── beowulf.txt
│   ├── completeworkswilliamshakespeare.txt
│   └── frankenstein.txt
└── chroma_db/              # ChromaDB persistence (created on first run)
```

## Usage

### Basic Usage

Simply run the script to ingest documents and execute demo queries:

```bash
python ingest_and_query.py
```

The script will:
1. Create/connect to the ChromaDB database
2. Read all `.txt` files from `data/`
3. Chunk documents into manageable pieces
4. Generate embeddings and store in ChromaDB
5. Execute sample semantic search queries

### Expected Output

```
  Added batch 1: 234 records
✅ Ingested 234 records from 5 file(s).

🔎 Query: Why does Macbeth decide to kill Duncan?
  • id=completeworkswilliamshakespeare__042  dist=0.4521  source=completeworkswilliamshakespeare.txt
    MACBETH. If it were done when 'tis done, then 'twere well It were done quickly...
  ...
```

### Customizing Queries

Edit the `search()` function calls at the bottom of `ingest_and_query.py`:

```python
# Basic search
search("Your question here", k=5)

# Search with metadata filtering
search("Another query", k=3, where={"source": "frankenstein.txt"})
```

### Query Parameters

- `query_text` (str): Natural language question
- `k` (int): Number of results to return (default: 4)
- `where` (dict, optional): Metadata filter, e.g., `{"source": "filename.txt"}`

## How It Works

### 1. Embedding Function

The script uses ChromaDB's default embedding function (all-MiniLM-L6-v2), which automatically generates vector embeddings for text.

To use a custom embedding function, uncomment the SentenceTransformers section:

```python
from chromadb.utils import embedding_functions
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
```

### 2. Document Chunking

Large documents are split into overlapping chunks to improve retrieval:

```python
chunk_text(text, max_chars=1500, overlap=200)
```

- **max_chars**: Maximum characters per chunk
- **overlap**: Characters shared between consecutive chunks (improves context continuity)

### 3. Persistence

ChromaDB uses `PersistentClient` to store vectors on disk:

```python
client = chromadb.PersistentClient(path="chroma_db")
```

The database persists across runs, so you don't need to re-ingest unless data changes.

### 4. Semantic Search

Queries are embedded using the same function as documents, then matched via cosine similarity:

```python
collection.query(
    query_texts=[query_text],
    n_results=k,
    where=metadata_filter  # optional
)
```

## Configuration

Key parameters you can adjust in `ingest_and_query.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DB_DIR` | `"chroma_db"` | ChromaDB storage location |
| `DOCS_DIR` | `"data"` | Source documents directory |
| `max_chars` | `1500` | Maximum characters per chunk |
| `overlap` | `200` | Overlap between chunks |
| `BATCH_SIZE` | `5000` | Records per ingestion batch |
| `collection name` | `"demo_texts"` | ChromaDB collection name |
| `hnsw:space` | `"cosine"` | Distance metric (cosine, l2, ip) |

## Adding Your Own Documents

1. Place `.txt` files in the `data/` directory
2. Run the script: `python ingest_and_query.py`
3. Your documents are automatically ingested and searchable

## Sample Queries to Try

Modify the search calls at the bottom of the script:

```python
# Literary analysis
search("What is the theme of revenge in the story?", k=4)

# Character questions
search("Who is the protagonist and what are their motivations?", k=3)

# Plot questions
search("What happens at the climax of the story?", k=5)

# Filter by specific book
search("Tell me about the monster", k=3, where={"source": "frankenstein.txt"})
```

## Resetting the Database

To start fresh and re-ingest everything:

```bash
rm -rf chroma_db/
python ingest_and_query.py
```

## Advanced Features

### Metadata Filtering

Filter results by source file or chunk number:

```python
# Only search in Shakespeare
search("famous soliloquy", k=5, where={"source": "completeworkswilliamshakespeare.txt"})

# Search only first chunks (early content)
search("beginning of story", k=5, where={"chunk": 0})
```

### Accessing Raw Results

The `query()` method returns a dictionary with:
- `ids`: Unique document IDs
- `documents`: Text content
- `metadatas`: Source file, chunk info
- `distances`: Similarity scores (lower = more similar)

```python
results = collection.query(query_texts=["your query"], n_results=5)
print(results)
```

## Technical Details

### Embedding Model

- **Model**: all-MiniLM-L6-v2 (via sentence-transformers)
- **Dimensions**: 384
- **Context**: Up to 256 tokens
- **Speed**: Very fast for real-time search

### Vector Database

- **Engine**: ChromaDB with HNSW index
- **Distance**: Cosine similarity
- **Persistence**: SQLite backend
- **Scalability**: Suitable for 100K+ documents

## Troubleshooting

**Issue**: "No .txt files found in ./data"
- Ensure `.txt` files exist in the `data/` directory
- Check file permissions

**Issue**: Poor search results
- Try adjusting chunk size (smaller chunks = more precise, larger = more context)
- Increase `k` parameter to retrieve more results
- Ensure query wording is similar to document language

**Issue**: Memory errors during ingestion
- Reduce `BATCH_SIZE` to process fewer records at once
- Process files individually instead of all at once

**Issue**: Database locked errors
- Ensure no other processes are accessing `chroma_db/`
- Delete `chroma_db/` and re-ingest

## Next Steps

- Integrate with an LLM for RAG (Retrieval-Augmented Generation)
- Add support for more file formats (PDF, DOCX, HTML)
- Implement hybrid search combining BM25 + vector search
- Build a web interface for querying
- Add document versioning and update tracking

## Related Projects

See other demos in this repository:
- **BookSearch**: Full RAG implementation with Ollama
- **FAISSDemo**: FAISS vector search alternative
- **Chromadev**: Another ChromaDB example

## License

Educational demo project.
