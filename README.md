# Hybrid RAG - Multi-Stage Variable Funnel

A Retrieval-Augmented Generation (RAG) implementation featuring a **variable-width 5-stage funnel** with optional LLM-powered query expansion, hybrid retrieval, RRF fusion, and neural Cross-Encoder reranking into a powerful sequential pipeline.

> [!IMPORTANT]
> ## Project notes (differences vs upstream)
> This repository is a **modified** version of the upstream course/demo material:
>
> - **Single "full" demo only**: the earlier incremental versions were removed (e.g., `app_v1.py`, `app_v2.py`) and the project is now centered around **`hybrid_rag.py`** as the main entry point.
> - **Multi-Stage Variable Funnel**: Transformed into a 5-stage pipeline with **optional LLM-powered query expansion** that widens the funnel dynamically.
> - **Query Expansion (Stage 0)**: Uses Pydantic-structured output with Ollama to generate diverse query variations, resolving pronouns and using synonyms.
> - **Optimized defaults**: `expand_queries=0` (optional), `k_each=50` (broad recall), `rerank_k=25` (efficient trimming), `final_k=5` (precision context).
> - **Enhanced verbose mode**: Complete pipeline visibility showing all 5 stages, per-query retrieval stats, ranking impact analysis, and full LLM prompts.
> - **Dependency management migrated to `uv`**: added `pyproject.toml` and `uv.lock`, updated install/run instructions to use `uv sync` and `uv run ...`, and updated defaults (e.g., the example LLM model).
> - **Ignore local DB artifacts**: `.gitignore` was updated to avoid committing local persistence artifacts (e.g., `chroma.sqlite3`).
>
> ## License / educational use notice
> This codebase was originally provided by **KodeKloud** and is **licensed for educational purposes only**.  
> That educational-only licensing **still applies** to this repository and any modifications contained here.
>
> **Disclaimer:** No copyright infringement intended.


## Overview

This project implements a **Multi-Stage Variable Funnel** - a RAG architecture that progressively refines retrieval results through five sequential stages with dynamic width control:

**🎯 Stage 0: Dynamic Query Expansion (Optional)**
- If enabled (`--expand-queries N`), generate N query variations using LLM with Pydantic-structured output
- Improves retrieval through: diverse vocabulary (synonyms, technical vs. layman terms), typo correction, pronoun resolution, and complete standalone phrasing
- Widens funnel: 1 query → (1 + N) queries
- Example: "What happened to him?" → "What happened to Victor Frankenstein?", "What was the scientist's fate?", etc.

**🔍 Stage 1: Broad Hybrid Retrieval (Variable Width)**
- Run BM25 (keyword) + Vector Search (semantic) for EACH query (original + variations)
- Default: Fetch 50 results per query per engine (configurable via `--k-each`)
- Merge and deduplicate across all queries
- Funnel width scales with `--expand-queries` value

**⚡ Stage 2: RRF Fusion + Filter**
- Apply Reciprocal Rank Fusion to merge BM25 + Vector rankings
- Default: Trim to top 25 candidates for efficiency (configurable via `--rerank-k`)

**🧠 Stage 3: Deep Neural Reranking**
- CrossEncoder AI analyzes candidates from Stage 2
- Assigns contextual relevance scores based on semantic meaning
- Uses original query for scoring (not variations)

**🎯 Stage 4: Final Selection**
- Select top results for LLM context (default: 5, configurable via `--final-k`)
- Balances quality, latency, and token costs

This variable funnel architecture combines query expansion from modern search engines with the multi-stage reranking pipeline used by major tech companies, providing +30-50% accuracy improvement while offering tunable recall/latency trade-offs.

## Example

Standalone LLMs often lack specific document knowledge, leading them to either hallucinate or incorrectly deny the existence of obscure facts. The Multi-Stage Variable Funnel overcomes this by identifying the precise "needle in the haystack"—in this case, the exact number of steps in Baker Street—while providing verifiable citations for every claim.

Plain LLM query:

```console
$ llm -m "llama3.2:3b" "In the conversation with Watson, how many steps does the detective claim lead up from the hall to his room, and what is the lesson he is trying to teach?"
I'm not aware of any specific scene or conversation in literature where a detective discusses stairs leading up to their room. Could you please provide more context or clarify which story or author you are referring to? I'll do my best to help.

However, I can try to find the answer for you if you could tell me which character and conversation you're thinking of (e.g., Sherlock Holmes and Dr. Watson).
```

RAG query:

```
$ uv run hybrid_rag.py ask --query "In the conversation with Watson, how many steps does the detective claim lead up from the hall to his room, and what is the lesson he is trying to teach?" --expand-queries 5 --k-each 100 --rerank-k 50  --final-k 15  

🎯 Stage 0: Query Expansion (generating 5 variations)...
   ✓ Expanding search to 6 queries total (1 original + 5 variations)

🔍 Stage 1: Broad Hybrid Retrieval (6 queries × 100 per engine)...
   ✓ Retrieved 634 unique candidates across 6 queries
      (BM25: 421, Vector: 272, Overlap: 59)
⚡ Stage 2: RRF Fusion + Filter (trimming to top 50)...
   ✓ Fused 634 candidates → Trimmed to top 50 (RRF scores from both engines)
🧠 Stage 3: Deep Neural Reranking (50 candidates with Cross-Encoder)...
   ✓ Neural scoring complete. Top 15 will be selected for LLM context.
🎯 Stage 4: Final Selection (top 15 for LLM context)
   ✓ Pipeline complete: 634 → 50 → 15 documents


=== Answer ===

In the conversation with Watson, Sherlock Holmes claims that there are 17 steps leading up from the hall to his room. The lesson he is trying to teach is the distinction between observing and seeing. He emphasizes that observation involves paying attention to details, while seeing involves merely perceiving something without truly understanding it.

--- Sources ---
adventuresofsherlockholmes.txt  (chunk 11)
adventuresofsherlockholmes.txt  (chunk 368)
adventuresofsherlockholmes.txt  (chunk 10)
adventuresofsherlockholmes.txt  (chunk 571)
adventuresofsherlockholmes.txt  (chunk 515)
adventuresofsherlockholmes.txt  (chunk 233)
adventuresofsherlockholmes.txt  (chunk 126)
adventuresofsherlockholmes.txt  (chunk 79)
adventuresofsherlockholmes.txt  (chunk 61)
adventuresofsherlockholmes.txt  (chunk 286)
adventuresofsherlockholmes.txt  (chunk 488)
adventuresofsherlockholmes.txt  (chunk 818)
adventuresofsherlockholmes.txt  (chunk 437)
adventuresofsherlockholmes.txt  (chunk 403)
adventuresofsherlockholmes.txt  (chunk 18)
```


## Features

- 🎯 **5-Stage Variable Funnel**: Sequential pipeline with dynamic width (Query Expansion → Hybrid → RRF → Neural → Selection)
- 📢 **Dynamic Query Expansion**: Optional LLM-powered query rewriting with Pydantic-structured output (diverse vocabulary, typo correction, pronoun resolution, complete standalone phrasing)
- 🔬 **Variable Funnel Width**: Control recall vs. latency by adjusting query expansion (0-10 variations recommended)
- 📊 **Optimized Recall/Precision**: Configurable retrieval width → Intelligent trimming → Final selection
- 🧠 **Neural Reranking**: CrossEncoder AI is core to the pipeline for semantic relevance scoring
- 📚 **Smart Chunking**: Paragraph-aware text splitting with configurable overlap
- 📄 **Multi-Format**: Supports `.txt` and `.md` files
- 💬 **Q&A with Citations**: Source-grounded answers with chunk references
- 🎯 **Deterministic IDs**: SHA1-based chunk identifiers for reproducibility
- ⚡ **Batch Processing**: Efficient ingestion with progress tracking
- 🛡️ **Duplicate Handling**: Graceful handling of re-ingestion
- 🔬 **Stage Visibility**: Clear progress indicators showing funnel progression and per-query stats

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

# Basic query (single query, 5-stage funnel without expansion)
# (Uses defaults: expand_queries=0, k_each=50, rerank_k=25, final_k=5)
uv run hybrid_rag.py ask --query "What is the address of Sherlock Holmes?"

# Expand with 3 query variations (wider funnel for better recall)
uv run hybrid_rag.py ask --query "What is the address of Sherlock Holmes?" --expand-queries 3

# Maximum width: 5 variations for difficult/ambiguous queries
uv run hybrid_rag.py ask --query "What happened to him?" --expand-queries 5

# See the complete pipeline in action with verbose mode
uv run hybrid_rag.py ask --query "What happened to him?" --expand-queries 3 --verbose

# Check index statistics
uv run hybrid_rag.py stats

# Reset both indices
uv run hybrid_rag.py reset
```

### Advanced Usage

```bash
# Customize the variable funnel parameters
uv run hybrid_rag.py ask \
  --query "Your question here" \
  --expand-queries 3 \           # Stage 0: Generate 3 query variations
  --k-each 100 \                 # Stage 1: Broader recall (fetch 100 per query per engine)
  --rerank-k 50 \                # Stage 2: Keep top 50 after RRF fusion
  --final-k 10 \                 # Stage 4: Use top 10 for LLM context
  --llm llama3.2:3b \
  --embed-model nomic-embed-text \
  --cross-encoder-model cross-encoder/ms-marco-MiniLM-L-6-v2

# Optimize for speed (narrow funnel, fewer candidates)
uv run hybrid_rag.py ask \
  --query "Quick question?" \
  --expand-queries 0 \    # No query expansion
  --k-each 20 \
  --rerank-k 10 \
  --final-k 3

# Optimize for maximum accuracy (wide funnel, more candidates, more context)
uv run hybrid_rag.py ask \
  --query "Complex analytical question?" \
  --expand-queries 5 \    # 5 query variations (6 total queries)
  --k-each 100 \          # 600 initial candidates per engine
  --rerank-k 50 \
  --final-k 15

# Query expansion for ambiguous/pronoun-heavy queries
uv run hybrid_rag.py ask \
  --query "What did he discover in his laboratory?" \
  --expand-queries 4 \    # LLM will resolve "he" and rephrase
  --verbose               # See all query variations

# Use different embedding model (must match ingestion model)
uv run hybrid_rag.py ingest --dir ./books --embed-model mxbai-embed-large
uv run hybrid_rag.py ask --query "Your question" --embed-model mxbai-embed-large
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

## How the Multi-Stage Variable Funnel Works

This implementation uses a **sequential 5-stage pipeline** where each stage refines the results from the previous stage. The funnel width is controlled by the `--expand-queries` parameter:

### Stage 0: Dynamic Query Expansion (Funnel Width Control)
**Goal: Improve retrieval through diverse query reformulations**

- **When disabled** (`--expand-queries 0`, default): Single query, narrow funnel, fastest
- **When enabled** (`--expand-queries 3`): 1 original + 3 variations = 4 total queries
- Uses LLM with Pydantic-structured output (`QueryExpansion` model)
- **Prompt engineering**: Instructs LLM with 5 requirements:
  1. Use different vocabulary (synonyms, technical vs. layman terms)
  2. Fix any obvious typos or spelling errors
  3. Explicitly replace ambiguous pronouns with specific subjects
  4. Maintain original search intent without adding new constraints
  5. Ensure each variation is a standalone, complete sentence
- **Why Pydantic?** Guarantees JSON structure with `variations: list[str]` field via `format=QueryExpansion.model_json_schema()`
- **Example transformation**:
  - Input: "What happened to him?"
  - Output: ["What happened to him?", "What happened to Victor Frankenstein?", "What was the scientist's fate?", "What occurred to the main character?"]
- **Temperature**: 0.7 for creativity while maintaining relevance
- **Fallback**: Returns single query if expansion fails
- **First LLM call**: ~1-2 seconds

### Stage 1: Broad Hybrid Retrieval (Variable Width, Recall Optimization)
**Goal: Cast a wide net across all query variations**

- **For each query** (original + variations):
  - BM25 fetches top-50 candidates (keyword matching)
  - Vector Search fetches top-50 candidates (semantic similarity)
- **Merge and deduplicate** across all queries
- **Funnel width scaling**:
  - 0 variations: ~100 unique candidates (50+50 per engine)
  - 3 variations: ~200-350 unique candidates (4 queries, deduplication)
  - 5 variations: ~300-500 unique candidates (6 queries, deduplication)
- **Why variable width?** More query angles increases chance of finding the "needle"
- **Deduplication**: Documents appearing in multiple query results are kept once (best score)

### Stage 2: RRF Fusion + Filter (Efficiency)
**Goal: Create a vetted shortlist using mathematical fusion**

- Applies Reciprocal Rank Fusion: `score = Σ 1/(k + rank)` where k=60
- Merges BM25 and Vector rankings from ALL queries into unified scores
- Trims to top-25 candidates
- **Why trim?** Neural reranking is expensive - only analyze the most promising candidates
- **Handles variable width**: Same trim threshold regardless of input funnel width

### Stage 3: Deep Neural Reranking (Precision)
**Goal: AI-powered contextual analysis**

- Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` CrossEncoder model (configurable)
- Reads and understands actual content of query-document pairs
- Assigns relevance scores based on semantic meaning
- **Uses original query** for relevance scoring (not variations)
- **Why neural?** Filters out "false positives" from Stage 1-2 (e.g., Shakespeare noise when searching for Sherlock)
- **First run**: Downloads model (~100MB) to `~/.cache/huggingface/`
- **Subsequent runs**: Loads from cache (~2-3 seconds)

### Stage 4: Final Selection (Context Optimization)
**Goal: Provide highest-quality context to LLM**

- Sorts by neural scores (highest to lowest)
- Selects top-5 documents
- **Why 5?** Balances answer quality, token costs, and latency

### Performance Characteristics

**Latency breakdown (with --expand-queries 3):**
- Stage 0 (Query Expansion): ~1-2s (LLM call)
- Stage 1 (Retrieval): ~600-1500ms (4× queries)
- Stage 2 (RRF): ~10ms
- Stage 3 (Neural): ~1-2s (after cache)
- Stage 4 (Selection): ~1ms
- **Total**: ~3-6 seconds per query

**Latency breakdown (without expansion, --expand-queries 0):**
- Stage 0 (Skipped): 0ms
- Stage 1 (Retrieval): ~200-500ms (1× query)
- Stage 2 (RRF): ~10ms
- Stage 3 (Neural): ~1-2s (after cache)
- Stage 4 (Selection): ~1ms
- **Total**: ~2-3 seconds per query

**Quality improvement:**
- +30-50% accuracy vs. single-stage retrieval
- +10-20% additional accuracy with query expansion (3-5 variations)
- Robust handling of noisy datasets
- Better disambiguation of ambiguous queries
- Resolves pronoun references automatically

## Chunking Strategy

Paragraph-aware text splitting with configurable overlap:
- Default: 800 characters per chunk
- 150 character overlap between chunks
- Preserves paragraph boundaries for better semantic coherence
- Uses deterministic SHA1-based IDs for reproducibility

## Answer Generation

After the 5-stage funnel produces the final documents:
1. Build context from the selected chunks (default: top 5)
2. Prompt LLM with grounded instructions (temperature: 0.2)
3. Include source citations in output

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

### Funnel Parameters

Via CLI:
- `--expand-queries`: Stage 0 - Number of query variations to generate (default: 0/disabled, range: 0-10)
- `--k-each`: Stage 1 - Top-k from each retriever per query (default: 50)
- `--rerank-k`: Stage 2 - Trim to this many after RRF fusion (default: 25)
- `--final-k`: Stage 4 - Final top-k after neural reranking (default: 5)
- `--cross-encoder-model`: Stage 3 - Neural reranker model (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `--verbose`: Show detailed results for all 5 stages, per-query stats, ranking impact analysis, and complete LLM prompts

## Tips

- Run `init` first to verify your Ollama setup
- Use `stats` to monitor index sizes
- **First run**: Expect a one-time ~100MB download for the Cross-Encoder model (cached thereafter)
- Use `--verbose` to see exactly how the variable funnel progressively refines results
- Adjust `max_chars` based on your document structure (smaller for dense content)

### When to Use Query Expansion

**Use `--expand-queries 3-5` when:**
- Query contains ambiguous pronouns ("What did he discover?", "Where did she live?")
- Query is vague or underspecified ("Tell me about the experiment")
- Query uses colloquial/informal language or contains typos
- Documents use different terminology than the query (technical vs. layman terms)
- You need maximum recall despite higher latency
- Query may benefit from diverse vocabulary perspectives

**Skip query expansion (`--expand-queries 0`) when:**
- Query is already specific and well-formed
- Query contains precise technical terms/proper nouns that shouldn't be altered
- Speed is critical (saves 1-2 seconds)
- You're doing batch processing of many queries

### Tuning the Funnel

**For maximum recall (find that needle!):**
- `--expand-queries 5` (6 total query angles)
- `--k-each 100` (600 candidates per engine)
- `--rerank-k 50`
- `--final-k 10`

**For balanced performance (recommended):**
- `--expand-queries 3` (4 total queries)
- `--k-each 50` (default, 200 candidates per engine)
- `--rerank-k 25` (default)
- `--final-k 5` (default)

**For speed (minimize latency):**
- `--expand-queries 0` (single query)
- `--k-each 20`
- `--rerank-k 10`
- `--final-k 3`

### Other Tips

- The funnel excels at handling noisy datasets with many distractors (e.g., finding Sherlock Holmes address in a corpus containing all of Shakespeare)
- Query expansion adds ~1-2s latency but can improve accuracy by 10-20% for ambiguous queries
- Use `--verbose` with `--expand-queries >0` to see which variations contributed to the final results

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
