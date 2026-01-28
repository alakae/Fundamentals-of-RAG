import argparse
import hashlib
import json
import os
import pickle
import re
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Iterator
from tqdm import tqdm
import chromadb
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import ollama


class QueryExpansion(BaseModel):
    """Structured output for LLM query expansion."""
    variations: list[str]


def expand_query(original_query: str, num_variations: int, llm_model: str, verbose: bool = False) -> List[str]:
    """
    Use LLM with structured output (Pydantic) to generate query variations.

    Args:
        original_query: The user's original query
        num_variations: Number of variations to generate
        llm_model: The LLM model to use for generation
        verbose: Whether to print debug info

    Returns:
        List containing [original_query] + generated variations
    """
    if num_variations <= 0:
        return [original_query]

    prompt = f"""You are an AI search expert. Your goal is to expand the user's query into {num_variations} diverse variations to improve document retrieval.

Requirements for each variation:
1. Use different vocabulary (synonyms, technical vs. layman terms).
2. Fix any obvious typos or spelling errors.
3. Explicitly replace ambiguous pronouns (it, they, he, she) with the specific subjects being discussed.
4. Maintain the original search intent without adding new constraints.
5. Ensure each variation is a standalone, complete sentence.
6. Do not include any JSON formatting, braces, or quotes inside the variation strings.

Original Query: "{original_query}"

Provide exactly {num_variations} variations in the requested JSON format."""


    try:
        response = ollama.chat(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            format=QueryExpansion.model_json_schema(),
            options={"temperature": 0.7}
        )

        # Parse JSON response
        result = json.loads(response["message"]["content"])
        variations = result.get("variations", [])

        # Return original + variations (limit to requested number)
        return [original_query] + variations[:num_variations]

    except Exception as e:
        if verbose:
            print(f"   ⚠ Query expansion failed: {e}")
            print(f"   → Falling back to single query")
        return [original_query]


def _iter_files(root: Path) -> Iterator[Path]:
    """Iterate over .txt and .md files in the given directory."""
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
            yield p

def read_text_files(root: Path) -> Dict[str, str]:
    """Read all .txt and .md files into a dict."""
    files = []
    if root.is_file() and root.suffix.lower() in {".txt", ".md"}:
        files = [root]
    else:
        files = list(_iter_files(root))
    out = {}
    for f in files:
        try:
            out[str(f)] = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            out[str(f)] = f.read_text(encoding="latin-1", errors="ignore")
    return out

def _split_paragraphs(text: str) -> List[str]:
    """Split on blank lines; keep non-empty paragraphs."""
    parts = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n")]
    return [p for p in parts if p]

def make_chunks(text: str, max_chars: int = 800, overlap: int = 150) -> List[str]:
    """Greedy paragraph packer with overlap between chunks."""
    paras = _split_paragraphs(text)
    chunks, buf, total = [], [], 0
    for p in paras:
        # +2 approximates the newlines we add when joining
        if buf and total + len(p) + 2 > max_chars:
            chunk = "\n\n".join(buf)
            chunks.append(chunk)
            tail = chunk[-overlap:] if overlap > 0 else ""
            buf, total = ([tail] if tail else []), len(tail)
        buf.append(p)
        total += len(p) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks

def chunk_text(text: str, chunk_size: int = 1024, overlap: int = 200) -> List[str]:
    """Legacy simple chunking - replaced by make_chunks for better results."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def tokenize(s: str) -> List[str]:
    # very simple tokenizer; good enough for BM25 demo
    return re.findall(r"[a-zA-Z0-9]+", s.lower())


def rrf_merge(list_a: List[str], list_b: List[str], k: int = 60, topn: int = 5) -> List[str]:
    """Reciprocal Rank Fusion for two ranked lists of IDs."""
    from collections import defaultdict
    scores = defaultdict(float)
    for lst in [list_a, list_b]:
        for rank, _id in enumerate(lst):
            scores[_id] += 1.0 / (k + rank + 1)
    return [x for x, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)][:topn]


def neural_rerank(query: str, candidates: List[Tuple[str, str]], model: CrossEncoder, final_k: int = 5) -> Tuple[List[str], List[Tuple[str, float]]]:
    """
    Neural reranking using Cross-Encoder.

    Args:
        query: The search query
        candidates: List of (doc_id, doc_text) tuples
        model: Loaded CrossEncoder model
        final_k: Number of top results to return

    Returns:
        Tuple of (list of doc_ids sorted by neural relevance scores, list of all (doc_id, score) tuples)
    """
    if not candidates:
        return [], []

    # Create query-document pairs for scoring
    pairs = [[query, doc_text] for _, doc_text in candidates]

    # Predict relevance scores (higher = more relevant)
    scores = model.predict(pairs)

    # Combine IDs with scores and sort by score (descending)
    scored_candidates = [(doc_id, float(score)) for (doc_id, _), score in zip(candidates, scores)]
    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    # Return top final_k IDs and all scored candidates
    return [doc_id for doc_id, _ in scored_candidates[:final_k]], scored_candidates


INDEX_DIR = Path("index")
INDEX_DIR.mkdir(exist_ok=True)
BM25_CORPUS_PKL = INDEX_DIR / "bm25_corpus_tokens.pkl"
BM25_IDS_PKL = INDEX_DIR / "bm25_ids.pkl"

CHROMA_DIR = Path(".chroma")
COLLECTION_NAME = "books"
LLM_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"


def ingest(dir_path: str, embedding_model: str = "nomic-embed-text"):
    src = Path(dir_path)
    docs = read_text_files(src)
    if not docs:
        raise SystemExit(f"No .txt/.md files found under: {src}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    all_ids, all_metadatas, all_docs = [], [], []
    bm25_tokens, bm25_ids = [], []

    print(f"[ingest] Reading and chunking {len(docs)} file(s)...")
    for file_path, text in docs.items():
        # Use improved paragraph-aware chunking
        chunks = make_chunks(text)

        # Use SHA1-based deterministic IDs
        base = hashlib.sha1(str(Path(file_path).resolve()).encode()).hexdigest()[:12]

        for idx, ch in enumerate(chunks):
            uid = f"{base}-{idx}"
            all_ids.append(uid)
            all_docs.append(ch)
            all_metadatas.append({"source": file_path, "chunk": idx})
            bm25_tokens.append(tokenize(ch))
            bm25_ids.append(uid)

    print(f"[ingest] Embedding {len(all_docs)} chunks with Ollama ({embedding_model})...")
    embeddings = []
    for ch in tqdm(all_docs):
        e = ollama.embeddings(model=embedding_model, prompt=ch)
        embeddings.append(e["embedding"])

    print("[ingest] Upserting into Chroma...")
    # batched add to avoid payload limits; handle duplicates gracefully
    BATCH = 256
    added = 0
    for i in range(0, len(all_ids), BATCH):
        batch_end = min(i + BATCH, len(all_ids))
        try:
            collection.add(
                ids=all_ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                documents=all_docs[i:batch_end],
                metadatas=all_metadatas[i:batch_end],
            )
            added += (batch_end - i)
        except Exception as e:
            # likely duplicates; add one by one to skip existing
            for j in range(i, batch_end):
                try:
                    collection.add(
                        ids=[all_ids[j]],
                        embeddings=[embeddings[j]],
                        documents=[all_docs[j]],
                        metadatas=[all_metadatas[j]]
                    )
                    added += 1
                except Exception:
                    pass

    print("[ingest] Writing BM25 corpus tokens...")
    with open(BM25_CORPUS_PKL, "wb") as f:
        pickle.dump(bm25_tokens, f)
    with open(BM25_IDS_PKL, "wb") as f:
        pickle.dump(bm25_ids, f)

    print(f"[ingest] Done. {added}/{len(all_ids)} chunks stored.")

def _embed(text: str) -> list[float]:
    """Support both prompt= and input= depending on client version."""
    try:
        return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]
    except TypeError:
        return ollama.embeddings(model=EMBED_MODEL, input=text)["embedding"]

def _generate(prompt: str) -> str:
    """Generate text using the LLM."""
    out = ollama.generate(model=LLM_MODEL, prompt=prompt, stream=False)
    return out.get("response", "")

def _get_collection():
    """Get or create the Chroma collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

def cmd_init():
    """Quick environment check (Ollama + Chroma)."""
    print("== Init: quick environment check ==")
    emb = _embed("hello world")
    print(f"Embedding length: {len(emb)} (OK)")
    resp = _generate("Reply with: RAG ready.")
    print(f"LLM said: {resp.strip()}")
    col = _get_collection()
    print(f"Chroma collection: {col.name} (OK)")

    # Check BM25 index
    if BM25_CORPUS_PKL.exists() and BM25_IDS_PKL.exists():
        print(f"BM25 index found: {INDEX_DIR}")
    else:
        print(f"BM25 index not found (run 'ingest' first)")

    print("Init complete ✅")

def cmd_stats():
    """Show number of chunks in the collection."""
    col = _get_collection()
    try:
        count = col.count()
    except Exception:
        count = "unknown"
    print(f"Chunks in Chroma collection: {count}")

    # BM25 stats
    if BM25_IDS_PKL.exists():
        with open(BM25_IDS_PKL, "rb") as f:
            bm25_ids = pickle.load(f)
        print(f"Chunks in BM25 index: {len(bm25_ids)}")
    else:
        print("BM25 index not found")

def cmd_reset():
    """Delete the local Chroma folder and BM25 index."""
    removed = []
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        removed.append(str(CHROMA_DIR))
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
        removed.append(str(INDEX_DIR))

    if removed:
        print(f"Removed: {', '.join(removed)} (index reset).")
    else:
        print("Nothing to reset.")

def _load_bm25() -> Tuple[BM25Okapi, List[str]]:
    with open(BM25_CORPUS_PKL, "rb") as f:
        tokens = pickle.load(f)
    with open(BM25_IDS_PKL, "rb") as f:
        ids = pickle.load(f)
    bm25 = BM25Okapi(tokens)
    return bm25, ids

def ask(query: str,
        llm_model: str = "llama3.2:3b",
        embedding_model: str = "nomic-embed-text",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        k_each: int = 50,
        rerank_k: int = 25,
        final_k: int = 5,
        expand_queries: int = 0,
        verbose: bool = False):
    """
    Multi-Stage Variable Funnel for RAG:
    Stage 0: Dynamic Query Expansion (optional, LLM-powered)
    Stage 1: Broad Hybrid Retrieval (BM25 + Vector × all queries)
    Stage 2: RRF Fusion and Trimming (merged → rerank_k)
    Stage 3: Neural Reranking (rerank_k → rerank_k scored)
    Stage 4: Final Selection (top final_k for LLM)
    """

    # ========== STAGE 0: DYNAMIC QUERY EXPANSION ==========
    if expand_queries > 0:
        print(f"\n🎯 Stage 0: Query Expansion (generating {expand_queries} variations)...")
        all_queries = expand_query(query, expand_queries, llm_model, verbose)
        print(f"   ✓ Expanding search to {len(all_queries)} queries total (1 original + {len(all_queries)-1} variations)")

        if verbose:
            print("\n   === Query Variations ===\n")
            print(f"   Original: {all_queries[0]}")
            for i, var in enumerate(all_queries[1:], 1):
                print(f"   Variation {i}: {var}")
            print()
    else:
        all_queries = [query]
        if verbose:
            print(f"\n🔍 Stage 0: Skipped (using single query)")

    # ========== STAGE 1: BROAD HYBRID RETRIEVAL (MULTI-QUERY) ==========
    print(f"\n🔍 Stage 1: Broad Hybrid Retrieval ({len(all_queries)} queries × {k_each} per engine)...")

    # Load BM25 and Chroma once
    bm25, bm25_ids = _load_bm25()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    # Track per-query statistics for verbose mode
    per_query_stats = []

    # Collect candidates from all query variations
    all_bm25_ids = []
    all_vec_ids = []
    all_bm25_scores = {}  # doc_id -> best score across all queries

    for q_idx, q in enumerate(all_queries):
        # BM25 retrieval
        q_tokens = tokenize(q)
        scores = bm25.get_scores(q_tokens)
        bm25_top_idx = list(reversed(sorted(range(len(scores)), key=lambda i: scores[i])))[:k_each]
        q_bm25_ids = [bm25_ids[i] for i in bm25_top_idx]
        q_bm25_scores = [scores[i] for i in bm25_top_idx]

        # Track best BM25 score for each document
        for doc_id, score in zip(q_bm25_ids, q_bm25_scores):
            if doc_id not in all_bm25_scores or score > all_bm25_scores[doc_id]:
                all_bm25_scores[doc_id] = score

        all_bm25_ids.extend(q_bm25_ids)

        # Vector retrieval
        q_emb = ollama.embeddings(model=embedding_model, prompt=q)["embedding"]
        vec = collection.query(query_embeddings=[q_emb], n_results=k_each)
        q_vec_ids = vec["ids"][0]
        all_vec_ids.extend(q_vec_ids)

        # Track statistics for this query
        if verbose or len(all_queries) > 1:
            unique_before = len(set(all_bm25_ids[:-len(q_bm25_ids)] + all_vec_ids[:-len(q_vec_ids)]))
            unique_after = len(set(all_bm25_ids + all_vec_ids))
            new_docs = unique_after - unique_before if q_idx > 0 else unique_after
            duplicate_docs = len(q_bm25_ids) + len(q_vec_ids) - new_docs

            per_query_stats.append({
                'query': q,
                'query_idx': q_idx,
                'bm25_count': len(q_bm25_ids),
                'vec_count': len(q_vec_ids),
                'new_docs': new_docs,
                'duplicate_docs': duplicate_docs
            })

    # Deduplicate while preserving order
    bm25_top_ids = list(dict.fromkeys(all_bm25_ids))
    vec_ids = list(dict.fromkeys(all_vec_ids))
    all_initial_ids = list(dict.fromkeys(bm25_top_ids + vec_ids))

    overlap_count = len(set(bm25_top_ids) & set(vec_ids))
    print(f"   ✓ Retrieved {len(all_initial_ids)} unique candidates across {len(all_queries)} queries")
    print(f"      (BM25: {len(bm25_top_ids)}, Vector: {len(vec_ids)}, Overlap: {overlap_count})")

    # Show detailed retrieval stats if verbose or multi-query
    if verbose and len(all_queries) > 1:
        print("\n   === Per-Query Retrieval Stats ===\n")
        for stat in per_query_stats:
            q_label = "original" if stat['query_idx'] == 0 else f"variation {stat['query_idx']}"
            print(f"   Query {stat['query_idx']} ({q_label}):")
            print(f"      \"{stat['query']}\"")
            print(f"      → {stat['new_docs']} new docs, {stat['duplicate_docs']} duplicates")
        print()

    # Show initial retrieval details if verbose (show top results from first query)
    if verbose:
        # For verbose output, show top results from the ORIGINAL query only
        q_tokens = tokenize(all_queries[0])
        scores = bm25.get_scores(q_tokens)
        bm25_top_idx = list(reversed(sorted(range(len(scores)), key=lambda i: scores[i])))[:min(k_each, 10)]
        display_bm25_ids = [bm25_ids[i] for i in bm25_top_idx]
        display_bm25_scores = [scores[i] for i in bm25_top_idx]

        q_emb = ollama.embeddings(model=embedding_model, prompt=all_queries[0])["embedding"]
        vec = collection.query(query_embeddings=[q_emb], n_results=min(k_each, 10))
        display_vec_ids = vec["ids"][0]
        display_vec_distances = vec["distances"][0] if "distances" in vec else None

        # Fetch metadata for display
        display_ids = list(dict.fromkeys(display_bm25_ids + display_vec_ids))
        if display_ids:
            initial_meta_data = collection.get(ids=display_ids)
            id_to_meta_verbose = dict(zip(initial_meta_data["ids"], initial_meta_data["metadatas"]))

            print(f"\n   === Detailed Stage 1 Results (Original Query) ===\n")
            print(f"   BM25 Top {len(display_bm25_ids)} Results:")
            for rank, (doc_id, score) in enumerate(zip(display_bm25_ids, display_bm25_scores), 1):
                meta = id_to_meta_verbose.get(doc_id, {})
                src = Path(meta.get("source", "unknown")).name
                chunk_num = meta.get("chunk", "?")
                print(f"     {rank}. {src} [chunk {chunk_num}] (score: {score:.4f})")

            print(f"\n   Vector Search Top {len(display_vec_ids)} Results:")
            for rank, doc_id in enumerate(display_vec_ids, 1):
                meta = id_to_meta_verbose.get(doc_id, {})
                src = Path(meta.get("source", "unknown")).name
                chunk_num = meta.get("chunk", "?")
                dist_info = f" (distance: {display_vec_distances[rank-1]:.4f})" if display_vec_distances else ""
                print(f"     {rank}. {src} [chunk {chunk_num}]{dist_info}")
            print()

    # ========== STAGE 2: RRF FUSION + TRIMMING ==========
    print(f"⚡ Stage 2: RRF Fusion + Filter (trimming to top {rerank_k})...")

    # Use RRF to merge and rank all candidates
    # Get all candidates with RRF scores (not limited yet)
    from collections import defaultdict
    rrf_scores = defaultdict(float)
    k = 60  # RRF constant
    for lst in [bm25_top_ids, vec_ids]:
        for rank, _id in enumerate(lst):
            rrf_scores[_id] += 1.0 / (k + rank + 1)

    # Sort by RRF score and trim to rerank_k
    rrf_ranked_all = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
    trimmed_ids = [doc_id for doc_id, _ in rrf_ranked_all[:rerank_k]]

    print(f"   ✓ Fused {len(all_initial_ids)} candidates → Trimmed to top {len(trimmed_ids)} (RRF scores from both engines)")

    if verbose:
        print("\n   === Detailed Stage 2 Results ===\n")
        print(f"   RRF-Ranked Top {rerank_k} (before neural reranking):")
        # Fetch metadata for trimmed results
        trimmed_meta_data = collection.get(ids=trimmed_ids)
        id_to_meta_trimmed = dict(zip(trimmed_meta_data["ids"], trimmed_meta_data["metadatas"]))

        for rank, (doc_id, rrf_score) in enumerate(rrf_ranked_all[:rerank_k], 1):
            meta = id_to_meta_trimmed.get(doc_id, {})
            src = Path(meta.get("source", "unknown")).name
            chunk_num = meta.get("chunk", "?")
            print(f"     {rank}. {src} [chunk {chunk_num}] (RRF score: {rrf_score:.4f})")
        print()

    # ========== STAGE 3: DEEP NEURAL RERANKING ==========
    print(f"🧠 Stage 3: Deep Neural Reranking ({rerank_k} candidates with Cross-Encoder)...")

    # Fetch documents for neural reranking
    candidates_data = collection.get(ids=trimmed_ids)
    candidates = [(doc_id, doc_text) for doc_id, doc_text in zip(candidates_data["ids"], candidates_data["documents"])]

    # Load Cross-Encoder model and perform neural reranking
    cross_encoder = CrossEncoder(cross_encoder_model)
    fused_ids, neural_scores = neural_rerank(query, candidates, cross_encoder, final_k=rerank_k)

    print(f"   ✓ Neural scoring complete. Top {final_k} will be selected for LLM context.")

    if verbose:
        print("\n   === Detailed Stage 3 Results ===\n")
        print(f"   Neural Reranking (all {len(neural_scores)} scored candidates):")
        for rank, (doc_id, score) in enumerate(neural_scores[:min(10, len(neural_scores))], 1):
            meta = id_to_meta_trimmed.get(doc_id, {})
            src = Path(meta.get("source", "unknown")).name
            chunk_num = meta.get("chunk", "?")
            print(f"     {rank}. {src} [chunk {chunk_num}] (neural score: {score:.4f})")
        if len(neural_scores) > 10:
            print(f"     ... (showing top 10 of {len(neural_scores)})")
        print()

    # ========== STAGE 4: FINAL SELECTION ==========
    print(f"🎯 Stage 4: Final Selection (top {final_k} for LLM context)")

    # Take top final_k from neural-reranked results
    final_ids = fused_ids[:final_k]

    print(f"   ✓ Pipeline complete: {len(all_initial_ids)} → {len(trimmed_ids)} → {len(final_ids)} documents\n")

    # Show ranking impact if verbose
    if verbose:
        print("   === Ranking Impact Analysis ===\n")
        # Create initial rank mappings
        bm25_ranks = {doc_id: rank + 1 for rank, doc_id in enumerate(bm25_top_ids)}
        vec_ranks = {doc_id: rank + 1 for rank, doc_id in enumerate(vec_ids)}
        rrf_ranks = {doc_id: rank + 1 for rank, doc_id in enumerate(trimmed_ids)}

        for final_rank, doc_id in enumerate(final_ids, 1):
            bm25_rank = bm25_ranks.get(doc_id, None)
            vec_rank = vec_ranks.get(doc_id, None)
            rrf_rank = rrf_ranks.get(doc_id, None)

            # Find neural score
            neural_score = next((score for _id, score in neural_scores if _id == doc_id), None)

            sources = []
            if bm25_rank is not None:
                sources.append(f"BM25 #{bm25_rank}")
            if vec_rank is not None:
                sources.append(f"Vec #{vec_rank}")
            if rrf_rank is not None:
                sources.append(f"RRF #{rrf_rank}")

            source_str = " + ".join(sources) if sources else "Not in top-k"

            meta = id_to_meta_trimmed.get(doc_id, {})
            src = Path(meta.get("source", "unknown")).name
            chunk_num = meta.get("chunk", "?")

            neural_str = f" | Neural: {neural_score:.4f}" if neural_score else ""
            print(f"     Final #{final_rank}: {src} [chunk {chunk_num}] ← {source_str}{neural_str}")
        print()

    # Fetch final docs for context
    got = collection.get(ids=final_ids)
    id_to_doc = dict(zip(got["ids"], got["documents"]))
    id_to_meta = dict(zip(got["ids"], got["metadatas"]))

    # Build context with simple headers
    sections = []
    for _id in final_ids:
        meta = id_to_meta[_id]
        src = Path(meta["source"]).name
        sections.append(f"Source: {src} [chunk {meta['chunk']}]\n{id_to_doc[_id]}")
    context = "\n\n---\n\n".join(sections)

    system = (
        "You are a concise assistant for a retrieval-augmented CLI.\n"
        "Answer ONLY using the provided context. If the answer is not present, say you don't know."
    )
    user = f"Context:\n\n{context}\n\nQuestion: {query}"

    if verbose:
        print("\n=== System Prompt ===\n")
        print(system)
        print("\n=== User Prompt ===\n")
        print(user)
        print("\n" + "=" * 50 + "\n")

    resp = ollama.chat(
        model=llm_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        options={"temperature": 0.2}
    )
    answer = resp["message"]["content"].strip()

    # Show the sources under the answer
    print("\n=== Answer ===\n")
    print(answer)
    print("\n--- Sources ---")
    for _id in final_ids:
        m = id_to_meta[_id]
        print(f"{Path(m['source']).name}  (chunk {m['chunk']})")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Hybrid RAG: Multi-Stage Variable Funnel (Query Expansion → BM25 + Vector → RRF → Neural → Selection)")
    sp = p.add_subparsers(dest="cmd", required=True)

    sp.add_parser("init", help="Quick environment check (Ollama + Chroma + BM25)")

    p_ing = sp.add_parser("ingest", help="Ingest .txt/.md files under a directory")
    p_ing.add_argument("--dir", required=True, help="Folder or file path to ingest")
    p_ing.add_argument("--embed-model", default=EMBED_MODEL, help="Embedding model name")

    p_ask = sp.add_parser("ask", help="Ask a question using multi-stage variable funnel")
    p_ask.add_argument("--query", required=True, help="Question string")
    p_ask.add_argument("--llm", default=LLM_MODEL, help="LLM model name")
    p_ask.add_argument("--embed-model", default=EMBED_MODEL, help="Embedding model name")
    p_ask.add_argument("--expand-queries", type=int, default=0,
                       help="Stage 0: Generate N query variations using LLM (0=disabled, 1-10 recommended) - widens funnel for better recall")
    p_ask.add_argument("--k-each", type=int, default=50,
                       help="Stage 1: Top-k from each retriever per query (BM25 + Vector) - default: 50 for broad recall")
    p_ask.add_argument("--rerank-k", type=int, default=25,
                       help="Stage 2: Trim RRF-fused results to this many candidates before neural reranking - default: 25")
    p_ask.add_argument("--final-k", type=int, default=5,
                       help="Stage 4: Final top-k after neural reranking to use as LLM context - default: 5")
    p_ask.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                       help="Cross-encoder model for Stage 3 neural reranking")
    p_ask.add_argument("--verbose", action="store_true",
                       help="Show detailed results for all 5 stages, per-query stats, ranking impact analysis, and complete LLM prompt")
    sp.add_parser("stats", help="Show number of chunks in both indices")
    sp.add_parser("reset", help="Delete Chroma and BM25 indices")

    args = p.parse_args()

    try:
        if args.cmd == "init":
            cmd_init()
        elif args.cmd == "ingest":
            ingest(args.dir, embedding_model=args.embed_model)
        elif args.cmd == "ask":
            ask(args.query, llm_model=args.llm, embedding_model=args.embed_model,
                cross_encoder_model=args.cross_encoder_model,
                k_each=args.k_each, rerank_k=args.rerank_k, final_k=args.final_k,
                expand_queries=args.expand_queries, verbose=args.verbose)
        elif args.cmd == "stats":
            cmd_stats()
        elif args.cmd == "reset":
            cmd_reset()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)

