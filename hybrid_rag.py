import argparse
import hashlib
import os
import pickle
import re
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Iterator
from tqdm import tqdm
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import ollama


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
        k_each: int = 6,
        final_k: int = 5,
        verbose: bool = False,
        reranker: str = "rrf"):

    # BM25 side
    bm25, bm25_ids = _load_bm25()
    q_tokens = tokenize(query)
    scores = bm25.get_scores(q_tokens)
    bm25_top_idx = list(reversed(sorted(range(len(scores)), key=lambda i: scores[i])))[:k_each]
    bm25_top_ids = [bm25_ids[i] for i in bm25_top_idx]
    bm25_top_scores = [scores[i] for i in bm25_top_idx]

    # Vector side (Chroma)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    q_emb = ollama.embeddings(model=embedding_model, prompt=query)["embedding"]
    vec = collection.query(query_embeddings=[q_emb], n_results=k_each)
    vec_ids = [doc_id for doc_id in vec["ids"][0]]
    vec_distances = vec["distances"][0] if "distances" in vec else None

    # Show initial retrieval results if verbose
    if verbose:
        # Fetch metadata for display
        all_initial_ids = list(dict.fromkeys(bm25_top_ids + vec_ids))
        initial_meta_data = collection.get(ids=all_initial_ids)
        id_to_meta_verbose = dict(zip(initial_meta_data["ids"], initial_meta_data["metadatas"]))

        print("\n=== Initial Retrieval Results ===\n")
        print(f"BM25 Top {k_each} Results:")
        for rank, (doc_id, score) in enumerate(zip(bm25_top_ids, bm25_top_scores), 1):
            meta = id_to_meta_verbose.get(doc_id, {})
            src = Path(meta.get("source", "unknown")).name
            chunk_num = meta.get("chunk", "?")
            print(f"  {rank}. {src} [chunk {chunk_num}] (score: {score:.4f})")

        print(f"\nVector Search Top {k_each} Results:")
        for rank, doc_id in enumerate(vec_ids, 1):
            meta = id_to_meta_verbose.get(doc_id, {})
            src = Path(meta.get("source", "unknown")).name
            chunk_num = meta.get("chunk", "?")
            dist_info = f" (distance: {vec_distances[rank-1]:.4f})" if vec_distances else ""
            print(f"  {rank}. {src} [chunk {chunk_num}]{dist_info}")
        print()

    # Merge/Rerank based on selected method
    neural_scores = None
    if reranker == "neural":
        # Combine unique candidates from both retrievers
        all_candidate_ids = list(dict.fromkeys(bm25_top_ids + vec_ids))

        if verbose:
            print(f"=== Neural Reranking ===\n")
            print(f"Combined candidate pool: {len(all_candidate_ids)} unique documents")
            print(f"  - From BM25: {len([x for x in all_candidate_ids if x in bm25_top_ids])}")
            print(f"  - From Vector: {len([x for x in all_candidate_ids if x in vec_ids])}")
            print(f"  - Overlap: {len(set(bm25_top_ids) & set(vec_ids))}\n")

        # Fetch documents for neural reranking
        candidates_data = collection.get(ids=all_candidate_ids)
        candidates = [(doc_id, doc_text) for doc_id, doc_text in zip(candidates_data["ids"], candidates_data["documents"])]

        # Load Cross-Encoder model and perform neural reranking
        cross_encoder = CrossEncoder(cross_encoder_model)
        fused_ids, neural_scores = neural_rerank(query, candidates, cross_encoder, final_k)

        if verbose:
            print(f"Neural Reranking Results (Top {final_k}):")
            for rank, (doc_id, score) in enumerate(neural_scores[:final_k], 1):
                meta = id_to_meta_verbose.get(doc_id, {})
                src = Path(meta.get("source", "unknown")).name
                chunk_num = meta.get("chunk", "?")
                print(f"  {rank}. {src} [chunk {chunk_num}] (score: {score:.4f})")
            print()
    else:
        # Default: RRF merge
        fused_ids = rrf_merge(bm25_top_ids, vec_ids, topn=final_k)

        if verbose:
            # Fetch metadata for RRF results
            rrf_meta_data = collection.get(ids=fused_ids)
            id_to_meta_rrf = dict(zip(rrf_meta_data["ids"], rrf_meta_data["metadatas"]))

            print(f"=== RRF Reranking ===\n")
            print(f"Reciprocal Rank Fusion with k=60")
            print(f"Final Top {final_k} Results:")
            for rank, doc_id in enumerate(fused_ids, 1):
                meta = id_to_meta_rrf.get(doc_id, {})
                src = Path(meta.get("source", "unknown")).name
                chunk_num = meta.get("chunk", "?")
                print(f"  {rank}. {src} [chunk {chunk_num}]")
            print()

    # Show ranking impact if verbose
    if verbose:
        print("=== Ranking Impact ===\n")
        # Create initial rank mappings
        bm25_ranks = {doc_id: rank + 1 for rank, doc_id in enumerate(bm25_top_ids)}
        vec_ranks = {doc_id: rank + 1 for rank, doc_id in enumerate(vec_ids)}

        # Fetch metadata for impact display (reuse if already fetched for RRF)
        if reranker == "rrf":
            id_to_meta_impact = id_to_meta_rrf
        else:
            id_to_meta_impact = id_to_meta_verbose

        for final_rank, doc_id in enumerate(fused_ids, 1):
            bm25_rank = bm25_ranks.get(doc_id, None)
            vec_rank = vec_ranks.get(doc_id, None)

            sources = []
            if bm25_rank is not None:
                sources.append(f"BM25 #{bm25_rank}")
            if vec_rank is not None:
                sources.append(f"Vec #{vec_rank}")

            if not sources:
                source_str = "Not in top-k"
            else:
                source_str = " + ".join(sources)

            meta = id_to_meta_impact.get(doc_id, {})
            src = Path(meta.get("source", "unknown")).name
            chunk_num = meta.get("chunk", "?")

            print(f"  Final #{final_rank}: {src} [chunk {chunk_num}] ← {source_str}")
        print()

    # Fetch fused docs for context
    got = collection.get(ids=fused_ids)
    id_to_doc = dict(zip(got["ids"], got["documents"]))
    id_to_meta = dict(zip(got["ids"], got["metadatas"]))

    # Build context with simple headers
    sections = []
    for _id in fused_ids:
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
    for _id in fused_ids:
        m = id_to_meta[_id]
        print(f"{Path(m['source']).name}  (chunk {m['chunk']})")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Hybrid RAG: BM25 + Vector Search with RRF or Neural Cross-Encoder Reranking")
    sp = p.add_subparsers(dest="cmd", required=True)

    sp.add_parser("init", help="Quick environment check (Ollama + Chroma + BM25)")

    p_ing = sp.add_parser("ingest", help="Ingest .txt/.md files under a directory")
    p_ing.add_argument("--dir", required=True, help="Folder or file path to ingest")
    p_ing.add_argument("--embed-model", default=EMBED_MODEL, help="Embedding model name")

    p_ask = sp.add_parser("ask", help="Ask a question using hybrid search (BM25 + Vector)")
    p_ask.add_argument("--query", required=True, help="Question string")
    p_ask.add_argument("--llm", default=LLM_MODEL, help="LLM model name")
    p_ask.add_argument("--embed-model", default=EMBED_MODEL, help="Embedding model name")
    p_ask.add_argument("--k-each", type=int, default=6, help="Top-k from each retriever")
    p_ask.add_argument("--final-k", type=int, default=5, help="Final top-k after fusion")
    p_ask.add_argument("--reranker", choices=["rrf", "neural"], default="rrf",
                       help="Reranking method: 'rrf' (mathematical, fast) or 'neural' (Cross-Encoder, more accurate)")
    p_ask.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                       help="Cross-encoder model for neural reranking")
    p_ask.add_argument("--verbose", action="store_true",
                       help="Show detailed retrieval results, reranking process, and complete prompt given to LLM")
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
                k_each=args.k_each, final_k=args.final_k, verbose=args.verbose,
                reranker=args.reranker)
        elif args.cmd == "stats":
            cmd_stats()
        elif args.cmd == "reset":
            cmd_reset()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)

