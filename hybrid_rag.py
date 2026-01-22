import argparse, os, re, json, pickle
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import chromadb
from rank_bm25 import BM25Okapi
import ollama


def read_text_files(root: Path) -> Dict[str, str]:
    files = []
    if root.is_file() and root.suffix.lower() == ".txt":
        files = [root]
    else:
        files = list(root.rglob("*.txt"))
    out = {}
    for f in files:
        try:
            out[str(f)] = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            out[str(f)] = f.read_text(encoding="latin-1", errors="ignore")
    return out

def chunk_text(text: str, chunk_size: int = 1024, overlap: int = 200) -> List[str]:
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


INDEX_DIR = Path("index")
INDEX_DIR.mkdir(exist_ok=True)
BM25_CORPUS_PKL = INDEX_DIR / "bm25_corpus_tokens.pkl"
BM25_IDS_PKL = INDEX_DIR / "bm25_ids.pkl"

CHROMA_DIR = Path(".chroma")
COLLECTION_NAME = "books"


def ingest(dir_path: str, embedding_model: str = "nomic-embed-text"):
    src = Path(dir_path)
    docs = read_text_files(src)
    if not docs:
        raise SystemExit(f"No .txt files found under: {src}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    all_ids, all_metadatas, all_docs = [], [], []
    bm25_tokens, bm25_ids = [], []

    print(f"[ingest] Reading and chunking {len(docs)} file(s)...")
    for file_path, text in docs.items():
        chunks = chunk_text(text)
        for idx, ch in enumerate(chunks):
            uid = f"{file_path}::chunk-{idx}"
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
    # batched add to avoid payload limits
    BATCH = 256
    for i in range(0, len(all_ids), BATCH):
        collection.add(
            ids=all_ids[i:i+BATCH],
            embeddings=embeddings[i:i+BATCH],
            documents=all_docs[i:i+BATCH],
            metadatas=all_metadatas[i:i+BATCH],
        )

    print("[ingest] Writing BM25 corpus tokens...")
    with open(BM25_CORPUS_PKL, "wb") as f:
        pickle.dump(bm25_tokens, f)
    with open(BM25_IDS_PKL, "wb") as f:
        pickle.dump(bm25_ids, f)

    print("[ingest] Done.")
    
def _load_bm25() -> Tuple[BM25Okapi, List[str]]:
    with open(BM25_CORPUS_PKL, "rb") as f:
        tokens = pickle.load(f)
    with open(BM25_IDS_PKL, "rb") as f:
        ids = pickle.load(f)
    bm25 = BM25Okapi(tokens)
    return bm25, ids

def ask(query: str,
        llm_model: str = "llama3.3:latest",
        embedding_model: str = "nomic-embed-text",
        k_each: int = 6,
        final_k: int = 5):

    # BM25 side
    bm25, bm25_ids = _load_bm25()
    q_tokens = tokenize(query)
    scores = bm25.get_scores(q_tokens)
    bm25_top_idx = list(reversed(sorted(range(len(scores)), key=lambda i: scores[i])))[:k_each]
    bm25_top_ids = [bm25_ids[i] for i in bm25_top_idx]

    # Vector side (Chroma)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    q_emb = ollama.embeddings(model=embedding_model, prompt=query)["embedding"]
    vec = collection.query(query_embeddings=[q_emb], n_results=k_each)
    vec_ids = [doc_id for doc_id in vec["ids"][0]]

    # Merge via RRF
    fused_ids = rrf_merge(bm25_top_ids, vec_ids, topn=final_k)

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
    p = argparse.ArgumentParser(description="Hybrid RAG (BM25 + Chroma + Ollama)")
    sp = p.add_subparsers(dest="cmd", required=True)

    p_ing = sp.add_parser("ingest")
    p_ing.add_argument("--dir", required=True, help="Folder or .txt file")
    p_ing.add_argument("--embed-model", default="nomic-embed-text")

    p_ask = sp.add_parser("ask")
    p_ask.add_argument("--query", required=True)
    p_ask.add_argument("--llm", default="llama3.3:latest")
    p_ask.add_argument("--embed-model", default="nomic-embed-text")
    p_ask.add_argument("--k-each", type=int, default=6)
    p_ask.add_argument("--final-k", type=int, default=5)

    args = p.parse_args()
    if args.cmd == "ingest":
        ingest(args.dir, embedding_model=args.embed_model)
    else:
        ask(args.query, llm_model=args.llm, embedding_model=args.embed_model,
            k_each=args.k_each, final_k=args.final_k)

