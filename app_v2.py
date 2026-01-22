import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from typing import Iterator

import chromadb
import ollama

CHROMA_PATH = Path("./.chroma")
COLLECTION_NAME = "hello_rag"   # keep same as v1 so the index persists
LLM_MODEL = "llama3.3:latest"
EMBED_MODEL = "nomic-embed-text"
TOP_K = 5   

def _embed(text: str) -> list[float]:
    """Support both prompt= and input= depending on client version."""
    try:
        return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]
    except TypeError:
        return ollama.embeddings(model=EMBED_MODEL, input=text)["embedding"]


def _generate(prompt: str) -> str:
    out = ollama.generate(model=LLM_MODEL, prompt=prompt, stream=False)
    return out.get("response", "")

def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

def _split_paragraphs(text: str) -> list[str]:
    # split on blank lines; keep non-empty paragraphs
    parts = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n")]
    return [p for p in parts if p]


def make_chunks(text: str, max_chars: int = 800, overlap: int = 150) -> list[str]:
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


def _iter_files(root: Path) -> Iterator[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
            yield p
            
def cmd_init():
    print("== Init: quick environment check ==")
    emb = _embed("hello world")
    print(f"Embedding length: {len(emb)} (OK)")
    resp = _generate("Reply with: RAG ready.")
    print(f"LLM said: {resp.strip()}")
    col = _get_collection()
    print(f"Chroma collection: {col.name} (OK)")
    print("Init complete ✅")
    
def cmd_ingest(dir_path: Path):
    col = _get_collection()
    files = list(_iter_files(dir_path))
    if not files:
        print(f"No .txt/.md found under {dir_path}")
        return

    total_chunks, added = 0, 0
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"(warn) Could not read {path}: {e}")
            continue

        chunks = make_chunks(text)
        total_chunks += len(chunks)
        if not chunks:
            continue

        # Deterministic ids: hash(path) + chunk index
        base = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]
        ids = [f"{base}-{i}" for i in range(len(chunks))]
        metadatas = [{"source": str(path), "chunk": i} for i in range(len(chunks))]
        embeddings = []
        for ch in chunks:
            embeddings.append(_embed(ch))

        # Try bulk add; on duplicate ids, fall back to per-chunk adds
        try:
            col.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
            added += len(ids)
        except Exception as e:
            # likely duplicates; add one by one to skip existing
            for i in range(len(ids)):
                try:
                    col.add(ids=[ids[i]], documents=[chunks[i]],
                            embeddings=[embeddings[i]], metadatas=[metadatas[i]])
                    added += 1
                except Exception:
                    pass
    print(f"Ingestion complete. {added}/{total_chunks} chunks stored.")

def _semantic_search(question: str, k: int = TOP_K):
    q_emb = _embed(question)
    col = _get_collection()
    res = col.query(
        query_embeddings=[q_emb],
        n_results=max(1, k),
        include=["documents", "metadatas", "distances"],
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    hits = []
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append({"text": doc, "meta": meta, "distance": float(dist)})
    return hits

def _build_prompt(question: str, hits: list[dict]) -> tuple[str, list[str]]:
    blocks, citations = [], []
    for i, h in enumerate(hits, 1):
        blocks.append(f"Source {i}:\n{h['text']}")
        src = f"[{i}] {h['meta'].get('source','unknown')}#chunk-{h['meta'].get('chunk',0)}"
        citations.append(src)
    ctx = "\n\n".join(blocks)
    prompt = (
        "You are a helpful assistant for DevOps teams. "
        "Answer the QUESTION using ONLY the CONTEXT. "
        "If the answer is not in the context, say you don't know. "
        "Cite sources in the form [1], [2], etc.\n\n"
        f"CONTEXT:\n{ctx}\n\n"
        f"QUESTION: {question}\n"
        "FINAL ANSWER:"
    )
    return prompt, citations

def cmd_ask(question: str, k: int):
    hits = _semantic_search(question, k=k)
    if not hits:
        print("No results found. Did you run ingest?")
        return
    prompt, citations = _build_prompt(question, hits)
    answer = _generate(prompt)
    print("\n=== Answer ===")
    print(answer.strip())
    print("\n=== Sources ===")
    for s in citations:
        print(s)
        
def cmd_stats():
    col = _get_collection()
    try:
        count = col.count()
    except Exception:
        count = "unknown"
    print(f"Chunks in collection: {count}")
    
def cmd_reset():
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)
        print(f"Removed {CHROMA_PATH} (index reset).")
    else:
        print("Nothing to reset.")
        
def main(argv=None):
    parser = argparse.ArgumentParser(description="Video 2: ingest local files and ask questions.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Quick environment check (Ollama + Chroma).")

    p_ingest = sub.add_parser("ingest", help="Ingest .txt/.md under a directory.")
    p_ingest.add_argument("--dir", type=Path, default=Path("data"))

    p_ask = sub.add_parser("ask", help="Ask a question over the ingested corpus.")
    p_ask.add_argument("q", type=str, help="Question string")
    p_ask.add_argument("--k", type=int, default=TOP_K, help="Top-k chunks to use")

    sub.add_parser("stats", help="Show number of chunks.")
    sub.add_parser("reset", help="Delete the local Chroma folder (.chroma).")

    args = parser.parse_args(argv)
    if args.cmd == "init":
        cmd_init()
    elif args.cmd == "ingest":
        cmd_ingest(args.dir)
    elif args.cmd == "ask":
        cmd_ask(args.q, args.k)
    elif args.cmd == "stats":
        cmd_stats()
    elif args.cmd == "reset":
        cmd_reset()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    