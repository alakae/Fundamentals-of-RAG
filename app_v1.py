import argparse
from pathlib import Path
import sys
import time

import chromadb
import ollama

CHROMA_PATH = Path("./.chroma")
COLLECTION_NAME = "hello_rag"
LLM_MODEL = "llama3.3:latest"
EMBED_MODEL = "nomic-embed-text"

def _embed(text: str) -> list[float]:
    """
    Use Ollama embeddings. Some versions expect 'prompt=', others 'input='.
    This tries both for compatibility.
    """
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

def cmd_init():
    print("== Init: environment check ==")

    # 1) Quick embedding check
    emb = _embed("hello world")
    print(f"Embedding length: {len(emb)} (OK)")

    # 2) Quick LLM generation check
    resp = _generate("Reply with: RAG ready.")
    print(f"LLM said: {resp.strip()}")

    # 3) Quick Chroma check
    col = _get_collection()
    print(f"Chroma collection: {col.name} (OK)")

    print("Init complete ✅")
    
def cmd_demo():
    print("== Demo: the tiniest RAG you can run ==")

    # 0) One tiny 'document' in memory (no files yet)
    doc_text = (
        "Runbook: Payments Service\n"
        "- SLO: p95 latency 200ms; error rate <0.1%\n"
        "- Rollback: run scripts/rollback.sh\n"
        "- Escalation: page #oncall and notify SRE.\n"
    )
    doc_id = "doc-1"
    print("Indexing 1 tiny in-memory doc...")

    # 1) Store in Chroma with our own embedding
    col = _get_collection()
    doc_emb = _embed(doc_text)
    try:
        col.add(
            ids=[doc_id],
            documents=[doc_text],
            embeddings=[doc_emb],
            metadatas=[{"source": "in-memory-demo"}],
        )
    except Exception as e:
        # If you rerun, the id might already exist — safe to ignore for this demo
        print(f"(note) add() raised {e!r}; continuing")

    # 2) Ask a question
    question = "What is the p95 latency target?"
    print(f"\nQ: {question}")

    # 3) Retrieve by semantic similarity (top 1 is fine)
    q_emb = _embed(question)
    result = col.query(
        query_embeddings=[q_emb],
        n_results=1,
        include=["documents", "metadatas", "distances"],
    )
    ctx = result["documents"][0][0]
    dist = result["distances"][0][0]
    src = result["metadatas"][0][0]["source"]
    print(f"\nRetrieved context (dist={dist:.3f}, source={src}):\n---\n{ctx}---\n")

    # 4) Ground the LLM on that context (simple prompt)
    prompt = (
        "You are a helpful assistant. Answer the QUESTION using ONLY the CONTEXT. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"CONTEXT:\n{ctx}\n\n"
        f"QUESTION: {question}\n"
        "FINAL ANSWER:"
    )
    answer = _generate(prompt)
    print("Answer:\n", answer.strip())
    print("\nDemo complete ✅")

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Hello RAG (Video 1): simple Ollama + Chroma demo."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="Check Ollama (LLM+embeddings) and Chroma reachability")
    sub.add_parser("demo", help="Index one doc and answer one question")

    args = parser.parse_args(argv)

    if args.cmd == "init":
        cmd_init()
    elif args.cmd == "demo":
        cmd_demo()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)