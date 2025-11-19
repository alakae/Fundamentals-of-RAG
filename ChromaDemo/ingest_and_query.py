import os
import glob
from pathlib import Path
import chromadb

# -------- (A) Choose persistence location --------
DB_DIR = Path("chroma_db")
DB_DIR.mkdir(exist_ok=True)

# Create a persistent client (data survives restarts)
client = chromadb.PersistentClient(path=str(DB_DIR))  # loads existing DB if present
# Docs: PersistentClient persists to .chroma/ or your path. 
# https://docs.trychroma.com/docs/run-chroma/persistent-client

# -------- (B) Choose an embedding function --------
# Option 1: Chroma’s default (no imports needed) -> comment out custom EF below to use default.
embedding_fn = None

# Option 2: SentenceTransformers (uncomment to use)
# from chromadb.utils import embedding_functions
# embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
#     model_name="all-MiniLM-L6-v2"
# )

# -------- (C) Get or create a collection --------
# If embedding_fn is None, Chroma uses its default embedding function.
collection = client.get_or_create_collection(
    name="demo_texts",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}   # cosine distance is typical for semantic search
)

# -------- (D) Helper: simple chunker (optional) --------
def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200):
    """Split long text into overlapping chunks to improve recall."""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap if (end - overlap) > start else end
    # keep non-empty only
    return [c for c in chunks if c]

# -------- (E) Read .txt files and prepare records --------
DOCS_DIR = Path("data")
paths = sorted(glob.glob(str(DOCS_DIR / "*.txt")))
documents = []
metadatas = []
ids = []

for p in paths:
    file_id_base = Path(p).stem  # e.g., "01_intro"
    with open(p, "r", encoding="utf-8") as f:
        raw = f.read()

    # chunk if long, else just one chunk
    chunks = chunk_text(raw, max_chars=1500, overlap=200) if len(raw) > 1800 else [raw]

    for idx, ch in enumerate(chunks):
        uid = f"{file_id_base}__{idx:03d}"
        documents.append(ch)
        metadatas.append({
            "source": os.path.basename(p),
            "chunk": idx
        })
        ids.append(uid)
        
# -------- (F) Upsert (add or replace by id) --------
# Tip: If you want strict "add" that fails on duplicates, use collection.add(...)
# get_or_create_collection + add can be repeated; but to avoid dupes, we remove existing ids first.
if ids:
    # Remove any existing records with these ids (idempotent re-run)
    try:
        collection.delete(ids=ids)
    except Exception:
        pass

    # Add documents in batches to avoid exceeding max batch size
    BATCH_SIZE = 5000  # Safe batch size under ChromaDB's limit
    total_records = len(ids)
    
    for i in range(0, total_records, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, total_records)
        batch_docs = documents[i:batch_end]
        batch_metas = metadatas[i:batch_end]
        batch_ids = ids[i:batch_end]
        
        collection.add(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids
        )
        print(f"  Added batch {i//BATCH_SIZE + 1}: {len(batch_ids)} records")
    
    print(f"✅ Ingested {total_records} records from {len(paths)} file(s).")
else:
    print("⚠️ No .txt files found in ./data")
    
# -------- (G) Run a few demo queries --------
def search(query_text, k=4, where=None):
    res = collection.query(
        query_texts=[query_text],
        n_results=k,
        where=where  # optional metadata filter dict
    )
    print(f"\n🔎 Query: {query_text}")
    for i in range(len(res["ids"][0])):
        doc = res["documents"][0][i]
        meta = res["metadatas"][0][i]
        id_ = res["ids"][0][i]
        dist = res.get("distances", [[None]])[0][i]
        print(f"  • id={id_}  dist={dist:.4f}  source={meta.get('source')}\n    {doc[:180].replace('\\n',' ')}...")
        
# Generic search
search("Why does Macbeth decide to kill Duncan?", k=3)
