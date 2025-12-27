import os
import chromadb
from sentence_transformers import SentenceTransformer

os.environ["ANONYMIZED_TELEMETRY"] = "False"

print("📦 Loading vector store...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

_chroma_client = None
_collection = None
_embedder = None

def get_collection():
    global _chroma_client, _collection

    if _collection is None:
        print("🔹 Initializing Chroma client...")
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _chroma_client.get_or_create_collection("portfolio")

    return _collection

def get_embedder():
    global _embedder

    if _embedder is None:
        print("🔹 Loading sentence transformer...")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")

    return _embedder

def retrieve_context(query: str, k: int = 6) -> str:
    collection = get_collection()
    embedder = get_embedder()

    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents"]
    )

    docs = results.get("documents", [[]])[0]

    if not docs:
        return ""

    return "\n\n".join(docs)