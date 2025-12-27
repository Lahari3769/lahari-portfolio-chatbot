import os
import chromadb
import requests

os.environ["ANONYMIZED_TELEMETRY"] = "False"

print("📦 Loading vector store...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
EMBEDDING_API_URL = "https://router.huggingface.co/v1/embeddings"

_chroma_client = None
_collection = None

def get_collection():
    global _chroma_client, _collection

    if _collection is None:
        print("🔹 Initializing Chroma client...")
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _chroma_client.get_or_create_collection("portfolio")

    return _collection

def get_embedding(text: str):
    """Get embeddings from Hugging Face API instead of local model"""
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "input": text
    }
    
    response = requests.post(EMBEDDING_API_URL, headers=headers, json=payload, timeout=10)
    response.raise_for_status()
    
    return response.json()["data"][0]["embedding"]

def retrieve_context(query: str, k: int = 6) -> str:
    collection = get_collection()
    
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents"]
    )

    docs = results.get("documents", [[]])[0]

    if not docs:
        return ""

    return "\n\n".join(docs)