import os
import chromadb
from sentence_transformers import SentenceTransformer

print("Loading vector DB and models...")

# ----------------------------
# Chroma DB (ABSOLUTE PATH)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("portfolio")

# ----------------------------
# Embedding Model
# ----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Vector DB loaded")

# ----------------------------
# Query Function (USED BY app.py)
# ----------------------------
def query_chroma(question: str, k: int = 4) -> str:
    embedding = embedder.encode(question).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=k
    )

    documents = results.get("documents", [[]])[0]

    if not documents:
        return ""

    # Simple context join (clean & fast)
    return "\n\n".join(documents)
