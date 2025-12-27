import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

print("🔄 Pre-loading vector store in WSGI...")

# Import and immediately trigger loading
from vector_store import get_collection, get_embedder

# Force load immediately
collection = get_collection()
embedder = get_embedder()

print(f"✅ Vector store ready! Collection has {collection.count()} documents")

# Now import the app
from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)