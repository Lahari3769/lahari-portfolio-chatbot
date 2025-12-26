from playwright.sync_api import sync_playwright
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import os
from urllib.parse import urlparse

BASE_URL = "http://localhost:5173"

# ----------------------------
# Chroma (ABSOLUTE PATH)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_or_create_collection("portfolio")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# Utils
# ----------------------------
def chunk_text(text, chunk_size=350):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

def normalize_url(url):
    parsed = urlparse(url)
    return parsed.path or "/"

SKIP_EXTENSIONS = (
    ".pdf", ".png", ".jpg", ".jpeg",
    ".svg", ".zip", ".rar", ".exe", ".dmg"
)

def should_skip(url: str) -> bool:
    return url.lower().endswith(SKIP_EXTENSIONS)

# ----------------------------
# Ingest
# ----------------------------
def ingest():
    visited = set()
    to_visit = {BASE_URL}
    total_chunks = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        while to_visit:
            url = to_visit.pop()
            if url in visited or should_skip(url):
                continue

            visited.add(url)

            try:
                page.goto(url, wait_until="networkidle", timeout=15000)
                page.wait_for_timeout(2000)
            except Exception as e:
                print(f"⚠️ Skipping {url}: {e}")
                continue

            # Collect internal links
            links = page.eval_on_selector_all(
                "a[href]",
                "els => els.map(e => e.href)"
            )

            for link in links:
                if link.startswith(BASE_URL) and not should_skip(link):
                    to_visit.add(link)

            # Extract visible text
            text = page.evaluate("() => document.body.innerText")

            if not text or len(text.strip()) < 200:
                print(f"⚠️ No meaningful content at {url}")
                continue

            chunks = chunk_text(text)

            for chunk in chunks:
                collection.add(
                    documents=[chunk],
                    embeddings=[embedder.encode(chunk).tolist()],
                    ids=[str(uuid.uuid4())],
                    metadatas=[{
                        "url": normalize_url(url),
                        "source": "portfolio"
                    }]
                )
                total_chunks += 1

            print(f"✅ Ingested {normalize_url(url)} ({len(chunks)} chunks)")

        browser.close()

    print(f"\n🎯 Total chunks ingested: {total_chunks}")

if __name__ == "__main__":
    ingest()
