from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import chromadb
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# -------------------------------
# Setup
# -------------------------------
load_dotenv()

app = Flask(__name__)

ALLOWED_ORIGIN = "https://majetilahari-portfolio.vercel.app"

CORS(
    app,
    resources={r"/*": {"origins": ALLOWED_ORIGIN}},
    supports_credentials=False
)

# -------------------------------
# FORCE CORS ON ALL RESPONSES (CRITICAL)
# -------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS, GET"
    return response

# -------------------------------
# ChromaDB (READ ONLY)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="portfolio",
    embedding_function=None
)

# -------------------------------
# HF Client
# -------------------------------
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

SYSTEM_PROMPT = """
You are an AI assistant for the portfolio of Majeti Lahari.

- Majeti Lahari is female (she/her).
- Use plain professional text.
- Do not use markdown or emojis.
- Answer ONLY from the given context.
- If missing info say exactly:
  "This information is not available in the portfolio."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

# -------------------------------
# Helpers
# -------------------------------
def retrieve_context(query, k=4):
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents"]
    )
    docs = results.get("documents", [[]])[0]
    return "\n\n".join(docs) if docs else ""

# -------------------------------
# Preflight
# -------------------------------
@app.route("/chat/stream", methods=["OPTIONS"])
def options():
    return make_response("", 204)

# -------------------------------
# Chat API
# -------------------------------
@app.route("/chat/stream", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True) or {}
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"answer": "Invalid question"}), 400

        context = retrieve_context(question)

        if not context:
            return jsonify({
                "answer": "This information is not available in the portfolio."
            })

        prompt = SYSTEM_PROMPT.format(
            context=context,
            question=question
        )

        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )

        text = response.choices[0].message.content.strip()

        if not text:
            text = "This information is not available in the portfolio."

        return jsonify({"answer": text})

    except Exception as e:
        print("SERVER ERROR:", e)
        return jsonify({"answer": "Something went wrong."}), 500

# -------------------------------
# Health
# -------------------------------
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# -------------------------------
# Entry
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
