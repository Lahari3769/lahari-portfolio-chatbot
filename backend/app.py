from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import chromadb
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# -------------------------------
# App Setup
# -------------------------------
load_dotenv()

app = Flask(__name__)

# Allow ONLY your Vercel domain
ALLOWED_ORIGIN = "https://majetilahari-portfolio.vercel.app"

CORS(
    app,
    resources={r"/chat/*": {"origins": ALLOWED_ORIGIN}},
    methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"]
)

# -------------------------------
# Vector DB
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection("portfolio")
print("Warming up ChromaDB...")
collection.count()
print("ChromaDB ready")

# -------------------------------
# Hugging Face Client
# -------------------------------
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# -------------------------------
# SYSTEM PROMPT
# -------------------------------
SYSTEM_PROMPT = """
You are an AI assistant for the portfolio of Majeti Lahari.

IMPORTANT:
- Majeti Lahari is female. Use she/her only.
- Do NOT use markdown, bullets, or emojis.
- Use plain professional text.
- Answer ONLY from the given context.
- If missing info, say exactly:
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
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return resp


def retrieve_context(query, k=4):
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    docs = results.get("documents", [[]])[0]
    return "\n\n".join(docs) if docs else ""


# -------------------------------
# OPTIONS (Preflight)
# -------------------------------
@app.route("/chat/stream", methods=["OPTIONS"])
def chat_options():
    return add_cors_headers(Response(status=204))


# -------------------------------
# Chat Endpoint (NON-STREAMING)
# -------------------------------
@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        resp = Response(
            "data:Invalid question\n\n",
            mimetype="text/event-stream"
        )
        return add_cors_headers(resp)

    context = retrieve_context(question)

    if not context:
        resp = Response(
            "data:This information is not available in the portfolio.\n\n",
            mimetype="text/event-stream"
        )
        return add_cors_headers(resp)

    prompt = SYSTEM_PROMPT.format(
        context=context,
        question=question
    )

    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400
        )

        text = response.choices[0].message.content.strip()
        if not text:
            text = "This information is not available in the portfolio."

        resp = Response(
            f"data:{text}\n\n",
            mimetype="text/event-stream"
        )
        return add_cors_headers(resp)

    except Exception as e:
        print("LLM ERROR:", e)
        resp = Response(
            "data:Something went wrong while generating the response.\n\n",
            mimetype="text/event-stream"
        )
        return add_cors_headers(resp)


# -------------------------------
# Health Check
# -------------------------------
@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
