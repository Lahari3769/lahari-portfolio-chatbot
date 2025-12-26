from flask import Flask, request, Response
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
CORS(
    app,
    origins=["https://majetilahari-portfolio.vercel.app"],
    supports_credentials=False,
    allow_headers=["Content-Type"],
    methods=["POST", "OPTIONS"]
)

# -------------------------------
# Vector DB & Embeddings
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection("portfolio")

# -------------------------------
# Hugging Face Mistral Client
# -------------------------------
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# -------------------------------
# SYSTEM PROMPT (Grounded RAG)
# -------------------------------
SYSTEM_PROMPT = """
You are an AI assistant for the portfolio of Majeti Lahari.

IMPORTANT IDENTITY RULES:
- Majeti Lahari is female.
- Refer to her using "she/her" pronouns consistently.
- Do NOT use "he", "they", or neutral references.

CRITICAL FORMATTING RULES (STRICT):
- DO NOT use Markdown
- DO NOT use *, **, -, #, or any Markdown symbols
- Use plain text only

STYLE RULES:
- Professional, recruiter-friendly tone
- Clean paragraphs
- No emojis
- No excessive formatting

CONTENT RULES:
- Use ONLY the provided context
- Answer strictly from retrieved content
- Do NOT add external knowledge
- If information is missing, say exactly:
  "This information is not available in the portfolio."

====================
CONTEXT:
{context}
====================

QUESTION:
{question}

ANSWER:
"""

# -------------------------------
# Retrieve Context
# -------------------------------
def retrieve_context(query, k=6):
    results = collection.query(
        query_texts=[query],
        n_results=k
    )

    docs = results.get("documents", [[]])[0]
    return "\n\n".join(docs) if docs else ""

# -------------------------------
# Chat Endpoint (SSE)
# -------------------------------
@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()

    if not question:
        return Response(
            "data:Invalid question\n\n",
            mimetype="text/event-stream"
        )

    context = retrieve_context(question)

    if not context:
        return Response(
            "data:This information is not available in the portfolio.\n\n",
            mimetype="text/event-stream"
        )

    prompt = SYSTEM_PROMPT.format(
        context=context,
        question=question
    )

    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )

        text = response.choices[0].message.content.strip()

        if not text:
            text = "This information is not available in the portfolio."

        return Response(
            f"data:{text}\n\n",
            mimetype="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "https://majetilahari-portfolio.vercel.app"
            }
        )

    except Exception as e:
        print("LLM ERROR:", e)
        return Response(
            "data:Something went wrong while generating the response.\n\n",
            mimetype="text/event-stream"
        )

# -------------------------------
# Health Check
# -------------------------------
@app.route("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
