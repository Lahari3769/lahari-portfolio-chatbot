import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv

print("🔥 RUNNING THIS app.py FILE 🔥")

# -------------------------------
# App Setup
# -------------------------------
load_dotenv()

app = Flask(__name__)

# Manual CORS - more reliable than flask-cors
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    response.headers['Access-Control-Max-Age'] = '3600'
    return response

print("Backend starting...")

# -------------------------------
# Pre-load Vector Store at Startup
# -------------------------------
print("🔄 Pre-loading vector store...")
try:
    from vector_store import retrieve_context
    print("✅ Vector store ready!")
except Exception as e:
    print(f"❌ Vector store loading failed: {e}")
    retrieve_context = None

# -------------------------------
# Hugging Face API Configuration
# -------------------------------
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def call_llm(prompt):
    """Call Hugging Face API directly using requests"""
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 500
    }
    
    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    
    return response.json()["choices"][0]["message"]["content"].strip()

print("Backend ready")

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
# Explicit OPTIONS handler for ALL routes
# -------------------------------
@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    return '', 204

# -------------------------------
# Chat Endpoint (JSON – Render-safe)
# -------------------------------
@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    print(f"📨 Received {request.method} request to /chat")
    
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        print("✅ OPTIONS request handled")
        return '', 204

    try:
        data = request.get_json(force=True)
        question = data.get("question", "").strip()
        print(f"❓ Question: {question}")

        if not question:
            return jsonify({"answer": "Please ask a question."}), 400

        if retrieve_context is None:
            return jsonify({"answer": "Vector store not loaded. Please try again later."}), 503

        context = retrieve_context(question)
        print(f"📚 Context retrieved: {len(context) if context else 0} chars")

        if not context:
            return jsonify({
                "answer": "This information is not available in the portfolio."
            })

        prompt = SYSTEM_PROMPT.format(
            context=context,
            question=question
        )

        print("🤖 Calling LLM...")
        answer = call_llm(prompt)
        print(f"✅ Answer generated: {len(answer)} chars")
        
        return jsonify({
            "answer": answer or "This information is not available in the portfolio."
        })

    except Exception as e:
        print(f"❌ LLM ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "answer": "Something went wrong on the server."
        }), 500

# -------------------------------
# Health Check
# -------------------------------
@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "vector_store": "loaded" if retrieve_context else "not_loaded"}

# -------------------------------
# Root endpoint for testing
# -------------------------------
@app.route("/", methods=["GET"])
def root():
    return {"message": "Lahari Portfolio Chatbot API", "status": "running"}

# -------------------------------
# Debug: print routes AFTER registration
# -------------------------------
print("📍 Registered routes:")
for rule in app.url_map.iter_rules():
    print(f"  {rule}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)