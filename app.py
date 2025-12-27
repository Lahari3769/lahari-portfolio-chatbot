import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv

print("🔥 RUNNING THIS app.py FILE 🔥")

# Set HuggingFace cache to persistent location
os.environ["HF_HOME"] = "/opt/render/project/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/opt/render/project/.cache/huggingface"

# -------------------------------
# App Setup
# -------------------------------
load_dotenv()

app = Flask(__name__)

# Manual CORS
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    response.headers['Access-Control-Max-Age'] = '3600'
    return response

print("Backend starting...")

# Import retrieve_context (lazy loads)
from vector_store import retrieve_context

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
# SYSTEM PROMPT
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
# OPTIONS handler
# -------------------------------
@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    return '', 204

# -------------------------------
# Chat Endpoint
# -------------------------------
@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    print(f"📨 Received {request.method} request to /chat")
    
    if request.method == "OPTIONS":
        print("✅ OPTIONS request handled")
        return '', 204

    try:
        data = request.get_json(force=True)
        question = data.get("question", "").strip()
        print(f"❓ Question: {question}")

        if not question:
            return jsonify({"answer": "Please ask a question."}), 400

        print("🔍 Retrieving context...")
        context = retrieve_context(question)
        print(f"📚 Context retrieved: {len(context) if context else 0} chars")

        if not context:
            return jsonify({
                "answer": "This information is not available in the portfolio."
            })

        prompt = SYSTEM_PROMPT.format(context=context, question=question)

        print("🤖 Calling LLM...")
        answer = call_llm(prompt)
        print(f"✅ Answer generated: {len(answer)} chars")
        
        return jsonify({"answer": answer or "This information is not available in the portfolio."})

    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"answer": "Something went wrong on the server."}), 500

# -------------------------------
# Health Check
# -------------------------------
@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}

# -------------------------------
# Root endpoint
# -------------------------------
@app.route("/", methods=["GET"])
def root():
    return {"message": "Lahari Portfolio Chatbot API", "status": "running"}

print("📍 Registered routes:")
for rule in app.url_map.iter_rules():
    print(f"  {rule}")