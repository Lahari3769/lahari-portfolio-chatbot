from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# ✅ Import ONLY the query function (code)
from vector_store import query_chroma

app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": "*"}},
)

print("Backend ready")

@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

@app.route("/chat/stream", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Please ask a question."}), 400

    try:
        answer = query_chroma(question)
        return jsonify({
            "answer": answer or "This information is not available in the portfolio."
        })
    except Exception as e:
        print("Chat error:", e)
        return jsonify({
            "answer": "Something went wrong on the server."
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
