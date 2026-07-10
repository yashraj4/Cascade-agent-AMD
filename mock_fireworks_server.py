"""
Mock Fireworks-compatible server for local testing.
"""
from flask import Flask, request, jsonify
import random

app = Flask(__name__)


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    body = request.get_json()
    model = body.get("model", "unknown")
    messages = body.get("messages", [])
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")

    # Canned response for testing
    fake_answer = f"[mock:{model}] answer for: {user_msg[:60]}"
    fake_tokens = random.randint(20, 150)

    return jsonify({
        "choices": [{"message": {"role": "assistant", "content": fake_answer}}],
        "usage": {"total_tokens": fake_tokens},
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
