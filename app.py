"""Serveur Flask — API du Portfolio Bot."""

import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

import config
from llm_interface import init_llm
from data_processing import build_index
from query_engine import create_query_engine, ask

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── App Flask ──
app = Flask(__name__)
CORS(app, origins=[o for o in config.ALLOWED_ORIGINS if o])

# ── Initialisation au démarrage ──
logger.info("Initialisation du LLM et de l'index...")
init_llm()
_index  = build_index()
_engine = create_query_engine(_index)
logger.info("Portfolio Bot prêt.")


# ── Routes ──

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Portfolio Bot is running."})


@app.route("/ask", methods=["POST"])
def ask_route():
    data = request.get_json(silent=True)

    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field."}), 400

    question = data["question"]

    if len(question) > 500:
        return jsonify({"error": "Question too long (max 500 chars)."}), 400

    answer = ask(_engine, question)
    return jsonify({"answer": answer})


@app.route("/rebuild", methods=["POST"])
def rebuild_route():
    """
    Reconstruit l'index depuis les PDFs (utile après ajout de nouveaux docs).
    Protégé par un token simple.
    """
    token = request.headers.get("X-Rebuild-Token", "")
    if token != config.OPENAI_API_KEY[:8]:   # utilise les 8 premiers chars de la clé comme token
        return jsonify({"error": "Unauthorized"}), 401

    global _index, _engine
    _index  = build_index(force_rebuild=True)
    _engine = create_query_engine(_index)
    return jsonify({"status": "Index rebuilt successfully."})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.FLASK_PORT, debug=config.FLASK_DEBUG)