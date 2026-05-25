"""Configuration centrale du Portfolio Bot."""

import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI ──
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ── LlamaIndex ──
EMBEDDING_MODEL  = "text-embedding-3-small"
LLM_MODEL        = "gpt-4o"
TEMPERATURE      = 0.3
MAX_NEW_TOKENS   = 512
SIMILARITY_TOP_K = 4

# ── Documents ──
DOCS_DIR = os.path.join(os.path.dirname(__file__), "documents")

# ── Flask ──
FLASK_PORT  = int(os.environ.get("PORT", 5000))
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

# ── CORS ──
ALLOWED_ORIGINS = [
    "http://localhost",
    "http://127.0.0.1",
    "https://samibennane.github.io",
    "https://samibennane.fr",
    os.environ.get("FRONTEND_URL", ""),
]

# ── Prompt système ──
SYSTEM_PROMPT = """
You are Sami Bennane's personal AI assistant, embedded on his portfolio website.
You answer questions about Sami — his background, projects, skills, education, and experience.
You speak in the same language as the user (French or English).
Be concise, warm, and professional. Always answer in first person on behalf of Sami.
If you don't know the answer based on the provided context, say so honestly.
Never invent information.
""".strip()