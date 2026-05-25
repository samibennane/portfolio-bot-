"""Moteur de requête RAG — LlamaIndex + OpenAI."""

import logging
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.core.query_engine import BaseQueryEngine

import config

logger = logging.getLogger(__name__)

PORTFOLIO_PROMPT = PromptTemplate(
    """
You are Sami Bennane's personal AI assistant on his portfolio website.
Use ONLY the context below to answer. Do not invent facts.
Reply in the same language as the question (French or English).
Be concise (2-4 sentences max) and warm.

Context:
{context_str}

Question: {query_str}
Answer:
""".strip()
)


def create_query_engine(index: VectorStoreIndex) -> BaseQueryEngine:
    """Crée le query engine RAG à partir de l'index."""
    engine = index.as_query_engine(
        streaming=False,
        similarity_top_k=config.SIMILARITY_TOP_K,
        text_qa_template=PORTFOLIO_PROMPT,
    )
    logger.info("Query engine RAG initialisé.")
    return engine


def ask(engine: BaseQueryEngine, question: str) -> str:
    """
    Pose une question au query engine et retourne la réponse textuelle.
    """
    if not question or not question.strip():
        return "Please ask me something about Sami!"

    try:
        response = engine.query(question.strip())
        answer = str(response).strip()
        return answer if answer else "I couldn't find relevant information to answer that."
    except Exception as e:
        logger.error(f"Erreur lors de la requête : {e}")
        return "Sorry, an error occurred while processing your question."