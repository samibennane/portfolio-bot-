"""Interface LLM — OpenAI via LlamaIndex."""

import logging
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

import config

logger = logging.getLogger(__name__)


def init_llm() -> None:
    """Configure LlamaIndex pour utiliser OpenAI (LLM + embeddings)."""
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY manquante dans les variables d'environnement.")

    Settings.llm = OpenAI(
        model=config.LLM_MODEL,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_NEW_TOKENS,
        api_key=config.OPENAI_API_KEY,
    )

    Settings.embed_model = OpenAIEmbedding(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
    )

    logger.info(f"LLM initialisé : {config.LLM_MODEL}")
    logger.info(f"Embedding initialisé : {config.EMBEDDING_MODEL}")