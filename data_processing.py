"""Chargement et indexation des documents PDF."""

import logging
import os

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)

from llama_index.readers.structured_data.base import StructuredDataReader

import config

logger = logging.getLogger(__name__)

# Dossier où LlamaIndex persiste l'index (évite de re-embedder à chaque démarrage)
INDEX_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "index_storage")


def load_documents():
    """Charge tous les PDFs du dossier docs/."""
    if not os.path.exists(config.DOCS_DIR):
        raise FileNotFoundError(f"Dossier docs/ introuvable : {config.DOCS_DIR}")

    xlsx_reader = StructuredDataReader(col_index=[0, 1])

    docs = SimpleDirectoryReader(
        input_dir=config.DOCS_DIR,
        required_exts=[".pdf", ".txt", ".xlsx"],
        file_extractor={".xlsx": xlsx_reader},
        recursive=True,
    ).load_data()

    if not docs:
        raise ValueError("Aucun document trouvé dans docs/. Ajoutes-y tes PDFs.")

    logger.info(f"{len(docs)} document(s) chargé(s) depuis {config.DOCS_DIR}")
    return docs


def build_index(force_rebuild: bool = False) -> VectorStoreIndex:
    """
    Construit ou charge l'index vectoriel.
    Si l'index existe déjà sur disque, le recharge (rapide).
    Si force_rebuild=True, reconstruit depuis les PDFs.
    """
    if not force_rebuild and os.path.exists(INDEX_PERSIST_DIR):
        logger.info("Index existant trouvé — chargement depuis le disque.")
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        return index

    logger.info("Construction de l'index vectoriel depuis les documents...")
    documents = load_documents()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)

    # Persistance sur disque
    os.makedirs(INDEX_PERSIST_DIR, exist_ok=True)
    index.storage_context.persist(persist_dir=INDEX_PERSIST_DIR)
    logger.info(f"Index sauvegardé dans {INDEX_PERSIST_DIR}")

    return index