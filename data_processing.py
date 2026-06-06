"""Chargement et indexation des documents PDF."""

import logging
import os

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.readers.structured_data.base import StructuredDataReader


import config


logger = logging.getLogger(__name__)

# Dossier où LlamaIndex persiste l'index (évite de re-embedder à chaque démarrage)
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_storage")
CHROMA_COLLECTION  = "portfolio_docs"


def _get_chroma_store() -> ChromaVectorStore:
    """Initialise le client ChromaDB avec HNSW et embedding OpenAI."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    ef = OpenAIEmbeddingFunction(
        api_key=config.OPENAI_API_KEY,
        model_name=config.EMBEDDING_MODEL,  # "text-embedding-3-small"
    )

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=ef,
        configuration={
            "hnsw": {
                "space": "cosine",
                "ef_construction": 200,
                "ef_construction": 200,
                "max_neighbors": 16,
            }
        }
    )

    return ChromaVectorStore(chroma_collection=collection) #here is the wrapping chroma_db to


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
    Construit ou charge l'index vectoriel via ChromaDB.

    - Si la collection existe et n'est pas vide → rechargement rapide.
    - Si force_rebuild=True ou collection vide → re-embedding depuis les documents.
    """
    vector_store    = _get_chroma_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    collection_count = vector_store.client.count()
    already_indexed  = not force_rebuild and collection_count > 0

    if already_indexed:
        logger.info(f"Collection ChromaDB existante ({collection_count} vecteurs) — rechargement.")
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
        )
    else:
        logger.info("Construction de l'index vectoriel depuis les documents...")
        documents = load_documents()

        parser = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50,
        )

        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[parser],
            show_progress=True,
        )
        logger.info(f"Index sauvegardé dans ChromaDB ({CHROMA_PERSIST_DIR})")

    return index