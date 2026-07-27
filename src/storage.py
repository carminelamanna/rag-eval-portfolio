from __future__ import annotations

import chromadb

from src.embedding import embed_chunks

CHROMA_PATH = "./chroma_db"
_client: chromadb.PersistentClient | None = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def store_chunks(
    chunks: list[str],
    embeddings: list[list[float]],
    collection_name: str,
) -> None:
    client = _get_client()

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(collection_name)
    ids = [str(i) for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, embeddings=embeddings)


def query_similar(
    query_text: str,
    collection_name: str,
    top_k: int = 3,
) -> list[dict[str, object]]:
    client = _get_client()
    collection = client.get_or_create_collection(collection_name)

    query_embedding = embed_chunks([query_text])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    return [
        {"id": doc_id, "document": document, "distance": distance}
        for doc_id, document, distance in zip(ids, documents, distances)
    ]
