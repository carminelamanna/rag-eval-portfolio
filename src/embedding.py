from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    if not chunks:
        return []

    model = _get_model()
    embeddings = model.encode(chunks, normalize_embeddings=True)
    return embeddings.tolist()
