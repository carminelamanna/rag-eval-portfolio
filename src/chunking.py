def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    if not text:
        return []

    stride = chunk_size - overlap
    chunks: list[str] = []

    for start in range(0, len(text), stride):
        chunk = text[start : start + chunk_size]
        if not chunk:
            break
        chunks.append(chunk)
        if start + chunk_size >= len(text):
            break

    return chunks
