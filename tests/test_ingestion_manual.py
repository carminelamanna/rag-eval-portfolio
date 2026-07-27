import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chunking import chunk_text
from src.embedding import embed_chunks
from src.loader import load_document
from src.storage import query_similar, store_chunks

SAMPLE_PATH = ROOT / "data" / "sample.txt"
COLLECTION_NAME = "sample"
EXAMPLE_QUERY = "Quali sono gli obblighi previsti dal regolamento?"


def main() -> None:
    print(f"Loading document from {SAMPLE_PATH}")
    text = load_document(str(SAMPLE_PATH))
    print(f"Loaded {len(text)} characters")

    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")

    embeddings = embed_chunks(chunks)
    print(f"Generated {len(embeddings)} embeddings")

    store_chunks(chunks, embeddings, COLLECTION_NAME)
    print(f"Stored chunks in collection '{COLLECTION_NAME}'")

    print(f"\nQuery: {EXAMPLE_QUERY}")
    results = query_similar(EXAMPLE_QUERY, COLLECTION_NAME, top_k=3)

    print("\nTop results (lower distance = more similar):")
    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"ID: {result['id']}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Chunk:\n{result['document']}")


if __name__ == "__main__":
    main()
