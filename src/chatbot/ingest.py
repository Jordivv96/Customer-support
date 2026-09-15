"""Ingestion script: chunk shipment docs and build a persistent vector store.

Usage:
    python -m src.chatbot.ingest --docs data/shipment_docs --out data/vector_store
"""
import argparse
from pathlib import Path
from typing import List

from .vector_store import Chunk, PersistentVectorStore


def chunk_text(text: str, source: str, chunk_size: int = 500, overlap: int = 50) -> List[Chunk]:
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, source=source))
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def load_documents(docs_dir: Path, chunk_size: int, overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for path in sorted(docs_dir.glob("**/*")):
        if path.suffix.lower() not in (".txt", ".md"):
            continue
        text = path.read_text(encoding="utf-8")
        chunks.extend(chunk_text(text, source=path.name, chunk_size=chunk_size, overlap=overlap))
    return chunks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", default="data/shipment_docs", help="Directory of .txt/.md docs to ingest")
    parser.add_argument("--out", default="data/vector_store", help="Output directory for the persisted index")
    parser.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args(argv)

    docs_dir = Path(args.docs)
    if not docs_dir.exists():
        raise SystemExit(f"Docs directory not found: {docs_dir}")

    chunks = load_documents(docs_dir, chunk_size=args.chunk_size, overlap=args.overlap)
    if not chunks:
        raise SystemExit(f"No .txt/.md documents found in {docs_dir}")

    store = PersistentVectorStore(embed_model_name=args.embed_model)
    store.add_chunks(chunks)
    store.build()
    store.save(args.out)

    backend = "faiss" if store._faiss_index is not None else "numpy"
    print(f"Ingested {len(chunks)} chunks from {docs_dir} into {args.out} (backend={backend})")


if __name__ == "__main__":
    main()
