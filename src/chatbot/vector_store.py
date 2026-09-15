"""Persistent vector store for shipment documents.

Uses FAISS when it's installed and falls back to a brute-force numpy
cosine-similarity index otherwise (this project's requirements.txt skips
faiss-cpu on macOS/Darwin), so ingestion and search work the same way on
any machine and the on-disk format is identical either way.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    text: str
    source: str


class PersistentVectorStore:
    """Embeds document chunks and persists them (+ index) to a directory."""

    INDEX_FILE = "index.faiss"
    EMB_FILE = "embeddings.npy"
    META_FILE = "metadata.json"

    def __init__(self, embed_model_name: str = "all-MiniLM-L6-v2"):
        self.embed_model_name = embed_model_name
        self._model: Optional[SentenceTransformer] = None
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self._faiss_index = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embed_model_name)
        return self._model

    def add_chunks(self, chunks: List[Chunk]) -> None:
        self.chunks.extend(chunks)

    def build(self) -> None:
        if not self.chunks:
            raise ValueError("No chunks to index; call add_chunks first")
        texts = [c.text for c in self.chunks]
        embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self.embeddings = embs.astype("float32")

        if faiss is not None:
            index = faiss.IndexFlatIP(self.embeddings.shape[1])
            index.add(self.embeddings)
            self._faiss_index = index

    def search(self, query: str, k: int = 3) -> List[Tuple[str, str, float]]:
        """Returns up to k (text, source, score) tuples, most relevant first."""
        if self.embeddings is None or not self.chunks:
            return []
        qv = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        qv = qv.astype("float32")
        k = min(k, len(self.chunks))

        if self._faiss_index is not None:
            scores, idx = self._faiss_index.search(qv.reshape(1, -1), k)
            idx, scores = idx[0], scores[0]
        else:
            sims = self.embeddings @ qv
            idx = np.argsort(-sims)[:k]
            scores = sims[idx]

        results = []
        for i, score in zip(idx, scores):
            if i < 0:
                continue
            chunk = self.chunks[int(i)]
            results.append((chunk.text, chunk.source, float(score)))
        return results

    def save(self, path: str) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / self.EMB_FILE, self.embeddings)
        with open(out / self.META_FILE, "w") as f:
            json.dump(
                {
                    "embed_model_name": self.embed_model_name,
                    "chunks": [{"text": c.text, "source": c.source} for c in self.chunks],
                    "backend": "faiss" if self._faiss_index is not None else "numpy",
                },
                f,
                indent=2,
            )
        if self._faiss_index is not None:
            faiss.write_index(self._faiss_index, str(out / self.INDEX_FILE))

    @classmethod
    def load(cls, path: str) -> "PersistentVectorStore":
        src = Path(path)
        with open(src / cls.META_FILE) as f:
            meta = json.load(f)

        store = cls(embed_model_name=meta["embed_model_name"])
        store.chunks = [Chunk(text=c["text"], source=c["source"]) for c in meta["chunks"]]
        store.embeddings = np.load(src / cls.EMB_FILE).astype("float32")

        if meta.get("backend") == "faiss" and faiss is not None and (src / cls.INDEX_FILE).exists():
            store._faiss_index = faiss.read_index(str(src / cls.INDEX_FILE))

        return store
