from typing import List, Optional

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:
    SentenceTransformer = None
    np = None


class SimpleRAG:
    """A minimal in-memory RAG store using sentence-transformers if available.

    This is a placeholder: for production use a persistent vector DB.
    """

    def __init__(self, embed_model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed")
        self.model = SentenceTransformer(embed_model_name)
        self.docs: List[str] = []
        self.embs = None

    def add(self, text: str):
        self.docs.append(text)

    def build_index(self):
        if len(self.docs) == 0:
            self.embs = None
            return
        self.embs = self.model.encode(self.docs, convert_to_numpy=True)

    def search(self, query: str, k: int = 3) -> List[str]:
        if self.embs is None:
            return []
        qv = self.model.encode([query], convert_to_numpy=True)[0]
        sims = (self.embs @ qv) / (
            (np.linalg.norm(self.embs, axis=1) * np.linalg.norm(qv)) + 1e-8
        )
        idx = list(reversed(sims.argsort()))[:k]
        return [self.docs[i] for i in idx]
