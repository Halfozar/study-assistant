"""Vector retrieval. 'tfidf' works offline; 'minilm' uses sentence-transformers."""
import numpy as np


class Retriever:
    def __init__(self, embedder: str = "tfidf"):
        self.embedder = embedder
        self.chunks = []
        self._matrix = None
        self._vec = None
        self._model = None

    def index(self, chunks: list):
        self.chunks = chunks
        if not chunks:
            self._matrix = None
            return
        if self.embedder == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
            self._matrix = self._vec.fit_transform(chunks)
        elif self.embedder == "minilm":
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self._matrix = np.asarray(self._model.encode(chunks, normalize_embeddings=True))
        else:
            raise ValueError(f"Unknown embedder: {self.embedder}")

    def search(self, query: str, k: int = 4) -> list:
        """Return [(chunk, cosine_score), ...] best first."""
        if self._matrix is None:
            return []
        if self.embedder == "tfidf":
            q = self._vec.transform([query])
            scores = (self._matrix @ q.T).toarray().ravel()  # rows are L2-normalised -> cosine
        else:
            q = self._model.encode([query], normalize_embeddings=True)[0]
            scores = self._matrix @ q
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]
