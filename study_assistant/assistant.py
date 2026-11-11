"""Main StudyAssistant: ingestion -> chunking -> retrieval -> generation."""
from .chunker import chunk_text
from .config import Config
from .ingest import clean_text, read_file
from .llm import NOT_FOUND, get_backend
from .retriever import Retriever


class StudyAssistant:
    def __init__(self, config: Config = None, backend=None):
        self.cfg = config or Config()
        self.backend = backend or get_backend(self.cfg)
        self.retriever = Retriever(self.cfg.embedder)
        self.chunks = []
        self.history = []   # [(role, text)] for the chat UI

    # ------------------------------------------------------------ loading
    def load_text(self, text: str):
        text = clean_text(text)
        self.chunks = chunk_text(text, self.cfg.chunk_size, self.cfg.chunk_overlap)
        self.retriever.index(self.chunks)
        return len(self.chunks)

    def load_file(self, path) -> int:
        return self.load_text(read_file(path))

    # ------------------------------------------------------------ helpers
    def _retrieve(self, query):
        if not self.cfg.use_rag:
            return []
        return self.retriever.search(query, self.cfg.top_k)

    def _gen(self, task, query, chunks, n=5):
        return self.backend.generate(task, query, chunks, level=self.cfg.level, n=n,
                                     temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens)

    # ------------------------------------------------------------ features
    def ask(self, question: str) -> dict:
        """Question answering with a similarity threshold to decline out-of-scope questions."""
        hits = self._retrieve(question)
        if self.cfg.use_rag:
            if not hits or hits[0][1] < self.cfg.threshold():
                return {"answer": NOT_FOUND, "sources": [], "scores": [hits[0][1]] if hits else [],
                        "declined": True}
        chunks = [c for c, _ in hits]
        answer = self._gen("answer", question, chunks)
        res = {"answer": answer, "sources": chunks, "scores": [round(s, 3) for _, s in hits],
               "declined": answer.strip().startswith(NOT_FOUND[:25])}
        self.history += [("user", question), ("assistant", answer)]
        return res

    def explain(self, topic: str) -> dict:
        hits = self._retrieve(topic)
        if self.cfg.use_rag and (not hits or hits[0][1] < self.cfg.threshold()):
            return {"answer": NOT_FOUND, "sources": [], "scores": [], "declined": True}
        chunks = [c for c, _ in hits]
        return {"answer": self._gen("explain", topic, chunks), "sources": chunks,
                "scores": [round(s, 3) for _, s in hits], "declined": False}

    def summarize(self, topic: str = None) -> dict:
        if topic:
            chunks = [c for c, _ in self._retrieve(topic)]
        else:  # whole document; cap at 12 evenly spaced chunks to stay within model context limits
            step = max(1, len(self.chunks) // 12)
            chunks = self.chunks[::step][:12]
        return {"answer": self._gen("summarize", topic or "", chunks, n=self.cfg.summary_sentences),
                "sources": chunks}

    def quiz(self, topic: str = None, n: int = None) -> dict:
        n = n or self.cfg.num_questions
        chunks = [c for c, _ in self._retrieve(topic)] if topic else self.chunks[:12]
        return {"answer": self._gen("quiz", topic or "", chunks, n=n), "sources": chunks}
