"""All tunable (testing) parameters live here."""
from dataclasses import dataclass, asdict
from typing import Optional

DEFAULT_MIN_SCORE = {"tfidf": 0.05, "minilm": 0.30}


@dataclass
class Config:
    # --- chunking ---
    chunk_size: int = 120        # words per chunk
    chunk_overlap: int = 30      # words shared between consecutive chunks
    # --- retrieval ---
    embedder: str = "tfidf"      # "tfidf" (offline) | "minilm" (sentence-transformers)
    top_k: int = 4               # chunks passed to the generator
    min_score: Optional[float] = None  # below this the assistant declines (None = embedder default)
    # --- generation ---
    backend: str = "extractive"  # "extractive" (offline) | "openai" | "hf"
    use_rag: bool = True         # False = no retrieval (baseline for comparison)
    temperature: float = 0.2
    max_tokens: int = 400
    openai_model: str = "gpt-4o-mini"
    hf_model: str = "google/flan-t5-base"
    # --- personalization ---
    level: str = "beginner"      # "beginner" | "intermediate" | "advanced"
    summary_sentences: int = 5
    num_questions: int = 5

    def threshold(self) -> float:
        if self.min_score is not None:
            return self.min_score
        return DEFAULT_MIN_SCORE.get(self.embedder, 0.1)

    def to_dict(self) -> dict:
        return asdict(self)
