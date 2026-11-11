"""Generation backends.

extractive : fully offline, no model download. Selects/rewrites sentences from the retrieved
             context (good for testing the pipeline and as a hallucination-free baseline).
openai     : OpenAI chat models (set OPENAI_API_KEY).
hf         : local Hugging Face seq2seq model (default google/flan-t5-base).
"""
import math
import os
import re
from collections import Counter

from .ingest import split_sentences

STOP = set("""a an the and or but if then of to in on at for with by from as is are was were be been being it its this
that these those which who whom what when where why how do does did can could should would will shall may might not no
so than too very into about over under between during also such their there they them he she his her we you your i our
us more most other some any each both all one two""".split())

NOT_FOUND = "I could not find this in your study material."


def tokens(text: str) -> list:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 2]


# ---------------------------------------------------------------- prompts
LEVEL_HINT = {
    "beginner": "Use very simple language, short sentences and one everyday analogy.",
    "intermediate": "Use clear language and define technical terms briefly.",
    "advanced": "Be precise and technical; assume strong prior knowledge.",
}


def build_prompt(task: str, query: str, context: str, level: str, n: int) -> str:
    rules = ("Use ONLY the study material below. If the answer is not in it, reply exactly: "
             f"'{NOT_FOUND}'\n\n") if context else ""
    material = f"STUDY MATERIAL:\n{context}\n\n" if context else ""
    if task == "answer":
        return f"{rules}{material}Question: {query}\nAnswer concisely:"
    if task == "explain":
        return (f"{rules}{material}Explain the topic '{query}'. {LEVEL_HINT[level]}\nExplanation:")
    if task == "summarize":
        return f"{rules}{material}Summarize the material in about {n} sentences.\nSummary:"
    if task == "quiz":
        return (f"{rules}{material}Write {n} quiz questions"
                f"{' about ' + query if query else ''}. Mix multiple-choice (with options A-D) and short-answer. "
                "After each question write 'Answer:' with the correct answer.\nQuiz:")
    raise ValueError(task)


# ---------------------------------------------------------------- backends
class Backend:
    name = "base"

    def generate(self, task, query, chunks, level="beginner", n=5, temperature=0.2, max_tokens=400) -> str:
        raise NotImplementedError


class ExtractiveBackend(Backend):
    name = "extractive"

    def _ranked_sentences(self, query, chunks):
        sents, seen = [], set()
        for c in chunks:
            for s in split_sentences(c):
                if s not in seen:
                    seen.add(s)
                    sents.append(s)
        q = set(tokens(query))
        df = Counter(w for s in sents for w in set(tokens(s)))
        n_s = max(len(sents), 1)
        scored = []
        for i, s in enumerate(sents):
            t = tokens(s)
            # idf-weighted overlap: rare query words count more than common ones
            overlap = sum(math.log(1 + n_s / df[w]) for w in set(t) if w in q)
            scored.append((overlap / (len(t) ** 0.5 + 1e-9), i, s))
        return sents, sorted(scored, reverse=True)

    def generate(self, task, query, chunks, level="beginner", n=5, temperature=0.2, max_tokens=400):
        if not chunks:
            return NOT_FOUND
        if task in ("answer", "explain"):
            k = 2 if task == "answer" else {"beginner": 2, "intermediate": 3, "advanced": 4}[level]
            _, ranked = self._ranked_sentences(query, chunks)
            best = ranked[0][0] if ranked else 0
            top = [r for r in ranked[:k] if r[0] > 0 and r[0] >= 0.5 * best]   # drop weak matches
            if not top:
                return NOT_FOUND
            text = " ".join(s for _, _, s in sorted(top, key=lambda r: r[1]))
            return ("In simple terms: " + text) if task == "explain" and level == "beginner" else text
        if task == "summarize":
            sents, seen = [], set()
            for c in chunks:                      # de-duplicate sentences repeated by chunk overlap
                for x in split_sentences(c):
                    if x not in seen:
                        seen.add(x); sents.append(x)
            freq = Counter(w for s in sents for w in tokens(s))
            scored = [(sum(freq[w] for w in tokens(s)) / (len(tokens(s)) + 1), i, s) for i, s in enumerate(sents)]
            top = sorted(sorted(scored, reverse=True)[:n], key=lambda r: r[1])
            return " ".join(s for _, _, s in top)
        if task == "quiz":
            return self._quiz(query, chunks, n)
        raise ValueError(task)

    def _quiz(self, query, chunks, n):
        _, ranked = self._ranked_sentences(query or "", chunks)
        pool = [s for _, _, s in ranked] if query else [s for _, _, s in ranked]
        vocab = Counter(w for c in chunks for w in tokens(c))
        out, used = [], 0
        for s in pool:
            if used >= n:
                break
            cands = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{5,}", s) if w.lower() not in STOP]
            if not cands:
                continue
            target = min(cands, key=lambda w: vocab[w.lower()])     # rarest term = most informative
            blanked = re.sub(re.escape(target), "_____", s, count=1)
            distract = [w for w, _ in vocab.most_common(60) if len(w) > 5 and w != target.lower()][:3]
            options = sorted({target.lower(), *distract})
            used += 1
            opts = "  ".join(f"{chr(65 + i)}) {o}" for i, o in enumerate(options))
            letter = chr(65 + options.index(target.lower()))
            out.append(f"Q{used}. Fill in the blank: {blanked}\n    {opts}\n    Answer: {letter}) {target.lower()}")
        return "\n\n".join(out) if out else NOT_FOUND


class OpenAIBackend(Backend):
    name = "openai"

    def __init__(self, model="gpt-4o-mini"):
        from openai import OpenAI
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Set the OPENAI_API_KEY environment variable.")
        self.client, self.model = OpenAI(), model

    def generate(self, task, query, chunks, level="beginner", n=5, temperature=0.2, max_tokens=400):
        prompt = build_prompt(task, query, "\n\n".join(chunks), level, n)
        r = self.client.chat.completions.create(
            model=self.model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": "You are a helpful, accurate study assistant."},
                      {"role": "user", "content": prompt}])
        return r.choices[0].message.content.strip()


class HFBackend(Backend):
    name = "hf"

    def __init__(self, model="google/flan-t5-base"):
        from transformers import pipeline
        self.pipe = pipeline("text2text-generation", model=model)

    def generate(self, task, query, chunks, level="beginner", n=5, temperature=0.2, max_tokens=400):
        prompt = build_prompt(task, query, "\n\n".join(chunks), level, n)
        out = self.pipe(prompt, max_new_tokens=min(max_tokens, 256), do_sample=temperature > 0.5,
                        temperature=max(temperature, 0.01))
        return out[0]["generated_text"].strip()


def get_backend(cfg) -> Backend:
    if cfg.backend == "extractive":
        return ExtractiveBackend()
    if cfg.backend == "openai":
        return OpenAIBackend(cfg.openai_model)
    if cfg.backend == "hf":
        return HFBackend(cfg.hf_model)
    raise ValueError(f"Unknown backend: {cfg.backend}")
