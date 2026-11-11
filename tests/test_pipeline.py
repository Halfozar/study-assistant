"""Run with:  python -m unittest discover -s tests -v   (or: pytest)"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from study_assistant import Config, StudyAssistant          # noqa: E402
from study_assistant.chunker import chunk_text               # noqa: E402
from study_assistant.llm import NOT_FOUND                    # noqa: E402

NOTES = ROOT / "data" / "sample_notes.txt"
TESTS = json.loads((ROOT / "data" / "test_set.json").read_text())


def make(**kw):
    bot = StudyAssistant(Config(**kw))
    bot.load_file(NOTES)
    return bot


class ChunkerTests(unittest.TestCase):
    def test_no_empty_and_size_bounded(self):
        text = (NOTES).read_text()
        chunks = chunk_text(text, 80, 20)
        self.assertTrue(chunks and all(c.strip() for c in chunks))
        # a chunk may exceed the size only by one long sentence
        self.assertTrue(all(len(c.split()) <= 80 + 40 for c in chunks))

    def test_overlap_shared(self):
        chunks = chunk_text(NOTES.read_text(), 60, 25)
        shared = sum(1 for a, b in zip(chunks, chunks[1:]) if set(a.split("\n")) & set(b.split("\n")))
        self.assertGreater(shared, 0)

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            chunk_text("a b c", 10, 10)
        with self.assertRaises(ValueError):
            chunk_text("a b c", 0, 0)

    def test_empty_text(self):
        self.assertEqual(chunk_text("", 50, 10), [])


class RetrievalTests(unittest.TestCase):
    def test_top_hit_contains_keyword(self):
        bot = make()
        hits = bot.retriever.search("What is the F1 score?", 1)
        self.assertIn("harmonic", hits[0][0].lower())

    def test_retrieval_hit_rate_at_least_80_percent(self):
        bot = make(top_k=4)
        ok = 0
        for item in TESTS["in_scope"]:
            ctx = " ".join(c for c, _ in bot.retriever.search(item["q"], 4)).lower()
            ok += any(k.lower() in ctx for k in item["keywords"])
        self.assertGreaterEqual(ok / len(TESTS["in_scope"]), 0.8)


class AssistantTests(unittest.TestCase):
    def test_answer_is_grounded(self):
        res = make().ask("Which algorithm is the most popular for clustering?")
        self.assertIn("k-means", res["answer"].lower())
        self.assertFalse(res["declined"])
        self.assertTrue(res["sources"])

    def test_out_of_scope_declined(self):
        bot = make()
        for o in TESTS["out_of_scope"]:
            res = bot.ask(o["q"])
            self.assertTrue(res["declined"], o["q"])
            self.assertEqual(res["answer"], NOT_FOUND)

    def test_no_rag_baseline_has_no_sources(self):
        res = make(use_rag=False).ask("What is overfitting?")
        self.assertEqual(res["sources"], [])

    def test_summary_length(self):
        out = make(summary_sentences=3).summarize()["answer"]
        self.assertTrue(out.strip())
        self.assertLessEqual(out.count(". ") + 1, 4)

    def test_quiz_has_requested_count_and_answers(self):
        out = make().quiz("neural networks", n=3)["answer"]
        self.assertEqual(out.count("Answer:"), 3)
        self.assertEqual(out.count("_____"), 3)

    def test_explain_levels(self):
        out = make(level="beginner").explain("gradient descent")["answer"]
        self.assertTrue(out.startswith("In simple terms"))

    def test_min_score_parameter_changes_behavior(self):
        strict = make(min_score=0.95).ask("What is overfitting?")
        self.assertTrue(strict["declined"])


if __name__ == "__main__":
    unittest.main()
