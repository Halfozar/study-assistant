# AI-Powered Personalized Study Assistant (GenAI + RAG)

Capstone project - Arya Suku (253007) and Ben Flis Ziya (253018), CSDA.

Upload notes (PDF / DOCX / TXT / MD) and the assistant can **summarize**, **explain concepts**, **answer questions**
from your material (with Retrieval-Augmented Generation), and **generate quizzes**. Questions that are not covered
by the material are declined instead of answered from guesswork.

## Architecture

```
file/text -> ingest -> sentence-aware chunking -> embeddings (TF-IDF or MiniLM) -> top-k retrieval
          -> prompt with retrieved context -> backend (extractive | OpenAI | local HF) -> answer + sources
```

| File | Purpose |
| --- | --- |
| `study_assistant/config.py` | every tunable parameter |
| `study_assistant/ingest.py`, `chunker.py` | loading and chunking |
| `study_assistant/retriever.py` | vector search (cosine similarity) |
| `study_assistant/llm.py` | generation backends + prompts |
| `study_assistant/assistant.py` | `ask`, `explain`, `summarize`, `quiz` |
| `app.py` | Streamlit web UI |
| `evaluate.py` | evaluation, parameter sweeps, RAG vs no-RAG |
| `tests/test_pipeline.py` | unit tests |
| `data/` | sample notes and a labelled test set |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py          # web app; tick "Use the bundled sample notes" to try it
```

Python usage:

```python
from study_assistant import Config, StudyAssistant
bot = StudyAssistant(Config())            # offline defaults
bot.load_file("data/sample_notes.txt")
print(bot.ask("What is overfitting?")["answer"])
print(bot.quiz("neural networks", n=3)["answer"])
```

### Backends

| Backend | Needs | Notes |
| --- | --- | --- |
| `extractive` (default) | nothing | offline; selects sentences from your material, so it cannot hallucinate |
| `openai` | `pip install openai`, `export OPENAI_API_KEY=...` | fluent generated answers (`gpt-4o-mini` by default) |
| `hf` | `pip install transformers torch` | local `google/flan-t5-base`, downloads the model once |

Better embeddings: `pip install sentence-transformers`, then use `--embedder minilm`.

## Testing parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `chunk_size` | 120 | words per chunk |
| `chunk_overlap` | 30 | words shared between chunks |
| `top_k` | 4 | chunks given to the generator |
| `min_score` | 0.05 tfidf / 0.30 minilm | below this similarity the question is declined |
| `embedder` | tfidf | `tfidf` or `minilm` |
| `backend` | extractive | `extractive`, `openai`, `hf` |
| `use_rag` | True | False = no-retrieval baseline |
| `temperature` | 0.2 | LLM backends only |
| `level` | beginner | explanation difficulty |
| `summary_sentences`, `num_questions` | 5, 5 | output size |

```bash
python -m unittest discover -s tests -v                 # unit tests
python evaluate.py                                      # one run with defaults
python evaluate.py --chunk-size 80 --overlap 20 --top-k 3
python evaluate.py --sweep                              # grid: chunk size x top-k
python evaluate.py --compare-rag --backend openai       # with vs without RAG
python evaluate.py --notes my_notes.pdf --tests my_tests.json
```

Each run writes `results/<run>.json` (per-question detail) and `results/summary.csv`.

### Metrics

- **retrieval_hit@k** - share of test questions whose retrieved chunks contain an expected keyword.
- **answer_keyword_recall** - share of expected keywords present in the answer.
- **faithfulness** - proxy: share of answer sentences whose content words appear in the retrieved context.
  It is a cheap automatic check, not a substitute for manual rating.
- **out_of_scope_declined** - share of unrelated questions the assistant correctly refuses.
- **avg_latency_s** - mean time per question.

### Test set format (`data/test_set.json`)

```json
{"in_scope": [{"q": "What is overfitting?", "keywords": ["noise", "training"]}],
 "out_of_scope": [{"q": "What is the capital of Australia?"}]}
```

Write your own 15-30 questions for your own course material for the report. The bundled set only covers the sample notes.

## Reference run (sample notes, offline backend)

Obtained with `python evaluate.py` (TF-IDF, extractive, chunk 120 / overlap 30 / top-k 4, 14 in-scope and 4 out-of-scope questions):

| chunks | retrieval_hit@k | answer_keyword_recall | faithfulness | out_of_scope_declined |
| --- | --- | --- | --- | --- |
| 6 | 1.00 | 0.86 | 1.00 | 1.00 |

These are measurements on a small, easy sample; rerun the evaluation on your own materials and report those numbers.
The extractive backend is faithful by construction (it copies sentences), so faithfulness is only a meaningful comparison
for the `openai` / `hf` backends - run `--compare-rag` with one of them to measure the real effect of RAG.

## Limitations

- Extractive answers can be choppy and cannot combine facts across sentences; use an LLM backend for fluent answers.
- Quiz questions from the extractive backend are fill-in-the-blank; LLM backends produce MCQ and short-answer questions.
- Scanned PDFs need OCR before upload. Always verify important answers against the source.
