"""Evaluate the study assistant with configurable testing parameters.

Examples
--------
python evaluate.py                                   # defaults (offline)
python evaluate.py --chunk-size 80 --overlap 20 --top-k 3
python evaluate.py --sweep                           # grid over chunk_size x top_k
python evaluate.py --compare-rag --backend openai    # RAG vs no-RAG (needs OPENAI_API_KEY)
python evaluate.py --embedder minilm --backend hf
python evaluate.py --notes my_notes.pdf --tests my_tests.json
"""
import argparse
import csv
import itertools
import json
import time
from pathlib import Path

from study_assistant import Config, StudyAssistant
from study_assistant.ingest import split_sentences
from study_assistant.llm import tokens


def faithfulness(answer: str, sources: list, thr: float = 0.6) -> float:
    """Proxy: share of answer sentences whose content words mostly occur in the retrieved context."""
    if not sources:
        return 0.0
    ctx = set(tokens(" ".join(sources)))
    sents = split_sentences(answer.replace("In simple terms:", "").strip())
    if not sents:
        return 0.0
    ok = 0
    for s in sents:
        t = tokens(s)
        if t and sum(w in ctx for w in t) / len(t) >= thr:
            ok += 1
    return ok / len(sents)


def run(cfg: Config, notes: str, tests: dict) -> dict:
    bot = StudyAssistant(cfg)
    n_chunks = bot.load_file(notes)
    hit = kw = faith = 0.0
    lat = []
    rows = []
    for item in tests["in_scope"]:
        kws = [k.lower() for k in item["keywords"]]
        t0 = time.time()
        res = bot.ask(item["q"])
        lat.append(time.time() - t0)
        retrieved = " ".join(res["sources"]).lower()
        h = float(any(k in retrieved for k in kws))                       # retrieval hit
        r = sum(k in res["answer"].lower() for k in kws) / len(kws)       # answer keyword recall
        f = faithfulness(res["answer"], res["sources"])
        hit += h; kw += r; faith += f
        rows.append({"question": item["q"], "retrieval_hit": h, "keyword_recall": round(r, 2),
                     "faithfulness": round(f, 2), "declined": res["declined"], "answer": res["answer"]})
    n = len(tests["in_scope"])
    oos = tests.get("out_of_scope", [])
    declined = sum(bot.ask(o["q"])["declined"] for o in oos)
    metrics = {
        "chunks": n_chunks,
        "retrieval_hit@k": round(hit / n, 3),
        "answer_keyword_recall": round(kw / n, 3),
        "faithfulness": round(faith / n, 3),
        "out_of_scope_declined": round(declined / len(oos), 3) if oos else None,
        "avg_latency_s": round(sum(lat) / len(lat), 3),
    }
    return {"config": cfg.to_dict(), "metrics": metrics, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default="data/sample_notes.txt")
    ap.add_argument("--tests", default="data/test_set.json")
    ap.add_argument("--chunk-size", type=int, default=120)
    ap.add_argument("--overlap", type=int, default=30)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--embedder", default="tfidf", choices=["tfidf", "minilm"])
    ap.add_argument("--backend", default="extractive", choices=["extractive", "openai", "hf"])
    ap.add_argument("--compare-rag", action="store_true", help="run with and without retrieval")
    ap.add_argument("--sweep", action="store_true", help="grid search chunk_size x top_k")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    tests = json.loads(Path(args.tests).read_text())
    Path(args.out).mkdir(exist_ok=True)

    def mk(**kw):
        base = dict(chunk_size=args.chunk_size, chunk_overlap=args.overlap, top_k=args.top_k,
                    min_score=args.min_score, embedder=args.embedder, backend=args.backend)
        base.update(kw)
        return Config(**base)

    configs = []
    if args.sweep:
        for cs, k in itertools.product([60, 120, 200], [1, 2, 4, 6]):
            configs.append((f"chunk{cs}_k{k}", mk(chunk_size=cs, chunk_overlap=cs // 4, top_k=k)))
    elif args.compare_rag:
        configs = [("with_rag", mk(use_rag=True)), ("no_rag", mk(use_rag=False))]
    else:
        configs = [("single_run", mk())]

    summary = []
    for name, cfg in configs:
        out = run(cfg, args.notes, tests)
        (Path(args.out) / f"{name}.json").write_text(json.dumps(out, indent=2))
        summary.append({"run": name, **out["metrics"]})

    cols = list(summary[0].keys())
    print("\n" + " | ".join(f"{c:>22}" for c in cols))
    for r in summary:
        print(" | ".join(f"{str(r[c]):>22}" for c in cols))
    with open(Path(args.out) / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(summary)
    print(f"\nSaved details to {args.out}/ (JSON per run + summary.csv)")


if __name__ == "__main__":
    main()
