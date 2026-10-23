#!/usr/bin/env python3
"""2x2 factorial for the drift detector: reference (dialogue's own window vs normal messages) x message
representation (TF-IDF+SVD vs te3-large embeddings). Same 5 CV folds as the Phase-2 CV protocol; every
configuration is selected inside the training folds. No API calls (embeddings are cached).

    python experiments/ocdd_factorial.py          # writes results/<FINAL_RUN>/experiments/ocdd_factorial.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_all  # noqa: E402
from cmot.data import load_seconvo  # noqa: E402
from cmot.dialogue_drift import build_spaces, evaluate as window_evaluate, score, tune_on_train  # noqa: E402
from cmot.llm_client import make_client  # noqa: E402
from cmot.ocdd_normal import EmbeddingSpace, MemoEmbedder, NormalOCDD, evaluate_detector, select_on_train  # noqa: E402


def main() -> None:
    run_all.load_dotenv(ROOT / ".env")
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    run = ROOT / (ROOT / "results" / "FINAL_RUN.txt").read_text(encoding="utf-8").strip()
    (run / "experiments").mkdir(exist_ok=True)
    seed, B = int(cfg["seed"]), cfg["statistics"]["bootstrap"]
    og, old = cfg["phase2"]["ocdd"], {**cfg["phase2"]["ocdd_grid"], "reference": ["dialogue"]}
    tol = cfg["phase2"]["localization_tolerance"]
    client = make_client(cfg["api"], ROOT / cfg["cache_dir"], offline=False)
    emb = MemoEmbedder(client, cfg["representations"]["embedding"]["provider"], cfg["representations"]["embedding"]["model"])
    convs = load_seconvo(ROOT / cfg["data"]["dir"], download=False)
    msgs = [m for c in convs for m in c.messages]
    emb(msgs)
    spaces = {"tfidf": build_spaces(msgs, sorted(set(og["tfidf_dims"]) | set(old["dims"])), seed),
              "emb": {d: EmbeddingSpace(emb, d, seed).fit(msgs) for d in og["emb_dims"]}}
    vecs = {(r, d): {c.conv_id: sp.transform(c.messages) for c in convs} for r, sd in spaces.items() for d, sp in sd.items()}
    y = np.array([c.label for c in convs])
    rc = cfg["protocols"]["repeated_cv"]
    folds = list(RepeatedStratifiedKFold(n_splits=rc["n_splits"], n_repeats=rc["n_repeats"],
                                         random_state=seed).split(np.zeros(len(y)), y))[:rc["n_splits"]]
    frames = {}
    for k, (a, b) in enumerate(folds):
        tr, te = [convs[i] for i in a], [convs[i] for i in b]
        for rep in ["tfidf", "emb"]:
            # within-dialogue window reference (previous design), tuned on the training fold
            sp = {d: spaces[rep][d] for d in (old["dims"] if rep == "tfidf" else og["emb_dims"])}
            _, best = tune_on_train(tr, sp, {**old, "dims": list(sp)}, seed)
            p, d = best["dialogue"]
            frames.setdefault(("within-dialogue window", rep), []).append(
                window_evaluate(te, vecs[(rep, d)], p, None, tol))
            # normal-message reference, selected on the training fold
            vs = {key: v for key, v in vecs.items() if key[0] == rep and key[1] in og[f"{rep}_dims"]}
            _, bn = select_on_train(tr, vs, {**og, "design": ["normal"]}, seed, og.get("outer_folds", 5))
            V = vecs[(bn.rep, bn.dims)]
            frames.setdefault(("normal-message reference", rep), []).append(
                evaluate_detector(NormalOCDD(bn, seed).fit(tr, V), te, V, tol))
        print(f"fold {k + 1}/{len(folds)} done", flush=True)
    rows = []
    for (ref, rep), parts in frames.items():
        ev = pd.concat(parts)
        lo, hi = run_all.drift_balanced_ci(ev, B, seed)
        rows.append({"reference": ref, "representation": "TF-IDF+SVD" if rep == "tfidf" else "te3-large embeddings",
                     **score(ev), "balanced_lo": lo, "balanced_hi": hi})
    res = pd.DataFrame(rows)
    res.to_csv(run / "experiments" / "ocdd_factorial.csv", index=False)
    pd.set_option("display.width", 200)
    print(res[["reference", "representation", "detect_rate", "localized_rate", "false_drift_rate", "balanced",
               "balanced_lo", "balanced_hi"]].round(3).to_string(index=False))
    print(f"spent: ${client.spent_usd:.2f}")


if __name__ == "__main__":
    main()
