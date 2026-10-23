#!/usr/bin/env python3
"""Paired bootstrap of macro-F1 differences for the Phase-2 decisions (no API calls):
Joint - Dual-LLM and Joint - Joint (no OCDD), per judge, CV (400 pooled held-out predictions) and external.
Writes results/<FINAL_RUN>/experiments/paired_f1.csv."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cmot.evaluation import holm, paired_bootstrap_diff  # noqa: E402


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    run = ROOT / (ROOT / "results" / "FINAL_RUN.txt").read_text(encoding="utf-8").strip()
    (run / "experiments").mkdir(exist_ok=True)
    B, seed = cfg["statistics"]["bootstrap"], int(cfg["seed"])
    rows = []
    for proto, f in [("cv", "phase2_cv_per_conversation.csv"), ("external", "external_phase2_per_conversation.csv")]:
        pc = pd.read_csv(run / "tables" / f)
        for judge, g in pc.groupby("judge"):
            w = {wf: d.set_index("conv_id").sort_index() for wf, d in g.groupby("workflow")}
            y = w["Joint"]["label"].to_numpy()
            fr = {wf: (d["outcome"] == "fraudulent").astype(int).to_numpy() for wf, d in w.items()}
            for other in ["Dual-LLM", "Joint (no OCDD)"]:
                rows.append({"protocol": proto, "judge": judge, "comparison": f"Joint - {other}",
                             **paired_bootstrap_diff(y, fr["Joint"], fr[other], "macro_f1", B, seed)})
    res = pd.DataFrame(rows)
    res["p_holm"] = np.nan
    for _, g in res.groupby(["protocol", "comparison"]):
        res.loc[g.index, "p_holm"] = holm(g["p_two_sided"].tolist())
    res.to_csv(run / "experiments" / "paired_f1.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
