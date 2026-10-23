#!/usr/bin/env python3
"""Does the OCDD output help the LLM judge? (Phase-2 judge-input experiment, CV setting)

Uses the Joint Phase-1 decisions and OCDD detections of the final run (5-fold CV over the 400 annotated
dialogues; results/<run>/tables/phase2_cv_per_conversation.csv) and re-queries low-cost judges with:

  full           original prompt, whole dialogue (as in the main results)
  loc_ocdd       + the message at which OCDD located the drift
  loc_fixed      + message 4 (control: a location that does not come from the detector)
  prefix_ocdd    dialogue truncated after the OCDD alarm message (decision at alarm time, early warning)
  prefix_fixed   dialogue truncated after message 4 (control)

OCDD variants judge flagged dialogues with a detected drift (the rest are 'spam-like'); fixed variants judge
every flagged dialogue that is long enough. Writes results/<run>/experiments/judge_inputs_*.csv.

    python experiments/judge_inputs.py            # uses results/FINAL_RUN.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_all  # noqa: E402
from cmot.data import load_seconvo  # noqa: E402
from cmot.evaluation import bootstrap_ci, compute_metrics, holm, paired_bootstrap_diff  # noqa: E402
from cmot.llm_client import ModelSpec, make_client  # noqa: E402
from cmot.llm_tasks import run_paper_judgments  # noqa: E402

FIXED = 4
VARIANTS = ["full", "loc_ocdd", "loc_fixed", "prefix_ocdd", "prefix_fixed"]


def main() -> None:
    run_all.load_dotenv(ROOT / ".env")
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    run = ROOT / (ROOT / "results" / "FINAL_RUN.txt").read_text(encoding="utf-8").strip()
    out = run / "experiments"
    out.mkdir(exist_ok=True)
    client = make_client(cfg["api"], ROOT / cfg["cache_dir"], offline=False)
    specs = {m["key"]: ModelSpec(**m) for m in cfg["llms"]}
    judges = [specs[k] for k in cfg["phase2"]["external_judges"]]  # the five low-cost judges
    s = cfg["llm_settings"]
    B, seed = cfg["statistics"]["bootstrap"], int(cfg["seed"])

    convs = {c.conv_id: c for c in load_seconvo(ROOT / cfg["data"]["dir"], download=False)}
    pc = pd.read_csv(run / "tables" / "phase2_cv_per_conversation.csv")
    base = pc[(pc.workflow == "Joint")].drop_duplicates("conv_id").set_index("conv_id")
    ids = sorted(convs)
    y = np.array([convs[i].label for i in ids])
    flagged = {i for i in ids if base.loc[i, "phase1"] == 1}
    drift_at = {i: int(base.loc[i, "drift_at"]) for i in ids if bool(base.loc[i, "drift"])}

    def plan(variant: str) -> dict[int, dict]:
        """conv_id -> judge-input arguments for the dialogues the variant sends to the judge."""
        if variant == "full":
            return {i: {} for i in flagged if i in drift_at}
        if variant.endswith("ocdd"):
            key = "location" if variant.startswith("loc") else "upto"
            return {i: {key: drift_at[i]} for i in flagged if i in drift_at}
        key = "location" if variant.startswith("loc") else "upto"
        return {i: {key: FIXED} for i in flagged if len(convs[i].messages) > FIXED}

    rows, per, preds = [], [], {}
    for spec in judges:
        for v in VARIANTS:
            pl = plan(v)
            jd = run_paper_judgments(client, [(spec, convs[i]) for i in sorted(pl)], s,
                                     tag=f"judge_input_{v}", variant=pl)
            jm = jd.drop_duplicates("conv_id").set_index("conv_id")
            outcome = []
            for i in ids:
                if i not in flagged:
                    o = "real"
                elif i not in pl:
                    o = "spam-like (no drift)"
                else:
                    jp = jm["judge_pred"].get(i, np.nan)
                    o = "fraudulent" if jp == 1 else ("genuine drift" if jp == 0 else "unjudged")
                outcome.append(o)
                per.append({"judge": spec.key, "variant": v, "conv_id": i, "label": convs[i].label,
                            "first_pii": convs[i].first_pii, "n_msgs": len(convs[i].messages),
                            "decision_at": (pl[i].get("upto") if i in pl and "upto" in pl[i] else
                                            (len(convs[i].messages) - 1 if i in pl else np.nan)), "outcome": o})
            outcome = np.array(outcome)
            fraud = (outcome == "fraudulent").astype(int)
            preds[(spec.key, v)] = fraud
            tt = np.array(["none" if (convs[i].first_pii is None or convs[i].first_pii < 0) else
                           ("fraudulent" if convs[i].label == 1 else "genuine") for i in ids])
            judged = np.isin(outcome, ["fraudulent", "genuine drift"])
            m = compute_metrics(y, fraud)
            lo, hi = bootstrap_ci(y, fraud, "macro_f1", B, seed)
            rows.append({"judge": spec.key, "variant": v, "judged": int(judged.sum()),
                         "type_acc_fraud": float((outcome[judged & (tt == "fraudulent")] == "fraudulent").mean()),
                         "type_acc_genuine": float((outcome[judged & (tt == "genuine")] == "genuine drift").mean()),
                         "fraud_precision": m["precision_mal"], "fraud_recall": m["recall_mal"],
                         "fraud_macro_f1": m["macro_f1"], "fraud_macro_f1_lo": lo, "fraud_macro_f1_hi": hi,
                         "cost_usd": float(jm["cost_usd"].sum()) if len(jm) else 0.0})
            print(f"{spec.key:22s} {v:13s} macro-F1={m['macro_f1']:.3f} judged={int(judged.sum())} "
                  f"(spent this run ${client.spent_usd:.2f})", flush=True)
    res = pd.DataFrame(rows)
    per = pd.DataFrame(per)
    # paired comparisons: OCDD-derived input vs its fixed-position control, and vs the original prompt
    tests = []
    for a, b in [("loc_ocdd", "loc_fixed"), ("prefix_ocdd", "prefix_fixed"), ("loc_ocdd", "full"),
                 ("prefix_ocdd", "full")]:
        for spec in judges:
            d = paired_bootstrap_diff(y, preds[(spec.key, a)], preds[(spec.key, b)], "macro_f1", B, seed)
            tests.append({"judge": spec.key, "comparison": f"{a} - {b}", **d})
    tests = pd.DataFrame(tests)
    tests["p_holm"] = np.nan
    for c, g in tests.groupby("comparison"):
        tests.loc[g.index, "p_holm"] = holm(g["p_two_sided"].tolist())
    # early warning: how far before the end of the dialogue the prefix decisions are taken
    ew = (per[per.variant.str.startswith("prefix") & (per.outcome == "fraudulent") & (per.label == 1)]
          .assign(msgs_before_end=lambda d: d.n_msgs - 1 - d.decision_at,
                  before_victim_reply=lambda d: d.decision_at <= d.first_pii + 1)
          .groupby(["judge", "variant"])[["msgs_before_end", "before_victim_reply"]].mean().reset_index())
    res.to_csv(out / "judge_inputs_summary.csv", index=False)
    per.to_csv(out / "judge_inputs_per_conversation.csv", index=False)
    tests.to_csv(out / "judge_inputs_tests.csv", index=False)
    ew.to_csv(out / "judge_inputs_early_warning.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))
    print(tests.round(3).to_string(index=False))
    print(ew.round(3).to_string(index=False))
    print(f"spent this run: ${client.spent_usd:.2f}")


if __name__ == "__main__":
    main()
