"""Phase 2 of the framework (Sects. 3.2-3.3, Figs. 2-3), within-dialogue.

    Phase 1 says 'fake' -> OCDD over the dialogue's messages
        no drift -> 'spam-like'
        drift    -> LLM judge -> 'fraudulent' | 'genuine drift' (spamming)
    Joint:             Phase 1 = ensemble, judge = any LLM
    Dual-LLM:          Phase 1 = the LLM's zero-shot decision, judge = the same LLM
    Joint (no OCDD):   ablation -- every conversation flagged by the ensemble goes straight to the judge

Scoring
    * final fraud decision: outcome == 'fraudulent' vs the malicious label (all test conversations),
      with bootstrap CI and paired McNemar tests (Joint vs Dual-LLM and Joint vs Joint (no OCDD),
      same judge; Holm-corrected per comparison family)
    * RQ2, drift-type accuracy: on judged conversations whose annotated drift type is known
      (fraudulent = malicious with a PII request, genuine = benign with a PII request)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Conversation
from .evaluation import bootstrap_ci, compute_metrics, holm, mcnemar_exact
from .llm_client import BudgetExceeded, ModelSpec
from .llm_tasks import run_paper_judgments
from .utils import LOG

OUTCOMES = ["real", "spam-like (no drift)", "genuine drift", "fraudulent", "unjudged"]
WORKFLOWS = ["Joint", "Dual-LLM", "Joint (no OCDD)"]


def truth_type(c: Conversation) -> str:
    if c.first_pii is None:
        return "unknown"  # external set: no drift annotation
    if c.first_pii < 0:
        return "none"
    return "fraudulent" if c.label == 1 else "genuine"


def run_phase2(test: list[Conversation], joint_pred: np.ndarray, joint_latency: float,
               llm_phase1: dict[str, pd.DataFrame], judges: list[ModelSpec], client, settings: dict,
               drift: dict[int, dict], B: int, seed: int, tag: str = "dialogue_judge",
               budget_hit: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    """drift: conv_id -> {'drift': bool, 'drift_at': int | None} from the selected OCDD.
    Returns (summary per workflow x judge, per-conversation outcomes, statistics, budget_hit).
    llm_phase1[key] must be aligned with `test` and contain pred / cost_usd / latency_s."""
    y = np.array([c.label for c in test])
    ids = [c.conv_id for c in test]
    has_drift = np.array([drift[i]["drift"] for i in ids])
    LOG.info("Phase 2: OCDD flagged a drift in %d/%d conversations", int(has_drift.sum()), len(test))
    by_id = {c.conv_id: c for c in test}
    rows, per_conv, final = [], [], {}
    for spec in judges:
        if spec.key not in llm_phase1 or budget_hit:
            continue
        p1_dual = llm_phase1[spec.key]["pred"].to_numpy().astype(int)
        go = sorted({i for i, f, d in zip(ids, joint_pred, has_drift) if f == 1}
                    | {i for i, f, d in zip(ids, p1_dual, has_drift) if f == 1 and d})
        try:
            jd = run_paper_judgments(client, [(spec, by_id[i]) for i in go], settings, tag=tag) if go else None
        except BudgetExceeded as exc:
            LOG.error("Budget reached in Phase 2 (%s): %s", spec.key, exc)
            budget_hit = True
            break
        except RuntimeError as exc:
            LOG.error("Phase 2 judge %s skipped (%s)", spec.key, exc)
            continue
        jmap = (jd.drop_duplicates("conv_id").set_index("conv_id") if jd is not None and len(jd)
                else pd.DataFrame(columns=["judge_pred", "cost_usd", "latency_s"]))
        for wf, p1, use_ocdd in [("Joint", joint_pred, True), ("Dual-LLM", p1_dual, True),
                                 ("Joint (no OCDD)", joint_pred, False)]:
            outs = []
            for c, f, d in zip(test, p1, has_drift):
                if f == 0:
                    o = "real"
                elif use_ocdd and not d:
                    o = "spam-like (no drift)"
                else:
                    jp = jmap["judge_pred"].get(c.conv_id, np.nan)
                    o = "fraudulent" if jp == 1 else ("genuine drift" if jp == 0 else "unjudged")
                outs.append(o)
                per_conv.append({"workflow": wf, "judge": spec.key, "conv_id": c.conv_id, "label": c.label,
                                 "truth_type": truth_type(c), "phase1": int(f), "drift": bool(d),
                                 "drift_at": drift[c.conv_id]["drift_at"], "first_pii": c.first_pii, "outcome": o})
            outs = np.array(outs)
            fraud = (outs == "fraudulent").astype(int)
            final[(wf, spec.key)] = fraud
            judged = np.isin(outs, ["fraudulent", "genuine drift", "unjudged"])
            tt = np.array([truth_type(c) for c in test])
            known = np.isin(outs, ["fraudulent", "genuine drift"]) & np.isin(tt, ["fraudulent", "genuine"])
            jsel = [i for i, o in zip(ids, judged) if o]
            judge_cost = float(jmap["cost_usd"].reindex(jsel).sum()) if len(jmap) else 0.0
            judge_lat = float(jmap["latency_s"].reindex(jsel).sum()) if len(jmap) else 0.0
            if wf == "Dual-LLM":
                p1_cost = float(llm_phase1[spec.key]["cost_usd"].sum()) / len(test)
                p1_lat = float(llm_phase1[spec.key]["latency_s"].mean())
                p1_calls = 1.0
            else:
                p1_cost, p1_lat, p1_calls = 0.0, joint_latency, 0.0
            m = compute_metrics(y, fraud)
            lo, hi = bootstrap_ci(y, fraud, "macro_f1", B, seed)
            rows.append({
                "workflow": wf, "judge": spec.key, "flagged": int(p1.sum()),
                "drift_detected": int(((p1 == 1) & has_drift).sum()) if use_ocdd else int(p1.sum()),
                "judged": int(judged.sum()),
                **{f"n_{o.split(' ')[0].replace('-', '_')}": int((outs == o).sum()) for o in OUTCOMES},
                "type_acc": float(((outs[known] == "fraudulent") == (tt[known] == "fraudulent")).mean()) if known.any() else np.nan,
                "type_acc_fraud": float((outs[known & (tt == "fraudulent")] == "fraudulent").mean())
                if (known & (tt == "fraudulent")).any() else np.nan,
                "type_acc_genuine": float((outs[known & (tt == "genuine")] == "genuine drift").mean())
                if (known & (tt == "genuine")).any() else np.nan,
                "fraud_accuracy": m["accuracy"], "fraud_precision": m["precision_mal"], "fraud_recall": m["recall_mal"],
                "fraud_f1": m["f1_mal"], "fraud_macro_f1": m["macro_f1"], "fraud_macro_f1_lo": lo,
                "fraud_macro_f1_hi": hi, "fraud_mcc": m["mcc"],
                "phase1_macro_f1": compute_metrics(y, p1)["macro_f1"],
                "llm_calls_per_conv": p1_calls + judged.sum() / len(test),
                "cost_per_1k_conv_usd": 1000 * (p1_cost + judge_cost / len(test)),
                "latency_per_conv_s": p1_lat + judge_lat / len(test),
            })
    res = pd.DataFrame(rows)
    stats = []
    for other, col in [("Dual-LLM", "mcnemar_p_holm"), ("Joint (no OCDD)", "mcnemar_p_holm_vs_no_ocdd")]:
        st = []
        for spec in judges:
            if ("Joint", spec.key) in final and (other, spec.key) in final:
                mc = mcnemar_exact(y, final[("Joint", spec.key)], final[(other, spec.key)])
                st.append({"judge": spec.key, "comparison": f"Joint vs {other}", **mc})
        st = pd.DataFrame(st)
        if len(st):
            st["p_holm"] = holm(st["p_value"].tolist())
            res = res.merge(st[["judge", "p_holm"]].rename(columns={"p_holm": col}), on="judge", how="left")
            stats.append(st)
    if "mcnemar_p_holm" in res:
        res.loc[res.workflow != "Joint", ["mcnemar_p_holm", "mcnemar_p_holm_vs_no_ocdd"]] = np.nan
    return res, pd.DataFrame(per_conv), (pd.concat(stats) if stats else pd.DataFrame()), budget_hit
