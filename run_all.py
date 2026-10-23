#!/usr/bin/env python3
"""Run every experiment of the CMOT joint fraud / concept-drift study and write
all tables, figures, logs and a manifest into results/<timestamp>/.

  python run_all.py                    # full run (needs CREATEAI_SERVICE_TOKEN, read from .env)
  python run_all.py --offline          # smoke test with MOCK LLM/embeddings (no token, no cost)
  python run_all.py --offline --synthetic --quick   # self-test without downloading data
  python run_all.py --skip-llm         # classical parts only (no API)

Structure:
  Phase 1   ensemble (majority vote of the 3 best models) vs LLM classifiers
  Phase 2   within-dialogue OCDD on flagged conversations -> LLM judge of the drift type
            Joint (ensemble Phase 1) vs Dual-LLM (same LLM in both phases)
  Protocol A  official 40/360 split   | Protocol B  5x5 repeated CV (Phase 1)
  Protocol C  external test: train on 400 annotated, test on 1,000 disjoint conversations
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import RepeatedStratifiedKFold

from cmot import figures as F
from cmot.classical import fit_predict, majority_vote, model_zoo, select_members
from cmot.data import load_seconvo, load_seconvo_external, make_synthetic
from cmot.dialogue_drift import build_spaces, evaluate as drift_evaluate, pooled_normal, score as drift_score, tune_on_train
from cmot.evaluation import (bootstrap_ci, compute_metrics, corrected_resampled_ttest, friedman_nemenyi, holm,
                             mcnemar_exact, paired_bootstrap_diff)
from cmot.features import build_views
from cmot.llm_client import BudgetExceeded, ModelSpec, make_client
from cmot.llm_tasks import PROMPT_VERSION, aggregate_repeats, run_classification, select_fewshot
from cmot.ocdd_normal import (EmbeddingSpace, FixedMessageRule, MemoEmbedder, NormalOCDD, evaluate_detector,
                              select_on_train)
from cmot.phase2 import run_phase2
from cmot.pipeline import evaluate_pipelines, routing_curve
from cmot.utils import LOG, environment_manifest, set_seed, setup_logging, stable_hash, write_json

ROOT = Path(__file__).resolve().parent
METRIC_COLS = ["accuracy", "balanced_accuracy", "precision_mal", "recall_mal", "f1_mal", "macro_precision",
               "macro_recall", "macro_f1", "mcc", "auroc"]
PHASE2_COLS = ["workflow", "judge", "flagged", "n_spam_like", "n_genuine", "n_fraudulent", "type_acc_fraud",
               "type_acc_genuine", "fraud_precision", "fraud_recall", "fraud_macro_f1", "fraud_macro_f1_lo",
               "fraud_macro_f1_hi", "mcnemar_p_holm", "mcnemar_p_holm_vs_no_ocdd", "llm_calls_per_conv",
               "cost_per_1k_conv_usd"]


def message_vector_sets(convs, embedder, og: dict, seed: int):
    """Message representations for OCDD, fitted label-free on the given conversations' messages:
    TF-IDF+SVD and (if an embedder is available) PCA-reduced te3-large embeddings."""
    msgs = [m for c in convs for m in c.messages]
    spaces = {("tfidf", d): sp for d, sp in build_spaces(msgs, og["tfidf_dims"], seed).items()}
    if embedder is not None:
        embedder(msgs)  # one batched cache read / request
        for d in og["emb_dims"]:
            spaces[("emb", d)] = EmbeddingSpace(embedder, d, seed).fit(msgs)
    vecs = {k: {c.conv_id: sp.transform(c.messages) for c in convs} for k, sp in spaces.items()}
    return spaces, vecs


def drift_balanced_ci(ev: pd.DataFrame, B: int, seed: int) -> tuple[float, float]:
    """Bootstrap CI (over dialogues) of the balanced drift score."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        d = ev.iloc[rng.integers(0, len(ev), len(ev))]
        pos, neg = d[d.has_drift], d[~d.has_drift]
        if len(pos) and len(neg):
            vals.append(np.mean([pos.localized.mean(), 1 - neg.detected.mean()]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def fit_ocdd(train, vec_sets, og: dict, seed: int):
    """Select the OCDD configuration on `train` (nested cross-fitting) and fit it on all of `train`."""
    tab, best = select_on_train(train, vec_sets, og, seed, og.get("outer_folds", 5))
    det = NormalOCDD(best, seed).fit(train, vec_sets[(best.rep, best.dims)])
    return tab, best, det


# --------------------------------------------------------------------------- helpers
def load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE lines). Existing environment variables win; later lines
    override earlier ones. Values are never logged."""
    if not path.exists():
        return
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.removeprefix("export ").split("=", 1)
        values[key.strip()] = val.strip().strip('"').strip("'")
    for key, val in values.items():
        if key and key not in os.environ:
            os.environ[key] = val


def latex_table(df: pd.DataFrame, path: Path, caption: str, label: str, float_fmt: str = "{:.3f}") -> None:
    cols = list(df.columns)
    lines = ["\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\small",
             "\\begin{tabular}{" + "l" + "c" * (len(cols) - 1) + "}", "\\toprule",
             " & ".join(f"\\textbf{{{str(c).replace('_', ' ')}}}" for c in cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("--" if np.isnan(v) else float_fmt.format(v))
            else:
                cells.append(str(v).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("·", "$\\cdot$"))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def save_table(df: pd.DataFrame, tables_dir: Path, name: str, caption: str, latex_cols: list[str] | None = None) -> None:
    df.to_csv(tables_dir / f"{name}.csv", index=False)
    latex_table(df[[c for c in (latex_cols or df.columns) if c in df.columns]], tables_dir / f"{name}.tex",
                caption, f"tab:{name}")


def _classical_cache_key(texts, conv_emb, y, tr, te, cfg, seed) -> str:
    """Deterministic key: split, labels, texts, embeddings, settings and the *source code*
    of every module that influences the classical results (edits invalidate the cache)."""
    import hashlib
    h = hashlib.sha256()
    for part in (np.asarray(tr).tobytes(), np.asarray(te).tobytes(), np.asarray(y).tobytes(),
                 "\x00".join(texts).encode("utf-8"),
                 b"" if conv_emb is None else np.ascontiguousarray(conv_emb).tobytes(),
                 stable_hash({"seed": seed, "rep": cfg["representations"], "cls": cfg["classical"]}).encode()):
        h.update(part)
    for mod in ("classical.py", "features.py", "impkmeans.py", "evaluation.py"):
        h.update((ROOT / "cmot" / mod).read_bytes())
    return h.hexdigest()[:32]


def classical_block(texts, conv_emb, y, tr, te, cfg, seed, emb_label):
    """Fit the whole zoo on every view; returns list of FitResult, ensemble dicts and view infos.
    Results are cached on disk (deterministic given the key), so re-runs skip the CV time."""
    import pickle
    cache_dir = ROOT / cfg["cache_dir"] / "classical"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{_classical_cache_key(texts, conv_emb, y, tr, te, cfg, seed)}.pkl"
    if path.exists():
        with path.open("rb") as fh:
            return pickle.load(fh)
    views = build_views(texts, conv_emb, tr, te, cfg["representations"], seed)
    fits, ens = [], {}
    ccfg = cfg["classical"]
    for v in views:
        vf = [fit_predict(md, v, y[tr], seed, ccfg["inner_folds"], ccfg["n_jobs"]) for md in model_zoo(seed, v)]
        fits += vf
        for strat in ("fixed", "inner_cv"):
            members = select_members(vf, strat, ccfg["ensemble"]["members"])
            pred, soft, unan = majority_vote(members)
            ens[(v.name, strat)] = {"pred": pred, "soft": soft, "unanimous": unan,
                                    "members": [m.name for m in members],
                                    "latency": float(sum(m.predict_s_per_conv for m in members))}
    view_infos = [SimpleNamespace(name=v.name, info=v.info) for v in views]
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump((fits, ens, view_infos), fh)
    os.replace(tmp, path)
    return fits, ens, view_infos


def joint_key(cfg) -> tuple[str, str]:
    p2 = cfg["phase2"]
    return f"TF-IDF+SVD ({p2.get('view_scope', 'transductive')})", p2.get("joint_ensemble", "inner_cv")


def phase2_judges(cfg, specs: list[ModelSpec], which: str = "judges") -> list[ModelSpec]:
    p2 = cfg["phase2"]
    if which == "external_judges" and p2.get("external_judges"):
        return [s for s in specs if s.key in p2["external_judges"]]
    excl = set(p2.get("exclude_judges", []))
    return [s for s in specs if s.key not in excl]


# --------------------------------------------------------------------------- Protocol C (external test)
def run_external(cfg, args, convs, client, specs, s, seed, run_dir, tab_dir, fw, budget_hit,
                 ocdd_bundle) -> dict:
    """Train on all 400 human-annotated conversations; test on the 1,000 disjoint SEConvo
    conversations (simulation-design labels; no drift annotation). Phase 2 re-uses the OCDD
    configuration and message space selected on the official training conversations."""
    LOG.info("=== Protocol C: external test (train = 400 annotated, test = 1,000 unannotated) ===")
    if args.synthetic:
        ext = make_synthetic(240, seed + 7)
        for i, c in enumerate(ext):
            c.conv_id, c.split, c.first_pii = 20_000 + i, "external", None
    else:
        ext = load_seconvo_external(ROOT / cfg["data"]["dir"])
    allc = list(convs) + ext
    texts_all = [c.plain_text() for c in allc]
    y_all = np.array([c.label for c in allc])
    tr, te = np.arange(len(convs)), np.arange(len(convs), len(allc))
    yte, ext_ids = y_all[te], np.array([c.conv_id for c in ext])
    emb_cfg = cfg["representations"]["embedding"]
    emb_all = (client.embed([c.transcript() for c in allc], emb_cfg["provider"], emb_cfg["model"])
               if client is not None and emb_cfg["enabled"] else None)
    fits, ens, _ = classical_block(texts_all, emb_all, y_all, tr, te, cfg, seed, emb_cfg.get("label"))
    B = cfg["statistics"]["bootstrap"]
    rows, preds = [], {}
    for f in fits:
        lab = f"{f.name} [{f.view}]"
        preds[lab] = f.pred
        rows.append({"system": lab, "group": "Classical", "view": f.view, **compute_metrics(yte, f.pred, f.score)})
    for (vname, strat), e in ens.items():
        lab = f"Ensemble-{strat} [{vname}]"
        preds[lab] = e["pred"]
        rows.append({"system": lab, "group": "Ensemble", "view": vname, "members": "+".join(e["members"]),
                     **compute_metrics(yte, e["pred"], e["soft"])})
    llm_ext = {}
    for spec in ([] if budget_hit else specs):
        try:
            raw = run_classification(client, [spec], ext, s, tag="external")
        except BudgetExceeded as exc:
            LOG.error("Budget reached during external LLM stage: %s", exc)
            budget_hit = True
            break
        except RuntimeError as exc:
            LOG.error("External stage: %s skipped (%s)", spec.key, exc)
            continue
        agg = aggregate_repeats(raw, s.get("invalid_policy", "benign")).set_index("conv_id").reindex(ext_ids)
        agg["pred"] = agg["pred"].fillna(0).astype(int)
        llm_ext[spec.key] = agg.reset_index()
        lab = f"{spec.key} (zero-shot)"
        preds[lab] = llm_ext[spec.key]["pred"].to_numpy()
        rows.append({"system": lab, "group": "LLM", "view": "transcript",
                     **compute_metrics(yte, preds[lab], llm_ext[spec.key]["p_malicious"].to_numpy()),
                     "cost_per_1k_conv_usd": 1000 * llm_ext[spec.key]["cost_usd"].mean(),
                     "latency_per_conv_s": llm_ext[spec.key]["latency_s"].mean()})
    tC = pd.DataFrame(rows)
    jv, js = joint_key(cfg)
    joint_lab = f"Ensemble-{js} [{jv}]"
    ci, mc = [], []
    for lab in tC["system"]:
        ci.append(bootstrap_ci(yte, preds[lab], "macro_f1", B, seed))
        mc.append(mcnemar_exact(yte, preds[lab], preds[joint_lab])["p_value"] if lab != joint_lab else np.nan)
    tC["macro_f1_lo"], tC["macro_f1_hi"] = zip(*ci)
    tC["mcnemar_p_vs_joint"] = mc
    m_ = tC["system"] != joint_lab
    tC.loc[m_, "mcnemar_p_holm"] = holm(tC.loc[m_, "mcnemar_p_vs_joint"].tolist())
    tC = tC.sort_values("macro_f1", ascending=False)
    save_table(tC, tab_dir, "table8_external_phase1",
               "External test, Phase 1: trained on the 400 human-annotated conversations, tested on the 1,000 "
               "disjoint SEConvo conversations (simulation-design labels). McNemar vs the Joint ensemble (Holm).",
               ["system", "accuracy", "precision_mal", "recall_mal", "macro_f1", "macro_f1_lo", "macro_f1_hi", "mcc",
                "mcnemar_p_holm"])
    fp = tC[(tC.group == "LLM") | (tC.view == jv) | tC.system.str.startswith("Ensemble")]
    F.forest_plot(fw, fp.assign(label=fp.system, value=fp.macro_f1, lo=fp.macro_f1_lo, hi=fp.macro_f1_hi),
                  name="fig9_external_phase1", title=f"External test set, Phase 1 (n = {len(te)})")

    # Phase 2 on the external set (low-cost judges; no drift annotation -> final decision only)
    joint = ens[(jv, js)]
    judges = [j for j in phase2_judges(cfg, specs, "external_judges") if j.key in llm_ext]
    # OCDD selected and fitted on the 400 annotated dialogues; message spaces fitted on their messages
    msg_spaces, vec_sets, og, embedder = ocdd_bundle
    selC, bestC, detC = fit_ocdd(list(convs), vec_sets, og, seed)
    selC.to_csv(tab_dir / "external_ocdd_selection.csv", index=False)
    if embedder is not None and bestC.rep == "emb":
        embedder([m for c in ext for m in c.messages])  # one batched request for the external messages
    sp = msg_spaces[(bestC.rep, bestC.dims)]
    drift_C = {c.conv_id: detC.detect(c, sp.transform(c.messages)) for c in ext}
    write_json(run_dir / "external_ocdd_selected.json", {**bestC.__dict__, "label": bestC.label,
                                                         "drift_rate": float(np.mean([d["drift"] for d in drift_C.values()]))})
    res, per_conv, _, budget_hit = run_phase2(ext, joint["pred"], joint["latency"], llm_ext, judges, client, s,
                                             drift_C, B, seed, tag="external_dialogue_judge", budget_hit=budget_hit)
    if len(res):
        save_table(res, tab_dir, "table9_external_phase2",
                   "External test, Phase 2 (within-dialogue OCDD -> LLM judge): final 'fraudulent attempt' "
                   "decision vs the malicious label. McNemar Joint vs Dual-LLM per judge (Holm).",
                   [c for c in PHASE2_COLS if not c.startswith("type_acc")])
        per_conv.to_csv(tab_dir / "external_phase2_per_conversation.csv", index=False)
        F.joint_vs_dual(fw, res, compute_metrics(yte, joint["pred"])["macro_f1"], name="fig10_external_phase2",
                        title=f"External test set, Phase 2 (n = {len(te)})")
    return {"n_external": int(len(te)), "llm_models": list(llm_ext), "phase2_judges": [j.key for j in judges],
            "budget_hit": budget_hit}


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--offline", action="store_true", help="mock LLM + mock embeddings (smoke test; no cost)")
    ap.add_argument("--synthetic", action="store_true", help="synthetic conversations instead of SEConvo")
    ap.add_argument("--skip-llm", action="store_true", help="no LLM or embedding calls")
    ap.add_argument("--no-cv", action="store_true", help="skip the repeated-CV protocol")
    ap.add_argument("--no-external", action="store_true", help="skip the external-test protocol")
    ap.add_argument("--quick", action="store_true", help="small CV / grid / bootstrap settings")
    ap.add_argument("--preflight-only", action="store_true", help="check token + model availability, then exit")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.quick:
        cfg["protocols"]["repeated_cv"].update({"n_splits": 3, "n_repeats": 1})
        cfg["statistics"].update({"bootstrap": 200, "random_routing_draws": 30})
        cfg["phase2"]["ocdd_grid"].update({"window": [2], "m": [1], "nu": [0.1], "dims": [5]})
    seed = int(cfg["seed"])
    mock = args.offline or args.synthetic
    stamp = time.strftime("%Y%m%d-%H%M%S") + ("_MOCK" if mock else "")
    run_dir = ROOT / cfg["output_dir"] / stamp
    fig_dir, tab_dir = run_dir / "figures", run_dir / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "run.log")
    set_seed(seed)
    LOG.info("Run directory: %s", run_dir)
    if mock:
        LOG.warning("MOCK MODE: outputs are for pipeline testing only and must not be reported.")

    # ---------------------------------------------------------------- data
    convs = make_synthetic(160, seed) if args.synthetic else load_seconvo(ROOT / cfg["data"]["dir"])
    ids = np.array([c.conv_id for c in convs])
    y = np.array([c.label for c in convs])
    texts = [c.plain_text() for c in convs]
    is_train = np.array([c.split == "train" for c in convs])
    tr, te = np.where(is_train)[0], np.where(~is_train)[0]
    train_convs, test_convs = [convs[i] for i in tr], [convs[i] for i in te]

    # ---------------------------------------------------------------- API client / embeddings
    client = None
    if not args.skip_llm:
        if not args.offline and not os.environ.get(cfg["api"].get("token_env", "CREATEAI_SERVICE_TOKEN")):
            raise SystemExit("Set CREATEAI_SERVICE_TOKEN (or use --offline / --skip-llm).")
        client = make_client(cfg["api"], ROOT / cfg["cache_dir"], offline=args.offline)
    emb_cfg = cfg["representations"]["embedding"]
    if client is not None:
        client.preflight([ModelSpec(**m) for m in cfg["llms"]],
                         f"{emb_cfg['provider']}/{emb_cfg['model']}" if emb_cfg["enabled"] else None)
    if args.preflight_only:
        LOG.info("Preflight only: exiting before any paid call.")
        return
    conv_emb = None
    if client is not None and emb_cfg["enabled"]:
        conv_emb = client.embed([c.transcript() for c in convs], emb_cfg["provider"], emb_cfg["model"])
    emb_label = emb_cfg.get("label", "Embedding")
    specs = [ModelSpec(**m) for m in cfg["llms"]] if client is not None else []
    s = cfg["llm_settings"]
    budget_hit = False

    # ---------------------------------------------------------------- Phase 1 LLMs (zero-shot all 400, few-shot test)
    llm_agg: dict[str, pd.DataFrame] = {}
    llm_few: dict[str, pd.DataFrame] = {}
    if specs:
        try:
            raw = run_classification(client, specs, convs, s, tag="zeroshot")
            raw.drop(columns=["text"]).to_csv(run_dir / "llm_zeroshot_calls.csv", index=False)
            raw[["model_key", "conv_id", "repeat", "text"]].to_json(run_dir / "llm_zeroshot_outputs.jsonl",
                                                                   orient="records", lines=True, force_ascii=False)
            agg = aggregate_repeats(raw, s.get("invalid_policy", "benign"))
            llm_agg = {k: g.reset_index(drop=True) for k, g in agg.groupby("model_key")}
            if s.get("fewshot", {}).get("enabled") and len(tr):
                shots = select_fewshot(train_convs, s["fewshot"]["k"], seed)
                write_json(run_dir / "fewshot_examples.json", [c.conv_id for c in shots])
                rawf = run_classification(client, specs, test_convs, {**s, "repeats": 1}, fewshot=shots, tag="fewshot")
                rawf.drop(columns=["text"]).to_csv(run_dir / "llm_fewshot_calls.csv", index=False)
                aggf = aggregate_repeats(rawf, s.get("invalid_policy", "benign"))
                llm_few = {k: g.reset_index(drop=True) for k, g in aggf.groupby("model_key")}
        except BudgetExceeded as exc:
            budget_hit = True
            LOG.error("%s -- continuing with what is cached; re-run to resume.", exc)

    def llm_on(idx: np.ndarray, key: str, src=None) -> pd.DataFrame:
        d = (src or llm_agg)[key].set_index("conv_id").reindex(ids[idx])
        d["pred"] = d["pred"].fillna(0).astype(int)
        return d.reset_index()

    # ================================================================ PROTOCOL A: official split, Phase 1
    LOG.info("=== Protocol A: official SEConvo split (train=%d, test=%d) ===", len(tr), len(te))
    yte = y[te]
    fits, ens, views = classical_block(texts, conv_emb, y, tr, te, cfg, seed, emb_label)
    write_json(run_dir / "views.json", {v.name: v.info for v in views})
    B = cfg["statistics"]["bootstrap"]
    jv, js = joint_key(cfg)
    joint = ens[(jv, js)]
    ref_lab = f"Ensemble-{js} [{jv}]"
    rows, preds = [], {}
    for f in fits:
        lab = f"{f.name} [{f.view}]"
        preds[lab] = f.pred
        rows.append({"system": lab, "group": "Classical", "view": f.view, **compute_metrics(yte, f.pred, f.score),
                     "inner_cv_macro_f1": f.inner_cv_score, "best_params": str(f.best_params),
                     "fit_s": f.fit_s, "latency_per_conv_s": f.predict_s_per_conv})
    for (vname, strat), e in ens.items():
        lab = f"Ensemble-{strat} [{vname}]"
        preds[lab] = e["pred"]
        rows.append({"system": lab, "group": "Ensemble", "view": vname, **compute_metrics(yte, e["pred"], e["soft"]),
                     "best_params": "+".join(e["members"]), "latency_per_conv_s": e["latency"]})
    for key in llm_agg:
        d = llm_on(te, key)
        lab = f"{key} (zero-shot)"
        preds[lab] = d["pred"].to_numpy()
        rows.append({"system": lab, "group": "LLM", "view": "transcript",
                     **compute_metrics(yte, preds[lab], d["p_malicious"].to_numpy()),
                     "invalid_rate": float((d["n_valid"] == 0).mean()), "latency_per_conv_s": d["latency_s"].mean(),
                     "cost_per_1k_conv_usd": 1000 * d["cost_usd"].mean(),
                     "model_returned": ";".join(sorted(set(map(str, d["model_returned"].dropna()))))})
    for key in llm_few:
        d = llm_on(te, key, llm_few)
        lab = f"{key} (few-shot k={s['fewshot']['k']})"
        preds[lab] = d["pred"].to_numpy()
        rows.append({"system": lab, "group": "LLM", "view": "transcript",
                     **compute_metrics(yte, preds[lab], d["p_malicious"].to_numpy()),
                     "invalid_rate": float((d["n_valid"] == 0).mean()), "latency_per_conv_s": d["latency_s"].mean(),
                     "cost_per_1k_conv_usd": 1000 * d["cost_usd"].mean()})
    tA = pd.DataFrame(rows)
    ci, mc = [], []
    for lab in tA["system"]:
        ci.append(bootstrap_ci(yte, preds[lab], "macro_f1", B, seed))
        mc.append(mcnemar_exact(yte, preds[lab], preds[ref_lab])["p_value"] if lab != ref_lab else np.nan)
    tA["macro_f1_lo"], tA["macro_f1_hi"] = zip(*ci)
    tA["mcnemar_p_vs_joint"] = mc
    mask = tA["system"] != ref_lab
    tA.loc[mask, "mcnemar_p_holm"] = holm(tA.loc[mask, "mcnemar_p_vs_joint"].tolist())
    tA = tA.sort_values("macro_f1", ascending=False)
    save_table(tA, tab_dir, "table1_official_phase1_all",
               "Official SEConvo split (40/360), Phase 1, all systems. Positive class = malicious; macro = "
               "unweighted class mean. McNemar vs the Joint ensemble (Holm).",
               ["system", "accuracy", "precision_mal", "recall_mal", "f1_mal", "macro_f1", "mcc", "auroc"])
    t2 = tA[(tA.group.isin(["Classical", "Ensemble"])) & (tA.view.str.startswith("TF-IDF+SVD"))]
    save_table(t2, tab_dir, "table2_classical_tfidf",
               "Phase 1 benchmark on TF-IDF+PCA, inductive vs transductive fitting.",
               ["system", "accuracy", "macro_precision", "macro_recall", "macro_f1", "mcc"])
    t3 = tA[(tA.group == "LLM") | (tA.system == ref_lab)]
    save_table(t3, tab_dir, "table3_llm_vs_ensemble", "Phase 1: LLM classifiers vs the Joint ensemble.",
               ["system", "accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_f1_lo", "macro_f1_hi",
                "mcc", "mcnemar_p_holm"])

    # ================================================================ PHASE 2 (within-dialogue OCDD + judge)
    LOG.info("=== Phase 2: OCDD (normal-message reference, selected on training dialogues) + LLM judge ===")
    p2 = cfg["phase2"]
    og = p2["ocdd"]
    tol = p2.get("localization_tolerance", 2)
    embedder = (MemoEmbedder(client, emb_cfg["provider"], emb_cfg["model"])
                if client is not None and emb_cfg["enabled"] else None)
    msg_spaces, vec_sets = message_vector_sets(convs, embedder, og, seed)
    old_grid = p2["ocdd_grid"]
    old_spaces = build_spaces([m for c in convs for m in c.messages], old_grid["dims"], seed)

    def old_window_detector(train_part):
        """Previous design: within-dialogue window reference, tuned in-sample on the training dialogues."""
        _, best_old = tune_on_train(train_part, old_spaces, {**old_grid, "reference": ["dialogue"]}, seed)
        p, d = best_old["dialogue"]
        return p, d

    def detection_rows(test_part, det, det_vecs, p_old, d_old, tag):
        vec_old = {c.conv_id: old_spaces[d_old].transform(c.messages) for c in test_part}
        frames = {"OCDD (normal reference)": evaluate_detector(det, test_part, det_vecs, tol),
                  "Within-dialogue window": drift_evaluate(test_part, vec_old, p_old, None, tol),
                  "Fixed message-4 rule": evaluate_detector(FixedMessageRule(4), test_part,
                                                            {c.conv_id: None for c in test_part}, tol)}
        return {k: v.assign(protocol=tag) for k, v in frames.items()}

    # ---- Protocol A: official split (selection on the 40 training dialogues)
    selA, bestA, detA = fit_ocdd(train_convs, vec_sets, og, seed)
    selA.to_csv(tab_dir / "phase2_ocdd_selection_official_train.csv", index=False)
    VA = vec_sets[(bestA.rep, bestA.dims)]
    p_oldA, d_oldA = old_window_detector(train_convs)
    framesA = detection_rows(test_convs, detA, VA, p_oldA, d_oldA, "official")
    det_frames = {k: v for k, v in framesA.items()}
    drift_A = {c.conv_id: detA.detect(c, VA[c.conv_id]) for c in test_convs}
    llm_te = {k: llm_on(te, k) for k in llm_agg}
    judges = phase2_judges(cfg, specs)
    p2_res, p2_conv, _, budget_hit = run_phase2(test_convs, joint["pred"], joint["latency"], llm_te, judges, client,
                                                s, drift_A, B, seed, budget_hit=budget_hit)
    if len(p2_res):
        save_table(p2_res, tab_dir, "table5_phase2_official",
                   "Phase 2 on the official test split with the OCDD configuration selected on the 40 training "
                   "dialogues. McNemar Joint vs Dual-LLM and vs Joint without OCDD per judge (Holm).", PHASE2_COLS)
        p2_conv.to_csv(tab_dir / "phase2_per_conversation.csv", index=False)

    # ---- Protocol B: Phase 2 under cross-validation over the 400 annotated dialogues
    rc = cfg["protocols"]["repeated_cv"]
    rskf = RepeatedStratifiedKFold(n_splits=rc["n_splits"], n_repeats=rc["n_repeats"], random_state=seed)
    cv_pred, cv_lat, drift_B, framesB, sel_rows = np.zeros(len(y), int), [], {}, [], []
    for fold, (ftr, fte) in enumerate(list(rskf.split(np.zeros(len(y)), y))[:rc["n_splits"]]):
        _, fens, _ = classical_block(texts, conv_emb, y, ftr, fte, cfg, seed + fold, emb_label)
        cv_pred[fte] = fens[(jv, js)]["pred"]
        cv_lat.append(fens[(jv, js)]["latency"])
        trc, tec = [convs[i] for i in ftr], [convs[i] for i in fte]
        tabB, bestB, detB = fit_ocdd(trc, vec_sets, og, seed)
        VB = vec_sets[(bestB.rep, bestB.dims)]
        drift_B.update({c.conv_id: detB.detect(c, VB[c.conv_id]) for c in tec})
        p_old, d_old = old_window_detector(trc)
        framesB.append({k: v.assign(fold=fold) for k, v in detection_rows(tec, detB, VB, p_old, d_old, "cv").items()})
        sel_rows.append({"fold": fold, "selected": bestB.label, "train_balanced": float(tabB.iloc[0].train_balanced),
                         "old_window": f"W={p_old.window} m={p_old.m} rho={p_old.rho} nu={p_old.nu} {p_old.speakers} d={d_old}"})
        LOG.info("  Phase 2 CV fold %d: OCDD %s", fold + 1, bestB.label)
    pd.DataFrame(sel_rows).to_csv(tab_dir / "phase2_ocdd_selection_cv.csv", index=False)
    det_rows = []
    for name in framesA:
        for proto, ev in [("official", framesA[name]), ("cv", pd.concat([f[name] for f in framesB]))]:
            sc = drift_score(ev)
            lo, hi = drift_balanced_ci(ev, B, seed)
            det_rows.append({"protocol": proto, "detector": name, **sc, "balanced_lo": lo, "balanced_hi": hi})
            ev.to_csv(tab_dir / f"phase2_detections_{proto}_{name.split(' ')[0].lower()}.csv", index=False)
    det_tab = pd.DataFrame(det_rows)
    save_table(det_tab, tab_dir, "table4_drift_detection",
               "Drift detection against the annotated first PII request: official split (selection on 40 training "
               "dialogues) and 5-fold CV over the 400 annotated dialogues (selection inside each training fold). "
               "Localized = drift flagged at or up to 2 messages after the onset.",
               ["protocol", "detector", "detect_rate", "localized_rate", "false_drift_rate", "balanced",
                "balanced_lo", "balanced_hi", "median_delay"])
    write_json(run_dir / "phase2_ocdd_selected.json",
               {"official": {**bestA.__dict__, "label": bestA.label}, "cv": sel_rows,
                "official_old_window": {**p_oldA.__dict__, "dims": d_oldA}})
    all_idx = np.arange(len(y))
    llm_all = {k: llm_on(all_idx, k) for k in llm_agg}
    p2cv_res, p2cv_conv, _, budget_hit = run_phase2(convs, cv_pred, float(np.mean(cv_lat)), llm_all, judges, client,
                                                    s, drift_B, B, seed, budget_hit=budget_hit)
    if len(p2cv_res):
        save_table(p2cv_res, tab_dir, "table5b_phase2_cv",
                   "Phase 2 under 5-fold CV over the 400 annotated dialogues (Joint ensemble and OCDD selected in "
                   "each training fold; pooled held-out predictions). McNemar Joint vs Dual-LLM and vs Joint "
                   "without OCDD per judge (Holm).", PHASE2_COLS)
        p2cv_conv.to_csv(tab_dir / "phase2_cv_per_conversation.csv", index=False)
        pd.crosstab([p2cv_conv.workflow, p2cv_conv.judge, p2cv_conv.outcome], p2cv_conv.truth_type).to_csv(
            tab_dir / "phase2_cv_outcome_vs_truth.csv")
    ocdd_bundle = (msg_spaces, vec_sets, og, embedder)

    # ---------------------------------------------------------------- extra analyses (labelled as such)
    empty_alarms = pd.DataFrame({"conv_id": ids[te], "alarm": False})
    pipes = evaluate_pipelines(yte, ids[te], joint["pred"], joint["soft"], joint["unanimous"], joint["latency"],
                               llm_te, empty_alarms, {}, cfg["statistics"]["random_routing_draws"], seed)
    pipes["label"] = pipes.apply(lambda r: r["system"] if r["llm"] == "-" else f"{r['system']} · {r['llm']}", axis=1)
    save_table(pipes, tab_dir, "tableS1_extra_uncertainty_routing",
               "Supplementary analysis: uncertainty-based routing of the Joint "
               "ensemble's non-unanimous votes to an LLM, with a matched random-routing control.",
               ["label", "accuracy", "recall_mal", "fpr", "macro_f1", "mcc", "llm_call_rate", "cost_per_1k_conv_usd"])
    curves = {k: routing_curve(yte, joint["pred"], joint["soft"], llm_te[k]["pred"].to_numpy(),
                               cfg["statistics"]["random_routing_draws"], seed) for k in llm_te}
    if curves:
        pd.concat([c.assign(llm=k) for k, c in curves.items()]).to_csv(tab_dir / "tableS2_routing_curves.csv", index=False)
    amb = np.array([convs[i].ambiguity for i in te], dtype=float)
    best_llm = max(llm_te, key=lambda k: compute_metrics(yte, llm_te[k]["pred"])["macro_f1"]) if llm_te else None
    sys_preds = {"Joint ensemble": joint["pred"]}
    if best_llm:
        sys_preds[best_llm] = llm_te[best_llm]["pred"].to_numpy()
    if len(p2_res):
        bj = p2_res[p2_res.workflow == "Joint"].sort_values("fraud_macro_f1").iloc[-1]["judge"]
        sel = p2_conv[(p2_conv.workflow == "Joint") & (p2_conv.judge == bj)].set_index("conv_id").reindex(ids[te])
        sys_preds[f"Joint + Phase 2 ({bj})"] = (sel["outcome"] == "fraudulent").astype(int).to_numpy()
    amb_rows = []
    for name, pr in sys_preds.items():
        for a in sorted(set(amb[~np.isnan(amb)])):
            m = amb == a
            amb_rows.append({"system": name, "ambiguity": a, "accuracy": float((pr[m] == yte[m]).mean()), "n": int(m.sum())})
    amb_df = pd.DataFrame(amb_rows)
    amb_df.to_csv(tab_dir / "tableS3_accuracy_by_ambiguity.csv", index=False)

    # ================================================================ PROTOCOL B: repeated CV (Phase 1)
    cv_long, cvstats = pd.DataFrame(), {}
    if cfg["protocols"]["repeated_cv"]["enabled"] and not args.no_cv:
        rc = cfg["protocols"]["repeated_cv"]
        LOG.info("=== Protocol B: repeated stratified %dx%d CV on all %d conversations ===",
                 rc["n_repeats"], rc["n_splits"], len(y))
        rskf = RepeatedStratifiedKFold(n_splits=rc["n_splits"], n_repeats=rc["n_repeats"], random_state=seed)
        recs = []
        for fold, (ftr, fte) in enumerate(rskf.split(np.zeros(len(y)), y)):
            ffits, fens, _ = classical_block(texts, conv_emb, y, ftr, fte, cfg, seed + fold, emb_label)
            for f in ffits:
                recs.append({"fold": fold, "system": f"{f.name} [{f.view}]", "group": "Classical",
                             **compute_metrics(y[fte], f.pred, f.score)})
            for (vname, strat), e in fens.items():
                recs.append({"fold": fold, "system": f"Ensemble-{strat} [{vname}]", "group": "Ensemble",
                             "members": "+".join(e["members"]), **compute_metrics(y[fte], e["pred"], e["soft"])})
            for k in llm_agg:
                d = llm_on(fte, k)
                recs.append({"fold": fold, "system": f"{k} (zero-shot)", "group": "LLM",
                             **compute_metrics(y[fte], d["pred"].to_numpy(), d["p_malicious"].to_numpy())})
            LOG.info("  fold %d/%d done", fold + 1, rc["n_splits"] * rc["n_repeats"])
        cv_long = pd.DataFrame(recs)
        cv_long.to_csv(tab_dir / "cv_per_fold.csv", index=False)
        summ = cv_long.groupby(["system", "group"])[METRIC_COLS].agg(["mean", "std"])
        summ.columns = [f"{a}_{b}" for a, b in summ.columns]
        summ = summ.reset_index().sort_values("macro_f1_mean", ascending=False)
        save_table(summ, tab_dir, "table6_repeated_cv",
                   f"Repeated stratified {rc['n_repeats']}x{rc['n_splits']}-fold CV, Phase 1 (mean and SD over folds).",
                   ["system", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std", "mcc_mean", "mcc_std"])
        piv = cv_long.pivot_table(index="fold", columns="system", values="macro_f1")
        keep = [c for c in piv.columns if f"[{jv}]" in c or "(zero-shot)" in c]
        cvstats = friedman_nemenyi(piv[keep])
        n_tr_cv = int(len(y) * (rc["n_splits"] - 1) / rc["n_splits"])
        tt = pd.DataFrame([{"system": c, **corrected_resampled_ttest((piv[c] - piv[ref_lab]).to_numpy(), n_tr_cv,
                                                                       len(y) - n_tr_cv)}
                           for c in piv.columns if c != ref_lab])
        tt["p_holm"] = holm(tt["p_value"].tolist())
        save_table(tt.sort_values("mean_diff", ascending=False), tab_dir, "table7_cv_tests_vs_joint",
                   "Repeated CV: Nadeau-Bengio corrected t-tests of macro-F1 against the Joint ensemble (Holm).")
        write_json(tab_dir / "cv_friedman_nemenyi.json", cvstats)

    # ================================================================ PROTOCOL C: external test
    external = {}
    fw = F.FigureWriter(fig_dir, mock=mock)
    if cfg["protocols"].get("external", {}).get("enabled") and not args.no_external:
        external = run_external(cfg, args, convs, client, specs, s, seed, run_dir, tab_dir, fw, budget_hit,
                                ocdd_bundle)
        budget_hit = budget_hit or external.get("budget_hit", False)

    # ================================================================ figures
    LOG.info("=== Figures ===")
    fp = tA[(tA.group == "LLM") | (tA.view == jv)]
    F.forest_plot(fw, fp.assign(label=fp.system, value=fp.macro_f1, lo=fp.macro_f1_lo, hi=fp.macro_f1_hi),
                  name="fig1_official_phase1", title=f"Official SEConvo test split, Phase 1 (n = {len(te)})")
    F.forest_plot(fw, tA[tA.group == "Classical"].assign(label=lambda d: d.system, value=lambda d: d.macro_f1,
                                                          lo=lambda d: d.macro_f1_lo, hi=lambda d: d.macro_f1_hi),
                  name="figS1_classical_all_views", title="All Phase-1 benchmark models")
    if len(cv_long):
        piv = cv_long.pivot_table(index="fold", columns="system", values="macro_f1")
        keep = [c for c in piv.columns if f"[{jv}]" in c or "(zero-shot)" in c]
        F.cv_boxplot(fw, piv[keep], dict(zip(cv_long.system, cv_long.group)), name="fig2_cv_distribution")
        if "cd" in cvstats:
            F.cd_diagram(fw, cvstats["avg_ranks"], cvstats["cd"], name="fig3_critical_difference")
    if len(p2_res):
        F.joint_vs_dual(fw, p2_res, compute_metrics(yte, joint["pred"])["macro_f1"], name="fig5_phase2_joint_vs_dual",
                        title=f"Official test split, Phase 2 (n = {len(te)})")
        ca = p2_res.assign(label=p2_res.workflow + " · " + p2_res.judge, group=p2_res.workflow.map(
            {"Joint": "Pipeline", "Dual-LLM": "LLM"}), macro_f1=p2_res.fraud_macro_f1)
        ca = pd.concat([ca, pd.DataFrame([{"label": "Joint Phase 1 only", "group": "Ensemble", "cost_per_1k_conv_usd": 0.0,
                                           "macro_f1": compute_metrics(yte, joint["pred"])["macro_f1"],
                                           "latency_per_conv_s": joint["latency"]}])], ignore_index=True)
        F.cost_accuracy(fw, ca, name="fig6_cost_accuracy")
    if curves:
        F.routing_curves(fw, curves, name="figS2_extra_routing_curves")
    cm_items = [(f"Joint ensemble ({jv})", compute_metrics(yte, joint["pred"]))]
    for k in list(llm_te)[:2]:
        cm_items.append((f"{k} (zero-shot)", compute_metrics(yte, llm_te[k]["pred"])))
    if len(p2_res):
        r = p2_res[p2_res.workflow == "Joint"].sort_values("fraud_macro_f1").iloc[-1]
        sel = p2_conv[(p2_conv.workflow == "Joint") & (p2_conv.judge == r.judge)].set_index("conv_id").reindex(ids[te])
        cm_items.append((f"Joint + Phase 2 ({r.judge})",
                         compute_metrics(yte, (sel["outcome"] == "fraudulent").astype(int).to_numpy())))
    F.confusion_grid(fw, cm_items, name="figS3_confusion_matrices")
    if len(amb_df):
        F.ambiguity_plot(fw, amb_df, name="figS4_accuracy_by_ambiguity")

    # ================================================================ report + manifest
    spent = client.spent_usd if client else 0.0
    manifest = {
        "mode": "MOCK" if mock else "REAL", "prompt_version": PROMPT_VERSION, "config": cfg,
        "environment": environment_manifest(), "n_conversations": int(len(y)),
        "official_split": [int(len(tr)), int(len(te))], "joint_ensemble": {"view": jv, "selection": js,
                                                                            "members": joint["members"]},
        "phase2_ocdd": {"official": bestA.label, "cv": [r["selected"] for r in sel_rows]},
        "phase2_judges": [j.key for j in judges], "llm_spend_this_run_usd": spent, "budget_hit": budget_hit,
        "llm_cache_entries": len(client.cache) if client else 0,
        "models_returned": {k: sorted(set(map(str, v["model_returned"].dropna()))) for k, v in llm_agg.items()},
        "external": external, "figures": fw.written,
    }
    write_json(run_dir / "manifest.json", manifest)
    md = (lambda df: df.round(4).to_markdown(index=False)) if _has_tabulate() else (lambda df: df.round(4).to_string())
    report = ["# CMOT re-evaluation report", "",
              f"Mode: **{manifest['mode']}** | prompt version {PROMPT_VERSION} | run {stamp}", "",
              f"Joint ensemble: {'+'.join(joint['members'])} on {jv}", "",
              f"## Official split ({len(tr)}/{len(te)}), Phase 1 (top 15)", "",
              md(tA[["system", "accuracy", "macro_f1", "macro_f1_lo", "macro_f1_hi", "mcc", "mcnemar_p_holm"]].head(15)), "",
              "## Phase 2: within-dialogue drift detection", "", md(det_tab.drop(columns=["n"], errors="ignore")), ""]
    if len(p2_res):
        report += ["## Phase 2: Joint vs Dual-LLM (final 'fraudulent attempt' decision)", "",
                   md(p2_res[[c for c in PHASE2_COLS if c in p2_res.columns]]), ""]
    if cvstats:
        report += ["## Repeated CV: Friedman/Nemenyi", "", str({k: v for k, v in cvstats.items() if k != "avg_ranks"}), ""]
    report += [f"LLM spend in this run: ${spent:.2f} (cache hits are free).", ""]
    (run_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    LOG.info("Done. Results in %s", run_dir)


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
