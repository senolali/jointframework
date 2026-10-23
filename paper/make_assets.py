#!/usr/bin/env python3
"""Build every number, table and figure used by the manuscript from the FINAL run.

    python paper/make_assets.py            # reads results/FINAL_RUN.txt

Writes paper/generated/numbers.tex (one macro per number cited in the text),
paper/generated/tab_*.tex (curated booktabs tables) and copies figures to paper/figures/.
No number in the manuscript is typed by hand.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
sys.path.insert(0, str(ROOT))
from cmot.data import load_seconvo, load_seconvo_external  # noqa: E402

RUN = ROOT / (ROOT / "results" / "FINAL_RUN.txt").read_text(encoding="utf-8").strip()
T = RUN / "tables"
GEN = PAPER / "generated"
FIG = PAPER / "figures"
GEN.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

JOINT = "Ensemble-inner_cv [TF-IDF+SVD (transductive)]"
macros: dict[str, str] = {}


def f3(x) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def pct(x) -> str:
    return f"{100 * x:.0f}\\%"


def pval(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def money(x) -> str:
    return f"{x:.2f}" if x >= 0.1 else f"{x:.3f}"


def M(name: str, value) -> None:
    assert name.isalpha(), name
    macros[name] = str(value)


def esc(s: str) -> str:
    return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def hcell(h: str) -> str:
    """Header cell; a newline in the text gives a two-line, centred header."""
    return "\\shortstack[c]{" + h.replace("\n", "\\\\{}") + "}" if "\n" in h else h  # {}: a following "[" is text


def table(path: str, header: list[str], rows: list[list[str]], colspec: str, caption: str, label: str,
          note: str = "", size: str = "\\small") -> None:
    lines = ["\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size,
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(hcell(h) for h in header) + " \\\\",
             "\\midrule"]
    for r in rows:
        lines.append("\\midrule" if r == ["MIDRULE"] else " & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines += ["\\par\\smallskip", "\\begin{minipage}{0.97\\linewidth}\\footnotesize " + note + "\\end{minipage}"]
    lines.append("\\end{table}")
    (GEN / path).write_text("\n".join(lines), encoding="utf-8")


def short(system: str) -> str:
    return (system.replace(" (zero-shot)", "").replace(" (few-shot k=4)", "")
            .replace("Ensemble-inner_cv", "Joint ensemble").replace("Ensemble-fixed", "kNN+SVM+ImpKmeans"))


# ---------------------------------------------------------------- data statistics
convs = load_seconvo(ROOT / "data" / "seconvo", download=False)
d = pd.DataFrame([{"split": c.split, "label": c.label, "n": len(c.messages),
                   "fp": c.first_pii if c.first_pii is not None else -1} for c in convs])
d["drift"] = np.where(d.fp < 0, "none", np.where(d.label == 1, "fraudulent", "genuine"))
M("NConv", len(d))
M("NMal", int(d.label.sum()))
M("NBen", int((d.label == 0).sum()))
M("NTrain", int((d.split == "train").sum()))
M("NTest", int((d.split == "test").sum()))
M("NTrainMal", int(((d.split == "train") & (d.label == 1)).sum()))
M("NTrainBen", int(((d.split == "train") & (d.label == 0)).sum()))
M("NTestMal", int(((d.split == "test") & (d.label == 1)).sum()))
M("NTestBen", int(((d.split == "test") & (d.label == 0)).sum()))
M("NDriftFraud", int((d.drift == "fraudulent").sum()))
M("NDriftGenuine", int((d.drift == "genuine").sum()))
M("NDriftNone", int((d.drift == "none").sum()))
M("MedianMsgs", int(d.n.median()))
mal_drift = d[d.drift == "fraudulent"]
M("ShareFraudOnsetTwo", pct((mal_drift.fp <= 2).mean()))
M("MedianOnsetFraud", int(mal_drift.fp.median()))
M("MedianOnsetGenuine", int(d[d.drift == "genuine"].fp.median()))
M("NExternal", 1000)
# corpus description (Section 4.1 and Table 1)
ext_convs = load_seconvo_external(ROOT / "data" / "seconvo")
words = [len(m.split()) for c in convs for m in c.messages]
M("MinMsgs", int(d.n.min()))
M("MaxMsgs", int(d.n.max()))
M("WordsPerMsg", f"{np.mean(words):.0f}")
M("NAmbHigh", sum(c.ambiguity == 3 for c in convs))
M("NScenCollab", sum(c.scenario == "Academic Collaboration" for c in convs))
M("NScenOther", sum(c.scenario == "Journalism" for c in convs))
rows = []
for name, cs in [("Training", [c for c in convs if c.split == "train"]),
                 ("Test", [c for c in convs if c.split == "test"]),
                 ("Held-out", ext_convs)]:
    n_ = [len(c.messages) for c in cs]
    w_ = [len(m.split()) for c in cs for m in c.messages]
    ann = cs is not ext_convs
    fp_ = [c.first_pii if (c.first_pii is not None and c.first_pii >= 0) else None for c in cs] if ann else []
    rows.append([name, str(len(cs)), f"{sum(c.label for c in cs)} / {sum(1 - c.label for c in cs)}",
                 f"{int(np.median(n_))} ({min(n_)}--{max(n_)})", f"{np.mean(w_):.0f}",
                 str(sum(f is not None and c.label == 1 for f, c in zip(fp_, cs))) if ann else "--",
                 str(sum(f is not None and c.label == 0 for f, c in zip(fp_, cs))) if ann else "--",
                 str(sum(f is None for f in fp_)) if ann else "--"])
table("tab_dataset.tex", ["Set", "Dialogues", "Malicious /\nbenign", "Messages per\ndialogue", "Words per\nmessage",
                          "Fraudulent\ndrift", "Genuine\ndrift", "No\ndrift"], rows, "lccccccc",
      "SEConvo conversations used in this study.", "tab:dataset",
      "Training and test: official split of the annotated conversations; held-out: the unannotated conversations. "
      "Messages per dialogue: median (range). Drift: the conversation contains a first PII request; its type follows "
      "the conversation label. The held-out conversations carry simulation-design labels and no drift annotation.",
      size="\\footnotesize\\setlength{\\tabcolsep}{4pt}")
# ---------------------------------------------------------------- Phase 1, official split
a = pd.read_csv(T / "table1_official_phase1_all.csv").set_index("system")
j = a.loc[JOINT]
M("JointMembers", "SVM, Random Forest and kNN")
M("JointAcc", f3(j.accuracy))
M("JointMF", f3(j.macro_f1))
M("JointMFlo", f3(j.macro_f1_lo))
M("JointMFhi", f3(j.macro_f1_hi))
ind = a.loc["Ensemble-inner_cv [TF-IDF+SVD (inductive)]"]
M("JointIndMF", f3(ind.macro_f1))
M("JointIndAcc", f3(ind.accuracy))
fx = a.loc["Ensemble-fixed [TF-IDF+SVD (transductive)]"]
M("FixedTransAcc", f3(fx.accuracy))
M("FixedTransMF", f3(fx.macro_f1))
M("ImpTransAcc", f3(a.loc["ImpKmeans [TF-IDF+SVD (transductive)]"].accuracy))
M("ImpIndAcc", f3(a.loc["ImpKmeans [TF-IDF+SVD (inductive)]"].accuracy))
models = ["kNN", "SVM", "Random Forest", "Naive Bayes", "XGBoost", "k-means", "DBSCAN", "GMM", "ImpKmeans"]
rows = []
for m in models:
    i, t = a.loc[f"{m} [TF-IDF+SVD (inductive)]"], a.loc[f"{m} [TF-IDF+SVD (transductive)]"]
    rows.append([m, f3(i.accuracy), f3(i.macro_f1), f3(t.accuracy), f3(t.macro_f1)])
rows.append(["MIDRULE"])
rows.append(["\\textbf{Joint ensemble (3 best)}", f3(ind.accuracy), f3(ind.macro_f1), f"\\textbf{{{f3(j.accuracy)}}}",
             f"\\textbf{{{f3(j.macro_f1)}}}"])
table("tab_phase1_classical.tex",
      ["Model", "Inductive\nacc.", "Inductive\nmacro-F1", "Transductive\nacc.", "Transductive\nmacro-F1"],
      rows, "lcccc", "Phase 1 candidate models on the official split (40 train / 360 test) with TF-IDF/SVD "
      "fitted on the training texts only (inductive) or on all texts without labels (transductive).",
      "tab:phase1classical",
      "Hyper-parameters and ensemble members are selected by stratified inner cross-validation on the training "
      "conversations; clusters are mapped to classes with the training labels. The Joint ensemble selected "
      "SVM, Random Forest and kNN.")

# LLMs vs Joint
llm_keys = sorted({s.replace(" (zero-shot)", "") for s in a.index if s.endswith("(zero-shot)")})
rows, zs_vals = [], []
for k in sorted(llm_keys, key=lambda k: -a.loc[f"{k} (zero-shot)"].macro_f1):
    z = a.loc[f"{k} (zero-shot)"]
    fs = a.loc[f"{k} (few-shot k=4)"] if f"{k} (few-shot k=4)" in a.index else None
    zs_vals.append((k, z.macro_f1))
    rows.append([esc(k), f3(z.accuracy), f"{f3(z.macro_f1)} [{f3(z.macro_f1_lo)}, {f3(z.macro_f1_hi)}]",
                 f3(fs.macro_f1) if fs is not None else "--", pval(z.mcnemar_p_holm), money(z.cost_per_1k_conv_usd)])
rows.append(["MIDRULE"])
rows.append(["\\textbf{Joint ensemble}", f3(j.accuracy), f"{f3(j.macro_f1)} [{f3(j.macro_f1_lo)}, {f3(j.macro_f1_hi)}]",
             "--", "--", "$\\approx$0"])
table("tab_phase1_llm.tex", ["Model", "Acc.", "Macro-F1\n[95\\% CI]", "Few-shot\nmacro-F1", "$p_{\\mathrm{Holm}}$",
                             "\\$ per 1k\nconv."], rows, "lccccc",
      "Phase 1 on the official test split: LLM classifiers (zero-shot unless stated) versus the Joint ensemble.",
      "tab:phase1llm", "$p_{\\mathrm{Holm}}$: exact McNemar test against the Joint ensemble, Holm-corrected over all "
      "Phase 1 systems. Few-shot: four class-balanced examples from the training split. Cost from the ASU "
      "CreateAI catalog prices (DeepSeek: provider list price).", size="\\footnotesize")
zs = sorted(zs_vals, key=lambda x: -x[1])
M("NLLM", len(llm_keys))
M("BestZSName", esc(zs[0][0]))
M("BestZSMF", f3(zs[0][1]))
M("LLMZSMin", f3(zs[-1][1]))
M("LLMZSMax", f3(zs[0][1]))
fsr = a[a.index.str.endswith("(few-shot k=4)")].sort_values("macro_f1", ascending=False)
M("BestFSName", esc(fsr.index[0].replace(" (few-shot k=4)", "")))
M("BestFSMF", f3(fsr.iloc[0].macro_f1))
M("BestFSp", pval(fsr.iloc[0].mcnemar_p_holm))
M("NLLMsigBetterJoint", int(((a.group == "LLM") & (a.mcnemar_p_holm < 0.05) & (a.macro_f1 > j.macro_f1)).sum()))

# ---------------------------------------------------------------- Protocol B
cv = pd.read_csv(T / "table6_repeated_cv.csv").set_index("system")
tt = pd.read_csv(T / "table7_cv_tests_vs_joint.csv").set_index("system")
M("CVJointMF", f3(cv.loc[JOINT].macro_f1_mean))
M("CVJointSD", f3(cv.loc[JOINT].macro_f1_std))
cvl = cv[cv.index.str.endswith("(zero-shot)")].sort_values("macro_f1_mean", ascending=False)
M("CVBestLLMName", esc(cvl.index[0].replace(" (zero-shot)", "")))
M("CVBestLLMMF", f3(cvl.iloc[0].macro_f1_mean))
M("CVBestLLMp", pval(tt.loc[cvl.index[0]].p_holm))
M("CVLLMMin", f3(cvl.macro_f1_mean.min()))
M("NCVsigWorse", int(((tt.index.str.endswith("(zero-shot)")) & (tt.p_holm < 0.05) & (tt.mean_diff < 0)).sum()))
rows = []
show = [JOINT, "SVM [TF-IDF+SVD (transductive)]", "Random Forest [TF-IDF+SVD (transductive)]",
        "kNN [TF-IDF+SVD (transductive)]",
        "Ensemble-inner_cv [TF-IDF+SVD (inductive)]"] + list(cvl.index)
for s in show:
    r = cv.loc[s]
    lab = short(s).replace("[TF-IDF+SVD (transductive)]", "(trans.)").replace("[TF-IDF+SVD (inductive)]", "(ind.)")
    rows.append([esc(lab), f"{f3(r.macro_f1_mean)} $\\pm$ {f3(r.macro_f1_std)}",
                 "--" if s == JOINT else f"{tt.loc[s].mean_diff:+.3f}", "--" if s == JOINT else pval(tt.loc[s].p_holm)])
table("tab_cv.tex", ["System", "Macro-F1 (mean $\\pm$ SD)", "$\\Delta$ vs Joint", "$p_{\\mathrm{Holm}}$"], rows,
      "lccc", "Phase 1 under 5$\\times$5 repeated stratified cross-validation (400 annotated conversations).",
      "tab:cv", "$p_{\\mathrm{Holm}}$: Nadeau--Bengio corrected resampled $t$-test against the Joint ensemble, "
      "Holm-corrected. LLMs are zero-shot and therefore scored on the same folds without training.",
      size="\\footnotesize")

# ---------------------------------------------------------------- Phase 2: drift detection
dt = pd.read_csv(T / "table4_drift_detection.csv")
NEW, OLD, RULE = "OCDD (normal reference)", "Within-dialogue window", "Fixed message-4 rule"
DET_LABEL = {NEW: "OCDD, normal-message reference", OLD: "OCDD, within-dialogue window",
             RULE: "Fixed rule: drift at message 4"}


def drow(proto: str, det: str):
    return dt[(dt.protocol == proto) & (dt.detector == det)].iloc[0]


cvn, ofn, cvo, cvr = drow("cv", NEW), drow("official", NEW), drow("cv", OLD), drow("cv", RULE)
M("OcddDetect", pct(cvn.detect_rate))
M("OcddLocal", pct(cvn.localized_rate))
M("OcddFalse", pct(cvn.false_drift_rate))
M("OcddBal", f3(cvn.balanced))
M("OcddBalLo", f3(cvn.balanced_lo))
M("OcddBalHi", f3(cvn.balanced_hi))
M("OcddDelay", int(cvn.median_delay))
M("OcddOffBal", f3(ofn.balanced))
M("OcddOffBalLo", f3(ofn.balanced_lo))
M("OcddOffBalHi", f3(ofn.balanced_hi))
M("OldBal", f3(cvo.balanced))
M("OldLocal", pct(cvo.localized_rate))
M("OldFalse", pct(cvo.false_drift_rate))
M("RuleBal", f3(cvr.balanced))
M("NDriftAll", int(cvn.n_drift))
M("NNoDriftAll", int(cvn.n_no_drift))
selcv = pd.read_csv(T / "phase2_ocdd_selection_cv.csv")
oc = json.loads((RUN / "external_ocdd_selected.json").read_text())  # selected on all 400 annotated dialogues
M("OcddDims", oc["dims"])
M("OcddNu", oc["nu"])
M("OcddQ", oc["q"])
M("OcddM", oc["m"])
M("OcddRho", oc["rho"])
M("OcddFoldsSame", int(selcv.selected.str.startswith(f"normal | embedding d={oc['dims']} |").sum()))
M("OcddQRange", "--".join(sorted({x.split("q=")[1].split(" ")[0] for x in selcv.selected})))
offsel = json.loads((RUN / "phase2_ocdd_selected.json").read_text())["official"]
M("OcddOffCfg", esc(offsel["label"]).replace("|", ","))
rows = []
SHORT = {NEW: "OCDD, normal-message ref.", OLD: "OCDD, dialogue window", RULE: "Fixed rule (message 4)"}
for det in [NEW, OLD, RULE]:
    o, c = drow("official", det), drow("cv", det)
    rows.append([SHORT[det], f3(o.balanced), pct(c.detect_rate), pct(c.localized_rate), pct(c.false_drift_rate),
                 f"{f3(c.balanced)} [{f3(c.balanced_lo)}, {f3(c.balanced_hi)}]"])
table("tab_ocdd.tex", ["Detector", "Official\nbalanced", "CV\ndetected", "CV\nlocalized", "CV false\ndrift",
                       "CV balanced\n[95\\% CI]"], rows, "lccccc",
      "Drift detection against the annotated drift onset. Official: configuration selected on the 40 training "
      f"dialogues, scored on the 360 test dialogues. CV: 5-fold cross-validation over the 400 annotated dialogues "
      f"({int(cvn.n_drift)} with and {int(cvn.n_no_drift)} without a drift), configuration selected inside each "
      "training fold.", "tab:ocdd",
      "Localized: drift flagged at the annotated onset or up to two messages later. Balanced: mean of the "
      "localization rate and one minus the false-drift rate; 0.5 is a nominal reference value (a detector that never fires scores 0.5, "
      "one that always fires at most 0.5).", size="\\footnotesize")

# ---------------------------------------------------------------- Phase 2: Joint vs Dual-LLM vs no OCDD (CV, 400)
p2 = pd.read_csv(T / "table5b_phase2_cv.csv")
jt, dd = p2[p2.workflow == "Joint"].set_index("judge"), p2[p2.workflow == "Dual-LLM"].set_index("judge")
nn = p2[p2.workflow == "Joint (no OCDD)"].set_index("judge")
order = jt.sort_values("fraud_macro_f1", ascending=False).index
EXP = RUN / "experiments"
pf = pd.read_csv(EXP / "paired_f1.csv")
pfd = pf[(pf.protocol == "cv") & (pf.comparison == "Joint - Dual-LLM")].set_index("judge")
pfn = pf[(pf.protocol == "cv") & (pf.comparison == "Joint - Joint (no OCDD)")].set_index("judge")
pfe = pf[(pf.protocol == "external") & (pf.comparison == "Joint - Dual-LLM")].set_index("judge")
pfen = pf[(pf.protocol == "external") & (pf.comparison == "Joint - Joint (no OCDD)")].set_index("judge")
sd = lambda x: f"{x:+.3f}"  # noqa: E731
rows = []
for k in order:
    r, q, n_ = jt.loc[k], dd.loc[k], nn.loc[k]
    b = pfd.loc[k]
    rows.append([esc(k), pct(r.type_acc_fraud), pct(r.type_acc_genuine), f3(r.fraud_macro_f1),
                 f3(n_.fraud_macro_f1), f3(q.fraud_macro_f1), f"{sd(b['diff'])} [{sd(b.ci_lo)}, {sd(b.ci_hi)}]",
                 money(r.cost_per_1k_conv_usd), money(q.cost_per_1k_conv_usd)])
M("DeltaDualMin", f3(pfd["diff"].min()))
M("DeltaDualMax", f3(pfd["diff"].max()))
M("DeltaDualMaxP", pval(pfd.p_holm.max()))
M("DeltaNoOcddMin", sd(pfn["diff"].min()))
M("DeltaNoOcddMax", sd(pfn["diff"].max()))
M("ExtDeltaDualMin", f3(pfe["diff"].min()))
M("ExtDeltaDualMax", f3(pfe["diff"].max()))
M("ExtDeltaNoOcddMin", sd(pfen["diff"].min()))
M("ExtDeltaNoOcddMax", sd(pfen["diff"].max()))
table("tab_phase2.tex", ["Judge LLM", "Fraud.\ndrift", "Genuine\ndrift", "Joint", "No\nOCDD", "Dual-\nLLM",
                         "$\\Delta$ vs Dual\n[95\\% CI]", "\\$/1k\nJoint", "\\$/1k\nDual"], rows, "lcccccccc",
      "Phase 2 under 5-fold cross-validation over the 400 annotated dialogues: drift-type accuracy of the judge and "
      "macro-F1 of the final \\emph{fraudulent attempt} decision for the Joint framework, the Joint framework "
      "without OCDD and the Dual-LLM framework.", "tab:phase2",
      "Drift-type columns: share of judged Joint conversations whose annotated drift type the judge recovered. "
      "$\\Delta$: macro-F1 of Joint minus Dual-LLM with the same judge, paired bootstrap CI; all Holm-adjusted "
      f"$p\\le{pval(pfd.p_holm.max())}$ (exact McNemar: $p_{{\\mathrm{{Holm}}}}\\le{pval(jt.mcnemar_p_holm.max())}$). "
      f"Joint vs no OCDD: $\\Delta$ between {sd(pfn['diff'].min())} and {sd(pfn['diff'].max())}, all not significant. "
      "Joint Phase 1 alone: macro-F1 "
      f"{f3(jt.iloc[0].phase1_macro_f1)}. Confidence intervals of each framework are shown in "
      "Figure~\\ref{fig:jointdual}.", size="\\footnotesize\\setlength{\\tabcolsep}{3pt}")
best = jt.loc[order[0]]
M("PTwoBestJudge", esc(order[0]))
M("PTwoBestMF", f3(best.fraud_macro_f1))
M("PTwoBestLo", f3(best.fraud_macro_f1_lo))
M("PTwoBestHi", f3(best.fraud_macro_f1_hi))
M("PTwoBestPrec", f3(best.fraud_precision))
M("PTwoBestCost", money(best.cost_per_1k_conv_usd))
M("PTwoPhaseOne", f3(best.phase1_macro_f1))
M("PTwoGain", f"{100 * (best.fraud_macro_f1 - best.phase1_macro_f1):.1f}")
M("NJudges", len(jt))
M("NJointWins", int((jt.fraud_macro_f1 > dd.reindex(jt.index).fraud_macro_f1).sum()))
M("NJointSig", int((jt.mcnemar_p_holm < 0.05).sum()))
M("PTwoMaxP", pval(jt.mcnemar_p_holm.max()))
M("NoOcddMaxDiff", f3((jt.fraud_macro_f1 - nn.reindex(jt.index).fraud_macro_f1).abs().max()))
M("NoOcddMinP", f3(jt.mcnemar_p_holm_vs_no_ocdd.min()))
M("JointDriftShare", pct((jt.drift_detected / jt.flagged).iloc[0]))
M("JointFlagged", int(jt.flagged.iloc[0]))
M("JointDriftN", int(jt.drift_detected.iloc[0]))
M("CallsSaved", pct(1 - (jt.llm_calls_per_conv / nn.reindex(jt.index).llm_calls_per_conv).iloc[0]))
M("TypeFraudMin", pct(jt.type_acc_fraud.min()))
M("TypeFraudMax", pct(jt.type_acc_fraud.max()))
M("TypeGenMin", pct(jt.type_acc_genuine.min()))
M("TypeGenMax", pct(jt.type_acc_genuine.max()))
g = jt.loc["Gemma 4 31B (ASU)"]
M("GemmaJointMF", f3(g.fraud_macro_f1))
M("GemmaJointCost", money(g.cost_per_1k_conv_usd))
M("GemmaDualCost", money(dd.loc["Gemma 4 31B (ASU)"].cost_per_1k_conv_usd))
nv = jt.loc["Nova 2 Lite"]
M("NovaBlocked", pct(nv.n_unjudged / nv.drift_detected))
cl = jt.loc["Claude 5 Sonnet"]
M("ClaudeRecall", pct(cl.fraud_recall))
M("ClaudeGenuine", pct(cl.type_acc_genuine))

# official split Phase 2 (OCDD selected on the 40 training dialogues) -> appendix
po = pd.read_csv(T / "table5_phase2_official.csv")
ojt, odd = po[po.workflow == "Joint"].set_index("judge"), po[po.workflow == "Dual-LLM"].set_index("judge")
onn = po[po.workflow == "Joint (no OCDD)"].set_index("judge")
rows = [[esc(k), f"{f3(ojt.loc[k].fraud_macro_f1)} [{f3(ojt.loc[k].fraud_macro_f1_lo)}, {f3(ojt.loc[k].fraud_macro_f1_hi)}]",
         f3(onn.loc[k].fraud_macro_f1), f3(odd.loc[k].fraud_macro_f1), pval(ojt.loc[k].mcnemar_p_holm)]
        for k in ojt.sort_values("fraud_macro_f1", ascending=False).index]
table("tab_phase2_official.tex", ["Judge LLM", "Joint [95\\% CI]", "No OCDD", "Dual-LLM", "$p_{\\mathrm{Holm}}$"], rows,
      "lcccc", "Phase 2 on the official test split (360 dialogues), with the OCDD configuration selected on the 40 "
      "training dialogues.", "tab:phase2official",
      "$p_{\\mathrm{Holm}}$: exact McNemar test Joint vs Dual-LLM (same judge), Holm-corrected.", size="\\footnotesize")
M("OffNJointWins", int((ojt.fraud_macro_f1 > odd.reindex(ojt.index).fraud_macro_f1).sum()))
M("OffMinP", pval(ojt.mcnemar_p_holm.min()))

# ---------------------------------------------------------------- follow-up experiments (experiments/*.py)
fac = pd.read_csv(EXP / "ocdd_factorial.csv")
fx = lambda ref, rp: fac[(fac.reference == ref) & (fac.representation == rp)].iloc[0]  # noqa: E731
WIN, NRM, TF, EM = "within-dialogue window", "normal-message reference", "TF-IDF+SVD", "te3-large embeddings"
M("FactWinTf", f3(fx(WIN, TF).balanced))
M("FactWinEmb", f3(fx(WIN, EM).balanced))
M("FactNormTf", f3(fx(NRM, TF).balanced))
M("FactNormTfLo", f3(fx(NRM, TF).balanced_lo))
M("FactNormTfHi", f3(fx(NRM, TF).balanced_hi))
M("FactNormEmb", f3(fx(NRM, EM).balanced))
rows = [[ref.capitalize(), *[f"{f3(fx(ref, rp).balanced)} [{f3(fx(ref, rp).balanced_lo)}, {f3(fx(ref, rp).balanced_hi)}] "
                             f"({pct(fx(ref, rp).localized_rate)} / {pct(fx(ref, rp).false_drift_rate)})" for rp in [TF, EM]]]
        for ref in [WIN, NRM]]
table("tab_ocdd_factorial.tex", ["Reference", "TF-IDF+SVD", "te3-large embeddings"], rows, "lcc",
      "Drift detection under 5-fold cross-validation: reference definition $\\times$ message representation "
      "(configurations selected inside each training fold).", "tab:ocddfactorial",
      "Balanced score [95\\% CI] (localization / false-drift rate).", size="\\scriptsize")
late = {}
for key, f in [("ocdd", "phase2_detections_cv_ocdd.csv"), ("rule", "phase2_detections_cv_fixed.csv"),
               ("win", "phase2_detections_cv_within-dialogue.csv")]:
    dd_ = pd.read_csv(T / f)
    late[key] = dd_[dd_.first_pii >= 6]
rng_ = np.random.default_rng(int(cfg_seed := json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))["config"]["seed"]))
bs = [late["ocdd"].localized.iloc[rng_.integers(0, len(late["ocdd"]), len(late["ocdd"]))].mean() for _ in range(2000)]
M("LateN", len(late["ocdd"]))
M("LateOcdd", pct(late["ocdd"].localized.mean()))
M("LateOcddLo", pct(np.percentile(bs, 2.5)))
M("LateOcddHi", pct(np.percentile(bs, 97.5)))
M("LateRule", pct(late["rule"].localized.mean()))
M("LateWin", pct(late["win"].localized.mean()))
M("ShareOnsetTwoFour", pct(pd.read_csv(T / "phase2_detections_cv_ocdd.csv").pipe(
    lambda d: d[d.has_drift].first_pii.isin([2, 4]).mean())))
ji = pd.read_csv(EXP / "judge_inputs_summary.csv")
jt_ = pd.read_csv(EXP / "judge_inputs_tests.csv")
ew = pd.read_csv(EXP / "judge_inputs_early_warning.csv")
VAR = ["full", "loc_ocdd", "loc_fixed", "prefix_ocdd", "prefix_fixed"]
rows = []
for k, g in ji.groupby("judge", sort=False):
    g = g.set_index("variant")
    rows.append([esc(k)] + [f3(g.loc[v, "fraud_macro_f1"]) for v in VAR])
table("tab_judge_inputs.tex", ["Judge LLM", "Whole\ndialogue", "+ OCDD\nlocation", "+ fixed\nlocation",
                               "Prefix to\nOCDD alarm", "Prefix to\nmessage 4"], rows, "lccccc",
      "Judge-input experiment (5-fold CV, Joint framework): macro-F1 of the final \\emph{fraudulent attempt} "
      "decision when the judge receives the whole dialogue, the dialogue plus the drift location (from OCDD or "
      "the fixed fourth message), or only the messages up to the alarm.", "tab:judgeinputs",
      "Fixed variants are controls that use no detector output. Paired bootstrap of OCDD minus fixed: no judge "
      "improves (Holm-adjusted).", size="\\footnotesize")
lc = jt_[jt_.comparison == "loc_ocdd - loc_fixed"]
pc_ = jt_[jt_.comparison == "prefix_ocdd - prefix_fixed"]
M("NJudgeInput", ji.judge.nunique())
M("LocBetter", int(((lc["diff"] > 0) & (lc.p_holm < 0.05)).sum()))
M("LocWorse", int(((lc["diff"] < 0) & (lc.p_holm < 0.05)).sum()))
M("PrefixBetter", int(((pc_["diff"] > 0) & (pc_.p_holm < 0.05)).sum()))
M("EWMsgs", f"{ew[ew.variant == 'prefix_ocdd'].msgs_before_end.mean():.1f}")
pfo = ji[ji.variant == "prefix_ocdd"].set_index("judge").fraud_macro_f1
ful = ji[ji.variant == "full"].set_index("judge").fraud_macro_f1
M("EWLossMin", f3((ful - pfo).min()))
M("EWLossMax", f3((ful - pfo).max()))
M("EWGainMax", f3((pfo - ful).max()))

# ---------------------------------------------------------------- Protocol C (external)
e1 = pd.read_csv(T / "table8_external_phase1.csv").set_index("system")
e2 = pd.read_csv(T / "table9_external_phase2.csv")
ej = e1.loc[JOINT]
M("ExtJointMF", f3(ej.macro_f1))
M("ExtJointLo", f3(ej.macro_f1_lo))
M("ExtJointHi", f3(ej.macro_f1_hi))
M("ExtJointAcc", f3(ej.accuracy))
el = e1[e1.group == "LLM"].sort_values("macro_f1", ascending=False)
M("ExtLLMMin", f3(el.macro_f1.min()))
M("ExtLLMMax", f3(el.macro_f1.max()))
M("ExtBestLLM", esc(el.index[0].replace(" (zero-shot)", "")))
M("ExtLLMPrecMin", f3(el.precision_mal.min()))
M("ExtLLMPrecMax", f3(el.precision_mal.max()))
M("ExtLLMRecMin", f3(el.recall_mal.min()))
M("ExtLLMRecMax", f3(el.recall_mal.max()))
M("ExtJointPrec", f3(ej.precision_mal))
M("ExtJointRec", f3(ej.recall_mal))
M("ExtIndSVM", f3(e1.loc["SVM [TF-IDF+SVD (inductive)]"].macro_f1))
M("NExtLLMsig", int((el.mcnemar_p_holm < 0.001).sum()))
ej2, ed2 = e2[e2.workflow == "Joint"].set_index("judge"), e2[e2.workflow == "Dual-LLM"].set_index("judge")
M("ExtPTwoJointMin", f3(ej2.fraud_macro_f1.min()))
M("ExtPTwoJointMax", f3(ej2.fraud_macro_f1.max()))
M("ExtPTwoDualMin", f3(ed2.fraud_macro_f1.min()))
M("ExtPTwoDualMax", f3(ed2.fraud_macro_f1.max()))
M("NExtJudges", len(ej2))
M("ExtPTwoMaxP", pval(ej2.mcnemar_p_holm.max()))
en2 = e2[e2.workflow == "Joint (no OCDD)"].set_index("judge")
M("ExtNoOcddMaxDiff", f3((ej2.fraud_macro_f1 - en2.reindex(ej2.index).fraud_macro_f1).abs().max()))
M("ExtDriftShare", pct((ej2.drift_detected / ej2.flagged).iloc[0]))
M("ExtDriftRate", pct(oc["drift_rate"]))
rows = []
for s in [JOINT, "SVM [TF-IDF+SVD (transductive)]", "SVM [TF-IDF+SVD (inductive)]", "SVM [te3-large]"] + list(el.index):
    r = e1.loc[s]
    lab = short(s).replace("[TF-IDF+SVD (transductive)]", "(TF-IDF trans.)").replace(
        "[TF-IDF+SVD (inductive)]", "(TF-IDF ind.)").replace("[te3-large]", "(te3-large)")
    rows.append([esc(lab), f3(r.precision_mal), f3(r.recall_mal), f"{f3(r.macro_f1)} [{f3(r.macro_f1_lo)}, {f3(r.macro_f1_hi)}]",
                 "--" if s == JOINT else pval(r.mcnemar_p_holm)])
table("tab_external_p1.tex", ["System", "Prec.", "Rec.", "Macro-F1 [95\\% CI]", "$p_{\\mathrm{Holm}}$"], rows,
      "lcccc", "External test, Phase 1: trained on the 400 annotated conversations and tested on 1,000 disjoint "
      "SEConvo conversations.", "tab:externalp1",
      "LLMs zero-shot. Precision and recall refer to the malicious class. $p_{\\mathrm{Holm}}$: exact McNemar "
      "test against the Joint ensemble.", size="\\footnotesize")
# ---------------------------------------------------------------- spend + manifest facts
man = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
p2cfg = man["config"]["phase2"]
judge_set = set(man.get("phase2_judges", []))
ext_judges = set(man.get("external", {}).get("phase2_judges", []))
rows = []
for mdl in man["config"]["llms"]:
    route = mdl.get("base_url") and f"{mdl['model']} (provider API)" or f"{mdl['provider']}/{mdl['model']}"
    roles = ["P1"] + (["J"] if mdl["key"] in judge_set else []) + (["J$_\\mathrm{ext}$"] if mdl["key"] in ext_judges else [])
    rows.append([esc(mdl["key"]), "\\texttt{" + esc(route) + "}", "yes" if mdl.get("open_weights") else "no",
                 f"{mdl['price_in']:.2f} / {mdl['price_out']:.2f}", ", ".join(roles)])
table("tab_models.tex", ["Model", "Endpoint", "Open weights", "\\$ per 1M tokens (in / out)", "Role"], rows, "llccc",
      "Large language models evaluated. All except DeepSeek were accessed through the ASU CreateAI platform.",
      "tab:models", "P1: Phase 1 classifier (zero- and few-shot); J: Phase 2 judge on the official split; "
      "J$_\\mathrm{ext}$: Phase 2 judge on the external test. DeepSeek was excluded as a judge after its API credit "
      "was exhausted; its Phase 1 results are reported. Decoding: temperature 0 where the model accepts it.",
      size="\\footnotesize")
M("NJudgeExt", len(ext_judges))

# ---------------------------------------------------------------- configuration details (from code + run)
from types import SimpleNamespace  # noqa: E402

from cmot.classical import model_zoo  # noqa: E402
from cmot.llm_tasks import CLASSIFY_TEMPLATE, SYSTEM_CLASSIFY  # noqa: E402

cfg = man["config"]


def grid_str(g: dict) -> str:
    parts = []
    for k, v in g.items():
        k = k.replace("clf__", "").replace("clusterer__", "")
        parts.append(f"{k} $\\in$ \\{{{', '.join(str(x) for x in v)}\\}}")
    return esc("; ".join(parts)).replace("\\$\\textbackslash in\\$", "$\\in$")


def best_str(s: str) -> str:
    try:
        d_ = eval(s, {"__builtins__": {}}, {})  # best_params were stored as a dict repr
    except Exception:
        return esc(s)
    return esc(", ".join(f"{k.replace('clf__', '').replace('clusterer__', '')}={v}" for k, v in d_.items()))


dummy = SimpleNamespace(S_train=object())
fixed = {"kNN": "standardized; Euclidean", "SVM": "standardized; class-balanced; probability outputs",
         "Random Forest": "300 trees; class-balanced", "Naive Bayes": "Complement NB on TF-IDF counts",
         "XGBoost": "300 trees; learning rate 0.05; subsample 0.9", "k-means": "10 initializations",
         "DBSCAN": "$\\varepsilon$ = median $k$-distance", "GMM": "diagonal covariance",
         "ImpKmeans": "official package; KDE in the 10-dimensional SVD space; 1 k-means iteration"}
rows = []
for md in model_zoo(int(cfg["seed"]), dummy):
    lab = f"{md.name} [TF-IDF+SVD (transductive)]"
    sel = best_str(a.loc[lab].best_params) if lab in a.index else "--"
    g = grid_str(md.grid).replace("\\textbackslash{}in", "$\\in$")
    rows.append([esc(md.name), fixed.get(md.name, ""), g, sel])
table("tab_hyper.tex", ["Model", "Fixed settings", "Inner-CV search grid", "Selected (official split)"], rows,
      "p{2.0cm}p{3.4cm}p{4.6cm}p{3.4cm}",
      "Phase 1 model configurations. Hyper-parameters are selected by 5-fold stratified inner cross-validation "
      "on the training conversations (criterion: macro-F1); the last column gives the values selected on the "
      "official split with the transductive TF-IDF/SVD representation.", "tab:hyper", size="\\scriptsize")

rep = cfg["representations"]["tfidf"]
og = cfg["phase2"]["ocdd"]
ls = cfg["llm_settings"]
rows = [
    ["Text representation", f"TF-IDF on word {rep['ngram_range'][0]}--{rep['ngram_range'][1]}-grams, sublinear tf, "
     f"English stop words, min.\\ document frequency {rep['min_df']}, at most {rep['max_features']:,} terms; "
     f"truncated SVD to {rep['svd_components']} components and $\\ell_2$ normalization. Fitted inductively "
     "(training texts) and transductively (all texts, no labels)."],
    ["OCDD message spaces", "TF-IDF + truncated SVD ($d\\in\\{" + ", ".join(map(str, og["tfidf_dims"])) +
     "\\}$) and PCA-reduced te3-large embeddings ($d\\in\\{" + ", ".join(map(str, og["emb_dims"])) +
     "\\}$) of individual messages, fitted without labels."],
    ["OCDD search grid", "design $\\in$ \\{normal-message reference, shift from the opening message\\}; outlier "
     "threshold at the $q$-quantile of held-out normal-message scores, $q\\in\\{" + ", ".join(map(str, og["q"])) +
     "\\}$; $(\\ell,\\rho)\\in\\{" + ", ".join(f"({a},{b})" for a, b in og["m_rho"]) + "\\}$; one-class SVM "
     "$\\nu\\in\\{" + ", ".join(map(str, og["nu"])) + "\\}$, RBF kernel, $\\gamma$ = scale, standardized inputs; "
     "nested 5-fold cross-fitting within the training dialogues."],
    ["OCDD selected", f"normal-message reference (initiator messages before the first PII request in the training "
     "dialogues); "
     f"embeddings, $d={oc['dims']}$; $\\nu={oc['nu']}$; $q={oc['q']}$; $\\ell={oc['m']}$, $\\rho={oc['rho']}$ "
     f"(selected on all 400 annotated dialogues; identical design in all CV folds); monitored: initiator "
     f"messages after the opening one; localization tolerance {cfg['phase2']['localization_tolerance']} messages."],
    ["LLM decoding", f"temperature {ls['temperature']} where accepted (GPT-6 models reject the parameter); maximum "
     "output tokens 600 (Claude, Llama, Gemma), 1,000 (Nova), 3,000 (GPT-6, Gemini, gpt-oss), 4,000 (DeepSeek); "
     "outputs cut by the limit are re-requested once with a 4$\\times$ budget (cap 32,000)."],
    ["LLM classification", f"JSON output (label, confidence, evidence turns, rationale); unparseable outputs "
     f"counted as {ls['invalid_policy']}; few-shot: {ls['fewshot']['k']} class-balanced training examples; "
     f"{ls['repeats']} run per input."],
    ["LLM judge", "judge prompt (Figure~\\ref{fig:prompt}), dialogue as JSON; verdict parsed from "
     "`fraudulent' / `genuine'."],
    ["Software", f"Python {man['environment']['python'].split()[0]}, scikit-learn "
     f"{man['environment']['packages']['sklearn']}, "
     f"impkmeans, XGBoost {man['environment']['packages'].get('xgboost')}; LLMs via the ASU CreateAI "
     "OpenAI-compatible API (DeepSeek via its own API)."],
]
lines = ["\\begin{table}[htbp]", "\\centering",
         "\\caption{Experimental configuration (generated from the run's configuration file).}",
         "\\label{tab:config}", "\\footnotesize", "\\begin{tabularx}{\\linewidth}{@{}p{3.1cm}X@{}}", "\\toprule",
         "\\textbf{Component} & \\textbf{Setting} \\\\", "\\midrule"]
lines += [f"{r[0]} & {r[1]} \\\\" for r in rows]
lines += ["\\bottomrule", "\\end{tabularx}", "\\end{table}"]
(GEN / "tab_config.tex").write_text("\n".join(lines), encoding="utf-8")


def tex_verb(s: str) -> str:
    return (s.replace("\\", "\\textbackslash{}").replace("{", "\\{").replace("}", "\\}").replace("_", "\\_")
            .replace("%", "\\%").replace("&", "\\&").replace("#", "\\#").replace('"', "\\textquotedbl{}")
            .replace("\n", "\\\\\n"))


# Joint vs Dual-LLM with the same judge: calls and latency (cited in the text)
J = "Gemma 4 31B (ASU)"
jo, do_ = jt.loc[J], dd.loc[J]
M("GemmaJointCalls", f"{jo.llm_calls_per_conv:.2f}")
M("GemmaDualCalls", f"{do_.llm_calls_per_conv:.2f}")
M("GemmaJointLat", f"{jo.latency_per_conv_s:.1f}")
M("GemmaDualLat", f"{do_.latency_per_conv_s:.1f}")
M("GemmaExtJointMF", f3(ej2.loc[J].fraud_macro_f1))
M("GemmaExtDualMF", f3(ed2.loc[J].fraud_macro_f1))

user = CLASSIFY_TEMPLATE.replace("{fewshot}", "[few-shot examples, if any]\n").replace("{transcript}", "<transcript>")
(GEN / "prompt_classify.tex").write_text(
    "\\textbf{System:} \\textit{``" + tex_verb(SYSTEM_CLASSIFY) + "''}\\\\[4pt]\n\\textbf{User:} \\textit{``"
    + tex_verb(user).replace("<transcript>", "\\texttt{[i] Name: message \\dots}") + "''}\n", encoding="utf-8")
M("RunID", RUN.name)
M("PythonVersion", man["environment"]["python"].split()[0])
M("SklearnVersion", man["environment"]["packages"]["sklearn"])
M("FewShotK", man["config"]["llm_settings"]["fewshot"]["k"])

(GEN / "numbers.tex").write_text("% AUTO-GENERATED by paper/make_assets.py from " + RUN.name + " -- do not edit\n" +
                                 "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n",
                                 encoding="utf-8")
from cmot import figures as F  # noqa: E402

for old in FIG.glob("fig*.pdf"):
    old.unlink()
fw = F.FigureWriter(FIG)
evcv = pd.read_csv(T / "phase2_detections_cv_ocdd.csv")
assert abs(evcv[evcv.has_drift].localized.mean() - cvn.localized_rate) < 1e-9
sc = dt[dt.protocol == "cv"].assign(detector=lambda d: d.detector.map(
    {NEW: "OCDD\nnormal ref.", OLD: "OCDD\nwithin-dialogue", RULE: "Fixed rule\n(message 4)"}))
F.drift_detectors(fw, evcv, sc.reset_index(drop=True), name="fig_dialogue_drift")
F.joint_vs_dual_panels(fw, [(f"(a) 5-fold CV over the {macros['NConv']} annotated dialogues", p2,
                             jt.iloc[0].phase1_macro_f1),
                            ("(b) External test (n = 1,000)", e2, ej.macro_f1)], name="fig_joint_vs_dual")
sys.path.insert(0, str(PAPER))
import make_diagrams  # noqa: E402  workflow diagrams (fig_framework, fig_selection)

make_diagrams.framework()
make_diagrams.selection()
print(f"{len(macros)} macros, tables and figures written from {RUN.name}")
