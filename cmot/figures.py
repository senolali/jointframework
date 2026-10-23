"""Publication figures (PDF vector + 300-dpi PNG). Colour is assigned by job:
categorical slots in fixed order (validated CVD-safe palette), grey for
reference/control elements; every multi-series figure has a legend."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
INK, INK2, GRID, CONTROL = "#0b0b0b", "#52514e", "#e4e3df", "#9b9a95"
GROUP_COLOR = {"Classical": BLUE, "LLM": ORANGE, "Ensemble": AQUA, "Pipeline": VIOLET}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})


class FigureWriter:
    def __init__(self, out_dir: Path, mock: bool = False):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.mock = mock
        self.written: list[str] = []

    def save(self, fig, name: str) -> None:
        if self.mock:
            fig.text(0.5, 0.5, "OFFLINE MOCK — NOT REAL RESULTS", ha="center", va="center", fontsize=20,
                     color=RED, alpha=0.25, rotation=20, transform=fig.transFigure)
        for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
            fig.savefig(self.out_dir / f"{name}.{ext}", **kw)
        plt.close(fig)
        self.written.append(name)


# ---------------------------------------------------------------------------
def forest_plot(fw: FigureWriter, df: pd.DataFrame, metric: str = "macro_f1", name: str = "fig1_model_comparison",
                title: str = "") -> None:
    """df columns: label, group, value, lo, hi (sorted inside)."""
    d = df.sort_values("value").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.2, 0.28 * len(d) + 1.2))
    for i, r in d.iterrows():
        c = GROUP_COLOR.get(r["group"], INK2)
        ax.plot([r["lo"], r["hi"]], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.plot(r["value"], i, "o", color=c, ms=7, mec="white", mew=1.5)
        ax.text(min(r["hi"] + 0.01, 1.0), i, f"{r['value']:.3f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(d)), d["label"])
    ax.set_xlabel({"macro_f1": "Macro-F1 (95% bootstrap CI)", "mcc": "MCC (95% bootstrap CI)"}.get(metric, metric))
    ax.set_xlim(max(0, d["lo"].min() - 0.05), 1.05)
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=GROUP_COLOR[g], label=g)
               for g in GROUP_COLOR if g in set(d["group"])]
    ax.legend(handles=handles, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
    if title:
        ax.set_title(title, loc="left")
    fw.save(fig, name)


def cv_boxplot(fw: FigureWriter, folds: pd.DataFrame, groups: dict[str, str], name: str = "fig2_cv_distribution") -> None:
    """folds: rows = fold, columns = model -> macro-F1."""
    order = folds.median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(6.2, 0.3 * len(order) + 1.2))
    bp = ax.boxplot([folds[m].dropna() for m in order], vert=False, widths=0.55, patch_artist=True,
                    medianprops={"color": INK, "lw": 1.5}, flierprops={"ms": 3, "mfc": CONTROL, "mec": CONTROL},
                    whiskerprops={"color": INK2}, capprops={"color": INK2})
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(GROUP_COLOR.get(groups.get(m, ""), CONTROL))
        patch.set_alpha(0.75)
        patch.set_edgecolor("white")
    ax.set_yticks(range(1, len(order) + 1), order)
    ax.set_xlabel("Macro-F1 across repeated stratified CV test folds")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g], alpha=0.75, label=g)
               for g in GROUP_COLOR if g in set(groups.get(m) for m in order)]
    ax.legend(handles=handles, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
    fw.save(fig, name)


def cd_diagram(fw: FigureWriter, avg_ranks: dict[str, float], cd: float, name: str = "fig3_critical_difference") -> None:
    items = sorted(avg_ranks.items(), key=lambda kv: kv[1])
    k = len(items)
    fig, ax = plt.subplots(figsize=(6.4, 0.22 * k + 1.4))
    ax.set_xlim(1, k)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.hlines(0.85, 1, k, color=INK2, lw=1)
    for r in range(1, k + 1):
        ax.vlines(r, 0.83, 0.87, color=INK2, lw=1)
        ax.text(r, 0.9, str(r), ha="center", fontsize=7.5, color=INK2)
    half = (k + 1) // 2
    for i, (m, r) in enumerate(items):
        left = i < half
        y = 0.75 - 0.6 * ((i if left else k - 1 - i) / max(1, half))
        x_end = 1 if left else k
        ax.plot([r, r, x_end], [0.85, y, y], color=INK, lw=0.8)
        ax.text(x_end + (-0.05 if left else 0.05), y, f"{m} ({r:.2f})", ha="right" if left else "left",
                va="center", fontsize=7.5)
    # cliques: maximal groups whose rank range < CD
    ranks = [r for _, r in items]
    cliques, y = [], 0.8
    for i in range(k):
        j = max(jj for jj in range(i, k) if ranks[jj] - ranks[i] < cd)
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for n, (a, b) in enumerate(cliques):
        yy = 0.8 - 0.035 * n
        ax.hlines(yy, ranks[a] - 0.03, ranks[b] + 0.03, color=RED, lw=3)
    ax.hlines(0.98, 1, 1 + cd, color=INK, lw=1.5)
    ax.text(1 + cd / 2, 1.0, f"CD = {cd:.2f}", ha="center", fontsize=8)
    fw.save(fig, name)


def cost_accuracy(fw: FigureWriter, df: pd.DataFrame, x: str = "cost_per_1k_conv_usd", name: str = "fig4_cost_accuracy",
                  xlabel: str = "LLM cost per 1,000 conversations (USD, log scale; no-LLM systems shown at 10$^{-3}$)") -> None:
    """df columns: label, group, macro_f1, x."""
    d = df.copy()
    d[x] = d[x].clip(lower=1e-3)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for g, gd in d.groupby("group"):
        ax.scatter(gd[x], gd["macro_f1"], s=46, color=GROUP_COLOR.get(g, INK2), edgecolor="white", lw=1.5,
                   label=g, zorder=3)
    # Pareto frontier (min cost, max F1)
    s = d.sort_values([x, "macro_f1"], ascending=[True, False])
    front, best = [], -1
    for _, r in s.iterrows():
        if r["macro_f1"] > best:
            front.append(r)
            best = r["macro_f1"]
    fr = pd.DataFrame(front)
    ax.step(fr[x], fr["macro_f1"], where="post", color=CONTROL, lw=1.2, ls="--", label="Pareto frontier", zorder=2)
    best = d.loc[d.groupby("group")["macro_f1"].idxmax()]
    to_label = pd.concat([fr, best]).drop_duplicates("label")
    for _, r in to_label.iterrows():
        ax.annotate(r["label"], (r[x], r["macro_f1"]), xytext=(4, -10), textcoords="offset points",
                    fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Macro-F1 (official test split)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
    fw.save(fig, name)


def routing_curves(fw: FigureWriter, curves: dict[str, pd.DataFrame], name: str = "fig5_routing_curve") -> None:
    keys = list(curves)
    ncol = min(3, len(keys))
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol + 0.6, 2.1 * nrow + 0.6), sharex=True, sharey=True,
                             squeeze=False)
    for ax, k in zip(axes.flat, keys):
        c = curves[k]
        u, r = c[c.strategy == "uncertainty"], c[c.strategy == "random"]
        ax.fill_between(r["frac"], r["lo"], r["hi"], color=CONTROL, alpha=0.3, lw=0)
        ax.plot(r["frac"], r["macro_f1"], color=CONTROL, lw=1.5, label="Random routing (95% band)")
        ax.plot(u["frac"], u["macro_f1"], color=AQUA, label="Uncertainty routing")
        ax.set_title(k, fontsize=8.5, loc="left")
    for ax in axes.flat[len(keys):]:
        ax.axis("off")
    fig.supxlabel("Fraction of conversations sent to the LLM", fontsize=9)
    fig.supylabel("Macro-F1", fontsize=9)
    axes.flat[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fw.save(fig, name)


def turn_monitor(fw: FigureWriter, tm: pd.DataFrame, name: str = "fig6_turn_level_ocdd") -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ok = tm["outlierness"].replace(-np.inf, np.nan).notna()
    fpr, tpr, _ = roc_curve(tm.loc[ok, "label"], tm.loc[ok, "outlierness"])
    from sklearn.metrics import auc
    a1.plot(fpr, tpr, color=BLUE, label=f"Turn-level OCDD (AUROC={auc(fpr, tpr):.3f})")
    a1.plot([0, 1], [0, 1], color=CONTROL, ls="--", lw=1, label="Chance")
    a1.set_xlabel("False-positive rate")
    a1.set_ylabel("True-positive rate")
    a1.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
    a1.set_title("(a) Malicious vs benign", loc="left")
    d = tm[(tm.label == 1) & tm.alarm & tm.first_pii.notna()]["delay_vs_first_pii"].astype(int)
    if len(d):
        bins = np.arange(d.min() - 0.5, d.max() + 1.5)
        a2.hist(d, bins=bins, color=AQUA, edgecolor="white", lw=2)
        a2.axvline(0, color=INK, lw=1, ls="--")
        a2.text(0.02, 0.95, f"median = {np.median(d):+.0f} msgs\nalarm ≤ first PII: {(d <= 0).mean():.0%}",
                transform=a2.transAxes, va="top", fontsize=7.5, color=INK2)
    a2.set_xlabel("Alarm message − first PII-request message")
    a2.set_ylabel("Malicious conversations")
    a2.set_title("(b) Onset timing", loc="left")
    fig.tight_layout()
    fw.save(fig, name)


def stream_traces(fw: FigureWriter, examples: dict, summary: pd.DataFrame, rho: float,
                  name: str = "fig7_stream_ocdd") -> None:
    kinds = list(examples)
    titles = {"prevalence": "Prevalence drift (malicious share ↑)", "topic": "Topic drift (scenario change, same share)",
              "none": "No drift (control)"}
    fig, axes = plt.subplots(len(kinds), 1, figsize=(6.8, 1.7 * len(kinds) + 0.6), squeeze=False)
    axes = axes[:, 0]
    for ax, k in zip(axes, kinds):
        e = examples[k]
        ax.plot(e["ratios"], color=BLUE, lw=1.4, label="Window outlier ratio")
        ax.axhline(rho, color=INK2, lw=1, ls=":", label=f"Threshold ρ={rho}")
        if e["cp"] is not None:
            ax.axvline(e["cp"], color=INK, lw=1.2, ls="--", label="True change point")
        for a in e["alarms"]:
            ax.axvline(a, color=ORANGE, lw=1.6, alpha=0.9)
        ax.plot([], [], color=ORANGE, label="OCDD alarm")
        ax.set_ylim(0, 1)
        ax.set_title(titles.get(k, k), loc="left", fontsize=8.5)
        ax.set_ylabel("Ratio")
    axes[len(kinds) - 1].set_xlabel("Stream position (conversations after reference window)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fw.save(fig, name)

    # Summary over all repeats: one measure per panel (no dual axes)
    s = summary.groupby("kind").agg(det=("detected", "mean"), delay=("delay", "median"),
                                    fa=("false_alarms_per_1000", "mean")).reindex(kinds)
    labels = [titles.get(k, k).split(" (")[0] for k in s.index]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, col, ttl in zip(axs, ["det", "delay", "fa"],
                            ["Detection rate", "Median delay (conversations)", "False alarms / 1000"]):
        vals = s[col].to_numpy(dtype=float)
        ax.bar(range(len(vals)), np.nan_to_num(vals), color=[BLUE if k != "none" else CONTROL for k in s.index],
               edgecolor="white", lw=2)
        for i, v in enumerate(vals):
            ax.text(i, np.nan_to_num(v), "n/a" if np.isnan(v) else f"{v:.2f}" if col == "det" else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.set_xticks(range(len(vals)), labels, fontsize=7.5)
        ax.set_title(ttl, loc="left", fontsize=8.5)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylim(0, 1.1)
    fig.tight_layout()
    fw.save(fig, name + "_summary")


def confusion_grid(fw: FigureWriter, items: list[tuple[str, dict]], name: str = "fig8_confusion_matrices") -> None:
    n = len(items)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.9 * ncol + 0.4, 1.9 * nrow + 0.3), squeeze=False)
    for ax, (label, m) in zip(axes.flat, items):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cm):
            frac = v / cm[i].sum()
            ax.text(j, i, str(v), ha="center", va="center", fontsize=9, color="white" if frac > 0.6 else INK)
        ax.set_xticks([0, 1], ["Benign", "Malicious"], fontsize=7)
        ax.set_yticks([0, 1], ["Benign", "Malicious"], fontsize=7)
        ax.grid(False)
        ax.set_title(label, fontsize=8, loc="left")
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.supxlabel("Predicted", fontsize=8.5)
    fig.supylabel("Human label", fontsize=8.5)
    fig.tight_layout()
    fw.save(fig, name)


def ambiguity_plot(fw: FigureWriter, df: pd.DataFrame, name: str = "fig9_accuracy_by_ambiguity") -> None:
    """df columns: system, ambiguity, accuracy, n."""
    systems = df["system"].unique().tolist()[:3]  # <=3 series keeps the categorical set CVD-safe
    amb = sorted(df["ambiguity"].dropna().unique())
    x = np.arange(len(amb))
    w = 0.8 / len(systems)
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for i, s in enumerate(systems):
        d = df[df.system == s].set_index("ambiguity").reindex(amb)
        ax.bar(x + (i - (len(systems) - 1) / 2) * w, d["accuracy"], w * 0.92, color=SERIES[i], label=s)
    ns = df.groupby("ambiguity")["n"].first().reindex(amb)
    ax.set_xticks(x, [f"{int(a)} ({['not', 'slightly', 'highly'][int(a) - 1]})\nn={int(n)}" for a, n in zip(amb, ns)])
    ax.set_xlabel("Annotator-rated ambiguity")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
    fw.save(fig, name)


def paper_workflow_outcomes(fw: FigureWriter, xt: pd.DataFrame, name: str = "fig10_paper_workflows") -> None:
    """Outcome counts (real / spamming / fraudulent) per workflow, each split into the human
    label: solid = malicious, hatched = benign. xt: long crosstab with workflow/judge columns."""
    d = xt.copy()
    d["label"] = d.apply(lambda r: f"Joint · judge {r['judge']}" if r["workflow"] == "Joint"
                         else f"Dual-LLM · {r['judge']}", axis=1)
    order = list(dict.fromkeys(d["label"]))
    colors = {"real": CONTROL, "spamming": YELLOW, "fraudulent": RED}
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(order) + 1.4))
    for i, lab in enumerate(order):
        g = d[d.label == lab].set_index("outcome")
        left = 0
        for o in ["real", "spamming", "fraudulent"]:
            if o not in g.index:
                continue
            mal, ben = int(g.loc[o].get("malicious", 0)), int(g.loc[o].get("benign", 0))
            ax.barh(i, mal, left=left, color=colors[o], edgecolor="white", lw=1)
            ax.barh(i, ben, left=left + mal, color=colors[o], alpha=0.45, hatch="///", edgecolor="white", lw=1)
            if o == "fraudulent" and mal + ben:
                ax.text(left + mal + ben + 3, i, f"{mal}/{mal + ben} mal.", va="center", fontsize=7, color=INK2)
            left += mal + ben
    ax.set_yticks(range(len(order)), order, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Test conversations (n = 360)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[o], label=o) for o in colors] + [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK2, hatch="///", label="benign (hatched)")]
    ax.legend(handles=handles, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fw.save(fig, name)


def dialogue_drift(fw: FigureWriter, det: pd.DataFrame, scores: pd.DataFrame, name: str = "fig6_dialogue_drift") -> None:
    """(a) OCDD drift message minus the annotated first PII request (malicious vs benign);
    (b) detection, localization and false-drift rates for the three reference definitions."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.1, 1]})
    d = det[det.has_drift & det.detected]
    if len(d):
        lo, hi = int(d.delay.min()), int(d.delay.max())
        bins = np.arange(lo - 0.5, hi + 1.5)
        for lab, col, nm in [(1, RED, "Malicious (fraudulent drift)"), (0, BLUE, "Benign (genuine drift)")]:
            x = d[d.label == lab].delay
            if len(x):
                a1.hist(x, bins=bins, color=col, alpha=0.75, edgecolor="white", lw=1.5, label=nm)
        a1.axvline(0, color=INK, lw=1, ls="--")
        a1.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)
    a1.set_xlabel("OCDD drift message − annotated first PII request")
    a1.set_ylabel("Conversations")
    a1.set_title("(a) Drift localization", loc="left")
    s = scores.set_index("reference")
    refs = [r for r in ["dialogue", "hybrid", "pooled"] if r in s.index]
    x = np.arange(len(refs))
    for j, (col, nm, colr) in enumerate([("localized_rate", "Localized (±2 msgs)", AQUA),
                                         ("false_drift_rate", "False drift (no-drift dialogues)", ORANGE)]):
        a2.bar(x + (j - 0.5) * 0.38, s.loc[refs, col], 0.36, color=colr, edgecolor="white", lw=1.5, label=nm)
    a2.set_xticks(x, [{"dialogue": "Within-dialogue", "hybrid": "Hybrid", "pooled": "Pooled"}[r] for r in refs],
                  fontsize=7.5)
    a2.set_ylim(0, 1.05)
    a2.set_ylabel("Rate")
    a2.grid(axis="x", visible=False)
    a2.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
    a2.set_title("(b) Reference definition", loc="left")
    fig.tight_layout()
    fw.save(fig, name)


def joint_vs_dual(fw: FigureWriter, res: pd.DataFrame, phase1: float, name: str = "fig7_joint_vs_dual",
                  title: str = "") -> None:
    """Final 'fraudulent attempt' decision (macro-F1, 95% CI) per judge LLM."""
    j = res[res.workflow == "Joint"].set_index("judge")
    dl = res[res.workflow == "Dual-LLM"].set_index("judge")
    order = j.sort_values("fraud_macro_f1").index.tolist()
    fig, ax = plt.subplots(figsize=(6.2, 0.36 * len(order) + 1.3))
    for i, k in enumerate(order):
        for df, col, off, lab in [(j, AQUA, 0.13, "Joint (ensemble → OCDD → judge)"),
                                  (dl, ORANGE, -0.13, "Dual-LLM (same LLM → OCDD → judge)")]:
            if k in df.index:
                r = df.loc[k]
                ax.plot([r.fraud_macro_f1_lo, r.fraud_macro_f1_hi], [i + off, i + off], color=col, lw=2)
                ax.plot(r.fraud_macro_f1, i + off, "o", color=col, ms=6.5, mec="white", mew=1.3,
                        label=lab if i == 0 else None)
    ax.axvline(phase1, color=CONTROL, ls="--", lw=1.2, label="Joint Phase 1 only")
    ax.set_yticks(range(len(order)), order, fontsize=8)
    ax.set_xlabel("Macro-F1 of the final 'fraudulent attempt' decision (95% bootstrap CI)")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    if title:
        ax.set_title(title, loc="left")
    fw.save(fig, name)


def drift_detectors(fw: FigureWriter, det: pd.DataFrame, scores: pd.DataFrame, name: str = "fig_dialogue_drift") -> None:
    """(a) detected drift message minus the annotated first PII request for the primary detector;
    (b) localization and false-drift rates (95% bootstrap CI of the balanced score in the labels) per detector."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.0, 1.15]})
    d = det[det.has_drift & det.detected]
    if len(d):
        lo, hi = int(d.delay.min()), int(d.delay.max())
        bins = np.arange(lo - 0.5, hi + 1.5)
        for lab, col, nm in [(1, RED, "Malicious (fraudulent drift)"), (0, BLUE, "Benign (genuine drift)")]:
            x = d[d.label == lab].delay
            if len(x):
                a1.hist(x, bins=bins, color=col, alpha=0.75, edgecolor="white", lw=1.5, label=nm)
        a1.axvline(0, color=INK, lw=1, ls="--")
        a1.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)
    a1.set_xlabel("Detected drift message − annotated first PII request")
    a1.set_ylabel("Conversations")
    a1.set_title("(a) Drift localization", loc="left")
    x = np.arange(len(scores))
    for j, (col, nm, colr) in enumerate([("localized_rate", "Localized (0 to +2 msgs)", AQUA),
                                         ("false_drift_rate", "False drift (no-drift dialogues)", ORANGE)]):
        a2.bar(x + (j - 0.5) * 0.38, scores[col], 0.36, color=colr, edgecolor="white", lw=1.5, label=nm)
    a2.set_xticks(x, [f"{r.detector}\nbal. {r.balanced:.2f}" for r in scores.itertuples()], fontsize=7)
    a2.set_ylim(0, 1.05)
    a2.set_ylabel("Rate")
    a2.grid(axis="x", visible=False)
    a2.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)
    a2.set_title("(b) Detectors", loc="left")
    fig.tight_layout()
    fw.save(fig, name)


def joint_vs_dual_panels(fw: FigureWriter, panels: list[tuple[str, pd.DataFrame, float]],
                         name: str = "fig_joint_vs_dual") -> None:
    """One panel per protocol: final 'fraudulent attempt' macro-F1 (95% CI) per judge, Joint vs Dual-LLM.
    panels: (title, results with workflow/judge/fraud_macro_f1[_lo|_hi], Joint Phase-1 macro-F1)."""
    n = max(len(r[r.workflow == "Joint"]) for _, r, _ in panels)
    ratios = [len(r[r.workflow == "Joint"]) + 1.2 for _, r, _ in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(6.4, 0.32 * sum(ratios) + 1.3), sharex=True,
                             gridspec_kw={"height_ratios": ratios}, squeeze=False)
    labels = {"Joint": "Joint (ensemble → OCDD → judge)", "Dual-LLM": "Dual-LLM (same LLM → OCDD → judge)",
              "Joint (no OCDD)": "Ensemble → judge (no OCDD)"}
    for ax, (ttl, res, p1) in zip(axes[:, 0], panels):
        j = res[res.workflow == "Joint"].set_index("judge")
        dl = res[res.workflow == "Dual-LLM"].set_index("judge")
        no = res[res.workflow == "Joint (no OCDD)"].set_index("judge")
        order = j.sort_values("fraud_macro_f1").index.tolist()
        for i, k in enumerate(order):
            series = [(j, AQUA, 0.22, "Joint"), (no, VIOLET, 0.0, "Joint (no OCDD)"), (dl, ORANGE, -0.22, "Dual-LLM")]
            for df, col, off, wf in series:
                if k in df.index:
                    r = df.loc[k]
                    ax.plot([r.fraud_macro_f1_lo, r.fraud_macro_f1_hi], [i + off, i + off], color=col, lw=2)
                    ax.plot(r.fraud_macro_f1, i + off, "o", color=col, ms=6, mec="white", mew=1.2,
                            label=labels[wf] if i == 0 else None)
        ax.axvline(p1, color=CONTROL, ls="--", lw=1.2, label="Joint Phase 1 only")
        ax.set_yticks(range(len(order)), order, fontsize=8)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.grid(axis="y", visible=False)
        ax.set_title(ttl, loc="left", fontsize=9)
    axes[-1, 0].set_xlabel("Macro-F1 of the final 'fraudulent attempt' decision (95% bootstrap CI)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.09 * 9 / max(n, 9), 1, 1))
    fig.legend(h, lab, fontsize=7.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fw.save(fig, name)
