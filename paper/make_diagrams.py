#!/usr/bin/env python3
"""Vector drawings of the framework and model-selection workflows.
Writes figures/fig_framework.{pdf,png} and figures/fig_selection.{pdf,png}.

    python paper/make_diagrams.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

FIG = Path(__file__).resolve().parent / "figures"
INK, MUTED = "#1f2937", "#4b5563"
# (fill, edge) per role
STYLE = {"data": ("#e8eef6", "#7b8fa8"), "proc": ("#d6e6f8", "#4f86c6"), "model": ("#d5efdf", "#3f9a64"),
         "dec": ("#ffeab0", "#d19a1a"), "llm": ("#e6dcf6", "#7e5bb5"), "alarm": ("#f8cdc8", "#c8473d"),
         "ok": ("#dcf2e2", "#4aa36a"), "out": ("#eef0f3", "#8a94a3"), "prompt": ("#fdf3dc", "#c9a14a")}
FS = 6.5
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS, "pdf.fonttype": 42})


def box(ax, x, y, w, h, text, kind="proc", bold=False, size=FS, dashed=False):
    fc, ec = STYLE[kind]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc=fc,
                                ec=ec, lw=1.0, ls="--" if dashed else "-", zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=INK, weight="bold" if bold else "normal",
            linespacing=1.3, zorder=3)


def diamond(ax, x, y, w, h, text):
    fc, ec = STYLE["dec"]
    ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)], closed=True, fc=fc, ec=ec,
                         lw=1.0, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=FS, color=INK, zorder=3)


def arrow(ax, p, q, label=None, at=None, ha="center", color=MUTED, conn="arc3"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=1.0, color=color, shrinkA=0,
                                 shrinkB=0, connectionstyle=conn, zorder=1))
    if label:
        ax.text(*at, label, ha=ha, va="center", fontsize=FS - 0.5, color=MUTED, zorder=3)


def person(ax, x, y, filled, label):
    ax.add_patch(Circle((x, y + 6.2), 1.4, fc=INK if filled else "white", ec=INK, lw=0.9, zorder=3))
    ax.plot([x, x], [y + 4.8, y + 1.6], color=INK, lw=0.9, zorder=3)
    ax.plot([x - 2, x, x + 2], [y + 3.8, y + 4.4, y + 3.8], color=INK, lw=0.9, zorder=3)
    ax.plot([x - 1.8, x, x + 1.8], [y - 1.2, y + 1.6, y - 1.2], color=INK, lw=0.9, zorder=3)
    ax.text(x, y - 3.2, label, ha="center", va="center", fontsize=FS, color=INK)


def bubble(ax, x, y, w, h, side):
    fc, ec = ("#d3efc4", "#6fb35a") if side == "left" else ("#f7b8b2", "#cf4f45")
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=fc, ec=ec,
                                lw=0.9, zorder=2))
    tx = x - w / 2 + 2 if side == "left" else x + w / 2 - 2
    tail = [(tx, y - h / 2), (tx + (1.6 if side == "left" else -1.6), y - h / 2), (tx - (0.6 if side == "left" else -0.6),
                                                                                    y - h / 2 - 2.2)]
    ax.add_patch(Polygon(tail, closed=True, fc=fc, ec=ec, lw=0.9, zorder=2))
    ax.add_patch(Rectangle((tail[0][0] - 0.3 if side == "left" else tail[1][0], y - h / 2 - 0.2), 2.2, 0.6, fc=fc, ec=fc,
                           lw=0, zorder=2))
    ax.text(x, y, "· · · ·", ha="center", va="center", fontsize=FS, color=INK, zorder=3)


def canvas(w_in, h_in, xmax, ymax):
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def save(fig, name):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 220})):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.03, **kw)
    plt.close(fig)


def framework():
    fig, ax = canvas(7.4, 4.69, 183, 116)
    # ---- online conversations (top band)
    xc = 84
    ax.plot([xc - 13, xc - 13], [84, 112], color="#c3c9d2", lw=0.8)
    ax.plot([xc + 13, xc + 13], [84, 112], color="#c3c9d2", lw=0.8)
    person(ax, xc - 13, 86, False, "Target")
    person(ax, xc + 13, 86, True, "Attacker")
    bubble(ax, xc + 4.5, 108.5, 13, 4.2, "right")
    bubble(ax, xc - 4.5, 103.5, 13, 4.2, "left")
    bubble(ax, xc + 4.5, 98.5, 13, 4.2, "right")
    bubble(ax, xc - 4.5, 93.5, 13, 4.2, "left")
    ax.text(xc - 22, 99, "· · ·", ha="center", va="center", fontsize=FS + 1, color=MUTED)
    ax.text(xc + 22, 99, "· · ·", ha="center", va="center", fontsize=FS + 1, color=MUTED)
    # stream of conversations
    yb = 73
    x0, x1 = 26, 142
    ax.plot([x0, x1], [yb + 3.2, yb + 3.2], color=INK, lw=0.9)
    ax.plot([x0, x1], [yb - 3.2, yb - 3.2], color=INK, lw=0.9)
    ax.plot([x0, xc - 22.5], [yb, yb], color=MUTED, lw=0.9, ls=(0, (1, 2)))
    ax.plot([xc + 22.5, x1], [yb, yb], color=MUTED, lw=0.9, ls=(0, (1, 2)))
    cells = ["$C_{n+2}$", "$C_{n+1}$", "$C_n$", "$C_{n-1}$", "$C_{n-2}$"]
    for i, lab in enumerate(cells):
        cx = xc + (i - 2) * 9
        fc, ec = STYLE["proc"] if i == 2 else ("white", "#7b8fa8")
        ax.add_patch(Rectangle((cx - 4.5, yb - 3.2), 9, 6.4, fc=fc, ec=ec, lw=0.9, zorder=2))
        ax.text(cx, yb, lab, ha="center", va="center", fontsize=FS, color=INK, zorder=3)
    ax.text(x0 - 3, yb, "New", ha="right", va="center", fontsize=FS, color=MUTED)
    ax.text(x1 + 3, yb, "Old", ha="left", va="center", fontsize=FS, color=MUTED)
    ax.text(x0, yb + 5.6, "Online conversations", ha="left", va="center", fontsize=FS, color=INK, weight="bold")
    ax.text(x1, yb + 5.6, "each $C_i$ = one complete dialogue", ha="right", va="center", fontsize=FS, color=MUTED)
    arrow(ax, (xc + 13, 82.5), (xc + 1.5, yb + 3.4), color="#cf4f45", conn="arc3,rad=-0.15")
    # ---- panel frames
    ax.plot([104, 104], [8, 64], color="#c3c9d2", lw=0.8, ls=(0, (3, 3)))
    ax.text(4, 64, "(a) Phase 1", ha="left", va="center", fontsize=FS + 1, weight="bold", color=INK)
    ax.text(108, 64, "(b) Phase 2", ha="left", va="center", fontsize=FS + 1, weight="bold", color=INK)
    # ---- phase 1
    box(ax, xc, 58, 26, 7, "TF-IDF + SVD", "proc")
    arrow(ax, (xc, yb - 3.4), (xc, 61.5))
    box(ax, xc, 39, 36, 14, "Ensemble model\n3 best: SVM · RF · kNN\n(majority vote)", "model")
    arrow(ax, (xc, 54.5), (xc, 46))
    box(ax, 10, 39, 18, 11, "Labelled\ndialogues", "data")
    box(ax, 30, 39, 15, 9, "TF-IDF\n+ SVD", "proc")
    box(ax, 52, 39, 22, 13, "Tune 9\ncandidates\n(inner CV)", "model")
    box(ax, 52, 56, 24, 6.5, "Parameter space", "out", size=FS - 0.5)
    arrow(ax, (19, 39), (22.5, 39))
    arrow(ax, (37.5, 39), (41, 39))
    arrow(ax, (52, 52.75), (52, 45.5))
    arrow(ax, (63, 39), (66, 39))
    diamond(ax, xc, 17, 24, 13, "Malicious?")
    arrow(ax, (xc, 32), (xc, 23.5))
    box(ax, 46, 17, 32, 10, "Benign\n(leaves the pipeline)", "ok")
    arrow(ax, (xc - 12, 17), (62, 17), "no", at=(67, 19.4))
    # ---- phase 2
    box(ax, 122, 17, 30, 14, "OCDD\n(normal-message\nreference)", "model")
    arrow(ax, (xc + 12, 17), (107, 17), "yes", at=(100, 19.4))
    diamond(ax, 122, 39, 18, 12, "Drift?")
    arrow(ax, (122, 24), (122, 33))
    box(ax, 122, 56, 22, 7.5, "Spam-like", "out")
    arrow(ax, (122, 45), (122, 52.25), "no", at=(123.8, 48.6), ha="left")
    box(ax, 146, 39, 20, 11, "LLM judge", "llm", bold=True)
    box(ax, 146, 56, 18, 7.5, "Prompt", "prompt")
    arrow(ax, (146, 52.25), (146, 44.5))
    arrow(ax, (131, 39), (136, 39), "yes", at=(133.5, 41.8))
    diamond(ax, 169, 39, 22, 14, "Fraudulent?")
    arrow(ax, (156, 39), (158, 39))
    box(ax, 169, 17, 22, 11, "Genuine drift\n(spamming)", "out")
    box(ax, 169, 58, 22, 11, "Fraudulent\nattempt", "alarm", bold=True)
    arrow(ax, (169, 46), (169, 52.5), "yes", at=(170.8, 49.3), ha="left")
    arrow(ax, (169, 32), (169, 22.5), "no", at=(170.8, 27.3), ha="left")
    box(ax, 90, 3.5, 168, 5.6,
        "Dual-LLM framework: an LLM replaces the ensemble in (a), and the same LLM acts as the judge in (b)",
        "out", size=FS - 0.5, dashed=True)
    save(fig, "fig_framework")


def selection():
    fig, ax = canvas(7.4, 2.9, 160, 62)
    box(ax, 9, 30, 15, 10, "Training\ndialogues", "data")
    box(ax, 27, 30, 15, 10, "TF-IDF\n+ SVD", "proc")
    box(ax, 57, 52, 62, 14, "Candidates\nsupervised: SVM, RF, kNN, NB, XGBoost\nclustering: k-means, ImpKmeans, DBSCAN, GMM",
        "model", size=FS - 0.3)
    box(ax, 57, 30, 28, 11, "Model training\n(5-fold inner CV)", "model", bold=True)
    box(ax, 57, 9, 24, 7, "Parameter space", "out", size=FS - 0.5)
    box(ax, 98, 30, 30, 11, "Rank by inner-CV\nmacro-F1; keep 3 best", "proc")
    box(ax, 127, 30, 22, 11, "Majority-vote\nensemble", "model", bold=True)
    box(ax, 127, 52, 24, 10, "Test dialogues\n(TF-IDF + SVD)", "data")
    diamond(ax, 150, 30, 19, 14, "Malicious?")
    box(ax, 150, 52, 17, 7, "Malicious", "alarm")
    box(ax, 150, 8, 17, 7, "Benign", "ok")
    arrow(ax, (16.5, 30), (19.5, 30))
    arrow(ax, (34.5, 30), (43, 30))
    arrow(ax, (57, 45), (57, 35.5))
    arrow(ax, (57, 12.5), (57, 24.5))
    arrow(ax, (71, 30), (83, 30), "best\nparams", at=(77, 34.5))
    arrow(ax, (113, 30), (116, 30))
    arrow(ax, (127, 47), (127, 35.5))
    arrow(ax, (138, 30), (140.5, 30))
    arrow(ax, (150, 37), (150, 48.5), "yes", at=(151.8, 42.5), ha="left")
    arrow(ax, (150, 23), (150, 11.5), "no", at=(151.8, 17), ha="left")
    save(fig, "fig_selection")


if __name__ == "__main__":
    framework()
    selection()
    print("diagrams written to", FIG)
