#!/usr/bin/env python3
"""Live progress dashboard for a running `run_all.py` (read-only; safe to start/stop anytime).

    python monitor.py            # refresh every 5 s, Ctrl+C to quit
    python monitor.py --once     # print one snapshot
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
STAGES = ["Embedding", "LLM classification [zeroshot]", "LLM classification [fewshot]", "Protocol A",
          "Drift E1", "Paper workflows", "Protocol B", "Drift E2", "Protocol C", "LLM classification [external]",
          "External judge", "Figures", "Done."]


def read_calls(path: Path, offset: int, acc: dict) -> int:
    """Incrementally parse new lines of the JSONL request log."""
    if not path.exists():
        return offset
    with path.open("r", encoding="utf-8") as fh:
        fh.seek(offset)
        for line in fh:
            if not line.endswith("\n"):
                break  # partially written line; read it next time
            offset += len(line.encode("utf-8"))
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            acc["count"][(rec.get("route"), rec.get("tag"))] += 1
            acc["cost"] += float(rec.get("cost_usd") or 0)
            acc["times"].append(float(rec.get("timestamp") or 0))
            acc["lat"][rec.get("route")].append(float(rec.get("latency_s") or 0))
    return offset


def latest_run() -> Path | None:
    runs = sorted((p for p in (ROOT / "results").glob("2*") if p.is_dir() and not p.name.endswith("_MOCK")),
                  key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def bar(done: int, total: int, width: int = 24) -> str:
    frac = 0 if total <= 0 else min(1.0, done / total)
    fill = int(round(frac * width))
    return "#" * fill + "." * (width - fill) + f" {done:>5}/{total:<5}"


def render(cfg: dict, acc: dict, run: Path | None, t0: float) -> str:
    models = cfg["llms"]
    n_conv, n_test = 400, 360
    few = cfg["llm_settings"].get("fewshot", {}).get("enabled", False)
    lines = []
    log_lines = (run / "run.log").read_text(encoding="utf-8", errors="replace").splitlines() if run else []
    stage, last, fold = "waiting for run", "", ""
    for ln in log_lines:
        for s in STAGES:
            if s in ln:
                stage = s
        m = re.search(r"fold (\d+)/(\d+) done", ln)
        if m:
            fold = f"{m.group(1)}/{m.group(2)}"
    if log_lines:
        last = log_lines[-1][:150]
    now = time.time()
    recent = [t for t in acc["times"] if now - t <= 120]
    rate = len(recent) / 2.0  # calls per minute over the last 2 min

    lines.append(f"CMOT live monitor   {time.strftime('%H:%M:%S')}   run: {run.name if run else '-'}")
    lines.append("=" * 100)
    lines.append(f"Stage: {stage}" + (f"   |  CV fold {fold}" if fold else ""))
    lines.append(f"Cached calls: {sum(acc['count'].values())}   Spend (cache total): ${acc['cost']:.2f}"
                 f"   Rate: {rate:.0f} calls/min")
    lines.append("")
    lines.append(f"{'Model':<20}{'zero-shot':<34}{'few-shot':<34}{'EXTERNAL (1000)':<34}{'judge':>7}  avg lat")
    remaining = 0
    for m in models:
        route = f"{m['provider']}/{m['model']}"
        zs = acc["count"][(route, "zeroshot")]
        fs = acc["count"][(route, "fewshot")]
        jd = acc["count"][(route, "judge")] + acc["count"][(route, "paper_judge")] + acc["count"][(route, "external_judge")]
        ex = acc["count"][(route, "external")]
        lat = acc["lat"][route]
        remaining += max(0, n_conv - zs) + (max(0, n_test - fs) if few else 0) + max(0, 1000 - ex)
        lines.append(f"{m['key'][:19]:<20}{bar(zs, n_conv, 18):<34}{(bar(fs, n_test, 18) if few else '-'):<34}"
                     f"{bar(ex, 1000, 18):<34}{jd:>7}"
                     f"  {(sum(lat) / len(lat)) if lat else 0:5.1f}s")
    lines.append("")
    if rate > 0 and remaining:
        lines.append(f"Classification calls remaining: {remaining}  ->  ETA ~{remaining / rate:.0f} min at current rate"
                     " (judge calls and CV come after)")
    lines.append(f"Last log line: {last}")
    lines.append("")
    lines.append("Ctrl+C closes this monitor only; the run keeps going.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    acc = {"count": Counter(), "cost": 0.0, "times": [], "lat": defaultdict(list)}
    offset, t0 = 0, time.time()
    cache = ROOT / cfg.get("cache_dir", "cache") / "llm_calls.jsonl"
    os.system("")  # enable ANSI escape codes on Windows consoles
    try:
        while True:
            offset = read_calls(cache, offset, acc)
            screen = render(cfg, acc, latest_run(), t0)
            if args.once:
                print(screen)
                return
            print("\033[2J\033[H" + screen, flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
