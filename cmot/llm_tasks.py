"""LLM prompts (versioned), parsing and batched execution.

Two tasks:
  1. Conversation classification (Phase-1 comparator / routed decision).
  2. Drift-conditioned judgment (Phase 2): the model is told at which turn the
     OCDD monitor raised an alarm and must decide whether the conversation is a
     social-engineering attempt or a benign shift. Unlike the original
     template, the output is a binary decision that is scored against the
     human label, so the stage is evaluated rather than only described.

Only the raw transcript (speaker names + messages) is given to the model; no
annotation fields are ever included.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .data import Conversation
from .llm_client import BudgetExceeded, CreateAIClient, ModelSpec
from .utils import LOG

PROMPT_VERSION = "cmot-v2.0"

SYSTEM_CLASSIFY = (
    "You are a security analyst who audits professional online conversations (e.g. LinkedIn-style "
    "outreach about research collaboration, funding, journalism or recruitment) for social-engineering "
    "attacks. A conversation is MALICIOUS if the initiating party tries to obtain sensitive personal, "
    "financial or organisational information, money, credentials or access under a pretext, or "
    "manipulates the other party (e.g. flattery, urgency, authority) toward such disclosure. A conversation "
    "is BENIGN if any information requested is proportionate and plausible for the stated legitimate "
    "purpose. Judge intent from the whole conversation."
)

CLASSIFY_TEMPLATE = (
    "{fewshot}Classify the conversation below.\n"
    "Return ONLY a JSON object with keys:\n"
    '  "label": "malicious" or "benign",\n'
    '  "confidence": number in [0,1] (probability that your label is correct),\n'
    '  "evidence_turns": list of message indices that support the decision,\n'
    '  "rationale": at most 40 words.\n\n'
    "<conversation>\n{transcript}\n</conversation>"
)

SYSTEM_JUDGE = (
    "You are a security analyst. An automated distribution-shift monitor (one-class drift detector) "
    "has raised an alarm on an online conversation that a first-stage classifier marked as suspicious. "
    "An alarm only means the conversation's content departed from previously seen benign behaviour; "
    "it is NOT evidence of fraud by itself. Natural topic changes are common."
)

JUDGE_TEMPLATE = (
    "The monitor first raised the alarm at message index {alarm_turn}.\n"
    "Decide whether the conversation is a social-engineering / fraudulent attempt or a benign "
    "shift in an otherwise legitimate conversation.\n"
    "Return ONLY a JSON object with keys:\n"
    '  "verdict": "fraudulent" or "benign_shift",\n'
    '  "confidence": number in [0,1],\n'
    '  "rationale": at most 40 words.\n\n'
    "<conversation>\n{transcript}\n</conversation>"
)


def build_fewshot_block(examples: list[Conversation], max_chars: int = 3000) -> str:
    if not examples:
        return ""
    parts = ["Labelled examples (from the training split only):\n"]
    for i, ex in enumerate(examples, 1):
        lab = "malicious" if ex.label else "benign"
        parts.append(f"--- Example {i} (label: {lab}) ---\n{ex.transcript(max_chars)}\n")
    parts.append("--- End of examples ---\n\n")
    return "\n".join(parts)


def classify_messages(conv: Conversation, fewshot: list[Conversation] | None = None,
                      max_chars: int = 20000) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_CLASSIFY},
        {"role": "user", "content": CLASSIFY_TEMPLATE.format(
            fewshot=build_fewshot_block(fewshot or []), transcript=conv.transcript(max_chars))},
    ]


def judge_messages(conv: Conversation, alarm_turn: int, max_chars: int = 20000) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_JUDGE},
        {"role": "user", "content": JUDGE_TEMPLATE.format(alarm_turn=alarm_turn, transcript=conv.transcript(max_chars))},
    ]


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    for candidate in (text, *(m.group(0) for m in [_JSON_RE.search(text)] if m)):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _conf(obj: dict) -> float:
    try:
        return float(min(max(float(obj.get("confidence", 0.75)), 0.0), 1.0))
    except (TypeError, ValueError):
        return 0.75


def parse_classification(text: str) -> tuple[int | None, float]:
    """Returns (label, p_malicious). label None = unparseable output."""
    obj = _extract_json(text)
    lab_raw = str(obj.get("label", "")).lower() if obj else (text or "").lower()
    if "malicious" in lab_raw and "benign" not in lab_raw:
        label = 1
    elif "benign" in lab_raw and "malicious" not in lab_raw:
        label = 0
    else:
        return None, float("nan")
    c = _conf(obj) if obj else 0.75
    c = max(c, 0.5)  # confidence refers to the emitted label
    return label, (c if label == 1 else 1 - c)


def parse_judgment(text: str) -> tuple[int | None, float]:
    obj = _extract_json(text)
    raw = str(obj.get("verdict", "")).lower() if obj else (text or "").lower()
    if "fraud" in raw:
        label = 1
    elif "benign" in raw:
        label = 0
    else:
        return None, float("nan")
    c = max(_conf(obj) if obj else 0.75, 0.5)
    return label, (c if label == 1 else 1 - c)


TRUNCATION_RETRY_FACTOR = 4
TRUNCATION_RETRY_CAP = 32000


def _retry_truncated(client: CreateAIClient, jobs: list[dict], rows: list[dict], parse, max_workers: int,
                     desc: str) -> list[dict]:
    """Protocol rule applied identically to every model: an output cut off by the token limit
    is re-requested once with a 4x larger budget, whether or not the cut-off text happens to
    parse (a truncated answer may contain the label word in its reasoning, not as a verdict).
    Truncation = finish_reason 'length' OR output_tokens >= max_tokens (the CreateAI gateway
    reports 'stop' for some providers even when the cap was hit). The deployment cost/latency
    of the conversation includes both attempts."""
    def truncated(r: dict) -> bool:
        cap = r.get("max_tokens") or 0
        return bool(r.get("ok")) and not r.get("retried_truncated") and (
            r.get("finish_reason") == "length" or (cap and int(r.get("output_tokens") or 0) >= cap))
    idx = [i for i, r in enumerate(rows) if truncated(r)]
    if not idx:
        return rows
    LOG.info("  %s: %d truncated outputs -> one retry with %dx token budget", desc, len(idx), TRUNCATION_RETRY_FACTOR)
    by_key = {(j["spec"].key, j["conv_id"], j["repeat"], j["tag"]): j for j in jobs}
    retry_jobs = []
    for i in idx:
        r = rows[i]
        j = dict(by_key[(r["model_key"], r["conv_id"], r["repeat"], r["tag"])])
        j["max_tokens"] = min(TRUNCATION_RETRY_CAP, (j.get("max_tokens") or j["spec"].max_tokens) * TRUNCATION_RETRY_FACTOR)
        retry_jobs.append(j)
    retried = {(r["model_key"], r["conv_id"], r["repeat"]): r for r in _run_jobs(client, retry_jobs, max_workers,
                                                                              f"{desc}/retry")}
    for i in idx:
        old = rows[i]
        new = retried[(old["model_key"], old["conv_id"], old["repeat"])]
        rows[i] = {**new, "retried_truncated": True,
                   "cost_usd": float(old.get("cost_usd") or 0) + float(new.get("cost_usd") or 0),
                   "latency_s": float(old.get("latency_s") or 0) + float(new.get("latency_s") or 0),
                   "output_tokens": int(old.get("output_tokens") or 0) + int(new.get("output_tokens") or 0)}
    return rows


def _run_jobs(client: CreateAIClient, jobs: list[dict], max_workers: int, desc: str) -> list[dict]:
    """jobs: dicts with spec, messages, temperature, tag, repeat + metadata."""
    out: list[dict] = []
    n_done, n_failed = 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(client.chat, j["spec"], j["messages"], j["temperature"],
                            j.get("max_tokens"), j["tag"], j["repeat"]): j for j in jobs}
        try:
            for fut in as_completed(futs):
                j = futs[fut]
                res = fut.result()
                meta = {k: v for k, v in j.items() if k not in ("spec", "messages")}
                out.append({**res, **meta, "model_key": j["spec"].key})  # job metadata wins over cached fields
                n_done += 1
                n_failed += 0 if res.get("ok") else 1
                if n_done >= 40 and n_failed / n_done > 0.5:
                    raise RuntimeError(f"{desc}: {n_failed}/{n_done} API calls failed (last error: "
                                       f"{res.get('error')}). Stopping; completed calls are cached.")
                if n_done % 50 == 0 or n_done == len(jobs):
                    LOG.info("  %s: %d/%d calls (spent this run: $%.2f)", desc, n_done, len(jobs), client.spent_usd)
        except (BudgetExceeded, RuntimeError):
            for f in futs:
                f.cancel()
            raise
    return out


def run_classification(client: CreateAIClient, specs: list[ModelSpec], convs: list[Conversation],
                       settings: dict, fewshot: list[Conversation] | None = None, tag: str = "zeroshot") -> pd.DataFrame:
    jobs = []
    for spec in specs:
        for conv in convs:
            msgs = classify_messages(conv, fewshot, settings.get("max_chars", 20000))
            for r in range(settings.get("repeats", 1)):
                jobs.append({"spec": spec, "messages": msgs, "temperature": settings.get("temperature", 0.0),
                             "max_tokens": spec.max_tokens, "tag": tag, "repeat": r, "conv_id": conv.conv_id})
    LOG.info("LLM classification [%s]: %d calls over %d models", tag, len(jobs), len(specs))
    rows = _run_jobs(client, jobs, settings.get("max_workers", 4), f"classify/{tag}")
    rows = _retry_truncated(client, jobs, rows, parse_classification, settings.get("max_workers", 4), f"classify/{tag}")
    df = pd.DataFrame(rows)
    parsed = df["text"].fillna("").map(parse_classification)
    df["pred"] = [p[0] for p in parsed]
    df["p_malicious"] = [p[1] for p in parsed]
    df["valid"] = df["pred"].notna() & df["ok"].astype(bool)
    return df


def aggregate_repeats(df: pd.DataFrame, invalid_policy: str = "benign") -> pd.DataFrame:
    """Majority vote over repeats; per-conversation latency/cost summed over repeats=0 only
    (cost of a single deployment call). Invalid outputs are mapped by `invalid_policy`."""
    def agg(g: pd.DataFrame) -> pd.Series:
        valid = g[g["valid"]]
        if len(valid):
            p = float(valid["p_malicious"].mean())
            pred = int(valid["pred"].mean() >= 0.5)
        else:
            p, pred = float("nan"), (1 if invalid_policy == "malicious" else 0)
        first = g[g["repeat"] == 0].iloc[0] if (g["repeat"] == 0).any() else g.iloc[0]
        return pd.Series({
            "pred": pred, "p_malicious": p, "n_valid": int(len(valid)), "n_calls": int(len(g)),
            "agreement": float((valid["pred"] == pred).mean()) if len(valid) else float("nan"),
            "latency_s": float(first["latency_s"]), "cost_usd": float(first["cost_usd"]),
            "input_tokens": int(first["input_tokens"]), "output_tokens": int(first["output_tokens"]),
            "model_returned": first.get("model_returned"),
        })
    return df.groupby(["model_key", "conv_id"]).apply(agg).reset_index()


def run_judgments(client: CreateAIClient, specs: list[ModelSpec], items: list[tuple[Conversation, int]],
                  settings: dict) -> pd.DataFrame:
    jobs = []
    for spec in specs:
        for conv, alarm_turn in items:
            jobs.append({"spec": spec, "messages": judge_messages(conv, alarm_turn, settings.get("max_chars", 20000)),
                         "temperature": settings.get("temperature", 0.0), "max_tokens": spec.max_tokens,
                         "tag": "judge", "repeat": 0, "conv_id": conv.conv_id, "alarm_turn": alarm_turn})
    if not jobs:
        return pd.DataFrame(columns=["model_key", "conv_id", "judge_pred", "judge_p", "latency_s", "cost_usd"])
    LOG.info("LLM drift judgment: %d calls", len(jobs))
    rows = _run_jobs(client, jobs, settings.get("max_workers", 4), "judge")
    rows = _retry_truncated(client, jobs, rows, parse_judgment, settings.get("max_workers", 4), "judge")
    df = pd.DataFrame(rows)
    parsed = df["text"].fillna("").map(parse_judgment)
    df["judge_pred"] = [p[0] for p in parsed]
    df["judge_p"] = [p[1] for p in parsed]
    return df


def select_fewshot(train: list[Conversation], k: int, seed: int) -> list[Conversation]:
    """Class-balanced few-shot examples drawn from the training split only."""
    rng = np.random.default_rng(seed)
    pos = [c for c in train if c.label == 1]
    neg = [c for c in train if c.label == 0]
    k_pos = k // 2 + k % 2
    ip = rng.choice(len(pos), min(k_pos, len(pos)), replace=False)
    ineg = rng.choice(len(neg), min(k - k_pos, len(neg)), replace=False)
    chosen = [pos[i] for i in ip] + [neg[i] for i in ineg]
    order = rng.permutation(len(chosen))
    return [chosen[i] for i in order]


# ---------------------------------------------------------------------------
# Drift-judgment prompt of the framework.
# Used by the paper-faithful Joint and Dual-LLM workflows.
PAPER_JUDGE_SYSTEM = "You are an AI expert in cybersecurity and fraud detection."
PAPER_JUDGE_USER = (
    "Analyze the following conversation JSON and determine if the detected concept drift is a genuine shift "
    "in behavior or a fraudulent attempt: {conversation_json}. If the conversation is a Fraudulent Attempt, "
    "return 'Fraudulent Attempt' and specify the type of conversation as 'fraudulent'. Otherwise, return "
    "'Genuine Concept Drift' and specify the type of conversation as 'genuine'."
)


LOCATION_NOTE = (" The drift detector located the start of the concept drift at message {k} of the list (counting "
                 "from 0), sent by {speaker}: \"{excerpt}\".")


def paper_judge_messages(conv: Conversation, location: int | None = None, upto: int | None = None) -> list[dict]:
    """Original judge prompt. Variants (experiments only): `location` appends the detected drift message;
    `upto` truncates the dialogue after that message (decision at alarm time, early warning)."""
    n = len(conv.messages) if upto is None else min(upto + 1, len(conv.messages))
    payload = json.dumps([{"Name": s, "Message": m} for s, m in zip(conv.speakers[:n], conv.messages[:n])],
                         ensure_ascii=False)
    user = PAPER_JUDGE_USER.format(conversation_json=payload)
    if location is not None and location < n:
        excerpt = conv.messages[location][:300] + ("..." if len(conv.messages[location]) > 300 else "")
        user += LOCATION_NOTE.format(k=location, speaker=conv.speakers[location], excerpt=excerpt)
    return [{"role": "system", "content": PAPER_JUDGE_SYSTEM}, {"role": "user", "content": user}]


def parse_paper_judgment(text: str) -> tuple[int | None, float]:
    """1 = 'Fraudulent Attempt' / 'fraudulent', 0 = 'Genuine Concept Drift' / 'genuine'.
    If both appear, the last type keyword wins (the prompt asks for the type at the end)."""
    t = (text or "").lower()
    hits = [(m.start(), 1) for m in re.finditer(r"fraudulent", t)] + \
           [(m.start(), 0) for m in re.finditer(r"genuine", t)]
    if not hits:
        return None, float("nan")
    label = max(hits)[1]
    return label, float(label)


def run_paper_judgments(client: CreateAIClient, pairs: list[tuple[ModelSpec, Conversation]],
                        settings: dict, tag: str = "paper_judge",
                        variant: dict[int, dict] | None = None) -> pd.DataFrame:
    """pairs: (judge model, conversation) -- each workflow decides which model judges what.
    variant: optional conv_id -> {'location': k, 'upto': t} for the judge-input experiments."""
    variant = variant or {}
    jobs = [{"spec": spec, "messages": paper_judge_messages(conv, **variant.get(conv.conv_id, {})),
             "temperature": settings.get("temperature", 0.0),
             "max_tokens": spec.max_tokens, "tag": tag, "repeat": 0, "conv_id": conv.conv_id}
            for spec, conv in pairs]
    cols = ["model_key", "conv_id", "judge_pred", "latency_s", "cost_usd", "text"]
    if not jobs:
        return pd.DataFrame(columns=cols)
    LOG.info("Paper-workflow LLM judgments: %d calls", len(jobs))
    rows = _run_jobs(client, jobs, settings.get("max_workers", 4), tag)
    rows = _retry_truncated(client, jobs, rows, parse_paper_judgment, settings.get("max_workers", 4), tag)
    df = pd.DataFrame(rows)
    df["judge_pred"] = [parse_paper_judgment(t)[0] for t in df["text"].fillna("")]
    return df
