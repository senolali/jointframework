"""SEConvo loading (Ai et al., 2024; Zenodo DOI 10.5281/zenodo.12170260, CC BY 4.0).

Only the 400 human-annotated conversations are used. The official release split
(40 train / 360 test) is used as the primary protocol. Annotation-only fields (FirstPII,
NumPIIMessages, PII, Ambiguity) are kept for *evaluation and stratification
only* and are never fed to any model as input features.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .utils import LOG, md5_file

ZENODO_URL = "https://zenodo.org/records/12170260/files/{name}?download=1"
FILES = {
    "annotated_train.json": "554f9698257934f415a09f09698e2905",
    "annotated_test.json": "9a40f6287bd0a49c8d17693526127ebe",
}
EXTERNAL_FILES = {
    "unannotated_train.json": "9ef599ab028186a44da441364ce52acb",
    "unannotated_test.json": "5cf3c2215ea00281ce574b46d4377191",
}


@dataclass
class Conversation:
    conv_id: int
    split: str  # "train" | "test" (official SEConvo release split)
    speakers: list[str]
    messages: list[str]
    label: int  # 1 = malicious (social-engineering attempt), 0 = benign
    scenario: str
    multi_agent: bool
    ambiguity: int | None = None
    first_pii: int | None = None  # message index of first PII request (annotation only)
    num_pii: int | None = None
    pii_types: list[str] = field(default_factory=list)

    @property
    def initiator(self) -> str:
        return self.speakers[0] if self.speakers else ""

    def initiator_turns(self) -> list[int]:
        return [i for i, s in enumerate(self.speakers) if s == self.initiator]

    def transcript(self, max_chars: int | None = None) -> str:
        lines = [f"[{i}] {s}: {m}" for i, (s, m) in enumerate(zip(self.speakers, self.messages))]
        text = "\n".join(lines)
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars] + "\n[...truncated...]"
        return text

    def plain_text(self) -> str:
        return "\n".join(self.messages)


def fix_mojibake(s: str) -> str:
    """Repair UTF-8 text that was decoded as cp1252 (e.g. 'â€”' -> '—')."""
    if not any(ch in s for ch in ("â", "Ã", "Â")):
        return s
    try:
        return s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        out = s
        for bad, good in {"â€”": "—", "â€“": "–", "â€™": "’", "â€œ": "“", "â€\x9d": "”", "â€˜": "‘", "â€¦": "…"}.items():
            out = out.replace(bad, good)
        return out


def download_seconvo(data_dir: Path, files: dict | None = None) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, md5 in (files or FILES).items():
        path = data_dir / name
        if path.exists() and md5_file(path) == md5:
            continue
        LOG.info("Downloading %s from Zenodo ...", name)
        urllib.request.urlretrieve(ZENODO_URL.format(name=name), path)
        got = md5_file(path)
        if got != md5:
            raise RuntimeError(f"MD5 mismatch for {name}: expected {md5}, got {got}")


def _ground_truth(entry: dict) -> dict:
    for value in entry.values():
        if isinstance(value, dict) and "IsMalicious" in value:
            return value
    raise KeyError("Ground-truth block with 'IsMalicious' not found")


def _parse_file(path: Path, split: str) -> list[Conversation]:
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    convs = []
    for entry in raw["Conversations"]:
        gt = _ground_truth(entry)
        turns = entry["Conversation"]
        pii = gt.get("PII") or []
        convs.append(
            Conversation(
                conv_id=int(gt["ConversationID"]),
                split=split,
                speakers=[fix_mojibake(t.get("Name", "")) for t in turns],
                messages=[fix_mojibake(t.get("Message", "")) for t in turns],
                label=int(bool(gt["IsMalicious"])),
                scenario=gt.get("Scenario", "unknown"),
                multi_agent=bool(gt.get("Multi", False)),
                ambiguity=gt.get("Ambiguity"),
                first_pii=gt.get("FirstPII"),
                num_pii=gt.get("NumPIIMessages"),
                pii_types=[p.get("Name", "") for p in pii if isinstance(p, dict)],
            )
        )
    return convs


def load_seconvo(data_dir: Path, download: bool = True) -> list[Conversation]:
    if download:
        download_seconvo(data_dir)
    convs = _parse_file(data_dir / "annotated_train.json", "train") + _parse_file(
        data_dir / "annotated_test.json", "test"
    )
    ids = [c.conv_id for c in convs]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate ConversationID values in SEConvo")
    n_tr = sum(c.split == "train" for c in convs)
    LOG.info(
        "SEConvo annotated: %d conversations (train=%d, test=%d, malicious=%d)",
        len(convs), n_tr, len(convs) - n_tr, sum(c.label for c in convs),
    )
    return convs


def load_seconvo_external(data_dir: Path, download: bool = True) -> list[Conversation]:
    """The 1,000 SEConvo conversations outside the annotated subset (IDs 0-999, disjoint from
    the annotated IDs 1000-1399). Their IsMalicious label is the simulation-design label
    (whether the initiator was instructed to attempt social engineering; 500/500), not a
    human judgment -- used here only as an independent external test set."""
    if download:
        download_seconvo(data_dir, EXTERNAL_FILES)
    convs = _parse_file(data_dir / "unannotated_train.json", "external") + _parse_file(
        data_dir / "unannotated_test.json", "external")
    LOG.info("SEConvo external (unannotated): %d conversations (malicious=%d)",
             len(convs), sum(c.label for c in convs))
    return convs


# ---------------------------------------------------------------------------
# Synthetic fixture (offline smoke tests only; never used for reported results)
# ---------------------------------------------------------------------------
_BENIGN = [
    "Hi {b}, I enjoyed your recent paper on {t}. Would you like to discuss a possible collaboration?",
    "Thanks {a}! Happy to talk. What part of {t} are you working on?",
    "We are exploring open datasets for {t}. Could we schedule a call through the university portal?",
    "Sure, my office hours are listed on the department page.",
    "Great, I will send an agenda via the official conference channel.",
]
_MALICIOUS = [
    "Hi {b}, your work on {t} is amazing, truly world class! We have a paid opportunity for you.",
    "Oh thank you {a}, that sounds interesting. Tell me more.",
    "To register you, please send your full name, date of birth and home address.",
    "Hmm, why do you need that?",
    "It is standard. Also we need your credit card for a small processing fee, urgently today.",
]
_TOPICS = ["renewable energy", "machine learning", "journalism ethics", "recruitment analytics"]
_SCEN = ["Academic Collaboration", "Academic Funding", "Journalism", "Recruitment"]


def make_synthetic(n: int = 120, seed: int = 0) -> list[Conversation]:
    rng = np.random.default_rng(seed)
    names = ["Alex", "Sam", "Jordan", "Riley", "Casey", "Morgan", "Taylor", "Jamie"]
    convs = []
    for i in range(n):
        label = int(rng.random() < 0.53)
        a, b = rng.choice(names, 2, replace=False)
        t = _TOPICS[i % 4]
        tmpl = list(_MALICIOUS if label else _BENIGN)
        if rng.random() < 0.15:  # label noise so that nothing is trivially separable
            tmpl = list(_BENIGN if label else _MALICIOUS)
        msgs = [m.format(a=a, b=b, t=t) for m in tmpl]
        speakers = [a if j % 2 == 0 else b for j in range(len(msgs))]
        convs.append(
            Conversation(
                conv_id=10_000 + i,
                split="train" if i < max(20, n // 10) else "test",
                speakers=speakers,
                messages=msgs,
                label=label,
                scenario=_SCEN[i % 4],
                multi_agent=bool(i % 2),
                ambiguity=int(rng.integers(1, 4)),
                first_pii=2 if label else None,
                num_pii=2 if label else 0,
            )
        )
    return convs
