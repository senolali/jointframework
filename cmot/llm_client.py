"""ASU CreateAI client (OpenAI-compatible /v1 routes) with caching, retries,
cost accounting, a hard budget guard and an explicit offline mock mode.

Design choices for provenance:
  * every request is logged (prompt hash, requested and *returned* model id,
    decoding parameters actually sent, token usage, cost, wall-clock latency);
  * decoding parameters are explicit in code (temperature=0 where supported);
  * models that reject `temperature` (reasoning models) are detected
    automatically and the omission is recorded per call.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

try:  # verify TLS against the OS trust store (needed behind antivirus HTTPS scanning, e.g. AVG/Avast)
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # pragma: no cover
    pass

from .utils import LOG, JsonlCache, chunks, stable_hash


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class ModelSpec:
    key: str  # display name used in tables/figures
    provider: str
    model: str
    family: str = ""
    price_in: float = 0.0  # USD per 1M input tokens (CreateAI catalog)
    price_out: float = 0.0
    supports_temperature: bool = True
    max_tokens: int = 600
    open_weights: bool = False
    base_url: str | None = None      # external OpenAI-compatible endpoint (None = CreateAI)
    api_key_env: str | None = None   # env variable holding that endpoint's key

    @property
    def route(self) -> str:
        return f"{self.provider}/{self.model}"


class CreateAIClient:
    def __init__(
        self,
        base_url: str,
        token: str | None,
        cache_dir: Path,
        timeout_s: float = 180.0,
        max_retries: int = 6,
        budget_usd: float = 50.0,
        offline: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.budget_usd = budget_usd
        self.offline = offline
        self.cache = JsonlCache(cache_dir / ("llm_calls_MOCK.jsonl" if offline else "llm_calls.jsonl"))
        self.emb_dir = cache_dir / ("embeddings_MOCK" if offline else "embeddings")
        self.emb_dir.mkdir(parents=True, exist_ok=True)
        self._spent_lock = threading.Lock()
        self.spent_usd = 0.0  # spent in *this* process (cache hits cost nothing)
        self._no_temperature: set[str] = set()
        self._session = requests.Session()
        if not offline and not token:
            raise RuntimeError(
                "CreateAI token missing. Set the environment variable named in config "
                "(default CREATEAI_SERVICE_TOKEN) or run with --offline for a mock smoke test."
            )

    # ------------------------------------------------------------- preflight
    def preflight(self, specs: list[ModelSpec], embedding_route: str | None = None) -> list[str]:
        """Fail fast before any paid call: verify the token and that every configured model
        is offered to this project. Returns the list of available model ids."""
        if self.offline:
            return [s.route for s in specs]
        resp = self._session.get(f"{self.base_url}/v1/models", headers=self._headers(), timeout=60)
        if resp.status_code in (401, 403):
            raise RuntimeError(f"CreateAI rejected the token (HTTP {resp.status_code}): {resp.text[:300]}\n"
                               "Check CREATEAI_SERVICE_TOKEN and that it belongs to the environment of the base URL.")
        resp.raise_for_status()
        available = sorted(m["id"] for m in resp.json().get("data", []))
        if embedding_route and embedding_route not in available:
            LOG.warning("Embedding model %s is not listed by /v1/models (listing may omit embedding "
                        "models); it will be tried and the run stops if it fails.", embedding_route)
        for spec in [s for s in specs if s.base_url]:
            r = self._session.get(f"{spec.base_url.rstrip('/')}/models", headers=self._headers(spec), timeout=60)
            if r.status_code != 200:
                raise RuntimeError(f"{spec.key}: external endpoint rejected the key (HTTP {r.status_code})")
            ids = [m.get("id") for m in r.json().get("data", [])]
            if spec.model not in ids:
                raise RuntimeError(f"{spec.key}: model {spec.model} not offered; available: {ids}")
            available.append(spec.route)
        wanted = [s.route for s in specs]
        missing = [r for r in wanted if r not in available]
        if missing:
            near = {r: [a for a in available if a.split("/")[0] == r.split("/")[0]][:12] for r in missing}
            raise RuntimeError("Models not available to this project: " + ", ".join(missing) +
                               "\nSame-provider models offered: " + str(near) +
                               "\nEdit the `llms` / `representations.embedding` entries in config.yaml.")
        LOG.info("Preflight OK: token valid, %d configured models available.", len(wanted))
        return available

    # ------------------------------------------------------------------ chat
    def chat(self, spec: ModelSpec, messages: list[dict], temperature: float | None,
             max_tokens: int | None = None, tag: str = "", repeat: int = 0) -> dict:
        max_tokens = max_tokens or spec.max_tokens
        send_temp = temperature if (spec.supports_temperature and spec.route not in self._no_temperature) else None
        key = stable_hash({"route": spec.route, "messages": messages, "temperature": temperature,
                           "max_tokens": max_tokens, "repeat": repeat, "offline": self.offline})
        hit = self.cache.get(key)
        if hit is not None and hit.get("ok"):
            return {**hit, "cached": True}

        if self.offline:
            rec = self._mock_chat(spec, messages, repeat)
        else:
            rec = self._remote_chat(spec, messages, send_temp, max_tokens)
        rec.update({"tag": tag, "route": spec.route, "repeat": repeat, "temperature_requested": temperature,
                    "max_tokens": max_tokens, "prompt_hash": stable_hash(messages)[:16]})
        cost = (rec.get("input_tokens", 0) * spec.price_in + rec.get("output_tokens", 0) * spec.price_out) / 1e6
        rec["cost_usd"] = cost
        if rec.get("ok"):
            self.cache.put(key, rec)
        with self._spent_lock:
            self.spent_usd += cost
            if self.spent_usd > self.budget_usd:
                raise BudgetExceeded(f"Budget of ${self.budget_usd:.2f} exceeded (spent ${self.spent_usd:.2f}).")
        return {**rec, "cached": False}

    def _headers(self, spec: ModelSpec | None = None) -> dict:
        token = self.token
        if spec is not None and spec.base_url:
            token = os.environ.get(spec.api_key_env or "", "")
            if not token:
                raise RuntimeError(f"{spec.key}: environment variable {spec.api_key_env} is not set")
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def _chat_target(self, spec: ModelSpec) -> tuple[str, str]:
        """(url, model field) for a spec: CreateAI uses provider/model, external APIs the bare id."""
        if spec.base_url:
            return f"{spec.base_url.rstrip('/')}/chat/completions", spec.model
        return f"{self.base_url}/v1/chat/completions", spec.route

    def _remote_chat(self, spec: ModelSpec, messages: list[dict], temperature: float | None, max_tokens: int) -> dict:
        url, model_field = self._chat_target(spec)
        body = {"model": model_field, "messages": messages, "max_tokens": max_tokens, "stream": False}
        if temperature is not None:
            body["temperature"] = temperature
        last_err = ""
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = self._session.post(url, headers=self._headers(spec), json=body, timeout=self.timeout_s)
            except requests.RequestException as exc:
                last_err = f"network: {exc}"
                self._sleep(attempt, 5)
                continue
            latency = time.perf_counter() - t0
            if resp.status_code == 200:
                data = resp.json()
                choice = (data.get("choices") or [{}])[0]
                text = (choice.get("message") or {}).get("content") or ""
                usage = data.get("usage") or {}
                return {
                    "ok": True,
                    "text": text,
                    "finish_reason": choice.get("finish_reason"),
                    "model_returned": data.get("model"),
                    "input_tokens": int(usage.get("prompt_tokens") or 0),
                    "output_tokens": int(usage.get("completion_tokens") or 0),
                    "latency_s": latency,
                    "temperature_sent": temperature,
                    "timestamp": time.time(),
                }
            err = self._error_payload(resp)
            last_err = f"HTTP {resp.status_code}: {err}"
            code = str(err.get("code") or err.get("type") or "")
            param = str(err.get("param") or "")
            msg = str(err.get("message") or "").lower()
            if resp.status_code == 400 and "temperature" in (param + msg) and "temperature" in body:
                LOG.warning("%s rejects temperature; retrying without it (recorded).", spec.route)
                self._no_temperature.add(spec.route)
                body.pop("temperature", None)
                temperature = None
                continue
            if resp.status_code == 429 or code in {"rate_limit_exceeded"}:
                if "monthly" in msg or "budget" in msg:
                    raise BudgetExceeded(f"CreateAI monthly budget exhausted: {err.get('message')}")
                self._sleep(attempt, 60)
                continue
            if resp.status_code >= 500:
                self._sleep(attempt, 10)
                continue
            break  # other 4xx: do not retry
        LOG.error("Request to %s failed: %s", spec.route, last_err)
        return {"ok": False, "text": "", "error": last_err, "input_tokens": 0, "output_tokens": 0,
                "latency_s": float("nan"), "model_returned": None, "temperature_sent": temperature}

    @staticmethod
    def _error_payload(resp: requests.Response) -> dict:
        try:
            data = resp.json()
        except ValueError:
            return {"message": resp.text[:300]}
        if isinstance(data.get("error"), dict):
            return data["error"]
        return {"message": data.get("message") or data.get("error") or str(data)[:300],
                "code": data.get("error_type")}

    @staticmethod
    def _sleep(attempt: int, base: float) -> None:
        delay = min(base * (2 ** attempt), 300) * (0.75 + 0.5 * random.random())
        time.sleep(delay)

    # ------------------------------------------------------------- embeddings
    def embed(self, texts: list[str], provider: str, model: str, batch_size: int = 64,
              max_chars: int = 24000) -> np.ndarray:
        """Embeddings with a per-text on-disk cache (npz shards keyed by sha256)."""
        route = f"{provider}/{model}"
        store_path = self.emb_dir / f"{route.replace('/', '__')}.npz"
        store: dict[str, np.ndarray] = {}
        if store_path.exists():
            with np.load(store_path) as z:
                store = {k: z[k] for k in z.files}
        keys = [hashlib.sha256(t[:max_chars].encode("utf-8")).hexdigest() for t in texts]
        missing = sorted({k: t for k, t in zip(keys, texts) if k not in store}.items())
        if missing:
            LOG.info("Embedding %d new texts with %s%s", len(missing), route, " (MOCK)" if self.offline else "")
            for b, batch in enumerate(chunks(missing, batch_size), 1):
                vecs = (self._mock_embed([t for _, t in batch]) if self.offline
                        else self._remote_embed([t[:max_chars] for _, t in batch], route))
                for (k, _), v in zip(batch, vecs):
                    store[k] = np.asarray(v, dtype=np.float32)
                if b % 10 == 0:  # checkpoint so an interrupted run does not redo finished batches
                    self._save_store(store_path, store)
                    LOG.info("  embedded %d/%d", min(b * batch_size, len(missing)), len(missing))
            self._save_store(store_path, store)
        return np.vstack([store[k] for k in keys])

    @staticmethod
    def _save_store(path: Path, store: dict) -> None:
        tmp = path.with_name(path.stem + ".tmp.npz")  # write-then-rename: never leaves a half-written cache
        np.savez_compressed(tmp, **store)
        os.replace(tmp, path)

    def _remote_embed(self, texts: list[str], route: str) -> list[list[float]]:
        url = f"{self.base_url}/v1/embeddings"
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.post(url, headers=self._headers(), json={"model": route, "input": texts},
                                          timeout=self.timeout_s)
            except requests.RequestException:
                self._sleep(attempt, 5)
                continue
            if resp.status_code == 200:
                data = sorted(resp.json()["data"], key=lambda d: d["index"])
                return [d["embedding"] for d in data]
            if resp.status_code == 429 or resp.status_code >= 500:
                self._sleep(attempt, 30)
                continue
            raise RuntimeError(f"Embedding request failed: HTTP {resp.status_code} {resp.text[:300]}")
        raise RuntimeError("Embedding request failed after retries")

    # ------------------------------------------------------------------ mocks
    @staticmethod
    def _mock_embed(texts: list[str], dim: int = 256) -> list[np.ndarray]:
        from sklearn.feature_extraction.text import HashingVectorizer

        hv = HashingVectorizer(n_features=dim, alternate_sign=True, norm="l2", ngram_range=(1, 2))
        return list(hv.transform(texts).toarray())

    _SUSPICIOUS = re.compile(
        r"birth|address|credit card|bank|password|fee|payment|ssn|social security|passport|urgent|"
        r"wire|gift card|account number|phone number|mother'?s maiden|billing",
        re.IGNORECASE,
    )

    def _mock_chat(self, spec: ModelSpec, messages: list[dict], repeat: int) -> dict:
        """Deterministic stand-in response so the full pipeline can be smoke-tested
        without a token. Mock outputs are tagged MOCK everywhere downstream."""
        user = messages[-1]["content"]
        convo = user.split("<conversation>")[-1]
        hits = len(self._SUSPICIOUS.findall(convo))
        seed = int(hashlib.md5(f"{spec.route}|{repeat}|{convo[:200]}".encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        score = 1 / (1 + np.exp(-(hits - 1.5))) + rng.gauss(0, 0.25)
        score = float(min(max(score, 0.01), 0.99))
        if "Genuine Concept Drift" in user:  # judge prompt
            text = ("Fraudulent Attempt. Type: 'fraudulent'" if score > 0.5
                    else "Genuine Concept Drift. Type: 'genuine'")
            n_in = sum(len(m["content"]) for m in messages) // 4
            return {"ok": True, "text": text, "finish_reason": "stop", "model_returned": f"MOCK::{spec.route}",
                    "input_tokens": n_in, "output_tokens": 12, "latency_s": 0.0, "temperature_sent": None,
                    "timestamp": time.time()}
        if "verdict" in user:
            out = {"verdict": "fraudulent" if score > 0.5 else "benign_shift", "confidence": round(abs(score - 0.5) * 2, 2),
                   "rationale": "mock"}
        else:
            out = {"label": "malicious" if score > 0.5 else "benign", "confidence": round(max(score, 1 - score), 2),
                   "evidence_turns": [], "rationale": "mock"}
        n_in = sum(len(m["content"]) for m in messages) // 4
        return {"ok": True, "text": json.dumps(out), "finish_reason": "stop", "model_returned": f"MOCK::{spec.route}",
                "input_tokens": n_in, "output_tokens": 40, "latency_s": 0.0, "temperature_sent": None,
                "timestamp": time.time()}


def make_client(api_cfg: dict, cache_dir: Path, offline: bool) -> CreateAIClient:
    token = os.environ.get(api_cfg.get("token_env", "CREATEAI_SERVICE_TOKEN"))
    base = os.environ.get("CREATEAI_BASE_URL", api_cfg.get("base_url", "https://api-main.aiml.asu.edu"))
    return CreateAIClient(
        base_url=base,
        token=token,
        cache_dir=cache_dir,
        timeout_s=api_cfg.get("timeout_s", 180),
        max_retries=api_cfg.get("max_retries", 6),
        budget_usd=api_cfg.get("budget_usd", 50.0),
        offline=offline,
    )
