"""In-memory exemplar store: numpy matrix + metadata list. Cosine kNN (k=7,
majority vote, margin = top-class share). Persists to data/exemplars.jsonl
(embedding stored per line so restart needs no re-embed). No vector DB."""

import json
import shutil
import threading
from datetime import datetime, timezone

import numpy as np

from router import config


def load_frozen_params() -> dict:
    frozen_file = config.ROOT / "config" / "frozen.json"
    if frozen_file.exists():
        try:
            with open(frozen_file) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "k": 7,
        "sim_exponent": 1.0,
        "margin_threshold": 0.15,
        "near_duplicate_cutoff": 0.90,
    }


FROZEN_PARAMS = load_frozen_params()
K = int(FROZEN_PARAMS.get("k", 7))
SIM_EXPONENT = float(FROZEN_PARAMS.get("sim_exponent", 1.0))
DEFAULT_MARGIN = float(FROZEN_PARAMS.get("margin_threshold", 0.15))
NEAR_DUP_CUTOFF = float(FROZEN_PARAMS.get("near_duplicate_cutoff", 0.90))


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


class ExemplarIndex:
    def __init__(self, max_cache_size: int = 2000):
        self.vecs: np.ndarray | None = None  # (n, d), L2-normalised
        self.meta: list[dict] = []
        self.lock = threading.Lock()
        self._cache: dict[str, tuple[str, float]] = {}
        self._max_cache_size = max_cache_size

    # -- persistence ---------------------------------------------------------

    def load(self, embed_fn=None) -> int:
        """Load exemplars.jsonl. Lines missing an embedding get re-embedded
        via embed_fn (batch)."""
        if not config.EXEMPLARS_PATH.exists():
            return 0
        rows = []
        with open(config.EXEMPLARS_PATH) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        missing = [r for r in rows if "embedding" not in r]
        if missing:
            if embed_fn is None:
                raise RuntimeError(f"{len(missing)} exemplars lack embeddings and no embed_fn given")
            vecs = embed_fn([r["text"] for r in missing])
            for r, v in zip(missing, vecs):
                r["embedding"] = v
            self._rewrite(rows)
        with self.lock:
            self.meta = [
                {
                    "id": r.get("id", f"ex_{i}"),
                    "text": r["text"],
                    "label": r["label"],
                    "source": r.get("source", "bootstrap"),
                    "category": r.get("category", "general"),
                    "ts": r.get("ts", datetime.now(timezone.utc).isoformat()),
                }
                for i, r in enumerate(rows)
            ]
            self.vecs = _norm(np.array([r["embedding"] for r in rows], dtype=np.float32))
            self._cache.clear()
        return len(rows)

    def _rewrite(self, rows: list[dict]):
        tmp = config.EXEMPLARS_PATH.with_suffix(".tmp")
        with open(tmp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        tmp.replace(config.EXEMPLARS_PATH)

    def add(
        self,
        text: str,
        label: str,
        source: str,
        embedding: list[float],
        persist: bool = True,
        category: str = "general",
    ):
        row = {
            "text": text,
            "label": label,
            "source": source,
            "category": category,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        v = _norm(np.array(embedding, dtype=np.float32).reshape(1, -1))
        with self.lock:
            self.meta.append(row)
            self.vecs = v if self.vecs is None else np.vstack([self.vecs, v])
            self._cache.clear()
            if persist:
                with open(config.EXEMPLARS_PATH, "a") as f:
                    f.write(json.dumps({**row, "embedding": embedding}) + "\n")

    # -- queries -------------------------------------------------------------

    def route(
        self,
        embedding: list[float],
        cost_weight: float | None = None,
        text: str | None = None,
        confidence_margin_threshold: float | None = None,
        return_category: bool = False,
    ) -> tuple[str, float] | tuple[str, float, str]:
        """Returns (label, margin) or (label, margin, category) if return_category=True.
        Margin = share of top class among k neighbours. Empty index -> ('reasoning', 0.0).
        cost_weight (lambda in [0, 1]) shifts borderline decisions toward 'simple'.
        Uses similarity-weighted voting with near-duplicate short-circuiting
        and confidence-gated guardrails on borderline decisions."""
        cache_key = text.strip() if text else None
        if cache_key and cache_key in self._cache:
            entry = self._cache[cache_key]
            lbl, mgn = entry[0], entry[1]
            cat = entry[2] if len(entry) > 2 else "general"
            return (lbl, mgn, cat) if return_category else (lbl, mgn)

        margin_thresh = DEFAULT_MARGIN if confidence_margin_threshold is None else confidence_margin_threshold

        with self.lock:
            if self.vecs is None or len(self.meta) == 0:
                return ("reasoning", 0.0, "general") if return_category else ("reasoning", 0.0)
            q = _norm(np.array(embedding, dtype=np.float32))
            sims = self.vecs @ q

            # Near-duplicate short-circuit (cos >= NEAR_DUP_CUTOFF): directly match exemplar
            max_idx = int(np.argmax(sims))
            if float(sims[max_idx]) >= NEAR_DUP_CUTOFF:
                label = self.meta[max_idx]["label"]
                margin = float(sims[max_idx])
                category = self.meta[max_idx].get("category", "general")
                if cache_key and len(self._cache) < self._max_cache_size:
                    self._cache[cache_key] = (label, margin, category)
                return (label, margin, category) if return_category else (label, margin)

            k = min(K, len(self.meta))
            top = np.argpartition(-sims, k - 1)[:k]
            
            lam = config.COST_WEIGHT if cost_weight is None else cost_weight
            s_w = 0.0
            r_w = 0.0
            cat_weights: dict[str, float] = {}
            for i in top:
                lbl = self.meta[i]["label"]
                cat = self.meta[i].get("category", "general")
                sim = max(0.0, float(sims[i]))
                sim_weight = (sim ** SIM_EXPONENT) if sim > 0 else 1e-4
                cat_weights[cat] = cat_weights.get(cat, 0.0) + sim_weight
                weight = sim_weight * ((1.0 - lam) if lbl == "reasoning" else (1.0 + lam))
                if lbl == "simple":
                    s_w += weight
                else:
                    r_w += weight

            top_category = max(cat_weights.items(), key=lambda x: x[1])[0] if cat_weights else "general"

            tot_w = s_w + r_w
            if tot_w == 0:
                label = "reasoning"
                margin = 0.0
            else:
                margin = abs(s_w - r_w) / tot_w
                raw_label = "simple" if s_w > r_w else "reasoning"
                label = "reasoning" if margin < margin_thresh else raw_label

            if cache_key and len(self._cache) < self._max_cache_size:
                self._cache[cache_key] = (label, margin, top_category)

            return (label, margin, top_category) if return_category else (label, margin)

    def nearest(self, embedding: list[float], top_n: int = 5) -> list[dict]:
        """Returns top_n nearest exemplar records with cosine similarity."""
        with self.lock:
            if self.vecs is None or len(self.meta) == 0:
                return []
            q = _norm(np.array(embedding, dtype=np.float32))
            sims = self.vecs @ q
            n = min(top_n, len(self.meta))
            top_indices = np.argsort(-sims)[:n]
            return [
                {
                    "text": self.meta[i]["text"],
                    "label": self.meta[i]["label"],
                    "source": self.meta[i].get("source", "bootstrap"),
                    "category": self.meta[i].get("category", "general"),
                    "similarity": round(float(sims[i]), 4),
                }
                for i in top_indices
            ]

    def max_sim(self, embedding: list[float], label: str) -> float:
        """Highest cosine vs existing exemplars of a route (judge dedup guard)."""
        with self.lock:
            if self.vecs is None:
                return 0.0
            mask = np.array([m["label"] == label for m in self.meta])
            if not mask.any():
                return 0.0
            q = _norm(np.array(embedding, dtype=np.float32))
            return float((self.vecs[mask] @ q).max())

    def max_conflicting_sim(self, embedding: list[float], keep_label: str) -> float:
        """Highest cosine among judge-sourced exemplars of a DIFFERENT label."""
        with self.lock:
            if self.vecs is None:
                return 0.0
            mask = np.array([m["label"] != keep_label and m["source"] == "judge"
                             for m in self.meta])
            if not mask.any():
                return 0.0
            q = _norm(np.array(embedding, dtype=np.float32))
            return float((self.vecs[mask] @ q).max())

    def remove_conflicting(self, embedding: list[float], keep_label: str,
                           threshold: float) -> int:
        """Drop judge-sourced exemplars that near-duplicate this embedding but
        carry a DIFFERENT label (verdict flip on a repeated query): latest
        clear verdict wins. Bootstrap exemplars are never touched."""
        with self.lock:
            if self.vecs is None:
                return 0
            q = _norm(np.array(embedding, dtype=np.float32))
            sims = self.vecs @ q
            drop = {
                i for i, (m, s) in enumerate(zip(self.meta, sims))
                if s >= threshold and m["label"] != keep_label and m["source"] == "judge"
            }
            if not drop:
                return 0
            keep = [i for i in range(len(self.meta)) if i not in drop]
            self.meta = [self.meta[i] for i in keep]
            self.vecs = self.vecs[keep] if keep else None
            # persist: normalized vectors are fine, cosine is norm-invariant
            self._rewrite([
                {**m, "embedding": self.vecs[i].tolist()}
                for i, m in enumerate(self.meta)
            ])
            return len(drop)

    def count(self, label: str | None = None, source: str | None = None) -> int:
        with self.lock:
            return sum(
                1
                for m in self.meta
                if (label is None or m["label"] == label)
                and (source is None or m["source"] == source)
            )

    def snapshot(self) -> str:
        config.SNAPSHOT_DIR.mkdir(exist_ok=True)
        name = datetime.now(timezone.utc).strftime("%H%M%S") + ".jsonl"
        with self.lock:
            shutil.copy(config.EXEMPLARS_PATH, config.SNAPSHOT_DIR / name)
        return name


_index: ExemplarIndex | None = None


def get_index() -> ExemplarIndex:
    global _index
    if _index is None:
        _index = ExemplarIndex()
    return _index
