from __future__ import annotations
import json
import math
from typing import List, Tuple, Optional
from urllib import request

from .config import Settings

HF_API_BASE = "https://api-inference.huggingface.co/models"


def _post_json(url: str, payload: dict, headers: dict, timeout: int) -> Optional[dict | list]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
        try:
            return json.loads(raw)
        except Exception:
            return None


def embed_texts(texts: List[str], settings: Settings) -> Optional[List[List[float]]]:
    if not texts:
        return []
    token = getattr(settings, "hf_api_token", None)
    model = getattr(settings, "embeddings_model", "sentence-transformers/all-MiniLM-L6-v2")
    if not token or not model:
        return None
    url = f"{HF_API_BASE}/{model}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"inputs": texts, "options": {"wait_for_model": True}}
    res = _post_json(url, payload, headers, int(getattr(settings, "llm_timeout_seconds", 25) or 25))
    if isinstance(res, list) and len(res) == len(texts):
        # Each item is a vector
        try:
            return [list(map(float, v[0] if isinstance(v[0], list) else v)) for v in res]
        except Exception:
            return None
    # Some servers return a dict
    return None


def _cos(a: List[float], b: List[float]) -> float:
    num = sum(x*y for x, y in zip(a, b))
    da = math.sqrt(sum(x*x for x in a))
    db = math.sqrt(sum(y*y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def cluster_texts(texts: List[str], embeddings: List[List[float]], k: int) -> List[List[int]]:
    """
    Simple farthest-first seeding + assignment clustering. Returns list of clusters, each a list of indices.
    """
    n = len(texts)
    if n == 0:
        return []
    k = max(1, min(k, n))

    # Farthest-first choose seeds
    seeds: List[int] = [0]
    sims = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = _cos(embeddings[i], embeddings[j])
            sims[i][j] = s
            sims[j][i] = s
    while len(seeds) < k:
        best_idx, best_score = -1, -1.0
        for i in range(n):
            # distance = 1 - max similarity to existing seeds
            d = 1.0 - max(sims[i][s] for s in seeds)
            if d > best_score and i not in seeds:
                best_score, best_idx = d, i
        if best_idx == -1: break
        seeds.append(best_idx)

    # Assign each to nearest seed
    clusters = [[] for _ in range(len(seeds))]
    for i in range(n):
        best_seed, best_sim = 0, -1.0
        for si, s in enumerate(seeds):
            if sims[i][s] > best_sim:
                best_sim, best_seed = sims[i][s], si
        clusters[best_seed].append(i)

    # Drop tiny clusters if we have many
    clusters = [c for c in clusters if len(c) > 0]
    return clusters


def summarize_themes(texts: List[str], clusters: List[List[int]], embeddings: List[List[float]], max_themes: int = 3) -> List[str]:
    themes: List[str] = []
    for c in clusters:
        # choose representative as the one with max average similarity within cluster
        best_idx, best_score = c[0], -1.0
        for i in c:
            score = sum(_cos(embeddings[i], embeddings[j]) for j in c if j != i) / max(1, (len(c)-1))
            if score > best_score:
                best_score, best_idx = score, i
        themes.append(texts[best_idx])
    # Deduplicate and cap
    dedup = []
    seen = set()
    for t in themes:
        key = t.lower()
        if key not in seen:
            seen.add(key); dedup.append(t)
    return dedup[:max_themes]

