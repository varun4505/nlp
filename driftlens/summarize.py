"""Affect-biased TextRank for a short timeline digest.

Sentences are nodes, TF-IDF cosine similarities are edge weights and the
scores come from PageRank power iteration. The teleport vector is biased
towards sentences that contain self-state words (topic-sensitive PageRank),
so the digest favours sentences about the writer's state over incidental
content. The digest is only for display in the prototype; it is not evaluated.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def textrank(sentences, bias=None, k=3, d=0.85, iters=100):
    n = len(sentences)
    if n == 0:
        return []
    if n <= k:
        return list(range(n))
    try:
        X = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    except ValueError:              # only stop words
        return list(range(k))
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, 0.0)
    rows = S.sum(1, keepdims=True)
    P = np.where(rows > 0, S / np.where(rows > 0, rows, 1), 1.0 / n)
    b = np.ones(n) if bias is None else np.asarray(bias, float) + 1e-3
    b = b / b.sum()
    r = np.full(n, 1.0 / n)
    for _ in range(iters):
        r_new = (1 - d) * b + d * P.T @ r
        if np.abs(r_new - r).sum() < 1e-9:
            break
        r = r_new
    return sorted(np.argsort(-r)[:k].tolist())
