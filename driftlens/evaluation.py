"""Metrics, grouped cross-validation and a paired bootstrap over timelines."""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def binary_views(labels):
    labels = list(labels)
    return {
        "switch": [int(l in ("S", "B")) for l in labels],
        "escalation": [int(l in ("E", "B")) for l in labels],
        "none": [int(l == "O") for l in labels],
    }


def scores(gold, pred):
    g, p = binary_views(gold), binary_views(pred)
    out = {}
    for k in g:
        out[f"f1_{k}"] = f1_score(g[k], p[k], zero_division=0)
        out[f"p_{k}"] = precision_score(g[k], p[k], zero_division=0)
        out[f"r_{k}"] = recall_score(g[k], p[k], zero_division=0)
    out["macro_f1"] = float(np.mean([out["f1_switch"], out["f1_escalation"], out["f1_none"]]))
    out["accuracy"] = float(np.mean([a == b for a, b in zip(gold, pred)]))
    return out


def fold_splits(n_groups, n_folds, seed):
    perm = np.random.RandomState(seed).permutation(n_groups)
    return [sorted(f) for f in np.array_split(perm, n_folds)]


def grouped_cv(X_seqs, y_seqs, factory, n_folds=5, repeats=5, seed=0):
    """Timeline-level K-fold CV repeated with different shuffles.

    Returns (per-repeat metric dicts, per-repeat out-of-fold predictions).
    Predictions from all folds are pooled before scoring each repeat.
    """
    n = len(X_seqs)
    metrics, all_preds = [], []
    for r in range(repeats):
        preds = [None] * n
        for test_idx in fold_splits(n, n_folds, seed + r):
            train_idx = [i for i in range(n) if i not in set(test_idx)]
            model = factory().fit([X_seqs[i] for i in train_idx], [y_seqs[i] for i in train_idx])
            out = model.predict([X_seqs[i] for i in test_idx])
            for i, p in zip(test_idx, out):
                preds[i] = list(p)
        gold = [s for ys in y_seqs for s in ys]
        flat = [s for ps in preds for s in ps]
        metrics.append(scores(gold, flat))
        all_preds.append(preds)
    return metrics, all_preds


def summarise(metrics):
    keys = metrics[0].keys()
    return {k: (float(np.mean([m[k] for m in metrics])), float(np.std([m[k] for m in metrics]))) for k in keys}


def paired_bootstrap(y_seqs, preds_a, preds_b, metric="macro_f1", n_boot=2000, seed=0):
    """One-sided p-value that system A is not better than B.

    Timelines are resampled with replacement; for each resample the metric is
    averaged over all CV repeats, so fold-assignment noise is integrated out.
    """
    rng = np.random.RandomState(seed)
    n = len(y_seqs)

    def value(idx, preds_list):
        vals = []
        for preds in preds_list:
            gold = [s for i in idx for s in y_seqs[i]]
            pr = [s for i in idx for s in preds[i]]
            vals.append(scores(gold, pr)[metric])
        return np.mean(vals)

    full = np.arange(n)
    observed = value(full, preds_a) - value(full, preds_b)
    worse = 0
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if value(idx, preds_a) - value(idx, preds_b) <= 0:
            worse += 1
    return float(observed), worse / n_boot
