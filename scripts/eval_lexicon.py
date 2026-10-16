"""Evaluate the WordNet self-state lexicon in-domain and on an external corpus.

In-domain (CLPsych, 236 posts with well-being and evidence annotations):
  * Spearman correlation between post scores and the 1-10 well-being rating
  * evidence recall: share of annotated maladaptive / adaptive evidence spans
    that contain at least one lexicon hit of the same polarity
  * hit precision: share of lexicon hits that fall inside an annotated span of
    the same polarity

External (eRisk 2026 Task 2, 909 users, 102 depressed):
  * ROC-AUC of single user-level scores for depressed vs control
  * 5-fold CV logistic regression on four interpretable scores, next to a
    TF-IDF bag-of-words reference model
"""
import json
import pickle
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftlens.data import load_erisk_users  # noqa: E402
from driftlens.paths import DATA_PROCESSED, RESULTS  # noqa: E402
from driftlens.textproc import FIRST_PERSON, clean  # noqa: E402

VARIANTS = ["seeds", "full", "no_wsd", "no_negation", "no_speller"]
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def spans_of(post, pol):
    block = post["evidence"].get(f"{pol}-state") or {}
    return [v["highlighted_evidence"] for v in block.values()
            if isinstance(v, dict) and v.get("highlighted_evidence")]


def locate(span, text):
    c = clean(span).lower()
    i = text.lower().find(c)
    if i >= 0:
        return (i, i + len(c)), None
    return None, set(WORD_RE.findall(c))


def boot_ci(x, y, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    x, y = np.asarray(x), np.asarray(y)
    vals = []
    for _ in range(n):
        idx = rng.randint(0, len(x), len(x))
        vals.append(spearmanr(x[idx], y[idx])[0])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def clpsych():
    res = {}
    for v in VARIANTS:
        with open(DATA_PROCESSED / f"clpsych_train_{v}.pkl", "rb") as fh:
            data = pickle.load(fh)
        posts = [p for d in data for p in d["posts"] if p["wellbeing"] is not None]
        wb = [p["wellbeing"] for p in posts]
        row = {"n_posts": len(posts)}
        for key in ("net", "mal", "ada"):
            vals = [p[key] for p in posts]
            row[f"spearman_{key}"] = float(spearmanr(vals, wb)[0])
        row["spearman_net_ci"] = boot_ci([p["net"] for p in posts], wb)
        if v == "full":
            row["spearman_vader"] = float(spearmanr([p["vader"] for p in posts], wb)[0])
            row["spearman_vader_ci"] = boot_ci([p["vader"] for p in posts], wb)
            row["spearman_len"] = float(spearmanr([p["n_words"] for p in posts], wb)[0])
        for pol, key in (("maladaptive", "mal"), ("adaptive", "ada")):
            n_spans = found = 0
            n_hits = hits_in = 0
            for p in posts:
                located = [locate(s, p["clean"]) for s in spans_of(p, pol)]
                hits = [h for h in p["hits"] if h["pol"] == key and h["w"] > 0]
                for rng_, toks in located:
                    n_spans += 1
                    if any((rng_ and h["start"] < rng_[1] and h["end"] > rng_[0]) or
                           (toks is not None and h["text"].lower() in toks) for h in hits):
                        found += 1
                for h in hits:
                    n_hits += 1
                    if any((r and h["start"] < r[1] and h["end"] > r[0]) or
                           (t is not None and h["text"].lower() in t) for r, t in located):
                        hits_in += 1
            row[f"{key}_span_recall"] = found / max(n_spans, 1)
            row[f"{key}_hit_precision"] = hits_in / max(n_hits, 1)
            row[f"{key}_n_spans"] = n_spans
            row[f"{key}_n_hits"] = n_hits
        row["coverage"] = float(np.mean([any(h["w"] > 0 for h in p["hits"]) for p in posts]))
        res[v] = row
        print(v, json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in row.items()}), flush=True)
    (RESULTS / "lexicon_clpsych_eval.json").write_text(json.dumps(res, indent=1))


def erisk(max_writings=100):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from driftlens.resources import lexicon

    users = load_erisk_users()
    y = np.array([u["label"] for u in users])
    texts = [[clean(t) for t in u["texts"][-max_writings:]] for u in users]
    vader = SentimentIntensityAnalyzer()
    res = {"n_users": len(users), "n_positive": int(y.sum()), "max_writings": max_writings, "variants": {}}

    fp, vad = [], []
    for ws in texts:
        toks = [t for w in ws for t in WORD_RE.findall(w.lower())]
        fp.append(sum(t in FIRST_PERSON for t in toks) / max(len(toks), 1))
        vad.append(float(np.mean([vader.polarity_scores(w)["compound"] for w in ws])) if ws else 0.0)
    fp, vad = np.array(fp), np.array(vad)
    res["auc_first_person_rate"] = float(roc_auc_score(y, fp))
    res["auc_neg_vader"] = float(roc_auc_score(y, -vad))

    skf = StratifiedKFold(5, shuffle=True, random_state=0)

    def cv_lr(X):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        prob = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]
        return {"auc": float(roc_auc_score(y, prob)), "f1": float(f1_score(y, prob >= 0.5))}

    for v in ("seeds", "full", "no_wsd"):
        t0 = time.time()
        lex = lexicon("seeds" if v == "seeds" else "full")
        mal, ada = [], []
        for ws in texts:
            mw = aw = n = 0.0
            for w in ws:
                s = lex.score(w, speller=None, use_wsd=(v != "no_wsd"))
                mw += s["mal"] * s["n_words"] / 100
                aw += s["ada"] * s["n_words"] / 100
                n += s["n_words"]
            mal.append(100 * mw / max(n, 1))
            ada.append(100 * aw / max(n, 1))
        mal, ada = np.array(mal), np.array(ada)
        row = {"auc_mal": float(roc_auc_score(y, mal)), "auc_neg_ada": float(roc_auc_score(y, -ada)),
               "auc_neg_net": float(roc_auc_score(y, mal - ada)),
               "lr_4_scores": cv_lr(np.c_[mal, ada, fp, vad]),
               "lr_lexicon_only": cv_lr(np.c_[mal, ada])}
        res["variants"][v] = row
        print(v, json.dumps(row), f"{time.time() - t0:.0f}s", flush=True)
    res["lr_fp_vader_only"] = cv_lr(np.c_[fp, vad])

    docs = [" ".join(ws) for ws in texts]
    tfidf = make_pipeline(TfidfVectorizer(min_df=5, max_features=50000, ngram_range=(1, 2), sublinear_tf=True),
                          LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0))
    prob = cross_val_predict(tfidf, docs, y, cv=skf, method="predict_proba")[:, 1]
    res["tfidf_lr"] = {"auc": float(roc_auc_score(y, prob)), "f1": float(f1_score(y, prob >= 0.5))}
    print("tfidf", res["tfidf_lr"], "fp", res["auc_first_person_rate"], "vader", res["auc_neg_vader"])
    (RESULTS / "lexicon_erisk_eval.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("clpsych", "all"):
        clpsych()
    if what in ("erisk", "all"):
        erisk()
