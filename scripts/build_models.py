"""Train and save every label-free resource used by DriftLens.

1. WordNet self-state lexicon          -> models/wal_lexicon.json
2. HMM POS tagger on Brown (universal) -> models/hmm_pos.pkl, results/pos_eval.json
3. Background n-gram LM on eRisk text  -> models/bg_lm.pkl,  results/lm_eval.json
4. Spelling dictionary + unigram prior -> models/speller.pkl

None of these touch the CLPsych labels, so they can be reused inside every
cross-validation fold without leakage. CLPsych *text* (no labels) is used
only as the perplexity test set.
"""
import json
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nltk  # noqa: E402
from nltk.corpus import brown, treebank, words as nltk_words, wordnet as wn  # noqa: E402

from driftlens.data import load_clpsych, load_erisk_users  # noqa: E402
from driftlens.lexicon import build_lexicon  # noqa: E402
from driftlens.ngram import NGramLM  # noqa: E402
from driftlens.paths import MODELS, RESULTS  # noqa: E402
from driftlens.postag import HMMTagger  # noqa: E402
from driftlens.textproc import CONTRACTIONS, lm_tokens  # noqa: E402

SEED = 13
BG_TOKENS = 1_500_000
DEV_TOKENS = 60_000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_lex():
    lx = build_lexicon()
    (MODELS / "wal_lexicon.json").write_text(json.dumps(lx, indent=1), encoding="utf-8")
    log(f"lexicon: {len(lx['entries'])} words, {len(lx['vetoed'])} vetoed")


def build_pos():
    sents = list(brown.tagged_sents(tagset="universal"))
    random.Random(SEED).shuffle(sents)
    cut = int(0.9 * len(sents))
    train, test = sents[:cut], sents[cut:]
    tb = list(treebank.tagged_sents(tagset="universal"))
    tb = [[(w, t) for w, t in s if t != "X" or not w.startswith("*")] for s in tb]   # drop traces
    tagger = HMMTagger().fit(train)
    res = {"train_sentences": len(train), "train_tokens": sum(map(len, train)),
           "hmm_brown_test": tagger.evaluate(test), "hmm_treebank": tagger.evaluate(tb)}

    def perceptron_acc(data):
        pred = nltk.pos_tag_sents([[w for w, _ in s] for s in data], tagset="universal")
        c = t = 0
        for gs, ps in zip(data, pred):
            for (_, g), (_, p) in zip(gs, ps):
                c += g == p
                t += 1
        return c / t

    res["perceptron_brown_test"] = perceptron_acc(test)
    res["perceptron_treebank"] = perceptron_acc(tb)
    with open(MODELS / "hmm_pos.pkl", "wb") as fh:
        pickle.dump(tagger, fh)
    (RESULTS / "pos_eval.json").write_text(json.dumps(res, indent=1))
    log(f"POS: {json.dumps(res)}")


def build_lm_and_speller():
    users = load_erisk_users()
    rng = random.Random(SEED)
    rng.shuffle(users)
    texts = [t for u in users for t in u["texts"]]
    rng.shuffle(texts)
    train, dev, n_tr, n_dev = [], [], 0, 0
    for t in texts:
        toks = lm_tokens(t)
        if not toks:
            continue
        if n_dev < DEV_TOKENS:
            dev.append(toks)
            n_dev += len(toks)
        elif n_tr < BG_TOKENS:
            train.append(toks)
            n_tr += len(toks)
        else:
            break
    log(f"LM corpus: {n_tr} train tokens, {n_dev} dev tokens")
    clp = [lm_tokens(p.text) for split in ("train", "test") for tl in load_clpsych(split) for p in tl.posts]
    clp = [c for c in clp if c]

    res = {"train_tokens": n_tr, "dev_tokens": n_dev, "clpsych_test_tokens": sum(map(len, clp)), "rows": []}
    k_grid = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    for n in (1, 2, 3):
        lm = NGramLM(n=n, min_count=2).fit(train)
        res["vocab"] = lm.V
        pp1 = lm.perplexity(clp, method="addk", k=1.0)
        res["rows"].append({"model": f"{n}-gram", "smoothing": "Laplace (add-1)", "k": 1.0,
                            "dev_pp": lm.perplexity(dev, method="addk", k=1.0), "clpsych_pp": pp1})
        best_k = min(k_grid, key=lambda k: lm.perplexity(dev, method="addk", k=k))
        res["rows"].append({"model": f"{n}-gram", "smoothing": "add-k (k tuned on dev)", "k": best_k,
                            "dev_pp": lm.perplexity(dev, method="addk", k=best_k),
                            "clpsych_pp": lm.perplexity(clp, method="addk", k=best_k)})
        if n >= 2:
            dev_pp, lam = lm.tune_lambdas(dev)
            res["rows"].append({"model": f"{n}-gram", "smoothing": "Jelinek-Mercer interpolation",
                                "lambdas": lam, "dev_pp": dev_pp,
                                "clpsych_pp": lm.perplexity(clp, method="interp")})
        log(f"LM n={n}: " + json.dumps(res["rows"][-1]))
        if n == 2:
            with open(MODELS / "bg_lm.pkl", "wb") as fh:
                pickle.dump(lm, fh)
            bigram = lm
        del lm
    (RESULTS / "lm_eval.json").write_text(json.dumps(res, indent=1))

    # ---------------------------------------------------------- speller
    corpus_counts = Counter(w for s in train for w in s if w.isalpha())
    brown_counts = Counter(w.lower() for w in brown.words() if w.isalpha())
    dictionary = set(brown_counts) | {w.lower() for w in nltk_words.words()}
    dictionary |= {l for l in wn.all_lemma_names() if l.isalpha()}
    dictionary |= {w for w, c in corpus_counts.items() if c >= 20}
    dictionary |= {t for v in CONTRACTIONS.values() for t in v.replace("'", " ").split()}
    counts = Counter({w: c for w, c in (corpus_counts + brown_counts).items() if w in dictionary})
    with open(MODELS / "speller.pkl", "wb") as fh:
        pickle.dump({"counts": dict(counts), "dictionary": dictionary}, fh)
    log(f"speller: dictionary {len(dictionary)} words, prior over {len(counts)} types")
    return bigram


if __name__ == "__main__":
    build_lex()
    build_pos()
    build_lm_and_speller()
    log("done")
