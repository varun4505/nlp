"""Intrinsic evaluation of the noisy-channel spelling corrector.

Real words are sampled (with their neighbours) from CLPsych post text and one
synthetic edit is applied, keeping only corruptions that are non-words. The
generator mixes keyboard-adjacent substitutions with random ones so the
keyboard-weighted channel is not tested only on errors it was designed for:

  adjacent-key substitution 25 %, random substitution 15 %, deletion 20 %,
  insertion 15 % (doubling or random letter), transposition 25 %.

The channel weight alpha is tuned on a dev split (500 items) and accuracy is
reported on the remaining test items.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftlens.data import load_clpsych  # noqa: E402
from driftlens.paths import RESULTS  # noqa: E402
from driftlens.resources import speller  # noqa: E402
from driftlens.spelling import LETTERS, ROWS, keys_adjacent  # noqa: E402
from driftlens.textproc import lm_tokens  # noqa: E402

N_ITEMS, N_DEV, SEED = 1500, 500, 7
ALPHAS = [1, 2, 3, 4, 6, 8]


def corrupt(w, rng):
    i = rng.randrange(len(w))
    r = rng.random()
    if r < 0.25:
        near = [c for c in "".join(ROWS) if keys_adjacent(w[i], c)]
        return w[:i] + rng.choice(near) + w[i + 1:] if near else None, "sub-adjacent"
    if r < 0.40:
        return w[:i] + rng.choice([c for c in LETTERS if c != w[i]]) + w[i + 1:], "sub-random"
    if r < 0.60:
        return w[:i] + w[i + 1:], "deletion"
    if r < 0.75:
        c = w[i] if rng.random() < 0.5 else rng.choice(LETTERS)
        return w[:i] + c + w[i:], "insertion"
    if i + 1 < len(w) and w[i] != w[i + 1]:
        return w[:i] + w[i + 1] + w[i] + w[i + 2:], "transposition"
    return None, None


def build_items(sp):
    rng = random.Random(SEED)
    posts = [lm_tokens(p.text) for s in ("train", "test") for tl in load_clpsych(s) for p in tl.posts]
    pool = [(toks, i) for toks in posts for i, t in enumerate(toks)
            if t.isalpha() and len(t) >= 4 and t in sp.dictionary]
    rng.shuffle(pool)
    items = []
    for toks, i in pool:
        bad, kind = corrupt(toks[i], rng)
        if bad and not sp.in_dictionary(bad):
            items.append({"gold": toks[i], "typo": bad, "kind": kind,
                          "prev": toks[i - 1] if i else None,
                          "next": toks[i + 1] if i + 1 < len(toks) else None})
        if len(items) == N_ITEMS:
            break
    return items, posts


def accuracy(sp, items, weighted, context, keep=False):
    ok = kept = 0
    by_kind = {}
    for it in items:
        ranked = sp.rank(it["typo"], it["prev"], it["next"], weighted=weighted, context=context, keep=keep)
        pred = ranked[0][0] if ranked else it["typo"]
        hit = pred == it["gold"]
        ok += hit
        kept += pred == it["typo"]
        k = by_kind.setdefault(it["kind"], [0, 0])
        k[0] += hit
        k[1] += 1
    return ok / len(items), kept / len(items), {k: v[0] / v[1] for k, v in by_kind.items()}


def main():
    sp = speller()
    items, posts = build_items(sp)
    dev, test = items[:N_DEV], items[N_DEV:]
    configs = {
        "Levenshtein + unigram prior": dict(weighted=False, context=False),
        "Weighted DL + unigram prior": dict(weighted=True, context=False),
        "Weighted DL + bigram context": dict(weighted=True, context=True),
        "Weighted DL + bigram context + keep option": dict(weighted=True, context=True, keep=True),
    }
    res = {"n_dev": len(dev), "n_test": len(test), "configs": {}}
    for name, cfg in configs.items():
        best_a = max(ALPHAS, key=lambda a: (setattr(sp, "alpha", a), accuracy(sp, dev, **cfg)[0])[1])
        sp.alpha = best_a
        acc, kept, kinds = accuracy(sp, test, **cfg)
        res["configs"][name] = {"alpha": best_a, "accuracy": acc, "left_unchanged": kept, "by_error_type": kinds}
        print(f"{name:45s} alpha={best_a} acc={acc:.3f} unchanged={kept:.3f} {kinds}", flush=True)

    # how often genuine out-of-dictionary tokens in real posts get rewritten
    sp.alpha = res["configs"]["Weighted DL + bigram context + keep option"]["alpha"]
    oov = [(toks[i - 1] if i else None, t, toks[i + 1] if i + 1 < len(toks) else None)
           for toks in posts for i, t in enumerate(toks) if sp.needs_check(t)]
    changed_keep = sum(sp.correct(t, p, n, context=True, keep=True) != t for p, t, n in oov)
    changed_forced = sum(sp.correct(t, p, n, context=True, keep=False) != t for p, t, n in oov)
    res["real_oov_tokens"] = len(oov)
    res["real_oov_changed_with_keep"] = changed_keep / max(len(oov), 1)
    res["real_oov_changed_forced"] = changed_forced / max(len(oov), 1)
    print("real OOV", len(oov), res["real_oov_changed_with_keep"], res["real_oov_changed_forced"])
    (RESULTS / "spelling_eval.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
