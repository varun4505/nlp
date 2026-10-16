"""Extract and cache per-post features for every CLPsych timeline.

Variants switch single components off so their contribution can be measured.
Features are label-free, so extracting them once outside the CV loop is safe.
"""
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftlens.data import load_clpsych  # noqa: E402
from driftlens.paths import DATA_PROCESSED  # noqa: E402
from driftlens.resources import extractor  # noqa: E402

VARIANTS = {
    "full": {},
    "seeds": {"lex_variant": "seeds"},
    "no_wsd": {"use_wsd": False},
    "no_negation": {"use_negation": False},
    "no_speller": {"use_speller": False},
}


def compact(tl, an, X):
    return {
        "timeline_id": tl.timeline_id, "X": X, "states": tl.states,
        "posts": [{
            "post_id": p.post_id, "wellbeing": p.wellbeing, "evidence": p.evidence, "date": p.date,
            "clean": a["lex"]["clean"], "mal": a["lex"]["mal"], "ada": a["lex"]["ada"], "net": a["lex"]["net"],
            "hits": [{k: h.get(k) for k in ("start", "end", "text", "lemma", "cat", "pol", "w", "negated")}
                     for h in a["lex"]["hits"]],
            "corrections": a["lex"]["corrections"], "self_evals": a["self_evals"],
            "vader": a["vader"], "n_words": a["n_words"], "features": a["features"],
        } for p, a in zip(tl.posts, an)],
    }


def run(name, kwargs, splits):
    fx = extractor(**kwargs)
    for split in splits:
        t0 = time.time()
        out = []
        for tl in load_clpsych(split):
            an, X = fx.timeline([(p.text, p.date) for p in tl.posts])
            out.append(compact(tl, an, X))
        with open(DATA_PROCESSED / f"clpsych_{split}_{name}.pkl", "wb") as fh:
            pickle.dump(out, fh)
        print(f"{name}/{split}: {len(out)} timelines in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(VARIANTS)
    for name in wanted:
        run(name, VARIANTS[name], ["train", "test"] if name == "full" else ["train"])
