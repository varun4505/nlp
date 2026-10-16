"""Moment-of-change sequence labelling experiments on the CLPsych training timelines.

Protocol: 5-fold cross-validation at the timeline level (no user appears in
both train and test), repeated with 5 different shuffles. Hyper-parameters are
library defaults fixed before running (C=1 for MaxEnt, c1=c2=0.1 for the CRF),
because 30 timelines are too few for a nested tuning loop.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftlens.evaluation import grouped_cv, paired_bootstrap, scores, summarise  # noqa: E402
from driftlens.features import FEATURES, GROUPS  # noqa: E402
from driftlens.paths import DATA_PROCESSED, RESULTS  # noqa: E402
from driftlens.sequence import CRF, GaussianHMM, HybridHMM, Majority, MaxEnt  # noqa: E402

REPEATS, FOLDS = 5, 5


def load(variant="full"):
    with open(DATA_PROCESSED / f"clpsych_train_{variant}.pkl", "rb") as fh:
        data = pickle.load(fh)
    return [d["X"] for d in data], [d["states"] for d in data], data


def cols(names):
    return [FEATURES.index(n) for n in names]


def subset(X_seqs, names):
    idx = cols(names)
    return [X[:, idx] for X in X_seqs]


def factories(names):
    return {
        "Majority (all O)": lambda: Majority(),
        "MaxEnt": lambda: MaxEnt(),
        "Gaussian HMM": lambda: GaussianHMM(),
        "Hybrid HMM (prior-divided)": lambda: HybridHMM(balanced=False),
        "Hybrid HMM (balanced)": lambda: HybridHMM(balanced=True),
        "CRF (Viterbi)": lambda: CRF(names),
        "CRF (prior-corrected)": lambda: CRF(names, decode="prior_corrected"),
    }


ABL_MODELS = {"MaxEnt", "Gaussian HMM", "Hybrid HMM (balanced)", "CRF (prior-corrected)"}
SEQ_MODELS = ["Gaussian HMM", "Hybrid HMM (balanced)", "Hybrid HMM (prior-divided)",
              "CRF (prior-corrected)", "CRF (Viterbi)"]


def run_models(X_seqs, y_seqs, names, which=None):
    out, preds = {}, {}
    for name, fac in factories(names).items():
        if which and name not in which:
            continue
        m, p = grouped_cv(subset(X_seqs, names), y_seqs, fac, FOLDS, REPEATS)
        out[name] = summarise(m)
        preds[name] = p
        print(f"  {name:22s} macroF1 {out[name]['macro_f1'][0]:.3f}±{out[name]['macro_f1'][1]:.3f} "
              f"S {out[name]['f1_switch'][0]:.3f} E {out[name]['f1_escalation'][0]:.3f}", flush=True)
    return out, preds


def main():
    X_seqs, y_seqs, data = load("full")
    res = {"n_timelines": len(y_seqs), "n_posts": sum(map(len, y_seqs))}

    print("Main comparison (all features)")
    res["main"], preds = run_models(X_seqs, y_seqs, FEATURES)

    print("Length-only baselines")
    res["length_only"], preds_len = run_models(X_seqs, y_seqs, GROUPS["LEN"],
                                               which={"MaxEnt", "Gaussian HMM", "Hybrid HMM (balanced)"})

    print("Feature-group ablations")
    res["ablation"] = {}
    for g in GROUPS:
        keep = [f for f in FEATURES if f not in GROUPS[g]]
        print(f" - without {g}")
        res["ablation"][f"-{g}"], _ = run_models(X_seqs, y_seqs, keep, which=ABL_MODELS)
    no_len = [f for f in FEATURES if f not in ("len_log", "z_len")]
    print(" - without any length feature")
    res["ablation"]["-LEN and z_len"], _ = run_models(X_seqs, y_seqs, no_len, which=ABL_MODELS)
    for g in ("LEX", "DRIFT", "SENT", "PERS"):
        print(f" - only {g}")
        res["ablation"][f"only {g}"], _ = run_models(X_seqs, y_seqs, GROUPS[g], which=ABL_MODELS)

    print("Lexicon variants (all features, lexicon built differently)")
    res["lexicon_variants"] = {}
    for v in ("seeds", "no_wsd", "no_negation", "no_speller"):
        Xv, yv, _ = load(v)
        print(f" - {v}")
        res["lexicon_variants"][v], _ = run_models(Xv, yv, FEATURES,
                                                   which={"Hybrid HMM (balanced)", "Gaussian HMM"})

    print("Paired bootstrap over timelines")
    res["bootstrap"] = {}
    best = max(SEQ_MODELS, key=lambda m: res["main"][m]["macro_f1"][0])
    res["best_sequence_model"] = best
    for other, p_other in (("MaxEnt", preds["MaxEnt"]), ("MaxEnt (length only)", preds_len["MaxEnt"]),
                           ("Majority", preds["Majority (all O)"])):
        for metric in ("macro_f1", "f1_escalation", "f1_switch"):
            diff, p = paired_bootstrap(y_seqs, preds[best], p_other, metric=metric, n_boot=1000)
            res["bootstrap"][f"{best} vs {other} [{metric}]"] = {"diff": diff, "p": p}
            print(f"  {best} vs {other} [{metric}]: diff {diff:+.3f}, p={p:.3f}")

    # length-stratified evaluation, recommended by the dataset analysis report
    lens = np.array([f for X in X_seqs for f in X[:, FEATURES.index("len_log")]])
    med = float(np.median(lens))
    gold = [s for ys in y_seqs for s in ys]
    res["length_stratified"] = {"median_len_log": med}
    for name in ("MaxEnt", best):
        for r_i, preds_r in enumerate(preds[name][:1]):
            flat = [s for ps in preds_r for s in ps]
            short = [i for i, l in enumerate(lens) if l <= med]
            long_ = [i for i, l in enumerate(lens) if l > med]
            res["length_stratified"][name] = {
                "short": scores([gold[i] for i in short], [flat[i] for i in short]),
                "long": scores([gold[i] for i in long_], [flat[i] for i in long_])}

    # per-state distribution of the drift and affect signals (descriptive statistics)
    res["signal_by_state"] = {}
    feats = np.vstack(X_seqs)
    states = np.array(gold)
    for f in ("novelty", "vocab_novelty", "net", "abs_d_net_hist", "vader", "len_log"):
        col = feats[:, FEATURES.index(f)]
        first = feats[:, FEATURES.index("is_first")] == 0     # novelty undefined for first posts
        row = {}
        for s in ("O", "S", "E", "B"):
            v = col[(states == s) & first]
            row[s] = {"median": float(np.median(v)), "mean": float(v.mean()), "n": int(len(v))}
        chg = col[(states != "O") & first]
        none = col[(states == "O") & first]
        row["mannwhitney_change_vs_none_p"] = float(mannwhitneyu(chg, none).pvalue)
        res["signal_by_state"][f] = row

    # learned transition matrix of the HMM trained on everything (for the paper figure)
    hmm = GaussianHMM().fit(X_seqs, y_seqs)
    res["transition_matrix"] = np.exp(hmm.log_A).round(3).tolist()

    (RESULTS / "sequence_eval.json").write_text(json.dumps(res, indent=1))
    with open(RESULTS / "sequence_oof_predictions.pkl", "wb") as fh:
        pickle.dump({"main": preds, "length_only": preds_len, "y": y_seqs,
                     "ids": [d["timeline_id"] for d in data]}, fh)
    print("saved results/sequence_eval.json")


if __name__ == "__main__":
    main()
