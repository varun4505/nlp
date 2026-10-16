"""Paper / slide figures and the extra analyses they need.

Colours: the reference categorical palette (validated for colour-vision
deficiency). States use slots 1-3 with a neutral grey for "no change".
"""
import json
import pickle
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftlens.evaluation import paired_bootstrap, scores  # noqa: E402
from driftlens.features import FEATURES  # noqa: E402
from driftlens.paths import DATA_PROCESSED, FIGURES, RESULTS  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"
GREY, INK, INK2, GRID = "#8f8d87", "#0b0b0b", "#52514e", "#e4e3df"
STATE_C = {"O": GREY, "S": BLUE, "E": ORANGE, "B": AQUA}
STATE_N = {"O": "No change", "S": "Switch", "E": "Escalation", "B": "Switch + escalation"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight",
})
W = 4.8   # LNCS text width in inches


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=220)
    plt.close(fig)
    print("saved", name)


def fig_models(r):
    order = ["Majority (all O)", "MaxEnt (length only)", "MaxEnt", "Gaussian HMM", "Hybrid HMM (prior-divided)",
             "Hybrid HMM (balanced)", "CRF (Viterbi)", "CRF (prior-corrected)"]
    rows = dict(r["main"])
    rows["MaxEnt (length only)"] = r["length_only"]["MaxEnt"]
    fig, ax = plt.subplots(figsize=(W, 2.7))
    y = np.arange(len(order))[::-1]
    for key, c, lab, off in (("f1_switch", BLUE, "F1 switch", 0.18), ("f1_escalation", ORANGE, "F1 escalation", 0.0),
                             ("macro_f1", INK, "Macro-F1 (mean ± sd)", -0.18)):
        m = [rows[o][key][0] for o in order]
        s = [rows[o][key][1] for o in order]
        if key == "macro_f1":
            ax.errorbar(m, y + off, xerr=s, fmt="D", ms=4.5, color=c, ecolor=c, elinewidth=1, capsize=2, label=lab)
        else:
            ax.plot(m, y + off, "o", ms=5, color=c, mec="white", mew=1, label=lab)
    ax.axvline(r["length_only"]["MaxEnt"]["macro_f1"][0], color=INK2, lw=0.8, ls=(0, (2, 2)))
    ax.text(r["length_only"]["MaxEnt"]["macro_f1"][0] + 0.008, -0.35, "length-only\nmacro-F1",
            color=INK2, fontsize=6.5, va="center")
    ax.set_yticks(y, order)
    ax.set_xlim(-0.02, 0.75)
    ax.set_xlabel("Score (pooled out-of-fold predictions, 5 × 5-fold CV)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3, fontsize=7)
    ax.grid(axis="y", visible=False)
    save(fig, "fig_models")


def fig_ablation(r):
    full = r["main"]
    names = ["-LEN", "-LEN and z_len", "-LEX", "-SENT", "-SELF", "-DRIFT", "-TIME", "-PERS"]
    labels = ["− length", "− all length features", "− lexicon", "− VADER", "− self-evaluation", "− LM drift",
              "− time", "− personal baselines"]
    fig, ax = plt.subplots(figsize=(W, 2.3))
    y = np.arange(len(names))[::-1]
    for model, c, off in (("Gaussian HMM", BLUE, 0.13), ("CRF (prior-corrected)", ORANGE, -0.13),
                          ("MaxEnt", GREY, 0.0)):
        d = [r["ablation"][n][model]["macro_f1"][0] - full[model]["macro_f1"][0] for n in names]
        ax.plot(d, y + off, "o", ms=5, color=c, mec="white", mew=1, label=model)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Δ macro-F1 when the feature group is removed")
    ax.legend(loc="lower left", fontsize=7)
    ax.grid(axis="y", visible=False)
    save(fig, "fig_ablation")


def fig_signals():
    with open(DATA_PROCESSED / "clpsych_train_full.pkl", "rb") as fh:
        data = pickle.load(fh)
    X = np.vstack([d["X"] for d in data])
    st = np.array([s for d in data for s in d["states"]])
    keep = X[:, FEATURES.index("is_first")] == 0
    panels = [("novelty", "Personal LM novelty\n(bits/token)"), ("z_net", "Self-state balance\n(z within user)"),
              ("z_len", "Post length\n(z within user)")]
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.9))
    for ax, (f, title) in zip(axes, panels):
        col = X[:, FEATURES.index(f)]
        vals = [col[(st == s) & keep] for s in "OSEB"]
        bp = ax.boxplot(vals, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=INK2, lw=0.8),
                        capprops=dict(color=INK2, lw=0.8), boxprops=dict(lw=0))
        for patch, s in zip(bp["boxes"], "OSEB"):
            patch.set_facecolor(STATE_C[s])
            patch.set_alpha(0.85)
        ax.set_xticks([1, 2, 3, 4], ["O", "S", "E", "B"])
        ax.set_title(title, fontsize=7.2)
        ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_signals")


def fig_transitions(r):
    A = np.array(r["transition_matrix"])
    fig, ax = plt.subplots(figsize=(2.3, 2.0))
    im = ax.imshow(A, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "b", ["#f3f7fd", "#9ec5f4", "#2a78d6", "#0d366b"]), vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if A[i, j] > 0.5 else INK)
    ax.set_xticks(range(4), list("OSEB"))
    ax.set_yticks(range(4), list("OSEB"))
    ax.set_xlabel("state at t")
    ax.set_ylabel("state at t−1")
    ax.grid(False)
    save(fig, "fig_transitions")


def fig_lm():
    r = json.loads((RESULTS / "lm_eval.json").read_text())
    groups = ["1-gram", "2-gram", "3-gram"]
    kinds = [("Laplace (add-1)", GREY), ("add-k (k tuned on dev)", BLUE), ("Jelinek-Mercer interpolation", ORANGE)]
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.0))
    x = np.arange(3)
    for i, (k, c) in enumerate(kinds):
        vals = []
        for g in groups:
            m = [row["clpsych_pp"] for row in r["rows"] if row["model"] == g and row["smoothing"] == k]
            vals.append(m[0] if m else np.nan)
        pos = x + (i - 1) * 0.26
        ax.bar(pos, vals, width=0.22, color=c, label=k.replace(" (k tuned on dev)", ""))
        for p, v in zip(pos, vals):
            if not np.isnan(v):
                ax.text(p, v * 1.08, f"{v:.0f}", ha="center", fontsize=6, color=INK2)
    ax.set_yscale("log")
    ax.set_xticks(x, groups)
    ax.set_ylabel("Perplexity (log scale)")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_lm")


def fig_lexicon():
    c = json.loads((RESULTS / "lexicon_clpsych_eval.json").read_text())
    e = json.loads((RESULTS / "lexicon_erisk_eval.json").read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 1.9), gridspec_kw=dict(width_ratios=[1, 1.15]))
    rows = [("VADER", c["full"]["spearman_vader"], c["full"]["spearman_vader_ci"], GREY),
            ("Seeds only", c["seeds"]["spearman_net"], c["seeds"]["spearman_net_ci"], GREY),
            ("WordNet-grown", c["full"]["spearman_net"], c["full"]["spearman_net_ci"], BLUE)]
    y = np.arange(len(rows))[::-1]
    for yy, (lab, v, ci, col) in zip(y, rows):
        a1.errorbar(v, yy, xerr=[[v - ci[0]], [ci[1] - v]], fmt="o", ms=5, color=col, elinewidth=1, capsize=2)
        a1.text(ci[1] + 0.015, yy, f"{v:.2f}", va="center", fontsize=6.5, color=INK2)
    a1.set_yticks(y, [r[0] for r in rows])
    a1.set_xlim(0, 0.68)
    a1.set_title("CLPsych: Spearman ρ with well-being", fontsize=7.2)
    a1.grid(axis="y", visible=False)
    rows2 = [("VADER (−compound)", e["auc_neg_vader"], GREY), ("First-person rate", e["auc_first_person_rate"], GREY),
             ("Seeds: maladaptive", e["variants"]["seeds"]["auc_mal"], GREY),
             ("WordNet-grown: maladaptive", e["variants"]["full"]["auc_mal"], BLUE),
             ("4 scores + LR", e["variants"]["full"]["lr_4_scores"]["auc"], BLUE),
             ("TF-IDF + LR (supervised)", e["tfidf_lr"]["auc"], ORANGE)]
    y = np.arange(len(rows2))[::-1]
    a2.barh(y, [r[1] - 0.5 for r in rows2], left=0.5, height=0.5, color=[r[2] for r in rows2])
    for yy, r in zip(y, rows2):
        a2.text(r[1] + 0.01, yy, f"{r[1]:.2f}", va="center", fontsize=6.5, color=INK2)
    a2.set_yticks(y, [r[0] for r in rows2], fontsize=6.8)
    a2.set_xlim(0.5, 1.05)
    a2.set_title("eRisk 2026: ROC-AUC, depressed vs control", fontsize=7.2)
    a2.grid(axis="y", visible=False)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_lexicon")


def fig_example(oof, tid="c62c65f481", model="CRF (prior-corrected)"):
    with open(DATA_PROCESSED / "clpsych_train_full.pkl", "rb") as fh:
        data = pickle.load(fh)
    i = oof["ids"].index(tid)
    d = data[i]
    pred = oof["main"][model][0][i]
    gold = d["states"]
    X = d["X"]
    x = np.arange(1, len(gold) + 1)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 2.4), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    a1.plot(x, X[:, FEATURES.index("z_net")], "-o", color=VIOLET, lw=2, ms=4, mec="white", label="self-state balance (z)")
    a1.plot(x, X[:, FEATURES.index("z_novelty")], "-o", color=YELLOW, lw=2, ms=4, mec="white",
            label="personal LM novelty (z)")
    a1.axhline(0, color=INK2, lw=0.6)
    a1.legend(ncol=2, loc="upper left", fontsize=6.8)
    a1.set_ylabel("z within user")
    for row, seq in ((1, gold), (0, pred)):
        a2.scatter(x, [row] * len(x), c=[STATE_C[s] for s in seq], marker="s", s=70)
    a2.set_yticks([0, 1], ["predicted\n(out-of-fold)", "gold"])
    a2.set_ylim(-0.6, 1.6)
    a2.set_xlabel("post number")
    a2.grid(False)
    handles = [plt.Line2D([], [], marker="s", ls="", color=STATE_C[s], label=STATE_N[s]) for s in "OSEB"]
    a2.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.55), fontsize=6.5)
    save(fig, "fig_example")


def extra_analyses(r, oof):
    """Length-stratified scores incl. the length-only baseline, and bootstrap vs length-only for the CRF."""
    with open(DATA_PROCESSED / "clpsych_train_full.pkl", "rb") as fh:
        data = pickle.load(fh)
    y = oof["y"]
    lens = np.array([row[FEATURES.index("len_log")] for d in data for row in d["X"]])
    med = np.median(lens)
    gold = [s for ys in y for s in ys]
    out = {"median_words": float(np.expm1(med))}
    systems = {"MaxEnt (length only)": oof["length_only"]["MaxEnt"], "MaxEnt": oof["main"]["MaxEnt"],
               "Gaussian HMM": oof["main"]["Gaussian HMM"], "CRF (prior-corrected)": oof["main"]["CRF (prior-corrected)"]}
    for name, preds_list in systems.items():
        res = {"short": [], "long": []}
        for preds in preds_list:
            flat = [s for ps in preds for s in ps]
            for part, idx in (("short", np.where(lens <= med)[0]), ("long", np.where(lens > med)[0])):
                res[part].append(scores([gold[i] for i in idx], [flat[i] for i in idx])["macro_f1"])
        out[name] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in res.items()}
    boot = {}
    for a in ("CRF (prior-corrected)",):
        for metric in ("macro_f1", "f1_escalation"):
            diff, p = paired_bootstrap(y, oof["main"][a], oof["length_only"]["MaxEnt"], metric=metric, n_boot=500)
            boot[f"{a} vs MaxEnt (length only) [{metric}]"] = {"diff": diff, "p": p}
    out["bootstrap_extra"] = boot
    (RESULTS / "extra_analyses.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def ieee_figures(r):
    """Single-column (3.45 in) versions for the IEEE two-column paper."""
    WI = 3.45
    c = json.loads((RESULTS / "lexicon_clpsych_eval.json").read_text())
    e = json.loads((RESULTS / "lexicon_erisk_eval.json").read_text())
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(WI, 2.9), gridspec_kw=dict(height_ratios=[1, 1.7]))
    rows = [("VADER", c["full"]["spearman_vader"], c["full"]["spearman_vader_ci"], GREY),
            ("Seeds only", c["seeds"]["spearman_net"], c["seeds"]["spearman_net_ci"], GREY),
            ("WordNet-grown", c["full"]["spearman_net"], c["full"]["spearman_net_ci"], BLUE)]
    y = np.arange(len(rows))[::-1]
    for yy, (lab, v, ci, col) in zip(y, rows):
        a1.errorbar(v, yy, xerr=[[v - ci[0]], [ci[1] - v]], fmt="o", ms=4.5, color=col, elinewidth=1, capsize=2)
        a1.text(ci[1] + 0.012, yy, f"{v:.2f}", va="center", fontsize=6.5, color=INK2)
    a1.set_yticks(y, [r_[0] for r_ in rows], fontsize=7)
    a1.set_xlim(0, 0.62)
    a1.set_title("(a) CLPsych: Spearman ρ with well-being", fontsize=7.2, loc="left")
    a1.grid(axis="y", visible=False)
    rows2 = [("VADER (−compound)", e["auc_neg_vader"], GREY), ("First-person rate", e["auc_first_person_rate"], GREY),
             ("Seeds: maladaptive", e["variants"]["seeds"]["auc_mal"], GREY),
             ("WordNet-grown: maladaptive", e["variants"]["full"]["auc_mal"], BLUE),
             ("4 scores + LR", e["variants"]["full"]["lr_4_scores"]["auc"], BLUE),
             ("TF-IDF + LR (supervised)", e["tfidf_lr"]["auc"], ORANGE)]
    y = np.arange(len(rows2))[::-1]
    a2.barh(y, [r_[1] - 0.5 for r_ in rows2], left=0.5, height=0.55, color=[r_[2] for r_ in rows2])
    for yy, r_ in zip(y, rows2):
        a2.text(r_[1] + 0.008, yy, f"{r_[1]:.2f}", va="center", fontsize=6.5, color=INK2)
    a2.set_yticks(y, [r_[0] for r_ in rows2], fontsize=7)
    a2.set_xlim(0.5, 1.03)
    a2.set_title("(b) eRisk 2026: ROC-AUC, depressed vs control", fontsize=7.2, loc="left")
    a2.grid(axis="y", visible=False)
    fig.tight_layout(h_pad=0.8)
    save(fig, "fig_lexicon_ieee")

    full = r["main"]
    names = ["-LEN", "-LEN and z_len", "-LEX", "-SENT", "-SELF", "-DRIFT", "-TIME", "-PERS"]
    labels = ["− length", "− all length", "− lexicon", "− VADER", "− self-eval.", "− LM drift", "− time",
              "− personal z"]
    fig, ax = plt.subplots(figsize=(WI, 2.2))
    y = np.arange(len(names))[::-1]
    for model, col, off, lab in (("Gaussian HMM", BLUE, 0.15, "Gaussian HMM"),
                                 ("CRF (prior-corrected)", ORANGE, -0.15, "CRF (prior-corr.)"),
                                 ("MaxEnt", GREY, 0.0, "MaxEnt")):
        d = [r["ablation"][nm][model]["macro_f1"][0] - full[model]["macro_f1"][0] for nm in names]
        ax.plot(d, y + off, "o", ms=4.5, color=col, mec="white", mew=0.8, label=lab)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlabel("Δ macro-F1 when the group is removed", fontsize=7)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, fontsize=6.3, handletextpad=0.2,
              columnspacing=0.8)
    ax.grid(axis="y", visible=False)
    save(fig, "fig_ablation_ieee")


if __name__ == "__main__":
    if "--ieee" in sys.argv:
        ieee_figures(json.loads((RESULTS / "sequence_eval.json").read_text()))
        sys.exit(0)
    r = json.loads((RESULTS / "sequence_eval.json").read_text())
    with open(RESULTS / "sequence_oof_predictions.pkl", "rb") as fh:
        oof = pickle.load(fh)
    fig_models(r)
    fig_ablation(r)
    fig_signals()
    fig_transitions(r)
    fig_lm()
    fig_lexicon()
    fig_example(oof)
    if "--skip-extra" not in sys.argv:
        extra_analyses(r, oof)
