"""Fit the timeline models on all 30 labelled CLPsych timelines for the prototype."""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftlens.features import FEATURES  # noqa: E402
from driftlens.paths import DATA_PROCESSED, MODELS  # noqa: E402
from driftlens.sequence import CRF, GaussianHMM, HybridHMM, MaxEnt  # noqa: E402


def main():
    with open(DATA_PROCESSED / "clpsych_train_full.pkl", "rb") as fh:
        data = pickle.load(fh)
    X, y = [d["X"] for d in data], [d["states"] for d in data]
    models = {
        "Gaussian HMM": GaussianHMM().fit(X, y),
        "CRF (prior-corrected)": CRF(FEATURES, decode="prior_corrected").fit(X, y),
        "Hybrid HMM (balanced)": HybridHMM(balanced=True).fit(X, y),
        "MaxEnt": MaxEnt().fit(X, y),
    }
    me = models["MaxEnt"]
    coef = {c: dict(zip(FEATURES, me.clf.coef_[i].round(3).tolist())) for i, c in enumerate(me.clf.classes_)}
    with open(MODELS / "timeline_models.pkl", "wb") as fh:
        pickle.dump({"models": models, "maxent_coef": coef, "features": FEATURES}, fh)
    print("saved", list(models))


if __name__ == "__main__":
    main()
