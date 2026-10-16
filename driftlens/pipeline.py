"""End-to-end analysis of one timeline, used by the Streamlit prototype."""
import pickle
import re

import nltk
import numpy as np

from .features import FEATURES
from .paths import MODELS
from .resources import extractor
from .sequence import STATES
from .summarize import textrank

STATE_NAMES = {"O": "No change", "S": "Switch", "E": "Escalation", "B": "Switch + escalation"}


class DriftLens:
    def __init__(self):
        self.fx = extractor()
        with open(MODELS / "timeline_models.pkl", "rb") as fh:
            bundle = pickle.load(fh)
        self.models = bundle["models"]
        self.maxent_coef = bundle.get("maxent_coef")

    def analyse(self, posts, model="CRF (prior-corrected)"):
        """posts: list of {"text": str, "date": datetime|None}."""
        an, X = self.fx.timeline([(p["text"], p.get("date")) for p in posts])
        m = self.models[model]
        pred = list(m.predict([X])[0])
        if hasattr(m, "marginals"):
            marg = m.marginals(X)
        else:
            marg = m.proba(X)
        # digest: sentences of the whole timeline, biased to self-state content
        sents, owner, bias = [], [], []
        for i, a in enumerate(an):
            for s in nltk.sent_tokenize(a["lex"]["clean"]):
                if len(s.split()) < 3:
                    continue
                low = s.lower()
                w = sum(1 for h in a["lex"]["hits"] if h["w"] > 0 and h["text"].lower() in low)
                sents.append(s)
                owner.append(i)
                bias.append(w)
        top = textrank(sents, bias, k=min(4, max(1, len(sents) // 4 + 1)))
        return {
            "analyses": an, "X": X, "features": FEATURES, "pred": pred,
            "marginals": np.asarray(marg), "states": STATES,
            "digest": [(owner[i], sents[i]) for i in top],
        }


def split_custom(text):
    """Split pasted text into posts: blank line or a line with '---' separates posts."""
    parts = re.split(r"\n\s*(?:---+)?\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]
