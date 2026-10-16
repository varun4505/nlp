"""Per-post feature extraction for a whole timeline.

Every feature is computed without CLPsych labels, so the extractor can run
once outside the cross-validation loop.

Self-evaluation extraction (information extraction)
---------------------------------------------------------------------
Posts are POS-tagged with the HMM tagger. First-person, copular and negation
words are then re-tagged, and a cascaded regular grammar (NLTK RegexpParser, a
finite-state approximation of a small CFG) chunks clauses such as "I am so
worthless", "I've been feeling hopeful" or "I hate myself". A chunk counts as
negative or positive by the lexicon polarity of its words, with a negation
inside the chunk flipping adaptive words.
"""
import math

import nltk
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from .ngram import personal_novelty, vocab_novelty
from .textproc import FIRST_PERSON, clean, expand_contractions, lm_tokens

COPULA = {"am", "'m", "m", "is", "was", "be", "been", "being", "feel", "feels", "felt", "feeling",
          "seem", "seems", "seemed", "get", "gets", "got", "getting", "become", "became", "look",
          "looked", "sound", "stay", "stayed", "remain", "remained", "keep", "kept"}
NEG_WORDS = {"not", "n't", "never", "no", "hardly", "barely"}
AUX = {"do", "does", "did", "will", "would", "can", "could", "should", "might", "must", "shall",
       "'ll", "'d", "'ve", "ca", "wo"}
SELF_GRAMMAR = r"""
SELF: {<FP><ADV|NEG|PRT>*<AUX|VERB|COP>*<COP><ADV|NEG|PRT>*<ADV|PRT|ADJ|VERB|CONJ|DET>*<ADJ|VERB|NOUN>}
      {<FP><ADV|NEG|AUX>*<VERB><FPO>}
      {<FP><ADV|NEG|AUX|COP>*<VERB>}
"""

FEATURES = ["len_log", "mal", "ada", "net", "crisis", "d_net_hist", "d_net_prev", "d_net_next",
            "abs_d_net_hist", "vader", "d_vader_prev", "d_vader_next", "fp_rate", "self_neg",
            "self_pos", "novelty", "vocab_novelty", "gap_log", "is_first",
            "z_len", "z_mal", "z_net", "z_vader", "z_fp", "z_novelty"]
GROUPS = {
    "LEN": ["len_log"],
    "LEX": ["mal", "ada", "net", "crisis", "d_net_hist", "d_net_prev", "d_net_next", "abs_d_net_hist"],
    "SENT": ["vader", "d_vader_prev", "d_vader_next"],
    "SELF": ["fp_rate", "self_neg", "self_pos"],
    "DRIFT": ["novelty", "vocab_novelty"],
    "TIME": ["gap_log", "is_first"],
    # the same signals expressed relative to the user's own timeline
    "PERS": ["z_len", "z_mal", "z_net", "z_vader", "z_fp", "z_novelty"],
}
PERSONAL_BASE = {"z_len": "len_log", "z_mal": "mal", "z_net": "net", "z_vader": "vader",
                 "z_fp": "fp_rate", "z_novelty": "novelty"}


def retag(tagged):
    out = []
    for w, t in tagged:
        lw = w.lower()
        if lw == "i":
            t = "FP"
        elif lw in ("me", "myself"):
            t = "FPO"
        elif lw in NEG_WORDS:
            t = "NEG"
        elif lw in COPULA:
            t = "COP"
        elif lw in AUX:
            t = "AUX"
        out.append((w, t))
    return out


class FeatureExtractor:
    def __init__(self, lexicon, tagger, bg_lm, speller=None, use_wsd=True, use_negation=True, beta=0.3):
        self.lexicon, self.tagger, self.bg_lm, self.speller = lexicon, tagger, bg_lm, speller
        self.use_wsd, self.use_negation, self.beta = use_wsd, use_negation, beta
        self.vader = SentimentIntensityAnalyzer()
        self.chunker = nltk.RegexpParser(SELF_GRAMMAR)

    # ------------------------------------------------------------- one post
    def self_evaluations(self, tagged):
        evals = []
        tree = self.chunker.parse(retag(tagged))
        for sub in tree.subtrees(lambda t: t.label() == "SELF"):
            words = [w for w, _ in sub.leaves()]
            negated = any(t == "NEG" for _, t in sub.leaves())
            score = 0.0
            for w in words:
                key = self.lexicon.lookup(w.lower())
                if key is None:
                    continue
                pol = self.lexicon.entries[key]["pol"]
                if pol == "ada":
                    score += -1 if negated else 1
                elif not negated:
                    score -= 1
            evals.append({"text": " ".join(words), "polarity": "neg" if score < 0 else "pos" if score > 0 else "neutral"})
        return evals

    def analyse_post(self, text):
        lex = self.lexicon.score(text, speller=self.speller, use_wsd=self.use_wsd,
                                 use_negation=self.use_negation)
        cleaned = lex["clean"]
        tagged = []
        for s in nltk.sent_tokenize(cleaned):
            toks = expand_contractions(nltk.word_tokenize(s))
            tagged.extend(self.tagger.tag(toks))
        words = [w for w, _ in tagged if any(c.isalpha() for c in w)]
        n = max(len(words), 1)
        fp = sum(1 for w in words if w.lower() in FIRST_PERSON)
        evals = self.self_evaluations(tagged)
        return {
            "lex": lex, "tagged": tagged, "self_evals": evals,
            "vader": self.vader.polarity_scores(cleaned)["compound"],
            "n_words": n, "fp_rate": fp / n,
            "self_neg": sum(e["polarity"] == "neg" for e in evals),
            "self_pos": sum(e["polarity"] == "pos" for e in evals),
            "lm_tokens": lm_tokens(text),
        }

    # --------------------------------------------------------- whole timeline
    def timeline(self, posts):
        """posts: list of (text, datetime|None). Returns (analyses, X[T, F])."""
        an = [self.analyse_post(t) for t, _ in posts]
        dates = [d for _, d in posts]
        toks = [a["lm_tokens"] for a in an]
        nov = personal_novelty(toks, self.bg_lm, beta=self.beta)
        stop = set(nltk.corpus.stopwords.words("english"))
        vnov = vocab_novelty(toks, stop)
        net = [a["lex"]["net"] for a in an]
        vad = [a["vader"] for a in an]
        T = len(posts)
        for t, a in enumerate(an):
            hist = float(np.mean(net[:t])) if t else net[t]
            gap = 0.0
            if t and dates[t] and dates[t - 1]:
                gap = max((dates[t] - dates[t - 1]).total_seconds() / 86400.0, 0.0)
            f = {
                "len_log": math.log1p(a["n_words"]),
                "mal": a["lex"]["mal"], "ada": a["lex"]["ada"], "net": net[t],
                "crisis": float(a["lex"]["crisis"] > 0),
                "d_net_hist": net[t] - hist,
                "d_net_prev": net[t] - net[t - 1] if t else 0.0,
                "d_net_next": net[t + 1] - net[t] if t + 1 < T else 0.0,
                "abs_d_net_hist": abs(net[t] - hist),
                "vader": vad[t],
                "d_vader_prev": vad[t] - vad[t - 1] if t else 0.0,
                "d_vader_next": vad[t + 1] - vad[t] if t + 1 < T else 0.0,
                "fp_rate": a["fp_rate"], "self_neg": a["self_neg"], "self_pos": a["self_pos"],
                "novelty": nov[t], "vocab_novelty": vnov[t],
                "gap_log": math.log1p(gap), "is_first": float(t == 0),
            }
            a["features"] = f
        # personal baseline: z-score each signal within this user's timeline, so a
        # post is described by how unusual it is *for this writer*
        for z, base in PERSONAL_BASE.items():
            vals = np.array([a["features"][base] for a in an])
            sd = vals.std()
            for a, v in zip(an, vals):
                a["features"][z] = float((v - vals.mean()) / sd) if sd > 1e-9 else 0.0
        rows = [[a["features"][k] for k in FEATURES] for a in an]
        return an, np.array(rows, dtype=float)
