"""Load the saved label-free resources and wire up a FeatureExtractor."""
import pickle
from functools import lru_cache

from .features import FeatureExtractor
from .lexicon import Lexicon
from .paths import MODELS
from .spelling import NoisyChannelSpeller


@lru_cache(maxsize=None)
def bg_lm():
    with open(MODELS / "bg_lm.pkl", "rb") as fh:
        return pickle.load(fh)


@lru_cache(maxsize=None)
def tagger():
    with open(MODELS / "hmm_pos.pkl", "rb") as fh:
        return pickle.load(fh)


@lru_cache(maxsize=None)
def speller():
    with open(MODELS / "speller.pkl", "rb") as fh:
        d = pickle.load(fh)
    return NoisyChannelSpeller(d["counts"], d["dictionary"], lm=bg_lm())


def lexicon(variant="full"):
    if variant == "seeds":
        return Lexicon.seeds_only()
    return Lexicon.load(MODELS / "wal_lexicon.json", use_phrases=(variant != "no_phrases"))


def extractor(lex_variant="full", use_speller=True, use_wsd=True, use_negation=True):
    return FeatureExtractor(lexicon(lex_variant), tagger(), bg_lm(),
                            speller=speller() if use_speller else None,
                            use_wsd=use_wsd, use_negation=use_negation)
