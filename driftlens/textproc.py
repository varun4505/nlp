"""Basic text processing for Reddit posts.

Cleaning removes markup that carries no linguistic content, sentence and word
segmentation use NLTK's Punkt and Treebank tokenisers, and two cheap
normalisation rules fix the most frequent Reddit spellings before the noisy
channel corrector sees the text.
"""
import html
import re

import nltk

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
USER_RE = re.compile(r"(?<!\w)/?u/[A-Za-z0-9_-]+")
SUB_RE = re.compile(r"(?<!\w)/?r/[A-Za-z0-9_]+")
ELONG_RE = re.compile(r"([a-z])\1{2,}", re.I)   # "sooooo" -> "soo"
WS_RE = re.compile(r"\s+")

# Apostrophe-less contractions that are frequent on Reddit. Only forms that are
# not ordinary English words are listed ("wont", "ill", "were" are excluded).
CONTRACTIONS = {
    "im": "i'm", "dont": "don't", "cant": "can't", "didnt": "didn't",
    "doesnt": "doesn't", "isnt": "isn't", "ive": "i've", "youre": "you're",
    "thats": "that's", "wasnt": "wasn't", "couldnt": "couldn't",
    "wouldnt": "wouldn't", "shouldnt": "shouldn't", "havent": "haven't",
    "hasnt": "hasn't", "arent": "aren't", "werent": "weren't",
    "theyre": "they're", "idk": "i don't know", "id've": "i'd have",
}

FIRST_PERSON = {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'd", "i'll"}


def clean(text):
    text = html.unescape(text or "")
    text = text.replace("​", " ")
    text = MD_LINK_RE.sub(r"\1", text)
    text = URL_RE.sub(" ", text)
    text = USER_RE.sub(" ", text)
    text = SUB_RE.sub(" ", text)
    text = re.sub(r"[*_~`>#]+", " ", text)          # markdown emphasis / quotes
    text = ELONG_RE.sub(r"\1\1", text)
    return WS_RE.sub(" ", text).strip()


def sentences(text):
    return [s for s in nltk.sent_tokenize(clean(text)) if s.strip()]


def expand_contractions(tokens):
    out = []
    for t in tokens:
        rep = CONTRACTIONS.get(t.lower())
        if rep:
            out.extend(nltk.word_tokenize(rep))
        else:
            out.append(t)
    return out


def words(text, lower=True):
    """Tokenise a (raw) post into word tokens."""
    toks = expand_contractions(nltk.word_tokenize(clean(text)))
    return [t.lower() for t in toks] if lower else toks


def is_word(tok):
    return any(c.isalpha() for c in tok)


def lm_tokens(text):
    """Lower-cased tokens used for the n-gram models (punctuation kept, numbers merged)."""
    return ["<num>" if re.fullmatch(r"[\d.,:/-]+\d", t) else t for t in words(text)]
