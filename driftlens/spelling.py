"""Noisy-channel spelling correction for non-word errors.

For an observed token x that is not in the dictionary, the corrector picks

    w* = argmax_w  log P(w | context) + log P(x | w)

P(w | context) is a unigram prior or a bigram context score from the
background language model. The channel P(x | w) is proportional to
exp(-alpha * d(x, w)), where d is a weighted Damerau-Levenshtein distance
(optimal string alignment). Edits are cheaper when they match common typing
slips: adjacent-key substitutions, doubled-letter insertions and deletions, and
transpositions.
"""
import math
import string

ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
_POS = {c: (r, i + 0.5 * r) for r, row in enumerate(ROWS) for i, c in enumerate(row)}
VOWELS = set("aeiou")
LETTERS = string.ascii_lowercase


def keys_adjacent(a, b):
    if a not in _POS or b not in _POS:
        return False
    (r1, c1), (r2, c2) = _POS[a], _POS[b]
    return abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1.0 and a != b


def sub_cost(a, b):
    if a == b:
        return 0.0
    if keys_adjacent(a, b):
        return 0.5
    if a in VOWELS and b in VOWELS:
        return 0.7
    return 1.0


def weighted_distance(x, w, weighted=True):
    """Weighted optimal-string-alignment distance from intended w to typed x."""
    n, m = len(w), len(x)
    d = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if weighted:
                sc = sub_cost(w[i - 1], x[j - 1])
                # deleting a letter that doubles its neighbour ("comming"->"coming" reversed)
                dc = 0.6 if (i > 1 and w[i - 1] == w[i - 2]) else 1.0
                ic = 0.6 if (j > 1 and x[j - 1] == x[j - 2]) else 1.0
                tc = 0.6
            else:
                sc = 0.0 if w[i - 1] == x[j - 1] else 1.0
                dc = ic = tc = 1.0
            d[i][j] = min(d[i - 1][j] + dc, d[i][j - 1] + ic, d[i - 1][j - 1] + sc)
            if i > 1 and j > 1 and w[i - 1] == x[j - 2] and w[i - 2] == x[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + tc)
    return d[n][m]


def edits1(word):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits if b for c in LETTERS]
    inserts = [a + c + b for a, b in splits for c in LETTERS]
    return set(deletes + transposes + replaces + inserts)


class NoisyChannelSpeller:
    def __init__(self, unigram_counts, dictionary, lm=None, alpha=4.0):
        self.counts = unigram_counts
        self.total = sum(unigram_counts.values())
        self.dictionary = dictionary
        self.lm = lm            # optional NGramLM for bigram context
        self.alpha = alpha
        self._cache = {}
        self._cand = {}

    def known(self, words):
        return {w for w in words if w in self.dictionary}

    def candidates(self, word):
        if word not in self._cand:
            c = self.known(edits1(word))
            if not c and len(word) <= 12:
                c = self.known(e2 for e1 in edits1(word) for e2 in edits1(e1))
            self._cand[word] = c
        return self._cand[word]

    def log_prior(self, w):
        return math.log((self.counts.get(w, 0) + 1) / (self.total + len(self.dictionary)))

    def _unk_mass(self):
        if not hasattr(self, "_unk_total"):
            oov = [w for w in self.dictionary if w not in self.lm.vocab]
            self._unk_total = sum(self.counts.get(w, 0) + 1 for w in oov)
        return self._unk_total

    def log_context(self, w, prev, nxt):
        """log P(w | prev) + log P(next | w) from the background bigram LM.

        Words outside the LM vocabulary share the <unk> probability in
        proportion to their unigram counts; otherwise every rare candidate would
        inherit the full (large) probability of <unk>.
        """
        lm = self.lm
        mw = lm.map(w)
        lp = math.log(lm.p_interp(mw, [lm.map(prev)] if prev else ["<s>"]))
        if mw == "<unk>":
            lp += math.log((self.counts.get(w, 0) + 1) / self._unk_mass())
        if nxt:
            lp += math.log(lm.p_interp(lm.map(nxt), [mw]))
        return lp

    def rank(self, word, prev=None, nxt=None, weighted=True, context=False, keep=False):
        """Return [(candidate, score, channel, prior)] best first.

        With ``keep=True`` the typed word itself competes as the "no error"
        hypothesis (channel cost 0, prior from its own corpus count), so rare
        slang and names survive unless a candidate is clearly more probable.
        """
        out = []
        for c in self.candidates(word):
            ch = -self.alpha * weighted_distance(word, c, weighted)
            pr = self.log_context(c, prev, nxt) if (context and self.lm) else self.log_prior(c)
            out.append((c, ch + pr, ch, pr))
        if keep and out:
            if context and self.lm:
                pr = self.log_context(word, prev, nxt)
            else:
                pr = self.log_prior(word)
            out.append((word, pr, 0.0, pr))
        return sorted(out, key=lambda t: -t[1])

    def in_dictionary(self, tok):
        if tok in self.dictionary:
            return True
        from nltk.corpus import wordnet as wn
        return any(wn.morphy(tok, p) for p in ("n", "v", "a", "r"))

    def needs_check(self, tok):
        return tok.isalpha() and len(tok) >= 3 and not self.in_dictionary(tok)

    def correct(self, word, prev=None, nxt=None, weighted=True, context=False, keep=True):
        key = (word, prev, nxt, weighted, context, keep)
        if key not in self._cache:
            ranked = self.rank(word, prev, nxt, weighted, context, keep)
            self._cache[key] = ranked[0][0] if ranked else word
        return self._cache[key]

    def correct_tokens(self, tokens, context=True):
        """Correct non-word tokens; returns (new_tokens, [(i, old, new)])."""
        out, changes = list(tokens), []
        for i, t in enumerate(tokens):
            if self.needs_check(t):
                prev = out[i - 1] if i else None
                nxt = tokens[i + 1] if i + 1 < len(tokens) else None
                new = self.correct(t, prev, nxt, weighted=True, context=context)
                if new != t:
                    out[i] = new
                    changes.append((i, t, new))
        return out, changes
