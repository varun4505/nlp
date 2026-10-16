"""Bigram HMM part-of-speech tagger with Viterbi decoding.

Transition and emission probabilities are relative frequencies from the Brown
corpus (universal tag set) with add-k smoothing. Unknown words are handled as
in TnT (Brants, 2000): the tag distribution of rare training words sharing
the longest known suffix, turned into an emission score by Bayes' rule
P(w|t) ~ P(t|suffix) / P(t). A capitalisation / digit / hyphen shape class is
used when no suffix matches.
"""
import math
from collections import Counter, defaultdict

import numpy as np

MAX_SUFFIX = 4
RARE = 10


def shape(w):
    if any(c.isdigit() for c in w):
        return "<DIGIT>"
    if "-" in w:
        return "<HYPH>"
    if w[:1].isupper():
        return "<CAP>"
    if not any(c.isalpha() for c in w):
        return "<PUNCT>"
    return "<LOWER>"


class HMMTagger:
    def __init__(self, k_trans=0.1, k_emit=0.01):
        self.k_trans, self.k_emit = k_trans, k_emit

    def fit(self, tagged_sents):
        tags = Counter()
        trans = Counter()
        emit = defaultdict(Counter)
        word_freq = Counter()
        for s in tagged_sents:
            prev = "<s>"
            for w, t in s:
                lw = w.lower()
                tags[t] += 1
                trans[(prev, t)] += 1
                emit[t][lw] += 1
                word_freq[lw] += 1
                prev = t
            trans[(prev, "</s>")] += 1
        self.tags = sorted(tags)
        self.T = len(self.tags)
        self.tag_index = {t: i for i, t in enumerate(self.tags)}
        states_from = ["<s>"] + self.tags
        # log transition matrix: rows = previous (incl. <s>), cols = current tag
        A = np.zeros((len(states_from), self.T))
        for i, p in enumerate(states_from):
            row = np.array([trans[(p, t)] for t in self.tags], float) + self.k_trans
            A[i] = np.log(row / row.sum())
        self.logA0, self.logA = A[0], A[1:]
        self.tag_prob = np.array([tags[t] for t in self.tags], float)
        self.tag_prob /= self.tag_prob.sum()
        self.vocab = set(word_freq)
        # emissions for known words
        self.emit = {}
        for t in self.tags:
            tot = sum(emit[t].values())
            self.emit[t] = (emit[t], tot)
        self.V = len(self.vocab)
        # suffix and shape models from rare words
        self.suffix = defaultdict(Counter)
        self.shape_model = defaultdict(Counter)
        for t in self.tags:
            for lw, c in emit[t].items():
                if word_freq[lw] <= RARE:
                    for k in range(1, min(MAX_SUFFIX, len(lw)) + 1):
                        self.suffix[lw[-k:]][t] += c
        for s in tagged_sents:
            for w, t in s:
                if word_freq[w.lower()] <= RARE:
                    self.shape_model[shape(w)][t] += 1
        self.known_tags = {}
        for t in self.tags:
            for lw in emit[t]:
                self.known_tags.setdefault(lw, set()).add(t)
        return self

    def _unknown_scores(self, w):
        lw = w.lower()
        dist = None
        for k in range(min(MAX_SUFFIX, len(lw)), 0, -1):
            if lw[-k:] in self.suffix and sum(self.suffix[lw[-k:]].values()) >= 5:
                dist = self.suffix[lw[-k:]]
                break
        if dist is None or shape(w) in ("<DIGIT>", "<PUNCT>", "<CAP>"):
            sd = self.shape_model.get(shape(w))
            if sd and (dist is None or shape(w) != "<CAP>"):
                dist = sd
            elif sd and shape(w) == "<CAP>" and dist is not None:
                # combine suffix evidence with the capitalisation cue
                dist = Counter({t: dist[t] + sd[t] for t in self.tags})
        tot = sum(dist.values()) if dist else 0
        p_t_given_x = np.array([(dist[t] if dist else 0) + 0.1 for t in self.tags], float)
        p_t_given_x /= p_t_given_x.sum()
        return np.log(p_t_given_x) - np.log(self.tag_prob) - math.log(self.V)

    def emission_scores(self, w):
        lw = w.lower()
        if lw not in self.vocab:
            return self._unknown_scores(w)
        out = np.empty(self.T)
        for i, t in enumerate(self.tags):
            cnt, tot = self.emit[t]
            out[i] = math.log((cnt.get(lw, 0) + self.k_emit) / (tot + self.k_emit * self.V))
        return out

    def viterbi(self, words):
        if not words:
            return []
        n = len(words)
        E = np.vstack([self.emission_scores(w) for w in words])
        delta = self.logA0 + E[0]
        back = np.zeros((n, self.T), int)
        for i in range(1, n):
            scores = delta[:, None] + self.logA           # prev x cur
            back[i] = scores.argmax(0)
            delta = scores.max(0) + E[i]
        best = [int(delta.argmax())]
        for i in range(n - 1, 0, -1):
            best.append(int(back[i][best[-1]]))
        best.reverse()
        return [self.tags[b] for b in best]

    def tag(self, words):
        return list(zip(words, self.viterbi(words)))

    def evaluate(self, tagged_sents):
        correct = total = unk_c = unk_t = 0
        for s in tagged_sents:
            words = [w for w, _ in s]
            pred = self.viterbi(words)
            for (w, g), p in zip(s, pred):
                total += 1
                correct += g == p
                if w.lower() not in self.vocab:
                    unk_t += 1
                    unk_c += g == p
        return {"accuracy": correct / total, "unknown_accuracy": unk_c / max(unk_t, 1),
                "tokens": total, "unknown_tokens": unk_t}
