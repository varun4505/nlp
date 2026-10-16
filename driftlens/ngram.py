"""N-gram language models written from scratch.

``NGramLM`` supports Laplace / add-k smoothing and Jelinek-Mercer
interpolation, and reports perplexity for intrinsic evaluation.

``personal_novelty`` is the project's drift signal. For post t of a user it
compares two models on the same text:

* the background LM alone, and
* a mixture of the background LM with a small add-k LM trained only on the
  user's own posts 1..t-1.

    novelty_t = H_mix(post_t) - H_bg(post_t)      (bits per token)

When the post re-uses the user's habitual words and phrases, the personal
component lowers the cross-entropy and novelty is negative. When the post
breaks from the user's own history, the personal component is useless and
novelty rises towards -log2(1 - beta). Subtracting H_bg removes the effect of
how intrinsically predictable the text is.
"""
import math
from collections import Counter, defaultdict

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


class NGramLM:
    def __init__(self, n=3, min_count=2):
        self.n = n
        self.min_count = min_count
        self.vocab = set()
        # flat tables keep memory low: counts[k][(w1..wk)] and ctx_totals[k][(w1..wk-1)]
        self.counts = [None] + [Counter() for _ in range(n)]
        self.ctx_totals = [None] + [Counter() for _ in range(n)]
        self.lambdas = None

    # ------------------------------------------------------------------ fit
    def fit(self, sentences):
        uni = Counter(w for s in sentences for w in s)
        self.vocab = {w for w, c in uni.items() if c >= self.min_count} | {UNK, EOS}
        for s in sentences:
            toks = [BOS] * (self.n - 1) + [self.map(w) for w in s] + [EOS]
            for i in range(self.n - 1, len(toks)):
                for k in range(1, self.n + 1):
                    gram = tuple(toks[i - k + 1:i + 1])
                    self.counts[k][gram] += 1
                    self.ctx_totals[k][gram[:-1]] += 1
        self.lambdas = [1.0 / self.n] * self.n
        return self

    def map(self, w):
        return w if w in self.vocab else UNK

    @property
    def V(self):
        return len(self.vocab)

    # ---------------------------------------------------------- probabilities
    def p_addk(self, w, ctx, k=1.0, order=None):
        """Add-k estimate of P(w | ctx) using the last (order-1) context words."""
        order = order or self.n
        ctx = tuple(ctx[len(ctx) - (order - 1):]) if order > 1 else ()
        c = self.counts[order].get(ctx + (w,), 0)
        return (c + k) / (self.ctx_totals[order].get(ctx, 0) + k * self.V)

    def p_interp(self, w, ctx, k_uni=0.01):
        """Jelinek-Mercer interpolation of ML estimates of every order."""
        p = self.lambdas[0] * self.p_addk(w, (), k=k_uni, order=1)
        for order in range(2, self.n + 1):
            c_ctx = tuple(ctx[len(ctx) - (order - 1):])
            tot = self.ctx_totals[order].get(c_ctx, 0)
            if tot:
                p += self.lambdas[order - 1] * self.counts[order].get(c_ctx + (w,), 0) / tot
        return p

    def sentence_logprob(self, sent, method="interp", k=1.0, order=None):
        order = order or self.n
        toks = [BOS] * (self.n - 1) + [self.map(w) for w in sent] + [EOS]
        lp = 0.0
        for i in range(self.n - 1, len(toks)):
            ctx = toks[i - (self.n - 1):i]
            if method == "interp":
                p = self.p_interp(toks[i], ctx)
            else:
                p = self.p_addk(toks[i], ctx, k=k, order=order)
            lp += math.log2(p)
        return lp, len(toks) - (self.n - 1)

    def perplexity(self, sentences, **kw):
        lp, n = 0.0, 0
        for s in sentences:
            a, b = self.sentence_logprob(s, **kw)
            lp += a
            n += b
        return 2 ** (-lp / n)

    def tune_lambdas(self, heldout, step=0.1):
        """Grid-search interpolation weights on held-out data (weights sum to 1)."""
        grid = [round(step * i, 3) for i in range(1, int(1 / step))]
        best = (float("inf"), self.lambdas)
        if self.n == 2:
            cands = [[a, round(1 - a, 3)] for a in grid]
        else:
            cands = [[a, b, round(1 - a - b, 3)] for a in grid for b in grid if a + b < 1 - 1e-9]
        for lam in cands:
            self.lambdas = lam
            pp = self.perplexity(heldout, method="interp")
            if pp < best[0]:
                best = (pp, lam)
        self.lambdas = best[1]
        return best


class PersonalLM:
    """Add-k bigram + unigram model over one user's earlier posts."""

    def __init__(self, bg: NGramLM, k=0.05, mu=0.6):
        self.bg, self.k, self.mu = bg, k, mu
        self.uni = Counter()
        self.bi = defaultdict(Counter)
        self.n = 0

    def add(self, sent):
        toks = [BOS] + [self.bg.map(w) for w in sent] + [EOS]
        for a, b in zip(toks, toks[1:]):
            self.uni[b] += 1
            self.bi[a][b] += 1
            self.n += 1

    def p(self, w, prev):
        V = self.bg.V
        pu = (self.uni[w] + self.k) / (self.n + self.k * V)
        row = self.bi[prev]
        tot = sum(row.values())
        pb = (row[w] + self.k) / (tot + self.k * V) if tot else pu
        return self.mu * pb + (1 - self.mu) * pu


def cross_entropies(sent, bg: NGramLM, personal: PersonalLM = None, beta=0.3):
    """Return (H_bg, H_mix) in bits/token for a single tokenised post."""
    toks = [BOS] * (bg.n - 1) + [bg.map(w) for w in sent] + [EOS]
    h_bg = h_mix = 0.0
    n = 0
    for i in range(bg.n - 1, len(toks)):
        ctx = toks[i - (bg.n - 1):i]
        pb = bg.p_interp(toks[i], ctx)
        h_bg -= math.log2(pb)
        if personal is not None and personal.n:
            pm = beta * personal.p(toks[i], toks[i - 1]) + (1 - beta) * pb
            h_mix -= math.log2(pm)
        n += 1
    h_bg /= n
    h_mix = h_mix / n if personal is not None and personal.n else h_bg
    return h_bg, h_mix


def personal_novelty(posts_tokens, bg: NGramLM, beta=0.3):
    """Novelty of each post relative to the same user's earlier posts.

    The first post has no history; it gets 0 (neutral) and is flagged by the
    caller through the position feature.
    """
    plm = PersonalLM(bg)
    out = []
    for toks in posts_tokens:
        if plm.n == 0:
            out.append(0.0)
        else:
            h_bg, h_mix = cross_entropies(toks, bg, plm, beta)
            out.append(h_mix - h_bg)
        plm.add(toks)
    return out


def vocab_novelty(posts_tokens, stop=frozenset()):
    """Share of a post's content-word types never used earlier by the same user."""
    seen, out = set(), []
    for toks in posts_tokens:
        types = {t for t in toks if t.isalpha() and t not in stop}
        out.append(len(types - seen) / len(types) if (types and seen) else 0.0)
        seen |= types
    return out
