"""Timeline sequence labelling: a user's timeline is read like a sentence.

States per post: O (no change), S (switch), E (escalation), B (both).
The four model families mirror the classical POS-tagging models
(generative vs. conditional, independent vs. sequential):

                 independent per post      sequence (Viterbi / forward-backward)
  generative     -                         GaussianHMM
  conditional    MaxEnt (logistic reg.)    HybridHMM (MaxEnt emissions), CRF

HybridHMM follows the hybrid connectionist/HMM recipe from speech recognition:
the MaxEnt posterior divided by the class prior, p(y|x)/p(y), is proportional
to the likelihood p(x|y), so it can replace the Gaussian emission inside a
standard HMM decoded with Viterbi.
"""
import numpy as np
import sklearn_crfsuite
from sklearn.linear_model import LogisticRegression

STATES = ["O", "S", "E", "B"]
IDX = {s: i for i, s in enumerate(STATES)}


class Standardizer:
    def fit(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        return self

    def transform(self, X):
        return (X - self.mu) / self.sd


def _transitions(y_seqs, add=1.0):
    K = len(STATES)
    pi = np.full(K, add)
    A = np.full((K, K), add)
    for ys in y_seqs:
        pi[IDX[ys[0]]] += 1
        for a, b in zip(ys, ys[1:]):
            A[IDX[a], IDX[b]] += 1
    return np.log(pi / pi.sum()), np.log(A / A.sum(1, keepdims=True))


def viterbi(log_pi, log_A, log_E):
    T, K = log_E.shape
    delta = log_pi + log_E[0]
    back = np.zeros((T, K), int)
    for t in range(1, T):
        s = delta[:, None] + log_A
        back[t] = s.argmax(0)
        delta = s.max(0) + log_E[t]
    path = [int(delta.argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t][path[-1]]))
    return [STATES[i] for i in reversed(path)]


def forward_backward(log_pi, log_A, log_E):
    """Posterior state marginals, used for per-post confidence in the demo."""
    from scipy.special import logsumexp
    T, K = log_E.shape
    la = np.zeros((T, K))
    lb = np.zeros((T, K))
    la[0] = log_pi + log_E[0]
    for t in range(1, T):
        la[t] = logsumexp(la[t - 1][:, None] + log_A, axis=0) + log_E[t]
    for t in range(T - 2, -1, -1):
        lb[t] = logsumexp(log_A + log_E[t + 1] + lb[t + 1], axis=1)
    post = la + lb
    post -= logsumexp(post, axis=1, keepdims=True)
    return np.exp(post)


class Majority:
    name = "Majority (all O)"

    def fit(self, X_seqs, y_seqs):
        return self

    def predict(self, X_seqs):
        return [["O"] * len(X) for X in X_seqs]


class MaxEnt:
    """Per-post multinomial logistic regression (maximum-entropy classifier)."""
    name = "MaxEnt"

    def __init__(self, C=1.0, balanced=True):
        self.C, self.balanced = C, balanced

    def fit(self, X_seqs, y_seqs):
        X = np.vstack(X_seqs)
        y = [s for ys in y_seqs for s in ys]
        self.sc = Standardizer().fit(X)
        self.clf = LogisticRegression(C=self.C, max_iter=5000,
                                      class_weight="balanced" if self.balanced else None)
        self.clf.fit(self.sc.transform(X), y)
        return self

    def proba(self, X):
        P = self.clf.predict_proba(self.sc.transform(X))
        out = np.full((len(X), len(STATES)), 1e-9)
        for j, c in enumerate(self.clf.classes_):
            out[:, IDX[c]] = P[:, j]
        return out

    def predict(self, X_seqs):
        return [[STATES[i] for i in self.proba(X).argmax(1)] for X in X_seqs]


class GaussianHMM:
    """Supervised HMM with diagonal-Gaussian emissions over standardised features."""
    name = "Gaussian HMM"

    def __init__(self, var_floor=0.05):
        self.var_floor = var_floor

    def fit(self, X_seqs, y_seqs):
        X = np.vstack(X_seqs)
        y = np.array([IDX[s] for ys in y_seqs for s in ys])
        self.sc = Standardizer().fit(X)
        Z = self.sc.transform(X)
        K, D = len(STATES), Z.shape[1]
        self.mu = np.zeros((K, D))
        self.var = np.ones((K, D))
        for k in range(K):
            if (y == k).sum() > 1:
                self.mu[k] = Z[y == k].mean(0)
                self.var[k] = np.maximum(Z[y == k].var(0), self.var_floor)
        self.log_pi, self.log_A = _transitions(y_seqs)
        return self

    def log_emissions(self, X):
        Z = self.sc.transform(X)
        return -0.5 * (((Z[:, None, :] - self.mu) ** 2) / self.var + np.log(2 * np.pi * self.var)).sum(2)

    def predict(self, X_seqs):
        return [viterbi(self.log_pi, self.log_A, self.log_emissions(X)) for X in X_seqs]

    def marginals(self, X):
        return forward_backward(self.log_pi, self.log_A, self.log_emissions(X))


class HybridHMM:
    """HMM whose emission scores come from a MaxEnt classifier.

    ``balanced=False``: plain MaxEnt posteriors divided by the empirical class
    prior, p(y|x)/p(y), the textbook scaled likelihood.
    ``balanced=True`` (default): MaxEnt trained with class-balanced weights,
    i.e. as if all states were equally frequent. Its posterior is then already
    proportional to p(x|y), and with 30 timelines it is a more stable estimate
    for rare states than dividing a heavily regularised posterior by a small prior.
    """
    name = "MaxEnt-HMM hybrid"

    def __init__(self, C=1.0, balanced=True):
        self.C, self.balanced = C, balanced

    def fit(self, X_seqs, y_seqs):
        self.me = MaxEnt(C=self.C, balanced=self.balanced).fit(X_seqs, y_seqs)
        y = [s for ys in y_seqs for s in ys]
        prior = np.array([y.count(s) + 1.0 for s in STATES])
        self.log_prior = np.zeros(len(STATES)) if self.balanced else np.log(prior / prior.sum())
        self.log_pi, self.log_A = _transitions(y_seqs)
        return self

    def log_emissions(self, X):
        return np.log(self.me.proba(X)) - self.log_prior

    def predict(self, X_seqs):
        return [viterbi(self.log_pi, self.log_A, self.log_emissions(X)) for X in X_seqs]

    def marginals(self, X):
        return forward_backward(self.log_pi, self.log_A, self.log_emissions(X))


class CRF:
    """Linear-chain conditional random field (CRFsuite, L-BFGS, elastic net)."""
    name = "Linear-chain CRF"

    def __init__(self, feature_names, c1=0.1, c2=0.1, decode="viterbi"):
        self.names, self.c1, self.c2, self.decode = feature_names, c1, c2, decode

    def _dicts(self, X):
        Z = self.sc.transform(X)
        return [dict({"bias": 1.0}, **{n: float(v) for n, v in zip(self.names, row)}) for row in Z]

    def fit(self, X_seqs, y_seqs):
        self.sc = Standardizer().fit(np.vstack(X_seqs))
        self.crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=self.c1, c2=self.c2, max_iterations=300,
                                        all_possible_transitions=True)
        self.crf.fit([self._dicts(X) for X in X_seqs], [list(ys) for ys in y_seqs])
        y = [s for ys in y_seqs for s in ys]
        self.prior = np.array([y.count(s) + 1.0 for s in STATES])
        self.prior /= self.prior.sum()
        return self

    def predict(self, X_seqs):
        if self.decode == "viterbi":
            return self.crf.predict([self._dicts(X) for X in X_seqs])
        # prior-corrected posterior decoding: the Bayes decision under a uniform
        # state prior, parameter-free counterpart of class-balanced training
        return [[STATES[i] for i in (self.marginals(X) / self.prior).argmax(1)] for X in X_seqs]

    def marginals(self, X):
        m = self.crf.predict_marginals_single(self._dicts(X))
        return np.array([[row.get(s, 0.0) for s in STATES] for row in m])
