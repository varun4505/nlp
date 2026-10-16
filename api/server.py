"""DriftLens API: the backend for the Next.js frontend in ../web.

It wraps the same driftlens library the Streamlit prototype uses, adds one
endpoint per NLP concept shown in the "NLP Lab", and a text-to-speech
endpoint that uses Microsoft Edge neural voices through the free `edge-tts`
package (no API key needed).

Run from the project folder:
    python -m uvicorn api.server:app --port 8000
"""
import asyncio
import base64
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import edge_tts  # noqa: E402
import nltk  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from nltk.corpus import stopwords, wordnet as wn, wordnet_ic  # noqa: E402
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # noqa: E402
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from driftlens import resources  # noqa: E402
from driftlens.features import retag  # noqa: E402
from driftlens.lexicon import MALADAPTIVE, adapted_lesk  # noqa: E402
from driftlens.ngram import BOS, EOS, UNK, PersonalLM, cross_entropies  # noqa: E402
from driftlens.pipeline import STATE_NAMES, DriftLens  # noqa: E402
from driftlens.spelling import sub_cost, weighted_distance  # noqa: E402
from driftlens.textproc import clean, expand_contractions, lm_tokens  # noqa: E402

app = FastAPI(title="DriftLens API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# NLTK's corpus readers (WordNet reads a zip file lazily) are not thread-safe, and
# FastAPI runs sync endpoints in a thread pool. The NLP endpoints take
# milliseconds, so run them one at a time; TTS is network-bound and stays concurrent.
_nlp_lock = asyncio.Lock()


@app.middleware("http")
async def serialise_nlp(request, call_next):
    if request.url.path in ("/api/tts", "/api/voices"):
        return await call_next(request)
    async with _nlp_lock:
        return await call_next(request)


@lru_cache(maxsize=None)
def engine():
    return DriftLens()


def f(x, nd=4):
    """JSON-safe rounded float."""
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, nd)


# =================================================================== analysis
class AnalyseReq(BaseModel):
    posts: list[str]
    model: str = "CRF (prior-corrected)"


CAT_LABEL = {"despair": "despair", "self_criticism": "self-criticism", "anxiety": "anxiety", "anger": "anger",
             "isolation": "loneliness", "exhaustion": "exhaustion", "crisis": "crisis language",
             "hope": "hope", "connection": "connection", "calm": "calm", "joy": "joy", "agency": "agency"}


def tfidf_keywords(texts, k=4):
    try:
        vec = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b")
        X = vec.fit_transform(texts)
    except ValueError:
        return [[] for _ in texts]
    vocab = np.array(vec.get_feature_names_out())
    out = []
    for i in range(X.shape[0]):
        row = X[i].toarray().ravel()
        idx = [j for j in np.argsort(-row)[:k] if row[j] > 0]
        out.append([{"term": vocab[j], "score": f(row[j], 3)} for j in idx])
    return out


def ordinal(n):
    return {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh",
            8: "eighth", 9: "ninth", 10: "tenth"}.get(n, f"number {n}")


def narrate(posts_out, digest_len):
    """Data-to-text generation: turn the numbers into a short spoken explanation.

    Content selection (which posts and words matter), then sentence planning
    with a few template variants, so the narration does not sound robotic.
    """
    rnd = random.Random(len(posts_out))
    n = len(posts_out)
    lines = [rnd.choice([f"I read these {n} posts the way a tagger reads a sentence, one post at a time.",
                         f"Here's what I found across these {n} posts, read in order like one long sentence."])]
    changes = [p for p in posts_out if p["state"] != "O"]
    calm = [p for p in posts_out if p["state"] == "O"]
    if calm and calm[0]["index"] == 0:
        lines.append(rnd.choice(["The timeline starts out fairly steady.",
                                 "At the start, nothing unusual stands out."]))
    if not changes:
        lines.append("I didn't find any moment of change. The language stays close to the writer's usual self "
                     "all the way through.")
    for p in changes[:3]:
        i = p["index"] + 1
        mal = [h["text"] for h in p["hits"] if h["pol"] == "mal" and h["w"] > 0][:3]
        ada = [h["text"] for h in p["hits"] if h["pol"] == "ada" and h["w"] > 0][:3]
        conf = int(round(p["confidence"] * 100))
        top = max(p["probs"].values())
        how = (f"about {conf} percent" if p["confidence"] >= top
               else f"{conf} percent, which is low, but changes are rare so the decoder gives them a boost")
        words = ""
        if mal:
            words = " Words like " + ", ".join(f"'{w}'" for w in mal) + " start to appear."
        elif ada:
            words = " Positive words like " + ", ".join(f"'{w}'" for w in ada) + " come back."
        if p["state"] == "S":
            lines.append(f"In the {ordinal(i)} post, the mood shifts suddenly.{words} "
                         f"The model calls this a switch, at {how}.")
        elif p["state"] == "E":
            lines.append(f"By the {ordinal(i)} post, things are building gradually rather than all at once.{words} "
                         f"That's an escalation, at {how}.")
        else:
            lines.append(f"The {ordinal(i)} post is both a sudden switch and part of a gradual build-up.{words} "
                         f"Confidence: {how}.")
    if len(changes) > 3:
        extra = len(changes) - 3
        lines.append(f"After that, {extra} more post{'s' if extra > 1 else ''} also show{'' if extra > 1 else 's'} change.")
    if any(p["crisis"] for p in posts_out):
        lines.append("Some posts contain crisis language, so in a real setting a human should look at this, "
                     "not a model.")
    first, last = posts_out[0]["signals"]["net"], posts_out[-1]["signals"]["net"]
    if n > 2:
        if last > first + 1:
            lines.append("By the end, the balance tips back toward adaptive, hopeful language.")
        elif last < first - 1:
            lines.append("By the end, the language is noticeably darker than where it began.")
    if digest_len:
        lines.append(f"I've also picked out {digest_len} key sentences as a short summary, using TextRank.")
    return " ".join(lines)


@app.get("/api/health")
def health():
    return {"ok": True, "models": list(engine().models)}


@app.post("/api/analyse")
def analyse(req: AnalyseReq):
    posts = [p.strip() for p in req.posts if p.strip()]
    if len(posts) < 2:
        raise HTTPException(400, "Add at least two posts.")
    if len(posts) > 40:
        raise HTTPException(400, "At most 40 posts.")
    eng = engine()
    if req.model not in eng.models:
        raise HTTPException(400, f"Unknown model {req.model}")
    res = eng.analyse([{"text": t, "date": None} for t in posts], model=req.model)
    states = res["states"]
    keywords = tfidf_keywords([a["lex"]["clean"] for a in res["analyses"]])
    out = []
    for i, a in enumerate(res["analyses"]):
        fe = a["features"]
        marg = res["marginals"][i]
        st = res["pred"][i]
        out.append({
            "index": i, "text": a["lex"]["clean"], "state": st,
            "confidence": f(marg[states.index(st)]),
            "probs": {s: f(marg[k]) for k, s in enumerate(states)},
            "hits": [{"start": h["start"], "end": h["end"], "text": h["text"], "cat": h["cat"],
                      "pol": h["pol"], "w": f(h["w"], 2), "negated": h["negated"],
                      "corrected_from": h.get("corrected_from")} for h in a["lex"]["hits"]],
            "corrections": [[o, n] for o, n in a["lex"]["corrections"]],
            "self_evals": [e for e in a["self_evals"] if e["polarity"] != "neutral"],
            "tagged": [[w, t] for w, t in a["tagged"]],
            "keywords": keywords[i],
            "crisis": a["lex"]["crisis"] > 0,
            "signals": {"net": f(fe["net"], 3), "mal": f(fe["mal"], 3), "ada": f(fe["ada"], 3),
                        "vader": f(fe["vader"], 3), "novelty": f(fe["novelty"], 3),
                        "vocab_novelty": f(fe["vocab_novelty"], 3), "fp_rate": f(fe["fp_rate"], 3),
                        "z_len": f(fe["z_len"], 3), "n_words": a["n_words"]},
        })
    cats = Counter()
    for a in res["analyses"]:
        for h in a["lex"]["hits"]:
            if h["w"] > 0:
                cats[h["cat"].replace("negated_", "")] += h["w"]
    digest = [{"post": int(o), "sentence": s} for o, s in res["digest"]]
    return {
        "model": req.model, "posts": out, "digest": digest,
        "categories": [{"cat": c, "label": CAT_LABEL.get(c, c), "weight": f(w, 2),
                        "pol": "mal" if c in MALADAPTIVE else "ada"} for c, w in cats.most_common()],
        "narration": narrate(out, len(digest)),
        "state_names": STATE_NAMES,
    }


# =========================================================== text processing
class TextReq(BaseModel):
    text: str


PENN = {"ADJ": "a", "VERB": "v", "NOUN": "n", "ADV": "r"}


@app.post("/api/tokenize")
def tokenize(req: TextReq):
    raw = req.text[:3000]
    cleaned = clean(raw)
    sents = nltk.sent_tokenize(cleaned)
    stop = set(stopwords.words("english"))
    porter, lanc, snow, lem = PorterStemmer(), LancasterStemmer(), SnowballStemmer("english"), WordNetLemmatizer()
    tagger = resources.tagger()
    rows = []
    for si, s in enumerate(sents):
        toks = expand_contractions(nltk.word_tokenize(s))
        for w, t in tagger.tag(toks):
            lw = w.lower()
            alpha = any(c.isalpha() for c in w)
            rows.append({
                "sent": si, "token": w, "lower": lw, "tag": t, "stop": lw in stop, "alpha": alpha,
                "porter": porter.stem(lw) if alpha else lw,
                "lancaster": lanc.stem(lw) if alpha else lw,
                "snowball": snow.stem(lw) if alpha else lw,
                "lemma": lem.lemmatize(lw, PENN.get(t, "n")) if alpha else lw,
            })
    freq = Counter(r["lower"] for r in rows if r["alpha"] and not r["stop"])
    return {"raw": raw, "clean": cleaned, "sentences": sents, "tokens": rows,
            "types": len({r["lower"] for r in rows}), "n_tokens": len(rows),
            "top": [{"word": w, "count": c} for w, c in freq.most_common(10)]}


# ============================================================== edit distance
class EditReq(BaseModel):
    source: str
    target: str
    weighted: bool = False
    sub_cost: float = 1.0


@app.post("/api/edit-distance")
def edit_distance(req: EditReq):
    """Full DP table + backtrace, so the frontend can draw the matrix."""
    a, b = req.source.lower()[:20], req.target.lower()[:20]
    n, m = len(a), len(b)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    P = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0], P[i][0] = float(i), "del"
    for j in range(1, m + 1):
        D[0][j], P[0][j] = float(j), "ins"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if req.weighted:
                sc = sub_cost(a[i - 1], b[j - 1])
                dc = 0.6 if (i > 1 and a[i - 1] == a[i - 2]) else 1.0
                ic = 0.6 if (j > 1 and b[j - 1] == b[j - 2]) else 1.0
            else:
                sc = 0.0 if a[i - 1] == b[j - 1] else req.sub_cost
                dc = ic = 1.0
            opts = [(D[i - 1][j - 1] + sc, "match" if a[i - 1] == b[j - 1] else "sub"),
                    (D[i - 1][j] + dc, "del"), (D[i][j - 1] + ic, "ins")]
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                opts.append((D[i - 2][j - 2] + (0.6 if req.weighted else 1.0), "swap"))
            D[i][j], P[i][j] = min(opts, key=lambda o: o[0])
    # backtrace
    i, j, path, ops = n, m, [], []
    while i > 0 or j > 0:
        path.append([i, j])
        op = P[i][j]
        if op in ("match", "sub"):
            ops.append({"op": op, "a": a[i - 1], "b": b[j - 1]})
            i, j = i - 1, j - 1
        elif op == "swap":
            ops.append({"op": "swap", "a": a[i - 2:i], "b": b[j - 2:j]})
            path.append([i - 1, j - 1])
            i, j = i - 2, j - 2
        elif op == "del":
            ops.append({"op": "del", "a": a[i - 1], "b": "-"})
            i -= 1
        else:
            ops.append({"op": "ins", "a": "-", "b": b[j - 1]})
            j -= 1
    path.append([0, 0])
    return {"source": a, "target": b, "matrix": [[f(x, 2) for x in r] for r in D],
            "distance": f(D[n][m], 2), "path": path[::-1], "ops": ops[::-1]}


class SpellReq(BaseModel):
    word: str
    prev: str = ""
    next: str = ""
    sentence: str = ""


@app.post("/api/spell")
def spell(req: SpellReq):
    sp = resources.speller()
    w = req.word.lower().strip()
    ranked = sp.rank(w, req.prev.lower() or None, req.next.lower() or None, weighted=True, context=True,
                     keep=True)[:8] if w else []
    cands = [{"candidate": c, "weighted": f(weighted_distance(w, c), 2),
              "levenshtein": f(weighted_distance(w, c, weighted=False), 2),
              "channel": f(ch, 2), "prior": f(pr, 2), "score": f(s, 2)} for c, s, ch, pr in ranked]
    fixed, changes = [], []
    if req.sentence.strip():
        toks = [t.lower() for t in expand_contractions(nltk.word_tokenize(req.sentence))]
        fixed, ch = sp.correct_tokens(toks)
        changes = [{"i": i, "from": o, "to": n} for i, o, n in ch]
    return {"word": w, "is_word": bool(w) and not sp.needs_check(w), "candidates": cands,
            "corrected": fixed, "changes": changes}


# ================================================================== n-grams
class NgramReq(BaseModel):
    sentence: str
    history: list[str] = []
    new_post: str = ""


@app.post("/api/ngram")
def ngram(req: NgramReq):
    lm = resources.bg_lm()
    toks = lm_tokens(req.sentence)[:60]
    rows, prev = [], BOS
    for t in toks + [EOS]:
        mt, mp = lm.map(t), lm.map(prev)
        p_interp = lm.p_interp(mt, [mp])
        rows.append({"token": t, "prev": prev, "oov": mt == UNK,
                     "p_mle": f(lm.counts[2].get((mp, mt), 0) / max(lm.ctx_totals[2].get((mp,), 0), 1), 6),
                     "p_laplace": f(lm.p_addk(mt, [mp], k=1.0, order=2), 6),
                     "p_interp": f(p_interp, 6), "bits": f(-math.log2(p_interp), 3),
                     "bigram_count": lm.counts[2].get((mp, mt), 0)})
        prev = t
    pp = 2 ** (sum(r["bits"] for r in rows) / len(rows)) if rows else None
    pp_lap = None
    if rows:
        pp_lap = 2 ** (-sum(math.log2(r["p_laplace"]) for r in rows if r["p_laplace"]) / len(rows))
    out = {"tokens": rows, "perplexity": f(pp, 2) if pp else None,
           "perplexity_laplace": f(pp_lap, 2) if pp_lap else None, "lambdas": lm.lambdas, "vocab": lm.V}
    hist = [h for h in req.history if h.strip()]
    if hist and req.new_post.strip():
        plm = PersonalLM(lm)
        for h in hist:
            plm.add(lm_tokens(h))
        h_bg, h_mix = cross_entropies(lm_tokens(req.new_post), lm, plm, beta=0.3)
        out["novelty"] = {"h_bg": f(h_bg, 3), "h_mix": f(h_mix, 3), "novelty": f(h_mix - h_bg, 3)}
    return out


@lru_cache(maxsize=None)
def bigram_index():
    lm = resources.bg_lm()
    idx = defaultdict(list)
    for (a, b), c in lm.counts[2].items():
        if b != UNK and c >= 2:
            idx[a].append((b, c))
    return idx


class GenReq(BaseModel):
    seed: str = "i"
    length: int = 20
    temperature: float = 1.0
    rng: int = 0


@app.post("/api/generate")
def generate(req: GenReq):
    """Sample text from the bigram LM (Shannon's game)."""
    lm, idx = resources.bg_lm(), bigram_index()
    rnd = random.Random(req.rng or None)
    seed = [t for t in lm_tokens(req.seed)] or []
    out = list(seed)
    prev = lm.map(seed[-1]) if seed else BOS
    steps = []
    for _ in range(max(1, min(req.length, 60))):
        nxt = idx.get(prev)
        if not nxt:
            break
        words, counts = zip(*nxt)
        w = np.array(counts, float) ** (1.0 / max(req.temperature, 0.05))
        w /= w.sum()
        top = np.argsort(-w)[:5]
        choice = words[int(rnd.choices(range(len(words)), weights=w.tolist())[0])]
        steps.append({"prev": prev, "chosen": choice,
                      "top": [{"w": words[k], "p": f(w[k], 3)} for k in top]})
        if choice == EOS:
            break
        out.append(choice)
        prev = choice
    return {"text": " ".join(out), "steps": steps}


@lru_cache(maxsize=None)
def _zipf():
    lm = resources.bg_lm()
    uni = sorted(((g[0], c) for g, c in lm.counts[1].items() if g[0] not in (EOS, UNK, BOS)),
                 key=lambda x: -x[1])
    pts = []
    for r in sorted(set(int(round(10 ** (k / 20))) for k in range(0, 90))):
        if r <= len(uni):
            pts.append({"rank": r, "freq": uni[r - 1][1], "word": uni[r - 1][0]})
    return {"points": pts, "top": [{"word": w, "count": c} for w, c in uni[:15]],
            "tokens": sum(c for _, c in uni), "types": len(uni)}


@app.api_route("/api/zipf", methods=["GET", "POST"])
def zipf():
    return _zipf()


# ============================================================ POS + Viterbi
class SentReq(BaseModel):
    sentence: str


@app.post("/api/pos")
def pos(req: SentReq):
    tg = resources.tagger()
    toks = expand_contractions(nltk.word_tokenize(req.sentence))[:25]
    if not toks:
        return {"tokens": [], "tags": tg.tags, "trellis": [], "best": [], "perceptron": []}
    E = np.vstack([tg.emission_scores(w) for w in toks])
    delta = tg.logA0 + E[0]
    deltas, backs = [delta.copy()], [[-1] * tg.T]
    for i in range(1, len(toks)):
        sc = delta[:, None] + tg.logA
        backs.append(sc.argmax(0).tolist())
        delta = sc.max(0) + E[i]
        deltas.append(delta.copy())
    best = [int(deltas[-1].argmax())]
    for i in range(len(toks) - 1, 0, -1):
        best.append(int(backs[i][best[-1]]))
    best.reverse()
    trellis = []
    for i, d in enumerate(deltas):
        # normalise each column so the frontend can shade cells by relative score
        col = d - d.max()
        trellis.append({"word": toks[i], "known": toks[i].lower() in tg.vocab,
                        "scores": [f(x, 2) for x in col], "emission": [f(x, 2) for x in E[i]],
                        "back": backs[i]})
    perc = nltk.pos_tag(toks, tagset="universal")
    return {"tokens": toks, "tags": tg.tags, "trellis": trellis, "best": [tg.tags[b] for b in best],
            "best_idx": best, "perceptron": [t for _, t in perc]}


# ======================================================= chunking, CFG, CKY
GRAMMAR = nltk.CFG.fromstring("""
    S -> FP VP
    VP -> COP ADJP | COP NEG ADJP | COP NP | COP NEG NP | V FPO | AUX NEG V | AUX NEG V NP | V NP
    ADJP -> ADJ | ADV ADJP | ADJ ADV | ADJ CONJ ADJP
    NP -> DET N | DET ADJP N | N
    FP -> 'FP'
    FPO -> 'FPO'
    COP -> 'COP'
    NEG -> 'NEG'
    AUX -> 'AUX'
    V -> 'VERB'
    ADJ -> 'ADJ'
    ADV -> 'ADV' | 'PRT'
    CONJ -> 'CONJ'
    DET -> 'DET'
    N -> 'NOUN'
""")


def tree_json(t, words=None, it=None):
    if it is None:
        it = iter(words or [])
    if isinstance(t, nltk.Tree):
        return {"label": t.label(), "children": [tree_json(c, words, it) for c in t]}
    if isinstance(t, tuple):
        return {"label": t[1], "children": [{"label": t[0], "children": []}]}
    w = next(it, None) if words else None
    return {"label": w if w else str(t), "tag": str(t) if w else None, "children": []}


def cky(cnf, tags):
    n = len(tags)
    lex = defaultdict(set)
    binary = defaultdict(set)
    unary = defaultdict(set)
    for p in cnf.productions():
        rhs = p.rhs()
        if len(rhs) == 1 and isinstance(rhs[0], str):
            lex[rhs[0]].add(str(p.lhs()))
        elif len(rhs) == 2:
            binary[(str(rhs[0]), str(rhs[1]))].add(str(p.lhs()))
        elif len(rhs) == 1:
            unary[str(rhs[0])].add(str(p.lhs()))

    def closure(s):
        s, frontier = set(s), list(s)
        while frontier:
            x = frontier.pop()
            for y in unary.get(x, ()):
                if y not in s:
                    s.add(y)
                    frontier.append(y)
        return s

    T = [[set() for _ in range(n + 1)] for _ in range(n + 1)]
    for i, t in enumerate(tags):
        T[i][i + 1] = closure(lex.get(t, set()))
    for span in range(2, n + 1):
        for i in range(0, n - span + 1):
            j = i + span
            for k in range(i + 1, j):
                for B in T[i][k]:
                    for C in T[k][j]:
                        T[i][j] |= binary.get((B, C), set())
            T[i][j] = closure(T[i][j])
    cells = [{"i": i, "j": j, "labels": sorted(T[i][j])} for i in range(n) for j in range(i + 1, n + 1)]
    return cells, str(cnf.start()) in T[0][n] if n else False


@app.post("/api/parse")
def parse(req: SentReq):
    eng, tg = engine(), resources.tagger()
    toks = expand_contractions(nltk.word_tokenize(req.sentence))[:20]
    tagged = tg.tag(toks)
    chunk_tree = eng.fx.chunker.parse(retag(tagged))
    evals = eng.fx.self_evaluations(tagged)
    ct = [(w, t) for w, t in retag(tagged) if t != "."]
    tags = [t for _, t in ct]
    words = [w for w, _ in ct]
    out = {"tagged": [[w, t] for w, t in retag(tagged)], "chunk_tree": tree_json(chunk_tree),
           "self_evals": evals, "tags": tags, "words": words}
    for key, parser in [("topdown", nltk.RecursiveDescentParser(GRAMMAR)),
                        ("bottomup", nltk.BottomUpChartParser(GRAMMAR))]:
        try:
            trees = list(parser.parse(tags))
        except ValueError:
            trees = []
        out[key] = tree_json(trees[0], words) if trees else None
    cnf = GRAMMAR.chomsky_normal_form()
    cells, ok = cky(cnf, tags)
    out["cky"] = {"cells": cells, "accepted": ok, "n": len(tags)}
    out["grammar"] = [str(p) for p in GRAMMAR.productions()]
    out["cnf"] = [str(p) for p in cnf.productions()]
    return out


# ================================================================== WordNet
class WordReq(BaseModel):
    word: str


@app.post("/api/wordnet")
def wordnet(req: WordReq):
    lex = engine().fx.lexicon
    w = req.word.lower().strip()
    key = lex.lookup(w) if w else None
    entry = None
    if key:
        e = lex.entries[key]
        entry = {"lemma": key, "cat": e["cat"], "label": CAT_LABEL.get(e["cat"], e["cat"]), "pol": e["pol"],
                 "weight": e["w"], "ratio": e["ratio"], "senses": e["senses"]}
    syns = []
    for s in wn.synsets(w)[:12]:
        path = s.hypernym_paths()[0] if s.hypernym_paths() else []
        syns.append({"name": s.name(), "pos": s.pos(), "definition": s.definition(),
                     "examples": s.examples()[:2], "lemmas": [l.name().replace("_", " ") for l in s.lemmas()][:6],
                     "hypernyms": [h.name() for h in path[-5:]],
                     "antonyms": sorted({a.name() for l in s.lemmas() for a in l.antonyms()}),
                     "in_lexicon": bool(entry and s.name() in entry["senses"])})
    return {"word": w, "lexicon": entry, "synsets": syns,
            "morphy": {p: wn.morphy(w, p) for p in ("n", "v", "a", "r")} if w else {}}


class SimReq(BaseModel):
    a: str
    b: str


@lru_cache(maxsize=None)
def brown_ic():
    return wordnet_ic.ic("ic-brown.dat")


@app.post("/api/similarity")
def similarity(req: SimReq):
    sa, sb = wn.synsets(req.a.strip()), wn.synsets(req.b.strip())
    pairs = [(x, y) for x in sa for y in sb if x.pos() == y.pos()]
    if not pairs:
        return {"pair": None}
    x, y = max(pairs, key=lambda p: p[0].path_similarity(p[1]) or 0)
    out = {"pair": [x.name(), y.name()], "defs": [x.definition(), y.definition()],
           "path": f(x.path_similarity(y) or 0, 3), "wup": f(x.wup_similarity(y) or 0, 3)}
    lcs = x.lowest_common_hypernyms(y)
    out["lcs"] = lcs[0].name() if lcs else None
    if x.pos() in "nv" and x.pos() == y.pos():
        ic = brown_ic()
        try:
            out.update({"lch": f(x.lch_similarity(y), 3), "resnik": f(x.res_similarity(y, ic), 3),
                        "lin": f(x.lin_similarity(y, ic), 3), "jcn": f(min(x.jcn_similarity(y, ic), 10), 3)})
        except Exception:
            pass
    return out


class WsdReq(BaseModel):
    sentence: str
    target: str


@app.post("/api/wsd")
def wsd(req: WsdReq):
    from nltk.wsd import lesk
    lex = engine().fx.lexicon
    target = req.target.lower().strip()
    ctx = [t.lower() for t in nltk.word_tokenize(req.sentence)]
    simple = lesk(ctx, target)
    key = lex.lookup(target)
    reached = set(lex.entries[key]["senses"]) if key else set()
    affect = any(t != target and lex.lookup(t) for t in ctx)
    adapted = adapted_lesk(target, ctx, reached, affect)

    def d(s):
        return {"name": s.name(), "definition": s.definition()} if s is not None else None

    return {"simple": d(simple), "adapted": d(adapted), "affect_context": affect,
            "counted": bool(adapted is not None and adapted.name() in reached)}


# ================================================================ sentiment
@lru_cache(maxsize=None)
def vader():
    return SentimentIntensityAnalyzer()


@app.post("/api/sentiment")
def sentiment(req: TextReq):
    lex = engine().fx.lexicon
    txt = req.text[:3000]
    sents = nltk.sent_tokenize(clean(txt))
    rows = []
    for s in sents:
        v = vader().polarity_scores(s)
        sc = lex.score(s)
        rows.append({"sentence": s, "vader": v, "lex_net": f(sc["net"], 2),
                     "hits": [{"text": h["text"], "cat": h["cat"], "pol": h["pol"], "w": f(h["w"], 2),
                               "negated": h["negated"]} for h in sc["hits"]]})
    whole = lex.score(txt)
    words = []
    for tok in nltk.word_tokenize(clean(txt)):
        val = vader().lexicon.get(tok.lower())
        if val is not None:
            words.append({"word": tok, "valence": val})
    return {"sentences": rows, "vader": vader().polarity_scores(clean(txt)),
            "lex": {"net": f(whole["net"], 2), "mal": f(whole["mal"], 2), "ada": f(whole["ada"], 2)},
            "vader_words": words[:30]}


# ================================================================== results
@lru_cache(maxsize=None)
def _results():
    R = ROOT / "results"

    def load(name):
        fp = R / name
        return json.loads(fp.read_text()) if fp.exists() else {}

    seq, lm, pos_, sp = load("sequence_eval.json"), load("lm_eval.json"), load("pos_eval.json"), load("spelling_eval.json")
    lc, le = load("lexicon_clpsych_eval.json"), load("lexicon_erisk_eval.json")

    def row(name, v):
        return {"model": name, "macro_f1": f(v["macro_f1"][0], 3), "sd": f(v["macro_f1"][1], 3),
                "f1_switch": f(v["f1_switch"][0], 3), "f1_escalation": f(v["f1_escalation"][0], 3),
                "f1_none": f(v["f1_none"][0], 3)}

    models = [row(k, v) for k, v in seq.get("main", {}).items()]
    if "length_only" in seq:
        models.append(row("MaxEnt, length only", seq["length_only"]["MaxEnt"]))
    crf = "CRF (prior-corrected)"
    abl = [{"group": k, "macro_f1": f(v[crf]["macro_f1"][0], 3)}
           for k, v in seq.get("ablation", {}).items() if k.startswith("-") and crf in v]
    return {
        "n_timelines": seq.get("n_timelines"), "n_posts": seq.get("n_posts"), "models": models,
        "ablation": abl, "crf_full": f(seq["main"][crf]["macro_f1"][0], 3) if crf in seq.get("main", {}) else None,
        "lm": [{"model": r["model"], "smoothing": r["smoothing"], "dev_pp": f(r["dev_pp"], 1)} for r in lm.get("rows", [])],
        "pos": {"hmm_brown": pos_.get("hmm_brown_test", {}).get("accuracy"),
                "hmm_unknown": pos_.get("hmm_brown_test", {}).get("unknown_accuracy"),
                "perceptron_brown": pos_.get("perceptron_brown_test")},
        "spelling": [{"config": k, "accuracy": v["accuracy"]} for k, v in sp.get("configs", {}).items()],
        "lexicon": {"spearman_lex": lc.get("full", {}).get("spearman_net"),
                    "spearman_vader": lc.get("full", {}).get("spearman_vader"),
                    "erisk_auc": le.get("variants", {}).get("full", {}).get("auc_mal"),
                    "erisk_users": le.get("n_users")},
    }


@app.get("/api/results")
def results():
    return _results()


# ====================================================================== TTS
class TtsReq(BaseModel):
    text: str
    voice: str = "en-US-AvaMultilingualNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"


VOICES = [
    {"id": "en-US-AvaMultilingualNeural", "name": "Ava", "accent": "US", "gender": "F"},
    {"id": "en-US-AndrewMultilingualNeural", "name": "Andrew", "accent": "US", "gender": "M"},
    {"id": "en-US-EmmaMultilingualNeural", "name": "Emma", "accent": "US", "gender": "F"},
    {"id": "en-US-BrianMultilingualNeural", "name": "Brian", "accent": "US", "gender": "M"},
    {"id": "en-IN-NeerjaExpressiveNeural", "name": "Neerja", "accent": "India", "gender": "F"},
    {"id": "en-IN-PrabhatNeural", "name": "Prabhat", "accent": "India", "gender": "M"},
    {"id": "en-GB-SoniaNeural", "name": "Sonia", "accent": "UK", "gender": "F"},
    {"id": "en-GB-RyanNeural", "name": "Ryan", "accent": "UK", "gender": "M"},
]
_tts_cache = {}
RATE_RE = re.compile(r"^[+-]\d{1,3}%$")
PITCH_RE = re.compile(r"^[+-]\d{1,3}Hz$")


@app.get("/api/voices")
def voices():
    return VOICES


@app.post("/api/tts")
async def tts(req: TtsReq):
    """Neural speech + word timings (for live captions) from Edge's free TTS service."""
    text = re.sub(r"\s+", " ", req.text).strip()[:4000]
    if not text:
        raise HTTPException(400, "empty text")
    if req.voice not in {v["id"] for v in VOICES}:
        raise HTTPException(400, "unknown voice")
    rate = req.rate if RATE_RE.match(req.rate) else "+0%"
    pitch = req.pitch if PITCH_RE.match(req.pitch) else "+0Hz"
    key = hashlib.sha1(json.dumps([text, req.voice, rate, pitch]).encode()).hexdigest()
    if key in _tts_cache:
        return _tts_cache[key]

    async def synth():
        com = edge_tts.Communicate(text, req.voice, rate=rate, pitch=pitch, boundary="WordBoundary")
        audio, words = bytearray(), []
        async for chunk in com.stream():
            if chunk["type"] == "audio":
                audio.extend(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                words.append({"t": chunk["offset"] / 1e4, "d": chunk["duration"] / 1e4, "text": chunk["text"]})
        return audio, words

    # the free service occasionally stalls: time out and retry instead of hanging
    last = None
    for attempt in range(3):
        try:
            audio, words = await asyncio.wait_for(synth(), timeout=12 + 6 * attempt)
            if audio:
                break
        except Exception as e:  # timeout, network down...
            last = e
    else:
        # the frontend falls back to the browser voice
        raise HTTPException(502, f"TTS service unavailable: {last!r}")
    out = {"audio": base64.b64encode(bytes(audio)).decode(), "mime": "audio/mpeg", "words": words}
    if len(_tts_cache) > 200:
        _tts_cache.clear()
    _tts_cache[key] = out
    return out


# Load the heavy resources before serving. NLTK's lazy WordNet loader is not
# thread-safe, so it must not be first touched by several requests at once.
@app.on_event("startup")
def _warm():
    wn.ensure_loaded()
    wn.lemma("happy.a.01.happy").count()   # loads the SemCor count file
    engine()
    resources.speller()
    bigram_index()
    _zipf()
    brown_ic()
    vader()
