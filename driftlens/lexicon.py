"""WordNet-grown self-state lexicon.

Construction
------------
1. Seeds are WordNet *synsets* (not bare words), so each seed fixes one sense:
   ``hopeless.a.01`` is the emotional sense, not "hopeless at maths".
2. Each seed is expanded along lexical relations (similar-to, also-see,
   attribute, derivational form, pertainym, verb group, hyponym). The weight
   decays multiplicatively per hop; nouns and verbs reached through hyponymy
   must also stay within a Wu-Palmer similarity threshold of their seed.
3. Every candidate word gets a *sense ratio*: the share of its WordNet usage
   (SemCor lemma counts, add-one smoothed) that falls in the reached synsets.
   Words whose everyday meaning is mostly unrelated ("blue", "down") get a low
   ratio and are down-weighted or dropped.
4. Expanded words are vetoed when SentiWordNet (synset level) or the VADER
   valence of the word clearly contradicts the category's polarity.

Scoring
-------
Tokens are looked up directly, after WordNet's inflectional morphology
(``morphy``) and, failing that, after a small derivational suffix stripper.
Words with a sense ratio below 0.7 are disambiguated with an adapted Lesk
algorithm inside their sentence and only counted if it picks a reached sense.
Negation scope is a two-state finite-state machine: a cue opens a scope that
closes after three words or at a clause boundary. A negated adaptive word
counts half towards the maladaptive side ("not happy"). A negated maladaptive
word is dropped, because "not sad" does not mean "happy".
"""
import json
import re
from collections import defaultdict

from nltk.corpus import wordnet as wn

from .textproc import clean

SEEDS = {
    # ---------------- maladaptive self-states
    "despair": ["hopeless.a.01", "gloomy.s.02", "miserable.s.01", "sad.a.01", "despair.n.01",
                "despair.n.02", "sadness.n.02", "grief.n.01", "cry.v.02", "empty.s.03", "dead.s.04"],
    "self_criticism": ["worthless.a.01", "useless.a.01", "pathetic.s.02", "stupid.a.01", "ashamed.a.01",
                       "shame.n.01", "guilt.n.02", "guilty.s.01", "failure.n.04", "hate.v.01"],
    "anxiety": ["anxious.s.02", "afraid.a.01", "frightened.s.01", "nervous.s.01", "anxiety.n.02",
                "panic.n.01", "fear.n.01", "apprehension.n.01", "worry.n.02", "worry.v.01", "panic.v.01"],
    "anger": ["angry.a.01", "angered.s.01", "annoyed.s.01", "resentful.a.01", "anger.n.01",
              "fury.n.01", "rage.v.03"],
    "isolation": ["alone.s.02", "lonely.s.02", "detached.s.02", "jilted.s.01", "unloved.a.01",
                  "loneliness.n.01", "forlornness.n.01", "isolation.n.02", "abandon.v.01", "reject.v.04"],
    "exhaustion": ["tired.a.01", "exhausted.s.01", "apathetic.s.02", "exhaustion.n.01", "fatigue.n.01",
                   "insomnia.n.01"],
    "crisis": ["self-destructive.s.01", "suicide.n.01"],
    # ---------------- adaptive self-states
    "hope": ["hopeful.a.01", "optimistic.a.01", "hope.n.02", "optimism.n.01", "hope.v.02"],
    "connection": ["grateful.a.01", "gratitude.n.01", "loved.a.01", "friend.n.01", "support.n.03",
                   "supported.a.01", "love.v.01", "thank.v.01"],
    "calm": ["calm.s.01", "relaxed.a.01", "peaceful.a.01", "contented.a.01", "safe.a.01", "relief.n.01"],
    "joy": ["happy.a.01", "glad.a.01", "excited.a.02", "joy.n.01", "happiness.n.02", "enjoy.v.01",
            "proud.a.01", "pride.n.01"],
    "agency": ["motivated.a.01", "determined.s.01", "achieve.v.01", "cope.v.01", "recuperate.v.04",
               "better.v.03", "advancement.n.03", "convalescence.n.01", "therapy.n.01", "pull_off.v.03"],
}
MALADAPTIVE = {"despair", "self_criticism", "anxiety", "anger", "isolation", "exhaustion", "crisis"}
POLARITY = {c: ("mal" if c in MALADAPTIVE else "ada") for c in SEEDS}
ANTONYM_TARGET = {"joy": "despair", "hope": "despair", "calm": "anxiety", "connection": "isolation",
                  "agency": "exhaustion", "despair": "joy", "anxiety": "calm", "isolation": "connection",
                  "exhaustion": "agency", "self_criticism": "agency", "anger": "calm"}

# Multi-word expressions WordNet cannot supply. Weight 1.5 marks crisis language.
PHRASES = [
    (r"\bkill(?:ing)? my ?self\b", "crisis", 1.5), (r"\bend(?:ing)? (?:it all|my life|things)\b", "crisis", 1.5),
    (r"\b(?:want|wanna|wanted|going|deserve) to die\b", "crisis", 1.5), (r"\bbetter off dead\b", "crisis", 1.5),
    (r"\bno reason to live\b", "crisis", 1.5), (r"\bself[- ]?harm\w*", "crisis", 1.5),
    (r"\bcut(?:ting)? my ?self\b", "crisis", 1.5), (r"\bsuicid\w*", "crisis", 1.5), (r"\boverdos\w*", "crisis", 1.5),
    (r"\bcan'?t go on\b", "crisis", 1.5), (r"\bdon'?t want to (?:live|be here|exist)\b", "crisis", 1.5),
    (r"\b(?:give|gave|giving) up\b", "exhaustion", 1.0), (r"\b(?:no ?one|nobody) cares\b", "isolation", 1.0),
    (r"\bhate my ?self\b", "self_criticism", 1.0), (r"\bno motivation\b", "exhaustion", 1.0),
    (r"\bfed up\b", "anger", 1.0), (r"\b(?:break|breaking|broke) down\b", "despair", 1.0),
    (r"\bcan'?t sleep\b", "exhaustion", 1.0), (r"\bnothing matters\b", "despair", 1.0),
    (r"\b(?:what'?s the|no) point\b", "despair", 1.0), (r"\bfeel(?:ing)? like (?:shit|crap)\b", "despair", 1.0),
    (r"\bfeel(?:ing|s)? better\b", "agency", 1.0), (r"\blook(?:ing)? forward\b", "hope", 1.0),
    (r"\breach(?:ed|ing)? out\b", "connection", 1.0), (r"\bthank(?:s| you)\b", "connection", 0.5),
    (r"\bproud of my ?self\b", "joy", 1.0), (r"\bgetting better\b", "agency", 1.0),
    (r"\bdoing (?:better|well|good)\b", "agency", 1.0), (r"\bgood day\b", "joy", 1.0),
    (r"\bmy (?:therapist|counsell?or|psychiatrist|psychologist)\b", "agency", 0.7),
]

NEG_CUES = {"not", "no", "never", "n't", "nothing", "nobody", "none", "neither", "nor", "without",
            "hardly", "cannot", "dont", "don't", "didn't", "doesn't", "isn't", "wasn't", "can't", "won't",
            "aren't", "weren't", "couldn't", "wouldn't", "shouldn't", "haven't", "hasn't"}
SCOPE_BREAKERS = {"but", "however", "though", "although", "yet"}
TOKEN_RE = re.compile(r"[A-Za-z]+(?:['\-][A-Za-z]+)*")
BOUNDARY_RE = re.compile(r"[.,;:!?()\n]")
SENT_END_RE = re.compile(r"[.!?\n]")
DERIV_RULES = [("iness", "y"), ("ness", ""), ("fully", "ful"), ("lessly", "less"), ("ily", "y"),
               ("ly", ""), ("ment", ""), ("ation", "e"), ("ed", ""), ("ing", "")]
STOP = {"be", "have", "do", "get", "go", "make", "one", "like", "well", "good", "right", "so", "use",
        "work", "time", "way", "thing", "feel", "feeling", "think", "want", "need", "know"}


# ============================================================ construction
def _neighbours(s):
    """Yield (synset, hop_weight, relation) for expansion."""
    pos = s.pos()
    if pos in ("a", "s"):
        for n in s.similar_tos():
            yield n, 0.85, "similar_to"
        for n in s.also_sees():
            yield n, 0.7, "also_see"
    if pos in ("n", "v"):
        for n in s.hyponyms():
            yield n, 0.8, "hyponym"
    if pos == "v":
        for n in s.verb_groups():
            yield n, 0.8, "verb_group"
    for n in s.attributes():
        yield n, 0.75, "attribute"
    for lem in s.lemmas():
        for d in lem.derivationally_related_forms():
            yield d.synset(), 0.85, "derivation"
        for p in lem.pertainyms():
            yield p.synset(), 0.8, "pertainym"


def _wup(a, b):
    try:
        return a.wup_similarity(b) or 0.0
    except Exception:
        return 0.0


def grow(seeds=SEEDS, max_hops=2, min_weight=0.45, min_wup=0.6):
    """Return {synset_name: (category, weight, relation_path)} reached from the seeds."""
    reached = {}

    def offer(syn, cat, w, path):
        cur = reached.get(syn.name())
        if cur is None or w > cur[1]:
            reached[syn.name()] = (cat, w, path)
            return True
        return False

    for cat, names in seeds.items():
        for name in names:
            seed = wn.synset(name)
            offer(seed, cat, 1.0, "seed")
            frontier = [(seed, 1.0, "seed")]
            for _ in range(max_hops):
                nxt = []
                for s, w, path in frontier:
                    for n, hw, rel in _neighbours(s):
                        nw = w * hw
                        if nw < min_weight:
                            continue
                        # crisis terms only admit morphological relatives: similar-to
                        # neighbours of "suicidal" drift to "dangerous", "risky", ...
                        if cat == "crisis" and rel != "derivation":
                            continue
                        if rel == "hyponym" and _wup(seed, n) < min_wup:
                            continue
                        if offer(n, cat, nw, f"{path}>{rel}"):
                            nxt.append((n, nw, f"{path}>{rel}"))
                frontier = nxt
            # antonyms of the seed seed the opposite polarity with a lower weight
            if cat in ANTONYM_TARGET:
                for lem in seed.lemmas():
                    for ant in lem.antonyms():
                        offer(ant.synset(), ANTONYM_TARGET[cat], 0.6, "antonym")
    return reached


def _polarity_veto(word, synset_name, pol, vader_lex):
    """True when an independent polarity resource contradicts the category.

    Two checks, applied to expanded (non-seed) words only: SentiWordNet at the
    synset level and the VADER valence of the surface word.
    """
    from nltk.corpus import sentiwordnet as swn
    try:
        s = swn.senti_synset(synset_name)
        pos, neg = s.pos_score(), s.neg_score()
        if pol == "ada" and neg - pos >= 0.25:
            return True
        if pol == "mal" and pos - neg >= 0.25:
            return True
    except Exception:
        pass
    v = vader_lex.get(word)
    if v is not None and ((pol == "ada" and v <= -1.0) or (pol == "mal" and v >= 1.0)):
        return True
    return False


def build_lexicon(seeds=SEEDS, min_ratio=0.2, min_seed_ratio=0.1, polarity_filter=True):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    vader_lex = SentimentIntensityAnalyzer().lexicon if polarity_filter else {}
    reached = grow(seeds)
    seed_names = {n for names in seeds.values() for n in names}
    cand = defaultdict(list)                      # word -> [(synset, cat, w)]
    vetoed = []
    for sname, (cat, w, path) in reached.items():
        syn = wn.synset(sname)
        for lem in syn.lemmas():
            word = lem.name().lower()
            # WordNet multi-word lemmas are mostly phrasal verbs ("back up", "all in")
            # that fire on neutral text, so only the curated PHRASES list is used.
            if "_" in word or len(word) < 3 or word in STOP:
                continue
            if (polarity_filter and sname not in seed_names
                    and _polarity_veto(word, sname, POLARITY[cat], vader_lex)):
                vetoed.append(word)
                continue
            cand[word].append((sname, cat, w))
    entries = {}
    for word, lst in cand.items():
        # only synsets that list the word itself; wn.synsets("depressed") would
        # otherwise also return the verb senses of "depress"
        all_syns = [s for s in wn.synsets(word) if any(l.name().lower() == word for l in s.lemmas())]
        reached_set = {s for s, _, _ in lst}

        def cnt(syn):
            return sum(l.count() for l in syn.lemmas() if l.name().lower() == word) + 1

        total = sum(cnt(s) for s in all_syns) or 1
        ratio = sum(cnt(s) for s in all_syns if s.name() in reached_set) / total
        is_seed = any(s in seed_names for s, _, _ in lst)
        if ratio < (min_seed_ratio if is_seed else min_ratio):
            continue
        # resolve the category by the strongest path; drop words torn between polarities
        by_pol = defaultdict(float)
        for s, c, w in lst:
            by_pol[POLARITY[c]] = max(by_pol[POLARITY[c]], w)
        if len(by_pol) == 2:
            hi, lo = sorted(by_pol.values(), reverse=True)
            if hi < 1.5 * lo:
                continue
        s_best, c_best, w_best = max(lst, key=lambda t: t[2])
        entries[word] = {
            "cat": c_best, "pol": POLARITY[c_best],
            "w": round(w_best * ratio ** 0.5, 3),
            "ratio": round(ratio, 3), "seed": is_seed,
            "senses": sorted(reached_set),
        }
    return {"entries": entries, "vetoed": sorted(set(vetoed) - set(entries))}


# ================================================================ WSD
_SIG_CACHE = {}
_STOPWORDS = None


def _stopwords():
    global _STOPWORDS
    if _STOPWORDS is None:
        from nltk.corpus import stopwords
        _STOPWORDS = set(stopwords.words("english")) | {"n't", "'s", "'m", "one", "someone", "something"}
    return _STOPWORDS


def _norm(tok):
    return wn.morphy(tok) or tok


def signature(syn):
    """Adapted-Lesk signature: gloss, examples and lemmas of the sense and its neighbours."""
    if syn.name() not in _SIG_CACHE:
        stop = _stopwords()
        related = [syn] + syn.similar_tos() + syn.hypernyms() + syn.also_sees()
        words = set()
        for s in related:
            text = " ".join([s.definition()] + s.examples() + [l.name().replace("_", " ") for l in s.lemmas()])
            words |= {_norm(t) for t in TOKEN_RE.findall(text.lower()) if t not in stop}
        _SIG_CACHE[syn.name()] = words
    return _SIG_CACHE[syn.name()]


def adapted_lesk(word, context, reached, affect_context=False):
    """Pick a WordNet sense of ``word`` for its context.

    Score = |signature(sense) & context| (+1 for senses in the lexicon when
    other affect words occur nearby, a one-domain-per-context bonus).
    Ties go to the more frequent sense (SemCor counts, then WordNet order).
    Returns None when no sense has any evidence.
    """
    stop = _stopwords()
    ctx = {_norm(t) for t in context if t not in stop and t != word}
    best, best_key = None, None
    for rank, s in enumerate(wn.synsets(word)):
        if not any(l.name().lower() == word for l in s.lemmas()):
            continue
        score = len(signature(s) & ctx) + (1 if (affect_context and s.name() in reached) else 0)
        freq = sum(l.count() for l in s.lemmas() if l.name().lower() == word)
        key = (score, freq, -rank)
        if best_key is None or key > best_key:
            best, best_key = s, key
    if best is None or best_key[0] == 0:
        return None
    return best


# ================================================================ scoring
class Lexicon:
    def __init__(self, entries, use_phrases=True):
        self.entries = entries
        self.phrases = [(re.compile(p), c, w) for p, c, w in PHRASES] if use_phrases else []

    @classmethod
    def load(cls, path, **kw):
        d = json.loads(open(path, encoding="utf-8").read())
        return cls(d["entries"], **kw)

    @classmethod
    def seeds_only(cls):
        """Baseline: seed synset lemmas only, no expansion, no ratio filter."""
        entries = {}
        for cat, names in SEEDS.items():
            for n in names:
                for lem in wn.synset(n).lemmas():
                    word = lem.name().lower()
                    if "_" not in word and word not in STOP:
                        entries[word] = {"cat": cat, "pol": POLARITY[cat], "w": 1.0, "ratio": 1.0,
                                         "seed": True, "senses": [n]}
        return cls(entries, use_phrases=False)

    def lookup(self, tok):
        t = tok.lower()
        if t in self.entries:
            return t
        for pos in ("a", "v", "n", "r"):
            m = wn.morphy(t, pos)
            if m and m in self.entries:
                return m
        for suf, rep in DERIV_RULES:
            if t.endswith(suf) and len(t) - len(suf) >= 3:
                stem = t[: len(t) - len(suf)] + rep
                if stem in self.entries:
                    return stem
                m = wn.morphy(stem)
                if m and m in self.entries:
                    return m
        return None

    def score(self, text, speller=None, use_wsd=True, use_negation=True):
        txt = clean(text)
        low = txt.lower()
        toks = [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(txt)]
        n_words = max(len(toks), 1)
        hits, corrections = [], []
        covered = set()

        # multi-word expressions first; their tokens are not scored twice
        for rx, cat, w in self.phrases:
            for m in rx.finditer(low):
                hits.append({"start": m.start(), "end": m.end(), "text": txt[m.start():m.end()],
                             "lemma": m.group(), "cat": cat, "pol": POLARITY[cat], "w": w,
                             "negated": False, "phrase": True})
                covered.update(range(m.start(), m.end()))

        neg_left = 0
        prev_end = 0
        lower_toks = [t.lower() for t, _, _ in toks]
        sent_id, sid, pe = [], 0, 0
        for tok, a, b in toks:
            if SENT_END_RE.search(txt[pe:a]):
                sid += 1
            sent_id.append(sid)
            pe = b
        for i, (tok, a, b) in enumerate(toks):
            # finite-state negation scope: NORMAL <-> NEGATED(k words left)
            if BOUNDARY_RE.search(txt[prev_end:a]) or lower_toks[i] in SCOPE_BREAKERS:
                neg_left = 0
            prev_end = b
            lt = lower_toks[i]
            if lt in NEG_CUES or lt.endswith("n't"):
                neg_left = 3
                continue
            negated = neg_left > 0
            if neg_left:
                neg_left -= 1
            if a in covered:
                continue
            key = self.lookup(lt)
            corrected_from = None
            proper_noun_like = tok[0].isupper() and i > 0 and sent_id[i - 1] == sent_id[i]
            if key is None and speller is not None and not proper_noun_like and speller.needs_check(lt):
                prev = lower_toks[i - 1] if i else None
                nxt = lower_toks[i + 1] if i + 1 < len(toks) else None
                fixed = speller.correct(lt, prev, nxt, context=True)
                if fixed != lt:
                    corrections.append((tok, fixed))
                    key = self.lookup(fixed)
                    corrected_from = tok if key else None
            if key is None:
                continue
            e = self.entries[key]
            w = e["w"]
            if use_wsd and e["ratio"] < 0.7:
                window = [lower_toks[j] for j in range(max(0, i - 8), min(len(toks), i + 9))
                          if sent_id[j] == sent_id[i]]
                affect = any(t != lt and self.lookup(t) for t in window)
                sense = adapted_lesk(key, window, set(e["senses"]), affect)
                if sense is not None and sense.name() not in e["senses"]:
                    continue
            pol, cat = e["pol"], e["cat"]
            if negated and use_negation:
                if pol == "mal":
                    hits.append({"start": a, "end": b, "text": tok, "lemma": key, "cat": cat, "pol": pol,
                                 "w": 0.0, "negated": True, "phrase": False, "corrected_from": corrected_from})
                    continue
                pol, cat, w = "mal", "negated_" + cat, w * 0.5
            hits.append({"start": a, "end": b, "text": tok, "lemma": key, "cat": cat, "pol": pol, "w": w,
                         "negated": negated and use_negation, "phrase": False,
                         "corrected_from": corrected_from})

        mal = sum(h["w"] for h in hits if h["pol"] == "mal")
        ada = sum(h["w"] for h in hits if h["pol"] == "ada")
        crisis = sum(1 for h in hits if h["cat"] == "crisis" and h["w"] > 0)
        per100 = 100.0 / n_words
        return {"clean": txt, "n_words": n_words, "mal": mal * per100, "ada": ada * per100,
                "net": (ada - mal) * per100, "crisis": crisis, "hits": sorted(hits, key=lambda h: h["start"]),
                "corrections": corrections}
