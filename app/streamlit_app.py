"""DriftLens prototype.

Run from the project folder:
    streamlit run app/streamlit_app.py
"""
import html
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nltk  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from driftlens import resources  # noqa: E402
from driftlens.data import load_clpsych  # noqa: E402
from driftlens.features import retag  # noqa: E402
from driftlens.lexicon import adapted_lesk  # noqa: E402
from driftlens.ngram import cross_entropies, PersonalLM  # noqa: E402
from driftlens.pipeline import STATE_NAMES, DriftLens, split_custom  # noqa: E402
from driftlens.spelling import weighted_distance  # noqa: E402
from driftlens.textproc import expand_contractions, lm_tokens  # noqa: E402

st.set_page_config(page_title="DriftLens", page_icon="📈", layout="wide")

STATE_COLOURS = {"O": "#8f8d87", "S": "#2a78d6", "E": "#eb6834", "B": "#1baf7a"}
EXAMPLE = """Finally finished my first semester at college. Tired but honestly proud of myself, my roommate and I celebrated with pizza.
---
Anyone have tips for organising notes? I keep losing track of deadlines and my planner is a mess lol.
---
Weird week. My dad lost his job and everyone at home is on edge. I've been trying to stay calm but I cant sleep.
---
I don't know why I'm writing this. I feel so empty and worthless lately. I skipped classes all week and I just stay in bed. Nobody would notice if I disappeared.
---
It keeps getting worse. I hate myself, I'm a failure and I'm so tired of pretending I'm fine. What's the point of any of this.
---
Update: I talked to my sister last night and she drove down to see me. I booked an appointment with the campus counsellor. Still scared but feeling a bit more hopeful."""


# ------------------------------------------------------------------ caching
@st.cache_resource(show_spinner="Loading language resources (lexicon, HMM tagger, n-gram LM, speller)...")
def get_engine():
    return DriftLens()


@st.cache_data(show_spinner=False)
def get_timelines(split):
    return {tl.timeline_id: tl for tl in load_clpsych(split)}


# ------------------------------------------------------------------ helpers
def highlight(text, hits, show_corrections=True):
    """Render lexicon hits as coloured spans (hits carry character offsets)."""
    out, pos = [], 0
    for h in sorted(hits, key=lambda x: x["start"]):
        if h["start"] < pos:
            continue
        out.append(html.escape(text[pos:h["start"]]))
        frag = html.escape(text[h["start"]:h["end"]])
        tip = f"{h['cat'].replace('_', ' ')} · weight {h['w']:.2f}"
        if h.get("corrected_from") and show_corrections:
            tip += f" · spelling corrected from '{h['corrected_from']}'"
        if h["negated"] and h["w"] == 0:
            style = "text-decoration:line-through;opacity:.6;"
            tip += " · negated (ignored)"
        elif h["cat"] == "crisis":
            style = "background:#dc2626;color:#fff;font-weight:600;"
        elif h["pol"] == "mal":
            style = "background:rgba(220,38,38,.18);border-bottom:2px solid #dc2626;"
        else:
            style = "background:rgba(22,163,74,.18);border-bottom:2px solid #16a34a;"
        if h.get("corrected_from") and show_corrections:
            style += "text-decoration:underline dotted;"
        out.append(f'<span title="{html.escape(tip)}" style="{style}padding:0 2px;border-radius:3px">{frag}</span>')
        pos = h["end"]
    out.append(html.escape(text[pos:]))
    return "<div style='line-height:1.7;font-size:0.97rem'>" + "".join(out) + "</div>"


def badge(state, prob=None):
    label = STATE_NAMES[state] + (f" · {prob:.0%}" if prob is not None else "")
    return (f"<span style='background:{STATE_COLOURS[state]};color:white;padding:2px 10px;"
            f"border-radius:12px;font-size:0.8rem;font-weight:600'>{html.escape(label)}</span>")


def timeline_figure(res, gold=None):
    an = res["analyses"]
    x = list(range(1, len(an) + 1))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06)
    net = [a["features"]["net"] for a in an]
    nov = [a["features"]["novelty"] for a in an]
    vad = [a["features"]["vader"] * 10 for a in an]
    fig.add_trace(go.Scatter(x=x, y=net, name="Self-state balance (adaptive − maladaptive)", mode="lines+markers",
                             line=dict(color="#9085e9", width=3)), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=vad, name="VADER compound ×10", mode="lines",
                             line=dict(color="#64748b", dash="dot")), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=[n * 10 for n in nov], name="Personal LM novelty ×10", mode="lines+markers",
                             line=dict(color="#eda100", width=2)), 1, 1)
    fig.add_hline(y=0, line_color="rgba(120,120,120,.4)", row=1, col=1)
    rows = [("Predicted", res["pred"])] + ([("Gold", gold)] if gold else [])
    for label, seq in rows:
        fig.add_trace(go.Scatter(
            x=x, y=[label] * len(x), mode="markers", showlegend=False,
            marker=dict(size=18, symbol="square", color=[STATE_COLOURS[s] for s in seq]),
            hovertext=[STATE_NAMES[s] for s in seq], hoverinfo="text+x"), 2, 1)
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.08),
                      hovermode="x unified")
    fig.update_xaxes(title_text="Post number", dtick=1, row=2, col=1)
    return fig


# ------------------------------------------------------------------ sidebar
engine = get_engine()
st.sidebar.title("DriftLens")
st.sidebar.caption("Moments of change in online timelines, detected with classical NLP.")
source = st.sidebar.radio("Timeline source", ["Write your own", "CLPsych sample (labelled)", "CLPsych test (unlabelled)"])
model_names = list(engine.models)
model_name = st.sidebar.selectbox("Sequence model", model_names, index=model_names.index("CRF (prior-corrected)"),
                                  help="CRF and Gaussian HMM are tied in cross-validation; the CRF gives "
                                       "better-calibrated probabilities (the HMM's saturate near 0 or 1).")
show_corr = st.sidebar.checkbox("Mark spelling corrections", value=True)
st.sidebar.divider()
st.sidebar.markdown(
    "<span style='background:rgba(220,38,38,.18);border-bottom:2px solid #dc2626;padding:0 4px'>maladaptive</span> "
    "<span style='background:rgba(22,163,74,.18);border-bottom:2px solid #16a34a;padding:0 4px'>adaptive</span> "
    "<span style='background:#dc2626;color:#fff;padding:0 4px'>crisis</span> "
    "<span style='text-decoration:line-through;opacity:.6'>negated</span>", unsafe_allow_html=True)
st.sidebar.markdown("".join(badge(s) + " " for s in STATE_NAMES), unsafe_allow_html=True)
st.sidebar.divider()
st.sidebar.caption("Research prototype. Not a diagnostic or crisis tool. "
                   "In India, Tele-MANAS offers free 24x7 support at **14416**.")

posts, gold, title = [], None, ""
if source == "Write your own":
    raw = st.sidebar.text_area("Posts (separate posts with a line containing ---)", EXAMPLE, height=300)
    posts = [{"text": t, "date": None} for t in split_custom(raw)]
    title = "Your timeline"
else:
    split = "train" if "labelled" in source and "unlabelled" not in source else "test"
    tls = get_timelines(split)
    tid = st.sidebar.selectbox("Timeline", list(tls), format_func=lambda k: f"{k} ({len(tls[k].posts)} posts)")
    tl = tls[tid]
    posts = [{"text": p.text, "date": p.date} for p in tl.posts]
    gold = tl.states if split == "train" else None
    title = f"Timeline {tid}"

tab_tl, tab_digest, tab_tools, tab_about = st.tabs(["Timeline analysis", "Digest", "NLP toolbox", "About"])

# ------------------------------------------------------------------ analysis
with tab_tl:
    if len(posts) < 2:
        st.info("Add at least two posts to analyse a timeline.")
    else:
        with st.spinner("Analysing timeline..."):
            res = engine.analyse(posts, model=model_name)
        pred, marg = res["pred"], res["marginals"]
        st.subheader(title)
        c = st.columns(4)
        c[0].metric("Posts", len(posts))
        c[1].metric("Predicted switches", sum(s in "SB" for s in pred))
        c[2].metric("Predicted escalation posts", sum(s in "EB" for s in pred))
        if gold:
            agree = np.mean([a == b for a, b in zip(pred, gold)])
            c[3].metric("Agreement with gold", f"{agree:.0%}")
        else:
            c[3].metric("Crisis-language posts", sum(a["lex"]["crisis"] > 0 for a in res["analyses"]))
        st.plotly_chart(timeline_figure(res, gold))
        if gold and model_name in ("Gaussian HMM", "CRF (prior-corrected)", "Hybrid HMM (balanced)", "MaxEnt"):
            st.caption("Note: sample timelines are part of the training data of the deployed model, "
                       "so agreement here is optimistic. Cross-validated scores are in the About tab.")
        for i, (p, a) in enumerate(zip(posts, res["analyses"])):
            k = res["states"].index(pred[i])
            head = f"Post {i + 1}"
            if p.get("date"):
                head += f" · {p['date']:%d %b %Y}"
            head += f" · predicted {STATE_NAMES[pred[i]]}"
            if gold:
                head += f" · gold {STATE_NAMES[gold[i]]}"
            with st.expander(head, expanded=(pred[i] != "O")):
                st.markdown(badge(pred[i], float(marg[i][k])) + (" &nbsp; gold: " + badge(gold[i]) if gold else ""),
                            unsafe_allow_html=True)
                st.markdown(highlight(a["lex"]["clean"], a["lex"]["hits"], show_corr), unsafe_allow_html=True)
                c1, c2, c3 = st.columns([1.1, 1, 1])
                with c1:
                    st.markdown("**Self-evaluations (chunk grammar)**")
                    ev = [e for e in a["self_evals"] if e["polarity"] != "neutral"]
                    if ev:
                        for e in ev:
                            icon = "🔻" if e["polarity"] == "neg" else "🔺"
                            st.markdown(f"{icon} _{html.escape(e['text'])}_")
                    else:
                        st.caption("none found")
                    if a["lex"]["corrections"]:
                        st.markdown("**Spelling (noisy channel)**")
                        st.caption(", ".join(f"{o} → {n}" for o, n in a["lex"]["corrections"][:8]))
                with c2:
                    f = a["features"]
                    st.markdown("**Signals**")
                    st.dataframe(pd.DataFrame({
                        "signal": ["self-state balance", "VADER", "personal novelty (bits)", "new-vocabulary share",
                                   "length vs own usual (z)", "first-person rate", "days since previous"],
                        "value": [f["net"], f["vader"], f["novelty"], f["vocab_novelty"], f["z_len"], f["fp_rate"],
                                  math.expm1(f["gap_log"])]}).round(2), hide_index=True)
                with c3:
                    st.markdown("**State probabilities**")
                    st.dataframe(pd.DataFrame({"state": [STATE_NAMES[s] for s in res["states"]],
                                               "p": marg[i]}).round(3), hide_index=True,
                                 column_config={"p": st.column_config.ProgressColumn("p", min_value=0, max_value=1)})

# ------------------------------------------------------------------ digest
with tab_digest:
    if len(posts) >= 2:
        res = engine.analyse(posts, model=model_name)
        st.subheader("Timeline digest (affect-biased TextRank)")
        st.caption("Extractive summary: sentences ranked by PageRank over TF-IDF similarity, "
                   "with teleport mass biased toward sentences containing self-state words.")
        for idx, sent in res["digest"]:
            st.markdown(f"> **Post {idx + 1}** ({STATE_NAMES[res['pred'][idx]]}): {html.escape(sent)}")
        cats = {}
        for a in res["analyses"]:
            for h in a["lex"]["hits"]:
                if h["w"] > 0:
                    cats[h["cat"]] = cats.get(h["cat"], 0) + h["w"]
        if cats:
            st.markdown("**Self-state categories across the timeline (summed lexicon weight)**")
            df = pd.DataFrame(sorted(cats.items(), key=lambda x: -x[1]), columns=["category", "weight"])
            st.bar_chart(df, x="category", y="weight", horizontal=True)

# ------------------------------------------------------------------ toolbox
with tab_tools:
    t1, t2, t3, t4 = st.tabs(["Spelling", "N-gram LM",
                              "POS, chunking & CFG", "WordNet & WSD"])
    sp, lm, tg = resources.speller(), resources.bg_lm(), resources.tagger()

    with t1:
        st.markdown("Noisy channel: **w\\* = argmax log P(w | context) + log P(x | w)** with a keyboard-aware "
                    "weighted Damerau–Levenshtein channel.")
        c1, c2, c3 = st.columns(3)
        prev = c1.text_input("Previous word", "feel")
        word = c2.text_input("Typed word", "lonley")
        nxt = c3.text_input("Next word", "and")
        w = word.lower().strip()
        if w:
            if not sp.needs_check(w):
                st.success(f"'{w}' is a dictionary word: no correction needed.")
            ranked = sp.rank(w, prev.lower() or None, nxt.lower() or None, weighted=True, context=True, keep=True)[:8]
            if ranked:
                st.dataframe(pd.DataFrame([{
                    "candidate": c, "weighted edit distance": round(weighted_distance(w, c), 2),
                    "Levenshtein": round(weighted_distance(w, c, weighted=False), 2),
                    "log channel": round(ch, 2), "log P(w|context)": round(pr, 2), "score": round(s, 2)}
                    for c, s, ch, pr in ranked]), hide_index=True)
        sent = st.text_input("Correct a whole sentence", "i realy dont know why i feel so hopless and anxous")
        toks = [t.lower() for t in expand_contractions(nltk.word_tokenize(sent))]
        fixed, changes = sp.correct_tokens(toks)
        st.markdown("**Corrected:** " + " ".join(fixed))
        if changes:
            st.caption("Changes: " + ", ".join(f"{o} → {n}" for _, o, n in changes))

    with t2:
        st.markdown("Background bigram LM (Jelinek–Mercer interpolation) trained on 1.5M tokens of eRisk Reddit text.")
        s = st.text_input("Sentence", "i feel so tired and alone tonight")
        toks = lm_tokens(s)
        if toks:
            rows, prev_t = [], "<s>"
            for t in toks + ["</s>"]:
                p = lm.p_interp(lm.map(t), [lm.map(prev_t)])
                rows.append({"token": t, "in vocab": t in lm.vocab or t == "</s>", "P(w | prev)": p,
                             "−log2 P": -math.log2(p)})
                prev_t = t
            df = pd.DataFrame(rows)
            st.dataframe(df.round(5), hide_index=True)
            pp = 2 ** df["−log2 P"].mean()
            st.metric("Perplexity", f"{pp:.1f}")
        st.markdown("---\n**Personal novelty** — how surprising is a new post given the same writer's history?")
        hist = st.text_area("Earlier posts by the same user (one per line)",
                            "just finished my shift, time for some gaming\nnew keyboard arrived, loving it\n"
                            "anyone playing the new update tonight?")
        new = st.text_input("New post", "i dont want to be here anymore, nothing feels real")
        plm = PersonalLM(lm)
        for line in hist.splitlines():
            if line.strip():
                plm.add(lm_tokens(line))
        if plm.n and lm_tokens(new):
            h_bg, h_mix = cross_entropies(lm_tokens(new), lm, plm, beta=0.3)
            c = st.columns(3)
            c[0].metric("H background (bits/token)", f"{h_bg:.2f}")
            c[1].metric("H with personal history", f"{h_mix:.2f}")
            c[2].metric("Novelty = H_mix − H_bg", f"{h_mix - h_bg:+.3f}")
            st.caption("Negative: the post reuses the writer's habitual language. Positive (up to −log2(1−β) ≈ "
                       "0.51): the personal history does not help, i.e. the writer sounds unlike themselves.")

    with t3:
        s = st.text_input("Sentence to tag", "I am not happy anymore and I hate myself")
        toks = expand_contractions(nltk.word_tokenize(s))
        tagged = tg.tag(toks)
        perc = nltk.pos_tag(toks, tagset="universal")
        st.markdown("**HMM (Viterbi) vs NLTK averaged perceptron**")
        st.dataframe(pd.DataFrame({"word": toks, "HMM tag": [t for _, t in tagged],
                                   "perceptron tag": [t for _, t in perc]}).T, hide_index=False)
        fx = engine.fx
        evs = fx.self_evaluations(tagged)
        st.markdown("**Self-evaluation chunks** (cascaded regular grammar over re-tagged tokens)")
        st.code(fx.chunker.parse(retag(tagged)).pformat(margin=100), language=None)
        for e in evs:
            st.markdown(f"- _{e['text']}_ → **{e['polarity']}**")
        st.markdown("**Context-free grammar for self-statements** — top-down vs bottom-up parsing")
        grammar = nltk.CFG.fromstring("""
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
        clause = st.text_input("Self-statement to parse", "I am not very happy")
        ct = retag(tg.tag(expand_contractions(nltk.word_tokenize(clause))))
        ct = [(w, t) for w, t in ct if t != "."]
        tags = [t for _, t in ct]
        st.caption("Tag sequence: " + " ".join(tags))
        cols = st.columns(2)
        for col, (name, parser) in zip(cols, [("Top-down (recursive descent)", nltk.RecursiveDescentParser(grammar)),
                                               ("Bottom-up (chart)", nltk.BottomUpChartParser(grammar))]):
            with col:
                st.markdown(f"*{name}*")
                try:
                    trees = list(parser.parse(tags))
                except ValueError as e:
                    trees = []
                    st.warning(str(e))
                if trees:
                    tr = trees[0].copy(deep=True)
                    for pos, (w, _) in zip(tr.treepositions("leaves"), ct):
                        tr[pos] = w
                    st.code(tr.pformat(margin=60), language=None)
                else:
                    st.info("No parse: the grammar only covers simple self-evaluations — a limitation of "
                            "hand-written CFGs that the finite-state chunker above avoids.")
        with st.expander("Grammar in Chomsky Normal Form"):
            st.code("\n".join(str(p) for p in grammar.chomsky_normal_form().productions()), language=None)

    with t4:
        from nltk.corpus import wordnet as wn, wordnet_ic
        lex = engine.fx.lexicon
        w = st.text_input("Word", "hopeless").lower().strip()
        if w:
            key = lex.lookup(w)
            if key:
                e = lex.entries[key]
                st.success(f"In lexicon as **{key}** → category **{e['cat']}** ({'adaptive' if e['pol'] == 'ada' else 'maladaptive'}), "
                           f"weight {e['w']}, sense ratio {e['ratio']}")
            else:
                st.info("Not in the self-state lexicon.")
            syns = wn.synsets(w)
            st.dataframe(pd.DataFrame([{"synset": s.name(), "definition": s.definition(),
                                        "in lexicon": bool(key and s.name() in lex.entries[key]["senses"])}
                                       for s in syns]), hide_index=True)
        st.markdown("**Word similarity** (noun or verb senses; IC from the Brown corpus)")
        c1, c2 = st.columns(2)
        a_w = c1.text_input("Word A", "sadness")
        b_w = c2.text_input("Word B", "grief")
        sa, sb = wn.synsets(a_w), wn.synsets(b_w)
        if sa and sb:
            ic = wordnet_ic.ic("ic-brown.dat")
            best = max(((x, y) for x in sa for y in sb if x.pos() == y.pos()),
                       key=lambda p: p[0].path_similarity(p[1]) or 0, default=None)
            if best:
                x, y = best
                rows = {"pair": f"{x.name()} – {y.name()}", "path": x.path_similarity(y),
                        "Wu-Palmer": x.wup_similarity(y)}
                if x.pos() in "nv":
                    rows.update({"Leacock-Chodorow": x.lch_similarity(y), "Resnik": x.res_similarity(y, ic),
                                 "Lin": x.lin_similarity(y, ic), "Jiang-Conrath": x.jcn_similarity(y, ic)})
                st.dataframe(pd.DataFrame([rows]).round(3), hide_index=True)
        st.markdown("**Word sense disambiguation**")
        wsd_s = st.text_input("Sentence", "I have been so anxious to see my friends again after the holidays")
        target = st.text_input("Target word", "anxious").lower()
        ctx = [t.lower() for t in nltk.word_tokenize(wsd_s)]
        from nltk.wsd import lesk
        simple = lesk(ctx, target)
        key = lex.lookup(target)
        reached = set(lex.entries[key]["senses"]) if key else set()
        affect = any(t != target and lex.lookup(t) for t in ctx)
        adapted = adapted_lesk(target, ctx, reached, affect)
        c1, c2 = st.columns(2)
        c1.markdown(f"*NLTK simplified Lesk*: `{simple.name() if simple else None}`  \n"
                    f"{simple.definition() if simple else ''}")
        c2.markdown(f"*Adapted Lesk (DriftLens)*: `{adapted.name() if adapted else None}`  \n"
                    f"{adapted.definition() if adapted else 'no evidence → fall back to prior weight'}")
        if adapted is not None and key:
            st.caption("Counted as self-state evidence." if adapted.name() in reached
                       else "Rejected: the chosen sense is not an affective sense.")

# ------------------------------------------------------------------ about
with tab_about:
    st.subheader("How DriftLens works")
    st.graphviz_chart("""
    digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Helvetica"];
      raw [label="Timeline of posts"]; clean [label="Cleaning, segmentation\\nnoisy-channel spelling"];
      lex [label="WordNet self-state lexicon\\n+ adapted Lesk + negation FSA"];
      pos [label="HMM POS tagger\\n+ self-evaluation grammar"]; lm [label="n-gram LM\\npersonal novelty"];
      vad [label="VADER sentiment"]; feat [label="25 per-post features\\n+ personal baselines"];
      seq [label="Sequence labeller\\nHMM / CRF (Viterbi)"]; out [label="Per-post state\\nO / S / E / B"];
      dig [label="TextRank digest"];
      raw -> clean -> {lex pos lm vad} -> feat -> seq -> out; clean -> dig; }""")
    st.markdown("""
| Area | Where it is used in DriftLens |
|---|---|
| Overview, NLTK | tokenisation, sentence segmentation, NLTK corpora |
| Text processing | Reddit cleaning, weighted edit distance, noisy-channel spelling correction |
| N-gram LMs | unigram/bigram/trigram LMs, Laplace, add-k, interpolation, perplexity, personal novelty |
| Morphology & CFG | morphy (inflection), derivational suffix FSA, negation-scope FSA, chunk grammar, CFG + CNF |
| POS tagging | HMM tagger with Viterbi; HMM, MaxEnt and CRF for timeline states |
| Lexical semantics | WordNet expansion, Wu-Palmer / Lin similarity, sense ratio, adapted Lesk WSD |
| Applications | sentiment (VADER vs lexicon), information extraction (self-evaluations), TextRank summarisation |
""")
    import json
    rp = ROOT / "results" / "sequence_eval.json"
    if rp.exists():
        r = json.loads(rp.read_text())
        rows = [{"model": k, "macro-F1": f"{v['macro_f1'][0]:.3f} ± {v['macro_f1'][1]:.3f}",
                 "F1 switch": round(v["f1_switch"][0], 3), "F1 escalation": round(v["f1_escalation"][0], 3)}
                for k, v in r["main"].items()]
        rows.append({"model": "MaxEnt, length only", "macro-F1": f"{r['length_only']['MaxEnt']['macro_f1'][0]:.3f}",
                     "F1 switch": round(r["length_only"]["MaxEnt"]["f1_switch"][0], 3),
                     "F1 escalation": round(r["length_only"]["MaxEnt"]["f1_escalation"][0], 3)})
        st.markdown(f"**Cross-validated results** ({r['n_timelines']} timelines, {r['n_posts']} posts, "
                    "5-fold × 5 repeats, split by timeline)")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.warning("DriftLens is a research prototype. It is not a diagnostic instrument and must not be used "
               "to make decisions about real people. If you or someone you know is struggling, contact "
               "Tele-MANAS (14416 or 1-800-891-4416, free, 24x7) or local emergency services.")
