# DriftLens: detecting moments of change in social-media timelines with classical NLP

DriftLens treats a user's sequence of posts the way a POS tagger treats a sentence. Each post gets a hidden
state: **O** no change, **S** switch (sudden mood shift), **E** escalation (gradual worsening or improvement),
**B** both. The states are decoded with HMM / CRF sequence models (Viterbi). They are fed by features built only
from classical NLP techniques:

| Area | Technique used |
|---|---|
| Text processing | NLTK tokenisation, Reddit cleaning, weighted Damerau–Levenshtein, noisy-channel spelling correction |
| Language models | unigram/bigram/trigram LMs, Laplace, add-k, Jelinek–Mercer interpolation, perplexity, **personal LM novelty** |
| Morphology & grammar | WordNet morphology, derivational suffix FSA, negation-scope FSA, chunk grammar, CFG + CNF (demo) |
| Sequence labelling | HMM POS tagger with Viterbi (from scratch); HMM, MaxEnt, MaxEnt–HMM hybrid and CRF for timeline states |
| Lexical semantics | WordNet-grown self-state lexicon, Wu–Palmer / Lin similarity, sense ratio, adapted Lesk WSD |
| Applications | VADER vs lexicon sentiment, information extraction of self-evaluations, affect-biased TextRank digest |

## Folder layout

```
driftlens/        library: textproc, spelling, ngram, postag, lexicon, features, sequence, evaluation,
                  summarize, pipeline, resources, data, paths
scripts/          prepare_erisk → build_models → extract_features → eval_* → train_final → make_figures
app/              Streamlit prototype (streamlit_app.py)
models/           trained resources (lexicon JSON, HMM tagger, bigram LM, speller, timeline models)
data/processed/   derived data (eRisk target writings, cached CLPsych features); do not share
results/          metrics JSON, logs, figures/ (PDF for LaTeX, PNG for slides)
paper/            Springer LNCS paper (driftlens_lncs.tex, 18 pp.) and IEEE conference paper
                  (driftlens_ieee.tex, 6 pp.), shared references.bib, compiled PDFs
presentation/     review deck (DriftLens_Review.pptx), its generator build_deck.js, app screenshots in assets/
```

## Run the prototype

```bash
cd 01_J_Component_DriftLens
pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

Open http://localhost:8501. The sidebar switches between your own text, labelled CLPsych timelines and
unlabelled test timelines. The *NLP toolbox* tab has one interactive panel per technique.

## Reproduce every number in the paper

```bash
python scripts/prepare_erisk.py        # ~30 s, extracts eRisk target-user writings
python scripts/build_models.py         # ~1 min, lexicon + HMM tagger + LMs + speller (+ LM/POS evals)
python scripts/extract_features.py     # ~20 s, CLPsych features for all ablation variants
python scripts/eval_spelling.py        # ~2 min
python scripts/eval_lexicon.py         # ~2 min (CLPsych + eRisk)
python scripts/eval_sequence.py        # ~35 min (5x5-fold CV, ablations, bootstrap)
python scripts/train_final.py          # models used by the app
python scripts/make_figures.py         # figures + length-stratified analysis
```

Paper: `cd paper && pdflatex driftlens_lncs && bibtex driftlens_lncs && pdflatex driftlens_lncs && pdflatex driftlens_lncs`
(Springer `llncs` class; also compiles on Overleaf).
IEEE version: same commands with `driftlens_ieee` (IEEEtran conference class). Its column-width figures come from
`python scripts/make_figures.py --ieee`.

Slides: `npm install pptxgenjs && node presentation/build_deck.js`

All random seeds are fixed. Everything runs on a CPU; no GPU or hosted API is used.

## Headline results (5 repeats × 5-fold CV over 30 timelines / 373 posts)

| System | Macro-F1 | F1 switch | F1 escalation |
|---|---|---|---|
| Majority (all O) | 0.266 | 0.000 | 0.000 |
| MaxEnt, post length only | 0.572 | 0.483 | 0.470 |
| MaxEnt, all features | 0.547 | 0.476 | 0.420 |
| Gaussian HMM, all features | 0.567 | 0.508 | 0.469 |
| CRF (prior-corrected), all features | 0.563 | 0.472 | 0.481 |

The honest summary: with 30 labelled timelines, no model beats a post-length baseline on macro-F1 by a
statistically reliable margin. Sequence models help most on escalation, which comes in runs of posts. The
WordNet lexicon tracks well-being better than VADER (Spearman 0.43 vs 0.36) and separates depressed from
control users on eRisk 2026 without any training labels (AUC 0.84). See the paper for the full analysis.

## Ethics

The CLPsych and eRisk data are under data-use agreements and stay on this machine. The prototype is a research
demonstration, not a diagnostic or crisis tool. In India, Tele-MANAS (14416) offers free 24x7 support.
