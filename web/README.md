# DriftLens web frontend

A Next.js frontend for DriftLens, backed by `../api/server.py`, a FastAPI wrapper around the same
`driftlens` library that the Streamlit prototype uses. No model logic lives in the frontend: every number
on screen comes from the Python code.

## Run

Double-click `../start_web.bat`. Or start the two parts by hand:

```bash
# terminal 1, from 01_J_Component_DriftLens/
pip install fastapi uvicorn edge-tts
python -m uvicorn api.server:app --port 8000

# terminal 2, from 01_J_Component_DriftLens/web/
npm install
npm run dev
```

Open http://localhost:3000. The API takes about 10 s to load the lexicon, tagger and language models.

## Pages

| Page | What it shows |
|---|---|
| `/` Overview | The core idea (a timeline is tagged like a sentence), the 7-step pipeline, a narrated 1-minute tour |
| `/analyze` | Paste posts → O/S/E/B tags, highlighted lexicon words, state probabilities, self-statements, TF-IDF keywords, signals chart, TextRank digest, spoken explanation |
| `/lab` | Eight live demos: tokenising, stemming vs lemmatising, edit-distance DP table, noisy-channel spelling, n-gram LM (smoothing, perplexity, generation, Zipf), HMM POS tagging with the Viterbi trellis, chunking + CFG + CKY chart, WordNet + similarity + Lesk WSD, VADER vs lexicon |
| `/results` | Cross-validated scores, ablation, LM perplexities, spelling accuracy, read from `../results/*.json` |

## NLP added for the frontend

- **Data-to-text generation (NLG).** `narrate()` in `api/server.py` picks the important facts (which posts change, which
  words drive the change, the confidence, crisis language, overall direction) and puts them into sentence templates
  with some variation. This is a classical NLG pipeline: content selection, then sentence planning, then surface realisation.
- **Text-to-speech.** `/api/tts` uses Microsoft Edge's neural voices through the free `edge-tts` package (no API
  key needed). It also returns word-boundary timings, which drive the live captions. Long narrations are split into
  sentence chunks, and each chunk is synthesised while the previous one plays. If the service is down (it needs internet),
  the browser's built-in voice takes over.
- **TF-IDF keywords** per post, **edit-distance backtrace**, **Viterbi trellis**, **CKY chart**, **bigram text generation**
  and **Zipf's law** are new views that use the existing models.

## Data

The example timelines are synthetic, written for the demo. The frontend never loads CLPsych or eRisk posts
(they fall under data-use agreements). For labelled CLPsych timelines, use the Streamlit app.
