// Builds presentation/DriftLens_Review.pptx
// Usage (from the project folder):  npm install pptxgenjs  &&  node presentation/build_deck.js
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const A = (p) => path.join(ROOT, "presentation", "assets", p);
const F = (p) => path.join(ROOT, "results", "figures", p);

const INK = "161A23", TEXT = "1B1F27", MUTED = "5A5F6B", CARD = "F2F3F5", RULE = "E3E5E8";
const BLUE = "2A78D6", ORANGE = "EB6834", AQUA = "1BAF7A", GREY = "8F8D87", SOFT = "C3C7D0";
const STATE = { O: GREY, S: BLUE, E: ORANGE, B: AQUA };
const HEAD = "Cambria", BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "DriftLens: detecting moments of change with classical NLP";
pres.author = "B Varun, Sagnik Basu, Arivoli A";

let n = 0;
function content(title) {
  const s = pres.addSlide();
  n += 1;
  s.background = { color: "FFFFFF" };
  s.addText(title, { x: 0.6, y: 0.4, w: 11.2, h: 0.8, fontFace: HEAD, fontSize: 32, bold: true, color: TEXT, margin: 0, isTextBox: true });
  strip(s, 12.02, 0.62, ["O", "S", "E", "B"], 0.16, 0.06);
  s.addText(`DriftLens`, { x: 0.6, y: 7.0, w: 6, h: 0.3, fontFace: BODY, fontSize: 10, color: MUTED, margin: 0, isTextBox: true });
  s.addText(String(n), { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontFace: BODY, fontSize: 10, color: MUTED, align: "right", margin: 0, isTextBox: true });
  return s;
}

function strip(s, x, y, states, size, gap) {
  states.forEach((st, i) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + i * (size + gap), y, w: size, h: size, fill: { color: STATE[st] }, line: { color: STATE[st], width: 0 }, rectRadius: size * 0.18 });
  });
}

function text(s, t, o) {
  s.addText(t, Object.assign({ fontFace: BODY, fontSize: 16, color: TEXT, margin: 0, isTextBox: true, valign: "top" }, o));
}

function bullets(s, items, o) {
  const runs = items.map((t, i) => ({ text: t, options: { bullet: { indent: 18 }, breakLine: i < items.length - 1, paraSpaceAfter: 8 } }));
  s.addText(runs, Object.assign({ fontFace: BODY, fontSize: 16, color: TEXT, margin: 0, isTextBox: true, valign: "top" }, o));
}

function card(s, x, y, w, h, fill) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill || CARD }, line: { color: fill || CARD, width: 0 }, rectRadius: 0.08 });
}

function numberCircle(s, x, y, label, color, d) {
  d = d || 0.5;
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { color, width: 0 } });
  s.addText(label, { x, y, w: d, h: d, fontFace: BODY, fontSize: 15, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0, isTextBox: true });
}

function stat(s, x, y, w, big, label, color) {
  text(s, big, { x, y, w, h: 0.9, fontFace: HEAD, fontSize: 48, bold: true, color: color || TEXT });
  text(s, label, { x, y: y + 0.95, w, h: 0.8, fontSize: 13, color: MUTED });
}

// one bar per category, coloured by group; zeros hidden by the number format
function groupedBars(s, cats, groups, o) {
  const data = groups.map((g) => ({ name: g.name, labels: cats, values: cats.map((_, i) => (g.idx.includes(i) ? g.values[i] : 0)) }));
  s.addChart(pres.charts.BAR, data, Object.assign({
    barDir: "bar", barGrouping: "stacked", chartColors: groups.map((g) => g.color),
    showValue: true, dataLabelPosition: "inEnd", dataLabelColor: "FFFFFF", dataLabelFontSize: 11, dataLabelFontFace: BODY,
    dataLabelFormatCode: "0.00;;;", catAxisLabelFontFace: BODY, catAxisLabelFontSize: 12, catAxisLabelColor: TEXT,
    catAxisOrientation: "maxMin", valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontFace: BODY, legendFontSize: 11, legendColor: MUTED, barGapWidthPct: 45,
  }, o));
}

// ------------------------------------------------------------------ 1 title
{
  const s = pres.addSlide();
  n += 1;
  s.background = { color: INK };
  text(s, "Detecting moments of change with classical NLP", { x: 0.8, y: 0.7, w: 11.5, h: 0.4, fontSize: 14, color: SOFT });
  text(s, "DriftLens", { x: 0.8, y: 1.6, w: 11, h: 1.2, fontFace: HEAD, fontSize: 66, bold: true, color: "FFFFFF" });
  text(s, "Reading a social-media timeline as a sentence to detect moments of change with classical NLP", { x: 0.8, y: 2.95, w: 9.5, h: 1.0, fontSize: 24, color: SOFT });
  strip(s, 0.8, 4.55, ["O", "O", "O", "S", "O", "O", "E", "E", "B", "E", "O", "O", "S", "O"], 0.42, 0.12);
  text(s, "no change   ·   switch   ·   escalation   ·   both", { x: 0.8, y: 5.1, w: 8, h: 0.35, fontSize: 12, color: SOFT });
  [["B Varun", "varun452005@gmail.com"], ["Sagnik Basu", "[email]"], ["Arivoli A", "[email]"]].forEach((a, i) => {
    const x = 0.8 + i * 3.9;
    text(s, a[0], { x, y: 5.75, w: 3.6, h: 0.4, fontSize: 18, bold: true, color: "FFFFFF" });
    text(s, "Vellore Institute of Technology, Vellore", { x, y: 6.18, w: 3.6, h: 0.3, fontSize: 12, color: SOFT });
    text(s, a[1], { x, y: 6.5, w: 3.6, h: 0.3, fontSize: 12, color: SOFT });
  });
  s.addNotes("Introduce the project in one sentence: DriftLens finds the posts where a person's mental state changes, using only classical NLP methods (n-grams, HMMs, CRFs, WordNet, edit distance). The coloured squares are a user's timeline: grey is no change, blue a switch, orange escalation, green both. Fill in the registration numbers before the review.");
}

// ------------------------------------------------------------------ 2 motivation
{
  const s = content("Why detect moments of change?");
  bullets(s, [
    "People describe their mental health on Reddit long before, or instead of, seeing a clinician.",
    "What matters clinically is the turning point, not a single label for the person.",
    "Switch: a sudden change of mood relative to the person's recent state.",
    "Escalation: a gradual worsening or improvement across several posts.",
    "Goal: mark every post in a timeline and show the evidence behind each mark.",
  ], { x: 0.6, y: 1.55, w: 5.9, h: 4.6, fontSize: 18 });
  card(s, 6.9, 1.55, 5.85, 3.9);
  text(s, "One user's timeline", { x: 7.2, y: 1.8, w: 5, h: 0.4, fontSize: 14, bold: true, color: MUTED });
  const seq = ["O", "O", "S", "O", "E", "E", "B", "O"];
  seq.forEach((st, i) => {
    const x = 7.25 + i * 0.66;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.9, w: 0.52, h: 0.52, fill: { color: STATE[st] }, line: { color: STATE[st], width: 0 }, rectRadius: 0.08 });
    text(s, "P" + (i + 1), { x, y: 3.5, w: 0.52, h: 0.3, fontSize: 11, color: MUTED, align: "center" });
  });
  text(s, "sudden\nswitch", { x: 8.35, y: 2.2, w: 1.1, h: 0.6, fontSize: 12, color: BLUE, align: "center", bold: true });
  text(s, "escalation run", { x: 9.8, y: 2.35, w: 2.0, h: 0.4, fontSize: 12, color: ORANGE, align: "center", bold: true });
  text(s, "A system must decide the state of each post using the posts around it, much like tagging each word of a sentence.", { x: 7.2, y: 4.3, w: 5.3, h: 1.4, fontSize: 15, color: TEXT });
  s.addNotes("Motivation. Static labels such as 'depressed or not' miss what clinicians look for: when things changed. The CLPsych shared tasks formalised this as switches (abrupt) and escalations (gradual). Point at the drawn timeline: P3 is a switch, P5 to P7 form an escalation run.");
}

// ------------------------------------------------------------------ 3 task and data
{
  const s = content("Task and data");
  const rows = [["O", "No change", "247 posts · 66%"], ["S", "Switch", "51 posts · 14%"], ["E", "Escalation", "48 posts · 13%"], ["B", "Switch + escalation", "27 posts · 7%"]];
  rows.forEach((r, i) => {
    const y = 1.7 + i * 1.02;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y, w: 0.6, h: 0.6, fill: { color: STATE[r[0]] }, line: { color: STATE[r[0]], width: 0 }, rectRadius: 0.1 });
    text(s, r[0], { x: 0.6, y, w: 0.6, h: 0.6, fontSize: 18, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    text(s, r[1], { x: 1.45, y: y - 0.02, w: 3.6, h: 0.35, fontSize: 18, bold: true });
    text(s, r[2], { x: 1.45, y: y + 0.33, w: 3.6, h: 0.3, fontSize: 14, color: MUTED });
  });
  stat(s, 5.7, 1.6, 2.2, "30", "labelled CLPsych timelines (Reddit)");
  stat(s, 8.1, 1.6, 2.2, "373", "posts with switch and escalation flags");
  stat(s, 10.5, 1.6, 2.2, "3.3", "posts in an average escalation run", ORANGE);
  stat(s, 5.7, 3.75, 2.2, "236", "posts with 1–10 well-being and evidence spans");
  stat(s, 8.1, 3.75, 2.2, "909", "eRisk 2026 users for external testing (102 depressed)");
  stat(s, 10.5, 3.75, 2.2, "1.5M", "eRisk tokens for the background language model");
  text(s, "10 further CLPsych timelines have no labels and are used only in the prototype. All data stay on the local machine under the data-use agreements.", { x: 0.6, y: 6.1, w: 12.1, h: 0.6, fontSize: 13, color: MUTED });
  s.addNotes("Data. CLPsych 2025 release: 30 labelled timelines, 373 posts. Two flags per post give four states. Two thirds of posts are 'no change', so the data are imbalanced. Escalations come in runs of about 3 posts, which is why a sequence model makes sense. eRisk 2026 is used twice: as background text for the language model and as an external test of the lexicon.");
}

// ------------------------------------------------------------------ 4 key idea
{
  const s = content("Key idea: a timeline is a sentence");
  const words = [["I", "PRON"], ["am", "VERB"], ["not", "ADV"], ["happy", "ADJ"]];
  text(s, "POS tagging", { x: 0.6, y: 1.5, w: 4, h: 0.4, fontSize: 14, bold: true, color: MUTED });
  words.forEach((w, i) => {
    const x = 0.6 + i * 1.3;
    card(s, x, 1.95, 1.15, 0.5, "E8EEF9");
    text(s, w[1], { x, y: 1.95, w: 1.15, h: 0.5, fontSize: 13, bold: true, color: BLUE, align: "center", valign: "middle" });
    text(s, w[0], { x, y: 2.55, w: 1.15, h: 0.4, fontSize: 18, align: "center" });
  });
  text(s, "DriftLens", { x: 0.6, y: 3.35, w: 4, h: 0.4, fontSize: 14, bold: true, color: MUTED });
  ["O", "S", "O", "E", "E"].forEach((st, i) => {
    const x = 0.6 + i * 1.04;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 3.8, w: 0.9, h: 0.5, fill: { color: STATE[st] }, line: { color: STATE[st], width: 0 }, rectRadius: 0.08 });
    text(s, st, { x, y: 3.8, w: 0.9, h: 0.5, fontSize: 15, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    text(s, "post " + (i + 1), { x, y: 4.4, w: 0.9, h: 0.4, fontSize: 13, align: "center", color: MUTED });
  });
  const hdr = { bold: true, color: "FFFFFF", fill: { color: INK }, fontFace: BODY, fontSize: 14 };
  const cell = { fontFace: BODY, fontSize: 14, color: TEXT };
  s.addTable([
    [{ text: "POS tagging", options: hdr }, { text: "DriftLens", options: hdr }],
    [{ text: "word", options: cell }, { text: "post", options: cell }],
    [{ text: "tag (NOUN, VERB, …)", options: cell }, { text: "state (O, S, E, B)", options: cell }],
    [{ text: "transition P(tag | previous tag)", options: cell }, { text: "persistence P(state | previous state)", options: cell }],
    [{ text: "emission P(word | tag)", options: cell }, { text: "p(post features | state)", options: cell }],
    [{ text: "Viterbi: best tag sequence", options: cell }, { text: "Viterbi: best state sequence", options: cell }],
  ], { x: 6.4, y: 1.55, w: 6.3, colW: [3.0, 3.3], rowH: 0.52, border: { type: "solid", color: RULE, pt: 1 }, fill: { color: "FFFFFF" } });
  text(s, "The same machinery taught for POS tagging (HMM, MaxEnt, CRF, Viterbi) is reused, but the tokens are posts and the tags are changes of state.", { x: 0.6, y: 5.3, w: 5.4, h: 1.2, fontSize: 15 });
  s.addNotes("This is the core idea. In POS tagging, the tag of a word depends on the word and on the previous tag. Here the state of a post depends on its features and on the previous state. Escalation tends to continue, a switch is usually followed by no change: exactly what an HMM transition matrix stores.");
}

// ------------------------------------------------------------------ 5 novelty
{
  const s = content("What is new in DriftLens");
  const items = [
    [BLUE, "Change detection as sequence labelling", "Gaussian HMM, MaxEnt–HMM hybrid and linear-chain CRF on identical features. A prior-corrected CRF decoding rule fixes the class imbalance without any tuning."],
    [ORANGE, "Personal language-model novelty", "How much the cross-entropy of a post changes when the writer's own earlier posts are mixed into a background n-gram model: does this person sound unlike themselves?"],
    [AQUA, "WordNet-grown self-state lexicon", "94 sense-specific seed synsets grow to 1,760 words, filtered by a sense-ratio prior and a polarity veto, and scored with adapted Lesk and a negation FSA."],
    [GREY, "An honest shortcut analysis", "Post length alone rivals every model. A length-stratified evaluation shows that the language features still add 0.09–0.13 macro-F1 within posts of similar length."],
  ];
  items.forEach((it, i) => {
    const x = 0.6 + (i % 2) * 6.2, y = 1.55 + Math.floor(i / 2) * 2.55;
    card(s, x, y, 5.95, 2.3);
    numberCircle(s, x + 0.3, y + 0.3, String(i + 1), it[0]);
    text(s, it[1], { x: x + 1.0, y: y + 0.32, w: 4.7, h: 0.5, fontSize: 18, bold: true });
    text(s, it[2], { x: x + 1.0, y: y + 0.85, w: 4.7, h: 1.35, fontSize: 14, color: TEXT });
  });
  s.addNotes("Four contributions. 1: the sequence-labelling formulation with four model families and a simple fix for class imbalance. 2: the personal novelty score, our attempt to measure 'sounding unlike yourself'. 3: a lexicon built from WordNet senses, not words. 4: we report the length shortcut instead of hiding it, and show where language still helps.");
}

// ------------------------------------------------------------------ 6 pipeline
{
  const s = content("System pipeline");
  const box = (x, y, w, h, t, fill, col) => {
    card(s, x, y, w, h, fill);
    text(s, t, { x: x + 0.1, y, w: w - 0.2, h, fontSize: 13, align: "center", valign: "middle", color: col || TEXT });
  };
  const arrow = (x1, y1, x2, y2) => s.addShape(pres.shapes.LINE, { x: x1, y: Math.min(y1, y2), w: x2 - x1, h: Math.abs(y2 - y1), flipV: y2 < y1, line: { color: MUTED, width: 1.25, endArrowType: "triangle" } });
  box(0.6, 3.25, 1.6, 1.0, "Timeline of posts");
  box(2.6, 3.0, 2.1, 1.5, "Cleaning, Punkt + Treebank tokens, noisy-channel spelling");
  const mids = [["VADER sentiment", "E8EEF9"], ["WordNet lexicon, adapted Lesk, negation FSA", "E8EEF9"], ["HMM POS tagger, self-evaluation grammar", "E8EEF9"], ["n-gram LM, personal novelty", "FCE9E1"]];
  mids.forEach((m, i) => {
    const y = 1.5 + i * 1.2;
    box(5.3, y, 2.7, 0.95, m[0], m[1]);
    arrow(4.7, 3.75, 5.3, y + 0.47);
    arrow(8.0, y + 0.47, 8.6, 3.75);
  });
  box(8.6, 3.0, 1.9, 1.5, "25 features per post + personal z-scores", "FCE9E1");
  box(10.9, 3.0, 1.85, 1.5, "HMM / CRF with Viterbi → O S E B", INK, "FFFFFF");
  arrow(2.2, 3.75, 2.6, 3.75);
  arrow(10.5, 3.75, 10.9, 3.75);
  text(s, "Every box is a classical method that runs on a laptop CPU: no GPU, no hosted API, no large language model. Orange boxes model change relative to the user.", { x: 0.6, y: 6.25, w: 12.1, h: 0.5, fontSize: 14, color: MUTED });
  s.addNotes("Walk left to right. Posts are cleaned and spell-corrected, then four label-free components score each post. None of them sees the CLPsych labels, so features can be computed once. The sequence model is the only supervised part. Everything runs on a CPU.");
}

// ------------------------------------------------------------------ 7 techniques
{
  const s = content("Every classical technique in one working system");
  const mods = [
    ["1", "Overview & NLTK", "tokenisation, Punkt sentence splitting, NLTK corpora"],
    ["2", "Text processing", "weighted edit distance, noisy-channel spelling correction"],
    ["3", "N-gram LMs", "Laplace, add-k, interpolation, perplexity, personal novelty"],
    ["4", "Morphology & CFG", "morphy, suffix FSA, negation FSA, chunk grammar, CFG + CNF"],
    ["5", "POS tagging", "HMM + Viterbi from scratch; HMM, MaxEnt and CRF for timelines"],
    ["6", "Lexical semantics", "WordNet expansion, Wu–Palmer, Lin, sense ratio, Lesk WSD"],
    ["7", "Applications", "sentiment, information extraction, TextRank summarisation"],
  ];
  const colors = [GREY, BLUE, ORANGE, AQUA, BLUE, ORANGE, AQUA];
  mods.forEach((m, i) => {
    const row = i < 4 ? 0 : 1, col = i < 4 ? i : i - 4;
    const x = 0.6 + col * 3.07 + (row ? 1.53 : 0), y = 1.6 + row * 2.45;
    card(s, x, y, 2.85, 2.15);
    numberCircle(s, x + 0.25, y + 0.25, m[0], colors[i], 0.62);
    text(s, m[1], { x: x + 1.0, y: y + 0.3, w: 1.75, h: 0.55, fontSize: 16, bold: true, valign: "middle" });
    text(s, m[2], { x: x + 0.25, y: y + 1.05, w: 2.4, h: 1.0, fontSize: 13, color: TEXT });
  });
  s.addNotes("Every technique contributes a working, evaluated component, and the prototype's toolbox tab lets you demonstrate each one live.");
}

// ------------------------------------------------------------------ 8 components
{
  const s = content("Each component, tested on its own");
  text(s, "Language-model perplexity on CLPsych posts (log scale)", { x: 0.6, y: 1.45, w: 5.8, h: 0.4, fontSize: 14, bold: true, color: MUTED });
  s.addChart(pres.charts.BAR, [
    { name: "Laplace (add-1)", labels: ["bigram", "trigram"], values: [1335, 10541] },
    { name: "add-k (tuned)", labels: ["bigram", "trigram"], values: [276, 1629] },
    { name: "Jelinek–Mercer", labels: ["bigram", "trigram"], values: [192, 176] },
  ], {
    x: 0.5, y: 1.85, w: 6.0, h: 4.2, barDir: "col", chartColors: [GREY, BLUE, ORANGE], valAxisLogScaleBase: 10,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "#,##0", dataLabelFontSize: 11, dataLabelColor: TEXT,
    catAxisLabelFontFace: BODY, catAxisLabelFontSize: 13, catAxisLabelColor: TEXT, valAxisHidden: true,
    valGridLine: { style: "none" }, showLegend: true, legendPos: "b", legendFontSize: 11, legendColor: MUTED, barGapWidthPct: 60,
  });
  card(s, 6.9, 1.55, 2.8, 4.5);
  stat(s, 7.15, 1.8, 2.4, "96.9%", "HMM tagger accuracy on held-out Brown (88.9% cross-domain on Treebank)", BLUE);
  text(s, "Bigram HMM, TnT-style suffix model for unknown words, Viterbi, written from scratch.", { x: 7.15, y: 3.75, w: 2.35, h: 1.8, fontSize: 13, color: TEXT });
  card(s, 9.95, 1.55, 2.8, 4.5);
  stat(s, 10.2, 1.8, 2.4, "95.3%", "spelling correction on 1,000 synthetic non-word errors (88.8% plain Levenshtein)", ORANGE);
  text(s, "Keyboard-weighted Damerau–Levenshtein channel with bigram context; a 'no error' option keeps slang.", { x: 10.2, y: 3.75, w: 2.35, h: 1.8, fontSize: 13, color: TEXT });
  text(s, "Laplace smoothing makes a trigram model worse than a unigram model (10,541 vs 604); interpolation gives the best perplexity (176).", { x: 0.6, y: 6.2, w: 12.1, h: 0.5, fontSize: 14, color: MUTED });
  s.addNotes("Component checks. Language models reproduce the classic smoothing result. Our from-scratch HMM tagger reaches 96.9% on Brown; NLTK's perceptron wins on Treebank because it was trained on WSJ, so each tagger is best in its own domain. The speller gains most from bigram context, especially on deletion errors.");
}

// ------------------------------------------------------------------ 9 lexicon construction
{
  const s = content("Growing a self-state lexicon from WordNet senses");
  const steps = [
    ["Seed senses", "94 synsets in 12 categories (7 maladaptive, 5 adaptive)"],
    ["Expand", "similar-to, derivation, attribute, hyponym (Wu–Palmer ≥ 0.6), ≤ 2 hops"],
    ["Sense ratio", "share of a word's usage in reached senses; drops down, blue, low"],
    ["Polarity veto", "SentiWordNet and VADER remove 312 contradicting words"],
    ["Score in context", "morphy, suffix FSA, adapted Lesk, negation FSA"],
  ];
  steps.forEach((st, i) => {
    const x = 0.6 + i * 2.47;
    card(s, x, 1.6, 2.25, 2.55);
    numberCircle(s, x + 0.2, 1.8, String(i + 1), [GREY, BLUE, ORANGE, AQUA, BLUE][i], 0.5);
    text(s, st[0], { x: x + 0.2, y: 2.45, w: 1.9, h: 0.4, fontSize: 16, bold: true });
    text(s, st[1], { x: x + 0.2, y: 2.9, w: 1.9, h: 1.2, fontSize: 13 });
  });
  stat(s, 0.6, 4.55, 3.0, "1,760", "words after filtering (1,076 maladaptive, 684 adaptive)", AQUA);
  card(s, 4.1, 4.5, 8.65, 2.2);
  text(s, [
    { text: "“I was so anxious to see the new film”", options: { bold: true, breakLine: true } },
    { text: "Lesk picks anxious = 'eagerly desirous' → not counted as anxiety", options: { color: MUTED, breakLine: true } },
    { text: "“I'm not happy”", options: { bold: true, breakLine: true } },
    { text: "negation FSA → counted as half a maladaptive hit", options: { color: MUTED, breakLine: true } },
    { text: "“worthlessness”", options: { bold: true, breakLine: true } },
    { text: "suffix FSA → worthless (self-criticism)", options: { color: MUTED } },
  ], { x: 4.4, y: 4.65, w: 8.1, h: 1.95, fontSize: 14 });
  s.addNotes("Seeds are senses, not words, so 'hopeless' the emotion is kept apart from 'hopeless at maths'. Expansion follows WordNet relations. The sense ratio uses SemCor counts to down-weight words that are mostly used in other senses. At scoring time, adapted Lesk rejects non-affective uses, as in the anxious example.");
}

// ------------------------------------------------------------------ 10 lexicon validity
{
  const s = content("Does the lexicon measure self-state?");
  text(s, "CLPsych: Spearman ρ with annotated well-being", { x: 0.6, y: 1.45, w: 5.6, h: 0.4, fontSize: 14, bold: true, color: MUTED });
  groupedBars(s, ["VADER", "WordNet seeds only", "WordNet-grown lexicon"],
    [{ name: "baseline", color: GREY, idx: [0, 1], values: [0.357, 0.306, 0] }, { name: "DriftLens", color: BLUE, idx: [2], values: [0, 0, 0.429] }],
    { x: 0.5, y: 1.85, w: 5.8, h: 3.3 });
  text(s, "eRisk 2026: ROC-AUC, depressed vs control users", { x: 6.8, y: 1.45, w: 5.9, h: 0.4, fontSize: 14, bold: true, color: MUTED });
  groupedBars(s, ["VADER", "First-person rate", "Seeds: maladaptive", "WordNet-grown: maladaptive", "TF-IDF + LR (supervised)"],
    [{ name: "label-free baseline", color: GREY, idx: [0, 1, 2], values: [0.522, 0.764, 0.825, 0, 0] },
     { name: "DriftLens (label-free)", color: BLUE, idx: [3], values: [0, 0, 0, 0.843, 0] },
     { name: "supervised reference", color: ORANGE, idx: [4], values: [0, 0, 0, 0, 0.952] }],
    { x: 6.7, y: 1.85, w: 6.1, h: 3.3 });
  bullets(s, [
    "Tracks well-being better than VADER (0.43 vs 0.36); length is unrelated (ρ = −0.06).",
    "Finds depressed users with no training labels (AUC 0.84). A supervised TF-IDF model is stronger, but its evidence cannot be explained word by word.",
    "Depressed users use more adaptive words (therapy, friends, hope): they write in support communities.",
  ], { x: 0.6, y: 5.35, w: 12.1, h: 1.5, fontSize: 14 });
  s.addNotes("Two independent validity checks. In-domain, the balance score correlates with the annotated well-being rating better than VADER and better than the seeds alone. Externally, on eRisk, the maladaptive density separates depressed from control users without labels. We also report that WSD, negation and spelling had no measurable effect on these aggregate numbers.");
}

// ------------------------------------------------------------------ 11 main results
{
  const s = content("Moment-of-change detection results");
  const cats = ["Majority (all O)", "MaxEnt, post length only", "MaxEnt, all features", "Gaussian HMM", "Hybrid HMM (prior-divided)", "CRF (Viterbi)", "CRF (prior-corrected)"];
  const vals = [0.266, 0.572, 0.547, 0.567, 0.542, 0.533, 0.563];
  groupedBars(s, cats, [
    { name: "baselines", color: GREY, idx: [0, 2], values: vals },
    { name: "length-only shortcut", color: ORANGE, idx: [1], values: vals },
    { name: "sequence models", color: BLUE, idx: [3, 4, 5, 6], values: vals },
  ], { x: 0.4, y: 1.45, w: 7.3, h: 5.2 });
  text(s, "Macro-F1, 5 × 5-fold cross-validation split by timeline", { x: 0.6, y: 6.6, w: 7, h: 0.3, fontSize: 11, color: MUTED });
  const pts = [
    ["0.567", "Gaussian HMM, best macro-F1; best switch F1 (0.508)", BLUE],
    ["0.481", "prior-corrected CRF, best escalation F1 (MaxEnt: 0.420)", BLUE],
    ["+0.030", "macro-F1 from prior-corrected decoding of the CRF", AQUA],
    ["0.572", "post length alone: no model beats it overall (p = 0.60)", ORANGE],
  ];
  pts.forEach((p, i) => {
    const y = 1.55 + i * 1.25;
    text(s, p[0], { x: 8.1, y, w: 1.7, h: 0.6, fontFace: HEAD, fontSize: 28, bold: true, color: p[2] });
    text(s, p[1], { x: 9.85, y: y + 0.05, w: 2.9, h: 1.0, fontSize: 14 });
  });
  s.addNotes("Main result. All models are far above the majority baseline. Sequence models help most on escalation, which comes in runs. The class-balance fix matters for the CRF. But a logistic regression on post length alone scores 0.572, and no model beats it overall. We treat that as a finding, next slide.");
}

// ------------------------------------------------------------------ 12 length shortcut
{
  const s = content("The length shortcut, and where language still helps");
  card(s, 0.6, 1.55, 4.2, 4.9);
  text(s, "150 vs 20", { x: 0.9, y: 1.8, w: 3.6, h: 0.9, fontFace: HEAD, fontSize: 44, bold: true, color: ORANGE });
  text(s, "median words in a change post vs an unchanged post (Mann–Whitney p < 10⁻¹⁹)", { x: 0.9, y: 2.75, w: 3.6, h: 0.8, fontSize: 13, color: MUTED });
  text(s, "People write at length when something has happened. Length is a real signal, but a model that relies on it misses a short, bleak post and flags every long, cheerful one.", { x: 0.9, y: 3.6, w: 3.6, h: 2.6, fontSize: 15 });
  text(s, "Macro-F1 within posts of similar length (split at 46 words)", { x: 5.3, y: 1.45, w: 7.4, h: 0.4, fontSize: 14, bold: true, color: MUTED });
  s.addChart(pres.charts.BAR, [
    { name: "MaxEnt, length only", labels: ["short posts", "long posts"], values: [0.311, 0.369] },
    { name: "CRF, all features", labels: ["short posts", "long posts"], values: [0.404, 0.495] },
  ], {
    x: 5.2, y: 1.85, w: 7.5, h: 3.9, barDir: "col", chartColors: [ORANGE, BLUE], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.00", dataLabelFontSize: 13, dataLabelColor: TEXT, catAxisLabelFontSize: 14, catAxisLabelColor: TEXT,
    valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 0.6, valGridLine: { style: "none" }, showLegend: true, legendPos: "b",
    legendFontSize: 12, legendColor: MUTED, barGapWidthPct: 70,
  });
  text(s, "Within each half the language features add +0.09 and +0.13 macro-F1. Length decides which half a post falls into; the features decide which posts inside it are moments of change.", { x: 5.3, y: 5.9, w: 7.4, h: 0.9, fontSize: 14 });
  s.addNotes("This is the most important slide scientifically. The overall tie hides a real effect. Once you compare posts of similar length, the length model collapses to 0.31 and 0.37, while the CRF with language features reaches 0.40 and 0.50.");
}

// ------------------------------------------------------------------ 13 what the model learned
{
  const s = content("What the model learned");
  s.addImage({ path: A("fig_transitions_hi.png"), x: 0.6, y: 1.5, w: 4.6, h: 4.64 });
  text(s, "HMM transition probabilities, rows = previous state", { x: 0.6, y: 6.2, w: 4.8, h: 0.4, fontSize: 12, color: MUTED });
  const pts = [
    [ORANGE, "Escalation persists", "E→E 0.42 and B→E 0.50: the transition matrix keeps a run going."],
    [BLUE, "Switches are isolated", "S→O 0.65; many switches are sudden improvements (median VADER +0.46)."],
    [AQUA, "Self-evaluations matter", "removing 'I feel so worthless'-type chunks costs the HMM 0.042 macro-F1."],
    [GREY, "Personal novelty: not yet", "higher for switch posts (0.26 vs 0.21 bits) but not significant (p = 0.25). Reported honestly."],
  ];
  pts.forEach((p, i) => {
    const y = 1.55 + i * 1.2;
    s.addShape(pres.shapes.OVAL, { x: 5.9, y: y + 0.08, w: 0.3, h: 0.3, fill: { color: p[0] }, line: { color: p[0], width: 0 } });
    text(s, p[1], { x: 6.4, y, w: 6.3, h: 0.4, fontSize: 17, bold: true });
    text(s, p[2], { x: 6.4, y: y + 0.42, w: 6.3, h: 0.7, fontSize: 14, color: TEXT });
  });
  s.addNotes("The learned transitions explain why sequence models help escalation. Among feature groups, the self-evaluation chunks from the HMM tagger and grammar help the HMM most. The personal novelty idea did not reach significance on this small dataset; we say so in the paper and suggest character n-grams as future work.");
}

// ------------------------------------------------------------------ 14 prototype timeline
{
  const s = content("Prototype: timeline view");
  s.addImage({ path: A("app_labelled_chart.png"), x: 0.6, y: 1.45, w: 9.0, h: 4.2 });
  text(s, "A labelled CLPsych timeline: signals over time with predicted and gold states. Post text is hidden here for privacy.", { x: 0.6, y: 5.75, w: 9.0, h: 0.6, fontSize: 12, color: MUTED });
  bullets(s, [
    "Paste your own posts or pick a CLPsych timeline.",
    "Choose HMM, CRF, hybrid or MaxEnt.",
    "Each post shows highlighted evidence, self-evaluations, spelling fixes and state probabilities.",
    "Runs locally on a CPU; models load in under a second.",
  ], { x: 9.95, y: 1.55, w: 2.8, h: 4.6, fontSize: 14 });
  text(s, "python -m streamlit run app/streamlit_app.py", { x: 0.6, y: 6.35, w: 9.0, h: 0.35, fontFace: "Courier New", fontSize: 12, color: TEXT });
  s.addNotes("Live demo here if possible: open the app, paste the example timeline, show the chart and expand post 4. Then switch to a labelled timeline to compare gold and predicted states.");
}

// ------------------------------------------------------------------ 15 prototype evidence + toolbox
{
  const s = content("Prototype: evidence and the NLP toolbox");
  s.addImage({ path: A("app_post_detail.png"), x: 0.6, y: 1.45, w: 6.1, h: 3.93 });
  text(s, "Per-post evidence for our own example posts: lexicon hits, self-evaluation chunks, signals and state probabilities.", { x: 0.6, y: 5.5, w: 6.1, h: 0.8, fontSize: 12, color: MUTED });
  s.addImage({ path: A("app_toolbox_cfg_crop.png"), x: 7.1, y: 1.45, w: 4.9, h: 3.92 });
  text(s, "Toolbox: HMM vs perceptron tags, chunk tree, top-down and bottom-up CFG parses. Other panels: noisy channel, n-gram LM, WordNet similarity and Lesk.", { x: 7.1, y: 5.5, w: 5.6, h: 0.8, fontSize: 12, color: MUTED });
  s.addNotes("The toolbox lets each technique be demonstrated interactively, for example typing a misspelled word to see the noisy-channel candidate table, or parsing a self-statement with the CFG.");
}

// ------------------------------------------------------------------ 16 limitations
{
  const s = content("Limitations and ethics");
  card(s, 0.6, 1.55, 5.95, 4.0);
  text(s, "Limitations", { x: 0.9, y: 1.75, w: 5.4, h: 0.5, fontSize: 20, bold: true });
  bullets(s, [
    "Only 30 labelled timelines: differences of 0.02 macro-F1 are within noise.",
    "Some design choices followed a first CV run; no untouched labelled test set exists.",
    "Lexicon categories and seeds are our reading of the CLPsych framework.",
    "English Reddit only; code-mixed Hindi–English would need new resources.",
  ], { x: 0.9, y: 2.35, w: 5.4, h: 3.0, fontSize: 15 });
  card(s, 6.8, 1.55, 5.95, 4.0);
  text(s, "Ethics", { x: 7.1, y: 1.75, w: 5.4, h: 0.5, fontSize: 20, bold: true });
  bullets(s, [
    "Data-use agreements: everything stays on the local machine; no post is quoted.",
    "A research prototype, not a diagnostic or monitoring tool.",
    "Shows its evidence so that a human can disagree with it.",
    "Displays the Tele-MANAS helpline (14416) for anyone who needs support.",
  ], { x: 7.1, y: 2.35, w: 5.4, h: 3.0, fontSize: 15 });
  s.addNotes("Be upfront about the small dataset and the post-hoc design changes. Stress that the tool is for research and teaching, not for making decisions about people.");
}

// ------------------------------------------------------------------ 17 conclusion
{
  const s = pres.addSlide();
  n += 1;
  s.background = { color: INK };
  text(s, "Conclusion", { x: 0.8, y: 0.6, w: 11, h: 0.9, fontFace: HEAD, fontSize: 40, bold: true, color: "FFFFFF" });
  const pts = [
    [BLUE, "Reading a timeline as a sentence works: HMM and CRF find moments of change far above chance, and sequence structure helps most on escalation."],
    [ORANGE, "Evaluation matters more than model choice: post length matches every model, and only a length-stratified analysis shows the +0.09 to +0.13 that language adds."],
    [AQUA, "A WordNet-grown lexicon beats a general sentiment tool on well-being and detects depression on eRisk without labels."],
  ];
  pts.forEach((p, i) => {
    const y = 1.8 + i * 1.15;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.8, y: y + 0.08, w: 0.32, h: 0.32, fill: { color: p[0] }, line: { color: p[0], width: 0 }, rectRadius: 0.06 });
    text(s, p[1], { x: 1.35, y, w: 11.2, h: 1.0, fontSize: 18, color: "FFFFFF" });
  });
  text(s, "Next: trend features over windows of posts, and character-level personal models to give the drift idea a fairer test.", { x: 0.8, y: 5.35, w: 11.5, h: 0.6, fontSize: 16, color: SOFT });
  text(s, "Thank you  ·  Questions?", { x: 0.8, y: 6.3, w: 11, h: 0.6, fontFace: HEAD, fontSize: 26, bold: true, color: "FFFFFF" });
  s.addNotes("Close with the three takeaways and invite questions. Likely questions: why not BERT (interpretability, CPU, small data); why the length baseline is so strong (event posts are longer); what personal novelty measures (cross-entropy change with the user's own history).");
}

pres.writeFile({ fileName: path.join(ROOT, "presentation", process.argv[2] || "DriftLens_Review.pptx") }).then((f) => console.log("wrote " + f));
