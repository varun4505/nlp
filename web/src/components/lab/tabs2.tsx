"use client";

import { useState } from "react";
import { ErrorNote, Explainer } from "@/components/ui";
import { Field, Panel, TreeView, useLive } from "./common";

const TAG_HELP: Record<string, string> = {
  ".": "punctuation", ADJ: "adjective", ADP: "adposition", ADV: "adverb", CONJ: "conjunction", DET: "determiner",
  NOUN: "noun", NUM: "number", PRON: "pronoun", PRT: "particle", VERB: "verb", X: "other",
};

// ----------------------------------------------------------- POS + Viterbi
export function PosLab() {
  const [s, setS] = useState("I am not happy anymore and I hate myself");
  const { data, error } = useLive("pos", { sentence: s });
  const [hover, setHover] = useState<[number, number] | null>(null);
  return (
    <div className="space-y-4">
      <Panel title="HMM tagger, decoded with Viterbi" sub="Bigram HMM trained on the Brown corpus (universal tags), add-k smoothing, TnT-style suffix model for unknown words.">
        <input className="input" value={s} onChange={(e) => setS(e.target.value)} />
      </Panel>
      <ErrorNote error={error} />
      {data && data.tokens.length > 0 && (
        <>
          <Panel title="Result" sub="HMM (ours, from scratch) vs NLTK's averaged perceptron. Disagreements are outlined.">
            <div className="flex flex-wrap gap-1.5">
              {data.tokens.map((w: string, i: number) => (
                <div key={i} className={`rounded-xl border px-2.5 py-1.5 text-center ${data.best[i] !== data.perceptron[i] ? "border-st-e border-2" : "border-line"}`}>
                  <div className="font-medium">{w}</div>
                  <div className="font-mono text-[0.7rem] text-accent" title={TAG_HELP[data.best[i]]}>
                    {data.best[i]}
                  </div>
                  <div className="font-mono text-[0.65rem] text-ink-3">{data.perceptron[i]}</div>
                </div>
              ))}
            </div>
          </Panel>
          <Panel
            title="The Viterbi trellis"
            sub="δₜ(j) = maxᵢ δₜ₋₁(i) · P(tagⱼ | tagᵢ) · P(wordₜ | tagⱼ). Each column is shaded by its log score relative to the best cell; the purple path is the back-trace. Hover a cell to see where it came from."
          >
            <div className="overflow-x-auto">
              <table className="border-separate border-spacing-1 text-xs">
                <thead>
                  <tr>
                    <th />
                    {data.trellis.map((c: any, i: number) => (
                      <th key={i} className="min-w-14 px-1 font-semibold">
                        {c.word}
                        {!c.known && <div className="font-normal text-st-e">unknown</div>}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.tags.map((tag: string, j: number) => (
                    <tr key={tag}>
                      <th className="pr-2 text-right font-mono text-ink-2" title={TAG_HELP[tag]}>
                        {tag}
                      </th>
                      {data.trellis.map((c: any, i: number) => {
                        const v = c.scores[j];
                        const best = data.best_idx[i] === j;
                        const fromHover = hover && hover[0] === i + 1 && data.trellis[i + 1]?.back[hover[1]] === j;
                        const strength = Math.max(0, 1 + v / 12);
                        return (
                          <td
                            key={i}
                            onMouseEnter={() => setHover([i, j])}
                            onMouseLeave={() => setHover(null)}
                            title={`δ (relative log) = ${v}, emission log P(${c.word}|${tag}) = ${c.emission[j]}${i ? `, best previous = ${data.tags[c.back[j]]}` : ""}`}
                            className={`h-7 rounded-md text-center font-mono ${best ? "font-bold text-white" : ""} ${fromHover ? "ring-2 ring-st-e" : ""}`}
                            style={{ background: best ? "var(--accent)" : `color-mix(in srgb, var(--accent) ${strength * strength * 45}%, var(--surface-2))` }}
                          >
                            {v > -30 ? v.toFixed(0) : "·"}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-3">
              <Explainer>
                A brute-force tagger would try 12ⁿ tag sequences ({data.tokens.length} words → about 10<sup>{Math.round(data.tokens.length * Math.log10(12))}</sup>). Viterbi needs only n × 12 × 12 steps because it keeps just the best path into each cell. DriftLens uses the <i>same</i> algorithm to tag timelines with O/S/E/B.
              </Explainer>
            </div>
          </Panel>
        </>
      )}
    </div>
  );
}

// ------------------------------------------------------- chunking, CFG, CKY
export function ParseLab() {
  const [s, setS] = useState("I am not very happy");
  const { data, error } = useLive("parse", { sentence: s });
  const [showG, setShowG] = useState(false);
  const cell = (i: number, j: number) => data?.cky.cells.find((c: any) => c.i === i && c.j === j);
  return (
    <div className="space-y-4">
      <Panel title="Self-statement" sub="Try: “I feel so worthless”, “I am not a failure”, “I hate myself”, “I am happy and calm”.">
        <input className="input" value={s} onChange={(e) => setS(e.target.value)} />
      </Panel>
      <ErrorNote error={error} />
      {data && (
        <>
          <Panel title="1 · Chunk grammar (cascaded regular expressions over tags)" sub="Tokens are re-tagged: FP = I, FPO = me/myself, COP = am/feel/seem…, NEG = not/never.">
            <div className="mb-3 flex flex-wrap gap-1">
              {data.tagged.map(([w, t]: [string, string], i: number) => (
                <span key={i} className="rounded-md border border-line px-1.5 py-0.5 text-xs">
                  {w} <span className="font-mono text-accent">{t}</span>
                </span>
              ))}
            </div>
            <pre className="mb-3 overflow-x-auto rounded-lg bg-surface-2 p-3 font-mono text-xs text-ink-2">{`SELF: {<FP><ADV|NEG|PRT>*<AUX|VERB|COP>*<COP><ADV|NEG|PRT>*<ADV|PRT|ADJ|VERB|CONJ|DET>*<ADJ|VERB|NOUN>}
      {<FP><ADV|NEG|AUX>*<VERB><FPO>}
      {<FP><ADV|NEG|AUX|COP>*<VERB>}`}</pre>
            <TreeView tree={data.chunk_tree} />
            <div className="mt-2 flex flex-wrap gap-2">
              {data.self_evals.map((e: any, i: number) => (
                <span key={i} className={`rounded-full px-3 py-1 text-sm ${e.polarity === "neg" ? "bg-mal-soft text-mal" : e.polarity === "pos" ? "bg-ada-soft text-ada" : "bg-surface-2"}`}>
                  “{e.text}” → {e.polarity}
                </span>
              ))}
            </div>
          </Panel>

          <div className="grid gap-4 lg:grid-cols-2">
            <Panel title="2 · Context-free grammar parse" sub={<>Tag sequence: <span className="font-mono">{data.tags.join(" ")}</span></>}>
              {data.topdown ? (
                <TreeView tree={data.topdown} />
              ) : (
                <Explainer>No parse. The hand-written CFG only covers simple self-evaluations, which is exactly why DriftLens uses the more forgiving chunker above.</Explainer>
              )}
              <div className="mt-2 text-xs text-ink-3">Top-down (recursive descent) and bottom-up (chart) parsers {data.topdown && data.bottomup ? "agree on this tree" : data.topdown || data.bottomup ? "disagree" : "both fail"}.</div>
              <button className="mt-2 text-xs font-medium text-accent" onClick={() => setShowG(!showG)}>
                {showG ? "Hide" : "Show"} grammar & Chomsky normal form
              </button>
              {showG && (
                <div className="mt-2 grid gap-2 sm:grid-cols-2">
                  <pre className="overflow-x-auto rounded-lg bg-surface-2 p-2 font-mono text-[0.65rem]">{data.grammar.join("\n")}</pre>
                  <pre className="overflow-x-auto rounded-lg bg-surface-2 p-2 font-mono text-[0.65rem]">{data.cnf.join("\n")}</pre>
                </div>
              )}
            </Panel>

            <Panel title={`3 · CKY chart ${data.cky.accepted ? "✓ accepted" : "✗ rejected"}`} sub="Grammar converted to CNF. Cell [i, j] lists every non-terminal that can span words i..j. S in the top cell ⇒ grammatical.">
              <div className="overflow-x-auto">
                <table className="border-separate border-spacing-1 text-[0.65rem]">
                  <tbody>
                    {Array.from({ length: data.cky.n }, (_, r) => {
                      const span = data.cky.n - r;
                      return (
                        <tr key={r}>
                          {Array.from({ length: data.cky.n }, (_, i) => {
                            const j = i + span;
                            if (j > data.cky.n) return <td key={i} />;
                            const c = cell(i, j);
                            const hasS = c?.labels.includes("S");
                            return (
                              <td
                                key={i}
                                className={`h-12 min-w-16 rounded-md px-1 text-center align-middle font-mono ${hasS ? "bg-accent font-bold text-white" : c?.labels.length ? "bg-accent-soft text-accent" : "bg-surface-2 text-ink-3"}`}
                              >
                                {c?.labels.length ? c.labels.slice(0, 3).join(" ") : "∅"}
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                    <tr>
                      {data.words.map((w: string, i: number) => (
                        <td key={i} className="pt-1 text-center text-xs font-medium">
                          {w}
                          <div className="font-mono text-[0.6rem] text-ink-3">{data.tags[i]}</div>
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>
            </Panel>
          </div>
        </>
      )}
    </div>
  );
}

// ------------------------------------------------------------------- WordNet
export function WordNetLab() {
  const [w, setW] = useState("hopeless");
  const [a, setA] = useState("sadness");
  const [b, setB] = useState("grief");
  const [ws, setWs] = useState("I have been so anxious to see my friends again after the holidays");
  const [wt, setWt] = useState("anxious");
  const wn = useLive("wordnet", { word: w });
  const sim = useLive("similarity", { a, b });
  const wsd = useLive("wsd", { sentence: ws, target: wt });
  const d = wn.data;
  return (
    <div className="space-y-4">
      <Panel title="Look up a word" sub="Try: blue, down, hopeless, tired, grateful, empty.">
        <input className="input" value={w} onChange={(e) => setW(e.target.value)} />
        {d && (
          <div className="mt-3">
            {d.lexicon ? (
              <div className={`mb-3 rounded-xl p-3 text-sm ${d.lexicon.pol === "mal" ? "bg-mal-soft" : "bg-ada-soft"}`}>
                In the DriftLens lexicon as <b>{d.lexicon.lemma}</b> → <b>{d.lexicon.label}</b> ({d.lexicon.pol === "mal" ? "maladaptive" : "adaptive"}) · weight {d.lexicon.weight} · sense ratio {d.lexicon.ratio}
                <div className="mt-1 text-xs text-ink-2">Sense ratio = share of the word&apos;s everyday usage (SemCor counts) that falls in the feeling senses. Low ratio ⇒ disambiguate with Lesk first.</div>
              </div>
            ) : (
              <div className="mb-3 rounded-xl bg-surface-2 p-3 text-sm text-ink-2">Not in the self-state lexicon.</div>
            )}
            <div className="grid gap-2 md:grid-cols-2">
              {d.synsets.map((s: any) => (
                <div key={s.name} className={`rounded-xl border p-3 text-sm ${s.in_lexicon ? "border-accent bg-accent-soft/40" : "border-line"}`}>
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono text-xs text-accent">{s.name}</span>
                    {s.in_lexicon && <span className="rounded-full bg-accent px-2 text-[0.65rem] text-white">feeling sense</span>}
                  </div>
                  <div className="mt-1">{s.definition}</div>
                  {s.examples[0] && <div className="mt-1 text-xs italic text-ink-3">“{s.examples[0]}”</div>}
                  <div className="mt-1.5 text-xs text-ink-2">
                    <b>Synonyms:</b> {s.lemmas.join(", ")}
                    {s.antonyms.length > 0 && (
                      <>
                        {" "}
                        · <b>Antonyms:</b> {s.antonyms.join(", ")}
                      </>
                    )}
                  </div>
                  {s.hypernyms.length > 1 && <div className="mt-1 font-mono text-[0.65rem] text-ink-3">{s.hypernyms.join(" › ")}</div>}
                </div>
              ))}
            </div>
          </div>
        )}
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Word similarity" sub="Best matching senses (same POS). IC-based measures use Brown corpus counts.">
          <div className="grid grid-cols-2 gap-2">
            <input className="input" value={a} onChange={(e) => setA(e.target.value)} />
            <input className="input" value={b} onChange={(e) => setB(e.target.value)} />
          </div>
          {sim.data?.pair ? (
            <div className="mt-3 space-y-2 text-sm">
              <div className="font-mono text-xs text-ink-3">
                {sim.data.pair[0]} ↔ {sim.data.pair[1]} · common ancestor: {sim.data.lcs}
              </div>
              {[
                ["Path", sim.data.path, 1],
                ["Wu–Palmer", sim.data.wup, 1],
                ["Lin", sim.data.lin, 1],
                ["Leacock–Chodorow", sim.data.lch, 3.7],
                ["Resnik", sim.data.resnik, 12],
              ]
                .filter(([, v]) => v !== undefined && v !== null)
                .map(([k, v, m]) => (
                  <div key={k as string} className="flex items-center gap-2">
                    <span className="w-36 text-ink-2">{k}</span>
                    <div className="h-2 flex-1 rounded-full bg-surface-2">
                      <div className="h-2 rounded-full bg-accent" style={{ width: `${Math.min(100, ((v as number) / (m as number)) * 100)}%` }} />
                    </div>
                    <span className="w-12 text-right font-mono">{v as number}</span>
                  </div>
                ))}
            </div>
          ) : (
            <div className="mt-3 text-sm text-ink-3">No comparable senses.</div>
          )}
        </Panel>

        <Panel title="Word sense disambiguation (Lesk)" sub="“anxious to see my friends” means eager, not worried, so it should not count as anxiety.">
          <Field label="Sentence">
            <input className="input" value={ws} onChange={(e) => setWs(e.target.value)} />
          </Field>
          <div className="mt-2">
            <Field label="Target word">
              <input className="input" value={wt} onChange={(e) => setWt(e.target.value)} />
            </Field>
          </div>
          {wsd.data && (
            <div className="mt-3 grid gap-2 sm:grid-cols-2">
              {[
                ["NLTK simplified Lesk", wsd.data.simple],
                ["Adapted Lesk (DriftLens)", wsd.data.adapted],
              ].map(([k, v]: any) => (
                <div key={k} className="rounded-xl bg-surface-2 p-3 text-sm">
                  <div className="text-xs text-ink-3">{k}</div>
                  <div className="font-mono text-xs text-accent">{v?.name ?? "—"}</div>
                  <div>{v?.definition ?? "no evidence → fall back to prior weight"}</div>
                </div>
              ))}
              <div className={`rounded-xl p-2 text-center text-sm sm:col-span-2 ${wsd.data.counted ? "bg-mal-soft" : "bg-ada-soft"}`}>{wsd.data.counted ? "Counted as self-state evidence" : "Not counted: chosen sense isn't a feeling sense"}</div>
            </div>
          )}
        </Panel>
      </div>
      <ErrorNote error={wn.error || sim.error || wsd.error} />
    </div>
  );
}

// ----------------------------------------------------------------- sentiment
export function SentimentLab() {
  const [t, setT] = useState("I'm not happy. I feel hopeless but my friends help. Honestly the party was great, yet I still feel empty inside.");
  const { data, error } = useLive("sentiment", { text: t });
  const tone = (v: number) => (v > 0.05 ? "text-ada" : v < -0.05 ? "text-mal" : "text-ink-3");
  return (
    <div className="space-y-4">
      <Panel title="VADER vs the DriftLens self-state lexicon">
        <textarea className="input h-24" value={t} onChange={(e) => setT(e.target.value)} />
      </Panel>
      <ErrorNote error={error} />
      {data && (
        <>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="card p-4">
              <div className="text-xs text-ink-3">VADER compound (−1 … +1)</div>
              <div className={`text-3xl font-semibold ${tone(data.vader.compound)}`}>{data.vader.compound.toFixed(2)}</div>
              <div className="mt-1 text-xs text-ink-3">
                pos {data.vader.pos} · neu {data.vader.neu} · neg {data.vader.neg}
              </div>
            </div>
            <div className="card p-4">
              <div className="text-xs text-ink-3">Self-state balance (adaptive − maladaptive per 100 words)</div>
              <div className={`text-3xl font-semibold ${tone(data.lex.net)}`}>{data.lex.net.toFixed(1)}</div>
              <div className="mt-1 text-xs text-ink-3">
                adaptive {data.lex.ada} · maladaptive {data.lex.mal}
              </div>
            </div>
          </div>
          <Panel title="Sentence by sentence">
            <div className="space-y-2">
              {data.sentences.map((s: any, i: number) => (
                <div key={i} className="grid gap-2 rounded-xl bg-surface-2 p-3 sm:grid-cols-[1fr_90px_90px]">
                  <div className="text-sm">
                    {s.sentence}
                    <div className="mt-1 flex flex-wrap gap-1">
                      {s.hits.map((h: any, k: number) => (
                        <span key={k} className={`rounded px-1.5 text-xs ${h.negated && h.w === 0 ? "line-through opacity-60" : h.pol === "mal" ? "bg-mal-soft text-mal" : "bg-ada-soft text-ada"}`}>
                          {h.text} · {h.cat.replace(/_/g, " ")}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-[0.65rem] text-ink-3">VADER</div>
                    <div className={`font-mono font-semibold ${tone(s.vader.compound)}`}>{s.vader.compound.toFixed(2)}</div>
                  </div>
                  <div className="text-center">
                    <div className="text-[0.65rem] text-ink-3">Lexicon</div>
                    <div className={`font-mono font-semibold ${tone(s.lex_net)}`}>{s.lex_net.toFixed(1)}</div>
                  </div>
                </div>
              ))}
            </div>
          </Panel>
          <Explainer>
            VADER scores general positivity (“great”, “party”, “help”). The DriftLens lexicon only counts words about the writer&apos;s own state, and its negation FSA turns “not happy” into half a maladaptive hit. On labelled CLPsych posts the lexicon correlated better with well-being (Spearman 0.43 vs 0.36).
          </Explainer>
        </>
      )}
    </div>
  );
}
