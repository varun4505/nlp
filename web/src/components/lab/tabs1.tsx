"use client";

import { useState } from "react";
import { Dices, Loader2 } from "lucide-react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "@/lib/api";
import { ErrorNote, Explainer } from "@/components/ui";
import { Field, Panel, useLive } from "./common";

// ------------------------------------------------------------ text processing
export function TextLab() {
  const [text, setText] = useState("I was running late, and honestly im feeling happier than yesterday!! The studies are hard but my friends' support helps.");
  const { data, error } = useLive("tokenize", { text });
  return (
    <div className="space-y-4">
      <Panel>
        <Field label="Any text">
          <textarea className="input h-24" value={text} onChange={(e) => setText(e.target.value)} />
        </Field>
      </Panel>
      <ErrorNote error={error} />
      {data && (
        <>
          <div className="grid gap-3 sm:grid-cols-4">
            {[
              ["Sentences", data.sentences.length],
              ["Tokens", data.n_tokens],
              ["Types (unique)", data.types],
              ["Type/token ratio", (data.types / Math.max(data.n_tokens, 1)).toFixed(2)],
            ].map(([k, v]) => (
              <div key={k} className="card p-4">
                <div className="text-xs text-ink-3">{k}</div>
                <div className="text-2xl font-semibold">{v}</div>
              </div>
            ))}
          </div>
          <Panel title="1 · Sentence segmentation (Punkt) → 2 · word tokenisation (Treebank)" sub="Contractions like “im” are expanded first; stop words are faded.">
            <div className="space-y-2">
              {data.sentences.map((s: string, si: number) => (
                <div key={si} className="flex flex-wrap gap-1 rounded-lg bg-surface-2 p-2">
                  <span className="mr-1 font-mono text-xs text-ink-3">S{si + 1}</span>
                  {data.tokens
                    .filter((t: any) => t.sent === si)
                    .map((t: any, i: number) => (
                      <span key={i} className={`rounded-md border border-line bg-surface px-1.5 py-0.5 text-sm ${t.stop ? "opacity-40" : ""}`}>
                        {t.token}
                      </span>
                    ))}
                </div>
              ))}
            </div>
          </Panel>
          <Panel title="3 · Stemming vs lemmatisation" sub="Red cells: the stemmer produced a non-word or changed the word where the lemmatiser did not.">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wider text-ink-3">
                    {["Token", "POS (HMM)", "Porter", "Lancaster", "Snowball", "WordNet lemma"].map((h) => (
                      <th key={h} className="px-2 py-2">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.tokens
                    .filter((t: any) => t.alpha && !t.stop)
                    .map((t: any, i: number) => (
                      <tr key={i} className="border-t border-line">
                        <td className="px-2 py-1.5 font-medium">{t.lower}</td>
                        <td className="px-2 py-1.5 font-mono text-xs text-accent">{t.tag}</td>
                        {["porter", "lancaster", "snowball"].map((k) => (
                          <td key={k} className={`px-2 py-1.5 font-mono ${t[k] !== t.lemma && t[k] !== t.lower ? "text-mal" : ""}`}>
                            {t[k]}
                          </td>
                        ))}
                        <td className="px-2 py-1.5 font-mono text-ada">{t.lemma}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      )}
      <Explainer>
        <b className="text-ink">Why it matters for DriftLens:</b> the lexicon lookup uses WordNet&apos;s <span className="kbd">morphy</span> (lemmatisation) plus a small derivational suffix stripper, so “hopelessness”, “hopelessly” and “hopeless” all hit the same entry.
      </Explainer>
    </div>
  );
}

// ------------------------------------------------------------- edit distance
export function EditLab() {
  const [a, setA] = useState("intention");
  const [b, setB] = useState("execution");
  const [weighted, setWeighted] = useState(false);
  const [sub2, setSub2] = useState(false);
  const { data, error } = useLive("edit-distance", { source: a, target: b, weighted, sub_cost: sub2 ? 2 : 1 }, 150);
  const onPath = new Set<string>((data?.path ?? []).map((p: number[]) => p.join(",")));
  const max = data ? Math.max(...data.matrix.flat()) : 1;
  return (
    <div className="space-y-4">
      <Panel>
        <div className="grid gap-3 sm:grid-cols-[1fr_1fr_auto]">
          <Field label="Source">
            <input className="input font-mono" value={a} onChange={(e) => setA(e.target.value)} maxLength={14} />
          </Field>
          <Field label="Target">
            <input className="input font-mono" value={b} onChange={(e) => setB(e.target.value)} maxLength={14} />
          </Field>
          <div className="flex flex-col justify-end gap-1.5 text-sm">
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={sub2} disabled={weighted} onChange={(e) => setSub2(e.target.checked)} /> Substitution costs 2 (J&amp;M)
            </label>
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={weighted} onChange={(e) => setWeighted(e.target.checked)} /> Keyboard-weighted (DriftLens)
            </label>
          </div>
        </div>
      </Panel>
      <ErrorNote error={error} />
      {data && (
        <div className="grid gap-4 lg:grid-cols-[auto_1fr]">
          <Panel title={`Distance = ${data.distance}`} sub="D[i][j] = min(D[i−1][j] + del, D[i][j−1] + ins, D[i−1][j−1] + sub). Highlighted: backtrace.">
            <div className="overflow-x-auto">
              <table className="border-separate border-spacing-0.5 font-mono text-xs">
                <thead>
                  <tr>
                    <th />
                    <th className="w-8 text-ink-3">#</th>
                    {data.target.split("").map((c: string, j: number) => (
                      <th key={j} className="w-8 text-accent">
                        {c}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.matrix.map((row: number[], i: number) => (
                    <tr key={i}>
                      <th className="pr-1 text-accent">{i === 0 ? "#" : data.source[i - 1]}</th>
                      {row.map((v, j) => {
                        const on = onPath.has(`${i},${j}`);
                        return (
                          <td
                            key={j}
                            className={`h-8 w-8 rounded-md text-center ${on ? "font-bold text-white" : ""}`}
                            style={{ background: on ? "var(--accent)" : `color-mix(in srgb, var(--accent) ${(v / max) * 22}%, var(--surface-2))` }}
                          >
                            {Number.isInteger(v) ? v : v.toFixed(1)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
          <Panel title="Alignment" sub="Reading the backtrace from top-left to bottom-right.">
            <div className="flex flex-wrap gap-1 font-mono">
              {data.ops.map((o: any, i: number) => (
                <div key={i} className={`rounded-lg border px-2 py-1 text-center ${o.op === "match" ? "border-line" : o.op === "sub" ? "border-st-s bg-[color-mix(in_srgb,var(--st-s)_12%,transparent)]" : o.op === "del" ? "border-mal bg-mal-soft" : o.op === "ins" ? "border-ada bg-ada-soft" : "border-accent bg-accent-soft"}`}>
                  <div>{o.a}</div>
                  <div>{o.b}</div>
                  <div className="text-[0.6rem] uppercase text-ink-3">{o.op}</div>
                </div>
              ))}
            </div>
            <div className="mt-4 flex flex-wrap gap-3 text-xs text-ink-3">
              <span>
                <b className="text-ada">ins</b> insert
              </span>
              <span>
                <b className="text-mal">del</b> delete
              </span>
              <span>
                <b style={{ color: "var(--st-s)" }}>sub</b> substitute
              </span>
              <span>
                <b className="text-accent">swap</b> transposition (Damerau)
              </span>
            </div>
            <div className="mt-4">
              <Explainer>
                With keyboard weights, a substitution between neighbouring keys (e→r) costs 0.5, vowel-for-vowel 0.7, doubled letters 0.6, and a swap 0.6. Try <span className="kbd">lonley → lonely</span>.
              </Explainer>
            </div>
          </Panel>
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ spelling
export function SpellLab() {
  const [prev, setPrev] = useState("feel");
  const [word, setWord] = useState("lonley");
  const [next, setNext] = useState("and");
  const [sentence, setSentence] = useState("i realy dont know why i feel so hopless and anxous");
  const { data, error } = useLive("spell", { word, prev, next, sentence });
  return (
    <div className="space-y-4">
      <Panel title="w* = argmax  log P(w | context) + log P(typed | w)" sub="Prior from the background bigram LM; channel from the weighted edit distance (α = 4).">
        <div className="grid gap-3 sm:grid-cols-3">
          <Field label="Previous word">
            <input className="input" value={prev} onChange={(e) => setPrev(e.target.value)} />
          </Field>
          <Field label="Typed word">
            <input className="input font-semibold" value={word} onChange={(e) => setWord(e.target.value)} />
          </Field>
          <Field label="Next word">
            <input className="input" value={next} onChange={(e) => setNext(e.target.value)} />
          </Field>
        </div>
      </Panel>
      <ErrorNote error={error} />
      {data && (
        <Panel title={data.is_word ? `“${data.word}” is already a dictionary word` : "Ranked candidates"} sub="Higher score is better. The typed word itself competes as the “no error” option.">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs uppercase tracking-wider text-ink-3">
                  {["Candidate", "Weighted DL", "Levenshtein", "log channel", "log P(w|context)", "Score"].map((h) => (
                    <th key={h} className="px-2 py-2">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.candidates.map((c: any, i: number) => (
                  <tr key={c.candidate} className={`border-t border-line ${i === 0 ? "bg-ada-soft font-semibold" : ""}`}>
                    <td className="px-2 py-1.5">{c.candidate}</td>
                    <td className="px-2 py-1.5 font-mono">{c.weighted}</td>
                    <td className="px-2 py-1.5 font-mono">{c.levenshtein}</td>
                    <td className="px-2 py-1.5 font-mono">{c.channel}</td>
                    <td className="px-2 py-1.5 font-mono">{c.prior}</td>
                    <td className="px-2 py-1.5 font-mono">{c.score}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}
      <Panel title="Correct a whole sentence">
        <input className="input" value={sentence} onChange={(e) => setSentence(e.target.value)} />
        {data?.corrected?.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1">
            {data.corrected.map((t: string, i: number) => {
              const ch = data.changes.find((c: any) => c.i === i);
              return (
                <span key={i} className={`rounded-md px-1.5 py-0.5 ${ch ? "bg-ada-soft" : "bg-surface-2"}`} title={ch ? `was “${ch.from}”` : undefined}>
                  {ch && <s className="mr-1 text-mal opacity-70">{ch.from}</s>}
                  {t}
                </span>
              );
            })}
          </div>
        )}
      </Panel>
    </div>
  );
}

// -------------------------------------------------------------------- n-gram
export function NgramLab() {
  const [s, setS] = useState("i feel so tired and alone tonight");
  const [hist, setHist] = useState("just finished my shift, time for some gaming\nnew keyboard arrived, loving it\nanyone playing the new update tonight?");
  const [np, setNp] = useState("i dont want to be here anymore, nothing feels real");
  const { data, error } = useLive("ngram", { sentence: s, history: hist.split("\n"), new_post: np });
  const zipf = useLive("zipf", {}, 0);
  const [seed, setSeed] = useState("i feel");
  const [temp, setTemp] = useState(1);
  const [gen, setGen] = useState<any>(null);
  const [genLoading, setGenLoading] = useState(false);
  async function generate() {
    setGenLoading(true);
    try {
      setGen(await api("generate", { seed, length: 25, temperature: temp, rng: Math.floor(Math.random() * 1e6) }));
    } finally {
      setGenLoading(false);
    }
  }
  return (
    <div className="space-y-4">
      <Panel title="Bigram probabilities" sub="Trained on 1.5M tokens of Reddit text. Compare unsmoothed MLE, Laplace (add-1) and Jelinek–Mercer interpolation.">
        <input className="input" value={s} onChange={(e) => setS(e.target.value)} />
      </Panel>
      <ErrorNote error={error} />
      {data && (
        <div className="grid gap-4 lg:grid-cols-[1fr_260px]">
          <Panel>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wider text-ink-3">
                    {["P(w | prev)", "count", "MLE", "Laplace", "Interpolated", "−log₂ P"].map((h) => (
                      <th key={h} className="px-2 py-2">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.tokens.map((t: any, i: number) => (
                    <tr key={i} className="border-t border-line">
                      <td className="px-2 py-1.5">
                        <span className="text-ink-3">{t.prev} →</span> <b>{t.token}</b> {t.oov && <span className="ml-1 rounded bg-mal-soft px-1 text-xs text-mal">OOV</span>}
                      </td>
                      <td className="px-2 py-1.5 font-mono">{t.bigram_count}</td>
                      <td className={`px-2 py-1.5 font-mono ${t.p_mle === 0 ? "text-mal" : ""}`}>{t.p_mle.toExponential(1)}</td>
                      <td className="px-2 py-1.5 font-mono">{t.p_laplace.toExponential(1)}</td>
                      <td className="px-2 py-1.5 font-mono font-semibold">{t.p_interp.toExponential(1)}</td>
                      <td className="px-2 py-1.5">
                        <div className="flex items-center gap-2">
                          <div className="h-2 rounded-full bg-accent" style={{ width: `${Math.min(t.bits * 5, 90)}px` }} />
                          <span className="font-mono text-xs">{t.bits.toFixed(1)}</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
          <div className="space-y-3">
            <div className="card p-4">
              <div className="text-xs text-ink-3">Perplexity, interpolated</div>
              <div className="text-3xl font-semibold text-accent">{data.perplexity}</div>
            </div>
            <div className="card p-4">
              <div className="text-xs text-ink-3">Perplexity, Laplace</div>
              <div className="text-3xl font-semibold">{data.perplexity_laplace}</div>
            </div>
            <Explainer>
              λ = {data.lambdas.join(" / ")} (unigram / bigram), tuned on held-out data. Any zero in the MLE column would make the whole sentence probability zero.
            </Explainer>
          </div>
        </div>
      )}

      <Panel title="Personal novelty: does this writer sound like themselves?" sub="H_mix uses a mixture of the background LM and a tiny LM of the writer's earlier posts (β = 0.3).">
        <div className="grid gap-3 md:grid-cols-2">
          <Field label="Earlier posts (one per line)">
            <textarea className="input h-24" value={hist} onChange={(e) => setHist(e.target.value)} />
          </Field>
          <Field label="New post">
            <textarea className="input h-24" value={np} onChange={(e) => setNp(e.target.value)} />
          </Field>
        </div>
        {data?.novelty && (
          <div className="mt-3 grid gap-3 sm:grid-cols-3">
            <div className="rounded-xl bg-surface-2 p-3">
              <div className="text-xs text-ink-3">H background</div>
              <div className="text-xl font-semibold">{data.novelty.h_bg} bits</div>
            </div>
            <div className="rounded-xl bg-surface-2 p-3">
              <div className="text-xs text-ink-3">H with personal history</div>
              <div className="text-xl font-semibold">{data.novelty.h_mix} bits</div>
            </div>
            <div className={`rounded-xl p-3 ${data.novelty.novelty > 0.2 ? "bg-mal-soft" : "bg-ada-soft"}`}>
              <div className="text-xs text-ink-3">Novelty = H_mix − H_bg</div>
              <div className="text-xl font-semibold">
                {data.novelty.novelty > 0 ? "+" : ""}
                {data.novelty.novelty} {data.novelty.novelty > 0.2 ? "· unlike usual self" : "· sounds like themselves"}
              </div>
            </div>
          </div>
        )}
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Shannon's game: generate from the bigram LM" sub="Each next word is sampled from P(w | previous word). Temperature < 1 is safer, > 1 is wilder.">
          <div className="flex flex-wrap items-end gap-2">
            <Field label="Start with">
              <input className="input !w-40" value={seed} onChange={(e) => setSeed(e.target.value)} />
            </Field>
            <Field label={`Temperature ${temp.toFixed(1)}`}>
              <input type="range" min={0.3} max={2} step={0.1} value={temp} onChange={(e) => setTemp(+e.target.value)} className="w-36 accent-[var(--accent)]" />
            </Field>
            <button className="btn btn-primary" onClick={generate}>
              {genLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Dices className="h-4 w-4" />} Generate
            </button>
          </div>
          {gen && (
            <>
              <p className="mt-4 rounded-xl bg-surface-2 p-3 font-serif text-xl italic">“{gen.text}”</p>
              <div className="mt-3 text-xs text-ink-3">First choices the model considered:</div>
              <div className="mt-1 space-y-1">
                {gen.steps.slice(0, 4).map((st: any, i: number) => (
                  <div key={i} className="flex flex-wrap items-center gap-1 text-xs">
                    <span className="w-16 truncate text-ink-3">{st.prev} →</span>
                    {st.top.map((t: any) => (
                      <span key={t.w} className={`rounded px-1.5 py-0.5 ${t.w === st.chosen ? "bg-accent text-white" : "bg-surface-2"}`}>
                        {t.w} {(t.p * 100).toFixed(0)}%
                      </span>
                    ))}
                  </div>
                ))}
              </div>
            </>
          )}
        </Panel>
        <Panel title="Zipf's law in our corpus" sub={zipf.data ? `${zipf.data.types.toLocaleString()} word types. On a log–log plot, frequency × rank ≈ constant.` : "Loading…"}>
          {zipf.data && (
            <div className="h-64">
              <ResponsiveContainer>
                <LineChart data={zipf.data.points} margin={{ left: 0, right: 10 }}>
                  <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" />
                  <XAxis dataKey="rank" scale="log" type="number" domain={["auto", "auto"]} stroke="var(--ink-3)" tickFormatter={(v) => v.toLocaleString()} />
                  <YAxis dataKey="freq" scale="log" type="number" domain={["auto", "auto"]} stroke="var(--ink-3)" tickFormatter={(v) => (v >= 1000 ? `${Math.round(v / 1000)}k` : v)} />
                  <Tooltip
                    contentStyle={{ background: "var(--surface)", border: "1px solid var(--line)", borderRadius: 12 }}
                    formatter={(v: any) => v.toLocaleString()}
                    labelFormatter={(l: any) => {
                      const p = zipf.data.points.find((x: any) => x.rank === l);
                      return `rank ${l}: “${p?.word}”`;
                    }}
                  />
                  <Line dataKey="freq" stroke="var(--accent)" strokeWidth={2.5} dot={{ r: 2 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}

