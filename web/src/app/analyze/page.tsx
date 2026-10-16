"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Loader2, Play, Quote, Wand2 } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, STATE_META } from "@/lib/api";
import { useNarrator } from "@/lib/narrator";
import { ErrorNote, Explainer, ListenButton, SectionHead, StateBadge, VoicePicker } from "@/components/ui";

// Synthetic example timelines (written for the demo; no real user data).
const EXAMPLES: Record<string, string> = {
  "Semester slump & recovery": `Finally finished my first semester at college. Tired but honestly proud of myself, my roommate and I celebrated with pizza.
---
Anyone have tips for organising notes? I keep losing track of deadlines and my planner is a mess lol.
---
Weird week. My dad lost his job and everyone at home is on edge. I've been trying to stay calm but I cant sleep.
---
I don't know why I'm writing this. I feel so empty and worthless lately. I skipped classes all week and I just stay in bed. Nobody would notice if I disappeared.
---
It keeps getting worse. I hate myself, I'm a failure and I'm so tired of pretending I'm fine. What's the point of any of this.
---
Update: I talked to my sister last night and she drove down to see me. I booked an appointment with the campus counsellor. Still scared but feeling a bit more hopeful.`,
  "Slow build-up": `New job starts Monday! Nervous but excited, the team seemed really friendly.
---
First week done. Lots to learn, staying late most days but it's fine.
---
Working weekends now. I'm tired all the time and I barely see my friends anymore.
---
I'm exhausted. I can't focus, I keep making mistakes and my manager noticed. I feel useless.
---
I haven't slept properly in weeks. I feel anxious every morning and alone at night. I don't think I can keep doing this.`,
  "Steady & calm": `Went hiking with my brother today, the weather was perfect and the view was amazing.
---
Trying out a new pasta recipe tonight. Wish me luck!
---
Finished the book my friend recommended. Really enjoyed it, any similar suggestions?
---
Had a relaxed weekend, cleaned my room and called my parents. Feeling grateful.`,
};

type Post = {
  index: number;
  text: string;
  state: string;
  confidence: number;
  probs: Record<string, number>;
  hits: { start: number; end: number; text: string; cat: string; pol: string; w: number; negated: boolean; corrected_from?: string | null }[];
  corrections: [string, string][];
  self_evals: { text: string; polarity: string }[];
  tagged: [string, string][];
  keywords: { term: string; score: number }[];
  crisis: boolean;
  signals: Record<string, number>;
};
type Result = { posts: Post[]; digest: { post: number; sentence: string }[]; categories: { cat: string; label: string; weight: number; pol: string }[]; narration: string; model: string };

const MODELS = ["CRF (prior-corrected)", "Gaussian HMM", "Hybrid HMM (balanced)", "MaxEnt"];

function Highlight({ p }: { p: Post }) {
  const out: React.ReactNode[] = [];
  let pos = 0;
  [...p.hits].sort((a, b) => a.start - b.start).forEach((h, i) => {
    if (h.start < pos) return;
    out.push(p.text.slice(pos, h.start));
    const negIgnored = h.negated && h.w === 0;
    const cls = negIgnored
      ? "line-through opacity-60"
      : h.cat === "crisis"
        ? "bg-mal text-white font-semibold"
        : h.pol === "mal"
          ? "bg-mal-soft border-b-2 border-mal"
          : "bg-ada-soft border-b-2 border-ada";
    const tip = `${h.cat.replace(/_/g, " ")} · weight ${h.w}${h.corrected_from ? ` · spelling fixed from “${h.corrected_from}”` : ""}${negIgnored ? " · negated, ignored" : ""}`;
    out.push(
      <span key={i} title={tip} className={`rounded px-0.5 ${cls} ${h.corrected_from ? "underline decoration-dotted" : ""}`}>
        {p.text.slice(h.start, h.end)}
      </span>
    );
    pos = h.end;
  });
  out.push(p.text.slice(pos));
  return <p className="leading-7">{out}</p>;
}

function PostCard({ p, open }: { p: Post; open: boolean }) {
  const [show, setShow] = useState(open);
  const [tags, setTags] = useState(false);
  const m = STATE_META[p.state];
  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: p.index * 0.05 }} className="card overflow-hidden">
      <button onClick={() => setShow(!show)} className="flex w-full items-center gap-3 p-4 text-left">
        <span className="grid h-9 w-9 shrink-0 place-items-center rounded-lg font-mono text-sm font-bold text-white" style={{ background: m.color }}>
          {p.state}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <b>Post {p.index + 1}</b>
            <StateBadge s={p.state} p={p.confidence} />
            {p.crisis && <span className="rounded-full bg-mal px-2 py-0.5 text-xs font-semibold text-white">crisis language</span>}
          </div>
          {!show && <div className="mt-0.5 truncate text-sm text-ink-3">{p.text}</div>}
        </div>
      </button>
      {show && (
        <div className="space-y-4 border-t border-line p-4">
          <Highlight p={p} />
          <div className="grid gap-4 md:grid-cols-3">
            <div>
              <div className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-ink-3">State probabilities (CRF / HMM marginals)</div>
              {Object.entries(p.probs).map(([s, v]) => (
                <div key={s} className="mb-1 flex items-center gap-2 text-xs">
                  <span className="w-28 shrink-0">{STATE_META[s].name}</span>
                  <div className="h-2 flex-1 rounded-full bg-surface-2">
                    <div className="h-2 rounded-full" style={{ width: `${v * 100}%`, background: STATE_META[s].color }} />
                  </div>
                  <span className="w-9 text-right font-mono">{Math.round(v * 100)}%</span>
                </div>
              ))}
              {Object.entries(p.probs).some(([s, v]) => s !== p.state && v > p.confidence) && (
                <div className="mt-2 rounded-lg bg-surface-2 px-2 py-1.5 text-xs text-ink-2">
                  Why not the highest bar? Changes are rare in the training data (about 1 post in 3), so the decoder divides by each state&apos;s prior. That boosts S/E/B, trading some precision for recall on the moments that matter.
                </div>
              )}
            </div>
            <div>
              <div className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-ink-3">Self-statements (chunk grammar)</div>
              {p.self_evals.length ? (
                p.self_evals.map((e, i) => (
                  <div key={i} className={`mb-1 rounded-md px-2 py-1 text-sm ${e.polarity === "neg" ? "bg-mal-soft" : "bg-ada-soft"}`}>
                    {e.polarity === "neg" ? "▼" : "▲"} <i>{e.text}</i>
                  </div>
                ))
              ) : (
                <div className="text-sm text-ink-3">none found</div>
              )}
              {p.corrections.length > 0 && (
                <>
                  <div className="mb-1 mt-3 text-xs font-semibold uppercase tracking-wider text-ink-3">Spelling fixes (noisy channel)</div>
                  <div className="text-sm">{p.corrections.map(([a, b]) => `${a} → ${b}`).join(", ")}</div>
                </>
              )}
              <div className="mb-1 mt-3 text-xs font-semibold uppercase tracking-wider text-ink-3">Keywords (TF-IDF)</div>
              <div className="flex flex-wrap gap-1">
                {p.keywords.map((k) => (
                  <span key={k.term} className="rounded-md bg-accent-soft px-2 py-0.5 text-xs text-accent">
                    {k.term}
                  </span>
                ))}
              </div>
            </div>
            <div>
              <div className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-ink-3">Signals</div>
              <table className="w-full text-sm">
                <tbody>
                  {[
                    ["Self-state balance", p.signals.net, "adaptive − maladaptive words per 100"],
                    ["VADER sentiment", p.signals.vader, "−1 … +1"],
                    ["Personal novelty", p.signals.novelty, "bits/token vs own history"],
                    ["New vocabulary", p.signals.vocab_novelty, "share of new content words"],
                    ["Length vs usual", p.signals.z_len, "z-score within this timeline"],
                    ["First-person rate", p.signals.fp_rate, "I, me, my…"],
                  ].map(([k, v, d]) => (
                    <tr key={k as string} title={d as string} className="border-b border-line last:border-0">
                      <td className="py-1 text-ink-2">{k}</td>
                      <td className="py-1 text-right font-mono">{(v as number)?.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <div>
            <button onClick={() => setTags(!tags)} className="text-xs font-medium text-accent">
              {tags ? "Hide" : "Show"} HMM part-of-speech tags
            </button>
            {tags && (
              <div className="mt-2 flex flex-wrap gap-1">
                {p.tagged.map(([w, t], i) => (
                  <span key={i} className="rounded-md border border-line px-1.5 py-0.5 text-xs">
                    {w} <span className="font-mono text-[0.65rem] text-accent">{t}</span>
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </motion.div>
  );
}

export default function Analyze() {
  const [ex, setEx] = useState(Object.keys(EXAMPLES)[0]);
  const [text, setText] = useState(EXAMPLES[ex]);
  const [model, setModel] = useState(MODELS[0]);
  const [res, setRes] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const n = useNarrator();

  const posts = text.split(/\n\s*-{3,}\s*\n|\n\s*\n/).map((s) => s.trim()).filter(Boolean);

  async function run(speakAfter = false) {
    setLoading(true);
    setErr(null);
    try {
      const r = await api<Result>("analyse", { posts, model });
      setRes(r);
      if (speakAfter) n.speak(r.narration, "analysis", "Explaining this timeline");
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setLoading(false);
    }
  }

  const chart = res?.posts.map((p) => ({
    post: p.index + 1,
    balance: p.signals.net,
    vader: +(p.signals.vader * 10).toFixed(2),
    novelty: +(p.signals.novelty * 10).toFixed(2),
  }));

  return (
    <div className="space-y-10">
      <SectionHead eyebrow="Analyse" title="Find the moments of change in a timeline">
        Each block below is one post, oldest first. Separate posts with a line containing <span className="kbd">---</span>. The examples are made up for the demo.
      </SectionHead>

      <div className="grid gap-5 lg:grid-cols-[1.3fr_1fr]">
        <div className="card p-4">
          <div className="mb-3 flex flex-wrap gap-2">
            {Object.keys(EXAMPLES).map((k) => (
              <button
                key={k}
                onClick={() => {
                  setEx(k);
                  setText(EXAMPLES[k]);
                  setRes(null);
                }}
                className={`rounded-full px-3 py-1 text-sm transition ${ex === k ? "bg-ink text-bg" : "bg-surface-2 text-ink-2 hover:bg-line"}`}
              >
                {k}
              </button>
            ))}
          </div>
          <textarea className="input h-80 font-mono text-sm leading-6" value={text} onChange={(e) => setText(e.target.value)} />
          <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
            <span className="text-sm text-ink-3">{posts.length} posts</span>
            <div className="flex flex-wrap gap-2">
              <select className="input !w-auto !py-1.5 text-sm" value={model} onChange={(e) => setModel(e.target.value)} aria-label="Sequence model">
                {MODELS.map((m) => (
                  <option key={m}>{m}</option>
                ))}
              </select>
              <button className="btn btn-ghost" disabled={loading || posts.length < 2} onClick={() => run(false)}>
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Wand2 className="h-4 w-4" />} Analyse
              </button>
              <button className="btn btn-primary" disabled={loading || posts.length < 2} onClick={() => run(true)}>
                <Play className="h-4 w-4" /> Analyse & explain aloud
              </button>
            </div>
          </div>
        </div>
        <div className="space-y-3">
          <Explainer>
            <b className="text-ink">What you&apos;ll get.</b> Each post is tagged <b>O</b> (no change), <b>S</b> (sudden switch), <b>E</b> (gradual escalation) or <b>B</b> (both). Words from the WordNet feelings lexicon are highlighted{" "}
            <span className="rounded bg-mal-soft px-1">maladaptive</span> or <span className="rounded bg-ada-soft px-1">adaptive</span>. Crossed-out words were negated (“not sad”).
          </Explainer>
          <Explainer>
            <b className="text-ink">Which model?</b> The <b>CRF</b> and <b>Gaussian HMM</b> tied in cross-validation. The CRF gives better-calibrated probabilities, so it&apos;s the default. <b>MaxEnt</b> ignores order, so it&apos;s a useful contrast.
          </Explainer>
          <Explainer>
            <b className="text-ink">The voice.</b> “Explain aloud” writes a short explanation from the results (data-to-text NLG) and reads it with a neural voice.
            <div className="mt-2">
              <VoicePicker />
            </div>
          </Explainer>
        </div>
      </div>

      <ErrorNote error={err} />

      {res && (
        <div className="space-y-8">
          {/* summary */}
          <div className="grid gap-3 sm:grid-cols-4">
            {[
              ["Posts", res.posts.length],
              ["Switches", res.posts.filter((p) => "SB".includes(p.state)).length],
              ["Escalation posts", res.posts.filter((p) => "EB".includes(p.state)).length],
              ["Crisis-language posts", res.posts.filter((p) => p.crisis).length],
            ].map(([k, v]) => (
              <div key={k as string} className="card p-4">
                <div className="text-sm text-ink-3">{k}</div>
                <div className="text-3xl font-semibold tabular-nums">{v}</div>
              </div>
            ))}
          </div>

          {/* narration */}
          <div className="card flex flex-wrap items-start gap-4 border-accent/40 bg-accent-soft/50 p-5">
            <Quote className="h-6 w-6 shrink-0 text-accent" />
            <div className="min-w-0 flex-1">
              <div className="mb-1 text-xs font-semibold uppercase tracking-wider text-accent">Plain-English explanation (generated from the results)</div>
              <p className="leading-relaxed">{res.narration}</p>
            </div>
            <ListenButton id="analysis" text={res.narration} title="Explaining this timeline" />
          </div>

          {/* chart + strip */}
          <div className="card p-5">
            <div className="mb-1 font-semibold">Signals over time</div>
            <div className="mb-4 text-sm text-ink-3">Self-state balance from the lexicon, VADER sentiment and personal LM novelty (both ×10 to share the axis).</div>
            <div className="h-72">
              <ResponsiveContainer>
                <LineChart data={chart} margin={{ left: -10, right: 10 }}>
                  <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" />
                  <XAxis dataKey="post" stroke="var(--ink-3)" tickFormatter={(v) => `#${v}`} />
                  <YAxis stroke="var(--ink-3)" />
                  <Tooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--line)", borderRadius: 12 }} />
                  <Legend />
                  <ReferenceLine y={0} stroke="var(--ink-3)" />
                  <Line type="monotone" dataKey="balance" name="Self-state balance" stroke="var(--accent)" strokeWidth={3} dot={{ r: 4 }} />
                  <Line type="monotone" dataKey="vader" name="VADER ×10" stroke="var(--ink-3)" strokeDasharray="5 4" dot={false} />
                  <Line type="monotone" dataKey="novelty" name="Novelty ×10" stroke="#e0a100" strokeWidth={2} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-3 flex items-center gap-2 pl-8">
              <span className="w-20 text-xs text-ink-3">Predicted</span>
              <div className="flex flex-1 gap-1">
                {res.posts.map((p) => (
                  <div key={p.index} title={`Post ${p.index + 1}: ${STATE_META[p.state].name}`} className="grid h-8 flex-1 place-items-center rounded-md font-mono text-xs font-bold text-white" style={{ background: STATE_META[p.state].color }}>
                    {p.state}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* posts */}
          <div>
            <h3 className="mb-3 text-xl font-semibold">Post by post</h3>
            <div className="space-y-2">
              {res.posts.map((p) => (
                <PostCard key={`${res.model}-${p.index}-${p.text.slice(0, 10)}`} p={p} open={p.state !== "O"} />
              ))}
            </div>
          </div>

          {/* digest + categories */}
          <div className="grid gap-5 lg:grid-cols-2">
            <div className="card p-5">
              <div className="font-semibold">Timeline digest</div>
              <div className="mb-3 text-sm text-ink-3">Affect-biased TextRank: PageRank over TF-IDF sentence similarity, nudged toward feeling words.</div>
              <div className="space-y-2">
                {res.digest.map((d, i) => (
                  <div key={i} className="rounded-lg border-l-4 bg-surface-2 px-3 py-2 text-sm" style={{ borderColor: STATE_META[res.posts[d.post].state].color }}>
                    <span className="text-ink-3">Post {d.post + 1} · </span>
                    {d.sentence}
                  </div>
                ))}
              </div>
              <div className="mt-3">
                <ListenButton id="digest" text={() => "Here's the short summary. " + res.digest.map((d) => d.sentence).join(" ")} title="Digest" label="Read the digest" />
              </div>
            </div>
            <div className="card p-5">
              <div className="font-semibold">Feelings across the timeline</div>
              <div className="mb-3 text-sm text-ink-3">Summed lexicon weight per self-state category.</div>
              <div className="h-64">
                <ResponsiveContainer>
                  <BarChart data={res.categories} layout="vertical" margin={{ left: 20 }}>
                    <XAxis type="number" stroke="var(--ink-3)" />
                    <YAxis type="category" dataKey="label" stroke="var(--ink-3)" width={100} tick={{ fontSize: 12 }} />
                    <Tooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--line)", borderRadius: 12 }} />
                    <Bar dataKey="weight" radius={[0, 6, 6, 0]}>
                      {res.categories.map((c) => (
                        <Cell key={c.cat} fill={c.pol === "mal" ? "var(--mal)" : "var(--ada)"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
