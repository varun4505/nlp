"use client";

import Link from "next/link";
import { useState } from "react";
import { motion } from "framer-motion";
import { ArrowRight, Brain, FlaskConical, LineChart } from "lucide-react";
import { ListenButton, SectionHead, VoicePicker } from "@/components/ui";
import { SCRIPTS } from "@/lib/scripts";
import { STATE_META } from "@/lib/api";

const SENT = [
  ["I", "PRON"],
  ["feel", "VERB"],
  ["so", "ADV"],
  ["hopeless", "ADJ"],
];
const TL = [
  ["Proud of my first semester!", "O"],
  ["Tips for organising notes?", "O"],
  ["Dad lost his job. Can't sleep.", "E"],
  ["I feel empty and worthless.", "B"],
  ["Booked a counsellor. A bit hopeful.", "S"],
];

const STEPS = [
  { k: "clean", n: "1", t: "Clean & correct", m: "Noisy channel", d: "Tokenise, normalise Reddit text, fix typos with a noisy-channel speller." },
  { k: "lexicon", n: "2", t: "Feelings lexicon", m: "WordNet · FSA", d: "Grow a self-state dictionary from WordNet senses; handle negation with an FSA." },
  { k: "lm", n: "3", t: "Language model", m: "N-grams", d: "Bigram LM + a personal LM per writer → how unlike themselves a post sounds." },
  { k: "grammar", n: "4", t: "Tagging & grammar", m: "HMM · chunking", d: "HMM POS tagger + chunk grammar finds self-statements like “I am worthless”." },
  { k: "features", n: "5", t: "25 features", m: "Feature design", d: "Sentiment, length, novelty, self-statements — each relative to the writer’s own baseline." },
  { k: "sequence", n: "6", t: "Sequence model", m: "HMM · CRF", d: "HMM / CRF reads the timeline post by post; Viterbi finds the best tag sequence." },
  { k: "output", n: "7", t: "Tags + digest", m: "TextRank", d: "O / S / E / B per post with confidence, plus a TextRank summary." },
] as const;

export default function Home() {
  const [step, setStep] = useState<string>("clean");
  const cur = STEPS.find((s) => s.k === step)!;
  return (
    <div className="space-y-24">
      {/* hero */}
      <section className="grid items-center gap-10 pt-6 lg:grid-cols-[1.15fr_1fr]">
        <div>
          <div className="eyebrow mb-3">Classical NLP · Moment-of-change detection</div>
          <h1 className="font-serif text-5xl leading-[1.04] tracking-tight sm:text-6xl">
            Read a person&apos;s posts <em className="text-accent">like a sentence.</em>
          </h1>
          <p className="mt-5 max-w-xl text-lg text-ink-2">
            DriftLens spots the moments when someone&apos;s online writing changes, a sudden switch or a slow
            escalation, using only classical NLP: no black-box deep learning, every step explainable.
          </p>
          <div className="mt-7 flex flex-wrap items-center gap-3">
            <ListenButton id="tour" text={SCRIPTS.tour} title="Guided tour" label="Play 1-minute guided tour" primary />
            <Link href="/analyze" className="btn btn-ghost">
              Try it on a timeline <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
          <div className="mt-4">
            <VoicePicker />
          </div>
        </div>

        {/* analogy */}
        <div className="card p-5 sm:p-6">
          <div className="text-xs font-semibold uppercase tracking-wider text-ink-3">A POS tagger tags words…</div>
          <div className="mt-3 flex flex-wrap gap-2">
            {SENT.map(([w, t], i) => (
              <motion.div key={w} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 * i }} className="rounded-xl border border-line bg-surface-2 px-3 py-2 text-center">
                <div className="font-medium">{w}</div>
                <div className="font-mono text-[0.7rem] text-accent">{t}</div>
              </motion.div>
            ))}
          </div>
          <div className="mt-6 text-xs font-semibold uppercase tracking-wider text-ink-3">…DriftLens tags posts</div>
          <div className="mt-3 space-y-2">
            {TL.map(([p, s], i) => (
              <motion.div key={p} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.5 + 0.12 * i }} className="flex items-center gap-3">
                <span className="w-5 text-right font-mono text-xs text-ink-3">{i + 1}</span>
                <div className="flex-1 truncate rounded-lg bg-surface-2 px-3 py-1.5 text-sm">{p}</div>
                <span className="grid h-7 w-7 shrink-0 place-items-center rounded-md font-mono text-xs font-bold text-white" style={{ background: STATE_META[s].color }}>
                  {s}
                </span>
              </motion.div>
            ))}
          </div>
          <div className="mt-5 grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
            {Object.entries(STATE_META).map(([k, m]) => (
              <div key={k} className="flex items-start gap-1.5">
                <span className="mt-0.5 h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: m.color }} />
                <span>
                  <b>{k}</b> {m.name}
                </span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* pipeline */}
      <section>
        <SectionHead eyebrow="How it works" title="Seven steps, all classical NLP" right={<ListenButton id={`step-${cur.k}`} text={SCRIPTS.pipeline[cur.k]} title={`Step ${cur.n}: ${cur.t}`} label={`Explain step ${cur.n}`} />}>
          Click a step to see what it does.
        </SectionHead>
        <div className="grid gap-2 sm:grid-cols-4 lg:grid-cols-7">
          {STEPS.map((s) => (
            <button
              key={s.k}
              onClick={() => setStep(s.k)}
              className={`card relative p-3 text-left transition hover:-translate-y-0.5 ${step === s.k ? "!border-accent ring-2 ring-accent-soft" : ""}`}
            >
              <div className={`mb-2 grid h-7 w-7 place-items-center rounded-full text-xs font-bold ${step === s.k ? "bg-accent text-white" : "bg-surface-2 text-ink-2"}`}>{s.n}</div>
              <div className="text-sm font-semibold leading-tight">{s.t}</div>
              <div className="mt-1 text-[0.7rem] text-ink-3">{s.m}</div>
            </button>
          ))}
        </div>
        <motion.div key={cur.k} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="card mt-3 flex flex-wrap items-center justify-between gap-4 p-5">
          <div className="max-w-3xl">
            <div className="text-sm text-ink-3">
              Step {cur.n} · {cur.m}
            </div>
            <div className="mt-1 text-lg font-medium">{cur.d}</div>
            <p className="mt-2 text-sm text-ink-2">{SCRIPTS.pipeline[cur.k]}</p>
          </div>
        </motion.div>
      </section>

      {/* CTA */}
      <section className="grid gap-4 md:grid-cols-3">
        {[
          { href: "/analyze", icon: Brain, t: "Analyse a timeline", d: "Paste posts or use an example. Get tags, highlighted words, confidence and a spoken explanation." },
          { href: "/lab", icon: FlaskConical, t: "NLP Lab", d: "Eight interactive demos: edit distance tables, Viterbi trellis, CKY parsing, WordNet, n-grams…" },
          { href: "/results", icon: LineChart, t: "Results", d: "Cross-validated scores, ablations and the honest takeaways, with charts." },
        ].map(({ href, icon: Icon, t, d }) => (
          <Link key={href} href={href} className="card group p-6 transition hover:-translate-y-1 hover:shadow-lg">
            <Icon className="h-6 w-6 text-accent" />
            <div className="mt-3 flex items-center gap-2 text-lg font-semibold">
              {t} <ArrowRight className="h-4 w-4 transition group-hover:translate-x-1" />
            </div>
            <p className="mt-1 text-sm text-ink-2">{d}</p>
          </Link>
        ))}
      </section>
    </div>
  );
}
