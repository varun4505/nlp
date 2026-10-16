"use client";

import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "@/lib/api";
import { ErrorNote, Explainer, ListenButton, SectionHead } from "@/components/ui";
import { SCRIPTS } from "@/lib/scripts";

const tip = { background: "var(--surface)", border: "1px solid var(--line)", borderRadius: 12 };

export default function Results() {
  const [r, setR] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api("results").then(setR).catch((e) => setErr(e.message));
  }, []);

  const GROUP: Record<string, string> = {
    "-LEN": "length", "-LEX": "lexicon", "-SENT": "VADER", "-SELF": "self-statements",
    "-DRIFT": "LM novelty", "-TIME": "timing", "-PERS": "personal baselines", "-LEN and z_len": "all length",
  };

  return (
    <div className="space-y-8">
      <SectionHead eyebrow="Evaluation" title="What worked, and what didn't" right={<ListenButton id="results" text={SCRIPTS.results} title="Results" label="Explain the results" />}>
        {r ? `${r.n_timelines} labelled CLPsych timelines, ${r.n_posts} posts. 5-fold cross-validation × 5 repeats, split by timeline so no person appears in both train and test.` : "Loading…"}
      </SectionHead>
      <ErrorNote error={err} />
      {r && (
        <>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {[
              ["Best macro-F1 (CRF)", r.crf_full?.toFixed(3), "vs 0.266 majority baseline"],
              ["Lexicon vs VADER", `${r.lexicon.spearman_lex?.toFixed(2)} vs ${r.lexicon.spearman_vader?.toFixed(2)}`, "Spearman ρ with well-being"],
              ["eRisk AUC, no labels", r.lexicon.erisk_auc?.toFixed(2), `${r.lexicon.erisk_users} users, depressed vs control`],
              ["HMM POS accuracy", `${(r.pos.hmm_brown * 100).toFixed(1)}%`, `Brown test · ${(r.pos.hmm_unknown * 100).toFixed(0)}% on unknown words`],
            ].map(([k, v, d]) => (
              <div key={k} className="card p-5">
                <div className="text-sm text-ink-3">{k}</div>
                <div className="mt-1 text-3xl font-semibold tracking-tight">{v}</div>
                <div className="mt-1 text-xs text-ink-3">{d}</div>
              </div>
            ))}
          </div>

          <div className="card p-5">
            <div className="font-semibold">Timeline models · macro-F1 over O / S / E</div>
            <div className="mb-3 text-sm text-ink-3">Grey line: the length-only baseline. Error bars are tiny (±0.01) because the five repeats are stable; the big uncertainty is the small number of timelines.</div>
            <div className="h-80">
              <ResponsiveContainer>
                <BarChart data={r.models} layout="vertical" margin={{ left: 60, right: 20 }}>
                  <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" domain={[0, 0.65]} stroke="var(--ink-3)" />
                  <YAxis type="category" dataKey="model" stroke="var(--ink-3)" width={150} tick={{ fontSize: 12 }} />
                  <Tooltip contentStyle={tip} />
                  <ReferenceLine x={0.572} stroke="var(--ink-3)" strokeDasharray="4 4" />
                  <Bar dataKey="macro_f1" name="macro-F1" radius={[0, 6, 6, 0]}>
                    {r.models.map((m: any) => (
                      <Cell key={m.model} fill={m.model.startsWith("Majority") ? "var(--st-o)" : m.model.includes("length") ? "var(--ink-3)" : m.model === "CRF (prior-corrected)" || m.model === "Gaussian HMM" ? "var(--accent)" : "color-mix(in srgb, var(--accent) 50%, var(--surface-2))"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="grid gap-5 lg:grid-cols-2">
            <div className="card p-5">
              <div className="font-semibold">Ablation: remove one feature group (CRF)</div>
              <div className="mb-3 text-sm text-ink-3">Change in macro-F1 from the full model ({r.crf_full}) when a group is removed. Green, below zero: that group was helping.</div>
              <div className="h-72">
                <ResponsiveContainer>
                  <BarChart data={r.ablation.map((a: any) => ({ ...a, name: GROUP[a.group] ?? a.group, delta: +(a.macro_f1 - r.crf_full).toFixed(3) }))} margin={{ bottom: 40 }}>
                    <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="name" stroke="var(--ink-3)" angle={-30} textAnchor="end" interval={0} tick={{ fontSize: 11 }} />
                    <YAxis stroke="var(--ink-3)" />
                    <Tooltip contentStyle={tip} />
                    <ReferenceLine y={0} stroke="var(--ink-3)" />
                    <Bar dataKey="delta" name="Δ macro-F1 when removed" radius={4}>
                      {r.ablation.map((a: any) => (
                        <Cell key={a.group} fill={a.macro_f1 < r.crf_full ? "var(--ada)" : "var(--mal)"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="card p-5">
              <div className="font-semibold">Language models · dev perplexity (lower is better)</div>
              <div className="mb-3 text-sm text-ink-3">Laplace over-smooths badly as n grows; interpolation wins.</div>
              <div className="h-72">
                <ResponsiveContainer>
                  <BarChart data={r.lm.map((x: any) => ({ ...x, name: `${x.model} ${x.smoothing.split(" ")[0]}` }))} margin={{ bottom: 40 }}>
                    <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="name" stroke="var(--ink-3)" angle={-30} textAnchor="end" interval={0} tick={{ fontSize: 11 }} />
                    <YAxis scale="log" domain={[100, "auto"]} stroke="var(--ink-3)" allowDataOverflow />
                    <Tooltip contentStyle={tip} />
                    <Bar dataKey="dev_pp" name="perplexity" radius={4}>
                      {r.lm.map((x: any, i: number) => (
                        <Cell key={i} fill={x.smoothing.startsWith("Jelinek") ? "var(--accent)" : "color-mix(in srgb, var(--accent) 35%, var(--surface-2))"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="card p-5">
            <div className="font-semibold">Spelling correction accuracy (1,000 synthetic typos)</div>
            <div className="mt-3 space-y-2">
              {r.spelling.map((s: any) => (
                <div key={s.config} className="flex items-center gap-3 text-sm">
                  <span className="w-72 shrink-0 text-ink-2">{s.config}</span>
                  <div className="h-3 flex-1 rounded-full bg-surface-2">
                    <div className="h-3 rounded-full bg-accent" style={{ width: `${((s.accuracy - 0.8) / 0.2) * 100}%` }} />
                  </div>
                  <span className="w-14 text-right font-mono">{(s.accuracy * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
            <div className="mt-2 text-xs text-ink-3">Axis starts at 80%. Keyboard weights + bigram context give the biggest jump.</div>
          </div>

          <div className="grid gap-3 md:grid-cols-3">
            <Explainer>
              <b className="text-ink">Honest takeaway 1.</b> With only 30 timelines, no model beats the post-length baseline on macro-F1 by a statistically reliable margin.
            </Explainer>
            <Explainer>
              <b className="text-ink">Takeaway 2.</b> Sequence models (HMM, CRF) help most on <b>escalation</b>, which comes in runs of posts. That&apos;s what reading the timeline “like a sentence” buys you.
            </Explainer>
            <Explainer>
              <b className="text-ink">Takeaway 3.</b> The WordNet lexicon is the strongest standalone component: it beats VADER on CLPsych and reaches AUC 0.84 on eRisk with zero training labels.
            </Explainer>
          </div>
        </>
      )}
    </div>
  );
}
