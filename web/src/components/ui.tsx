"use client";

import { Loader2, Pause, Play, Square, Volume2 } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useNarrator } from "@/lib/narrator";
import { STATE_META } from "@/lib/api";

export function Nav() {
  const path = usePathname();
  const links = [
    ["/", "Overview"],
    ["/analyze", "Analyse"],
    ["/lab", "NLP Lab"],
    ["/results", "Results"],
  ];
  return (
    <header className="sticky top-0 z-40 border-b border-line bg-bg/85 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-3 px-4 py-3">
        <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight">
          <span className="grid h-7 w-7 place-items-center rounded-lg bg-accent text-white">
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2.4">
              <path d="M3 17c3 0 4-8 7-8s3 6 6 6 3-9 5-9" strokeLinecap="round" />
            </svg>
          </span>
          DriftLens
        </Link>
        <nav className="flex items-center gap-1 overflow-x-auto text-sm">
          {links.map(([href, label]) => {
            const active = href === "/" ? path === "/" : path.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={`whitespace-nowrap rounded-full px-3 py-1.5 transition ${
                  active ? "bg-ink text-bg" : "text-ink-2 hover:bg-surface-2"
                }`}
              >
                {label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}

export function ListenButton({
  id,
  text,
  title,
  label = "Listen",
  className = "",
  primary = false,
}: {
  id: string;
  text: string | (() => string);
  title?: string;
  label?: string;
  className?: string;
  primary?: boolean;
}) {
  const n = useNarrator();
  const active = n.speakingId === id;
  const loading = n.loadingId === id;
  return (
    <button
      type="button"
      onClick={() => (active || loading ? n.stop() : n.speak(typeof text === "function" ? text() : text, id, title ?? label))}
      className={`btn ${primary ? "btn-primary" : "btn-ghost"} ${active && !primary ? "!border-accent !text-accent" : ""} ${className}`}
      aria-label={active ? "Stop narration" : `Listen: ${title ?? label}`}
    >
      {loading ? (
        <Loader2 className="h-4 w-4 animate-spin" />
      ) : active ? (
        <span className="wave" aria-hidden>
          <span />
          <span />
          <span />
          <span />
        </span>
      ) : (
        <Volume2 className="h-4 w-4" />
      )}
      {active ? "Stop" : loading ? "Preparing voice…" : label}
    </button>
  );
}

export function NarratorBar() {
  const n = useNarrator();
  const on = n.speakingId || n.loadingId;
  if (!on) return null;
  const start = Math.max(0, n.wordIndex - 9);
  const shown = n.words.slice(start, start + 18);
  return (
    <div className="fixed inset-x-0 bottom-4 z-50 flex justify-center px-4">
      <div className="card flex w-full max-w-3xl items-center gap-3 px-4 py-3 shadow-xl">
        <button className="btn btn-primary !px-3" onClick={n.togglePause} aria-label={n.paused ? "Resume" : "Pause"}>
          {n.loadingId ? <Loader2 className="h-4 w-4 animate-spin" /> : n.paused ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}
        </button>
        <div className="min-w-0 flex-1">
          <div className="text-xs text-ink-3">
            {n.title || "Narration"} · {n.fallback ? "browser voice (offline fallback)" : n.voices.find((v) => v.id === n.voice)?.name + " · neural voice"}
          </div>
          <div className="truncate text-[0.95rem]">
            {n.loadingId ? (
              <span className="text-ink-3">Synthesising speech…</span>
            ) : (
              shown.map((w, i) => (
                <span
                  key={start + i}
                  className={start + i === n.wordIndex ? "rounded bg-accent-soft px-0.5 text-accent" : start + i < n.wordIndex ? "text-ink" : "text-ink-3"}
                >
                  {w.text}{" "}
                </span>
              ))
            )}
          </div>
        </div>
        <button className="btn btn-ghost !px-3" onClick={n.stop} aria-label="Stop">
          <Square className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}

export function VoicePicker() {
  const n = useNarrator();
  return (
    <div className="flex flex-wrap items-center gap-2 text-sm">
      <select className="input !w-auto !py-1.5" value={n.voice} onChange={(e) => n.setVoice(e.target.value)} aria-label="Voice">
        {n.voices.map((v) => (
          <option key={v.id} value={v.id}>
            {v.name} ({v.accent}, {v.gender === "F" ? "female" : "male"})
          </option>
        ))}
      </select>
      <select className="input !w-auto !py-1.5" value={n.rate} onChange={(e) => n.setRate(Number(e.target.value))} aria-label="Speed">
        <option value={-15}>Slower</option>
        <option value={0}>Normal speed</option>
        <option value={12}>Faster</option>
      </select>
    </div>
  );
}

export function StateBadge({ s, p }: { s: string; p?: number }) {
  const m = STATE_META[s];
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-semibold text-white" style={{ background: m.color }}>
      {m.short} · {m.name}
      {p !== undefined && <span className="opacity-80">{Math.round(p * 100)}%</span>}
    </span>
  );
}

export function SectionHead({ eyebrow, title, children, right }: { eyebrow?: string; title: string; children?: React.ReactNode; right?: React.ReactNode }) {
  return (
    <div className="mb-5 flex flex-wrap items-end justify-between gap-4">
      <div className="max-w-2xl">
        {eyebrow && <div className="eyebrow mb-1">{eyebrow}</div>}
        <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">{title}</h2>
        {children && <p className="mt-2 text-ink-2">{children}</p>}
      </div>
      {right}
    </div>
  );
}

export function Explainer({ children }: { children: React.ReactNode }) {
  return <div className="rounded-xl border border-line bg-surface-2 px-4 py-3 text-sm leading-relaxed text-ink-2">{children}</div>;
}

export function ErrorNote({ error }: { error: string | null }) {
  if (!error) return null;
  return (
    <div className="rounded-xl border border-mal/40 bg-mal-soft px-4 py-3 text-sm text-mal">
      {error.includes("ECONNREFUSED") || error.startsWith("500") || error.startsWith("502")
        ? "Can't reach the Python backend. Start it with: python -m uvicorn api.server:app --port 8000"
        : error}
    </div>
  );
}
