"use client";

import { useEffect, useState } from "react";
import { BookOpen, GitBranch, Grid3x3, MessageSquareText, Network, ScrollText, SpellCheck, Type } from "lucide-react";
import { ListenButton, SectionHead } from "@/components/ui";
import { SCRIPTS } from "@/lib/scripts";
import { EditLab, NgramLab, SpellLab, TextLab } from "@/components/lab/tabs1";
import { ParseLab, PosLab, SentimentLab, WordNetLab } from "@/components/lab/tabs2";

const TABS = [
  { k: "text", t: "Text processing", m: "Tokens, stems, lemmas", icon: Type, C: TextLab },
  { k: "edit", t: "Edit distance", m: "Dynamic programming", icon: Grid3x3, C: EditLab },
  { k: "spell", t: "Spelling", m: "Noisy channel", icon: SpellCheck, C: SpellLab },
  { k: "ngram", t: "N-gram LM", m: "Smoothing, perplexity", icon: ScrollText, C: NgramLab },
  { k: "pos", t: "POS & Viterbi", m: "HMM tagger", icon: GitBranch, C: PosLab },
  { k: "parse", t: "Parsing & CKY", m: "Chunking, CFG", icon: Network, C: ParseLab },
  { k: "wordnet", t: "WordNet & WSD", m: "Lexical semantics", icon: BookOpen, C: WordNetLab },
  { k: "sentiment", t: "Sentiment", m: "VADER vs lexicon", icon: MessageSquareText, C: SentimentLab },
] as const;

export default function Lab() {
  const [tab, setTab] = useState<string>("text");
  useEffect(() => {
    const h = window.location.hash.slice(1);
    if (TABS.some((t) => t.k === h)) setTab(h);
  }, []);
  const cur = TABS.find((t) => t.k === tab)!;
  const C = cur.C;
  return (
    <div>
      <SectionHead eyebrow="NLP Lab" title="Every technique, one at a time" right={<ListenButton id={`lab-${cur.k}`} text={SCRIPTS.lab[cur.k as keyof typeof SCRIPTS.lab]} title={cur.t} label={`Explain ${cur.t.toLowerCase()}`} />}>
        Live demos running the same code as the DriftLens pipeline. Type anything; results update as you go.
      </SectionHead>
      <div className="mb-6 flex gap-2 overflow-x-auto pb-1">
        {TABS.map(({ k, t, m, icon: Icon }) => (
          <button
            key={k}
            onClick={() => {
              setTab(k);
              history.replaceState(null, "", `#${k}`);
            }}
            className={`flex shrink-0 items-center gap-2 rounded-2xl border px-3 py-2 text-left transition ${tab === k ? "border-accent bg-accent text-white" : "border-line bg-surface hover:bg-surface-2"}`}
          >
            <Icon className="h-4 w-4" />
            <span>
              <span className="block text-sm font-semibold leading-tight">{t}</span>
              <span className={`block text-[0.68rem] ${tab === k ? "text-white/75" : "text-ink-3"}`}>{m}</span>
            </span>
          </button>
        ))}
      </div>
      <C key={cur.k} />
    </div>
  );
}
