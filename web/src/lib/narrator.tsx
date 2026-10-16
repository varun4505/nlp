"use client";

// Global narrator: one voice for the whole site. Speech comes from the backend's
// /api/tts (Microsoft Edge neural voices via edge-tts), which also returns word
// timings so we can show live captions. If the backend or network is down we fall
// back to the browser's built-in speechSynthesis.

import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";

export type Voice = { id: string; name: string; accent: string; gender: string };
type Word = { t: number; d: number; text: string };

type Ctx = {
  speak: (text: string, id?: string, title?: string) => Promise<void>;
  stop: () => void;
  togglePause: () => void;
  speakingId: string | null;
  loadingId: string | null;
  paused: boolean;
  title: string;
  words: Word[];
  wordIndex: number;
  fallback: boolean;
  voices: Voice[];
  voice: string;
  setVoice: (v: string) => void;
  rate: number;
  setRate: (r: number) => void;
};

const NarratorCtx = createContext<Ctx | null>(null);

const DEFAULT_VOICES: Voice[] = [
  { id: "en-US-AvaMultilingualNeural", name: "Ava", accent: "US", gender: "F" },
  { id: "en-US-AndrewMultilingualNeural", name: "Andrew", accent: "US", gender: "M" },
  { id: "en-IN-NeerjaExpressiveNeural", name: "Neerja", accent: "India", gender: "F" },
  { id: "en-IN-PrabhatNeural", name: "Prabhat", accent: "India", gender: "M" },
];

/** Split narration into chunks of whole sentences, roughly 280 characters each. */
function splitChunks(text: string, max = 280) {
  const sents = text.replace(/\s+/g, " ").trim().match(/[^.!?]+[.!?]+["')\]]*\s*|[^.!?]+$/g) ?? [text];
  const out: string[] = [];
  let cur = "";
  for (const s of sents) {
    if (cur && (cur + s).length > max) {
      out.push(cur.trim());
      cur = "";
    }
    cur += s;
  }
  if (cur.trim()) out.push(cur.trim());
  return out;
}

function readPref(key: string) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}
function writePref(key: string, v: string) {
  try {
    localStorage.setItem(key, v);
  } catch {}
}

export function NarratorProvider({ children }: { children: React.ReactNode }) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const rafRef = useRef<number>(0);
  const tokenRef = useRef(0);
  const [speakingId, setSpeakingId] = useState<string | null>(null);
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [paused, setPaused] = useState(false);
  const [title, setTitle] = useState("");
  const [words, setWords] = useState<Word[]>([]);
  const [wordIndex, setWordIndex] = useState(-1);
  const [fallback, setFallback] = useState(false);
  const [voices, setVoices] = useState<Voice[]>(DEFAULT_VOICES);
  const [voice, setVoiceState] = useState(DEFAULT_VOICES[0].id);
  const [rate, setRateState] = useState(0);

  useEffect(() => {
    const v = readPref("dl-voice");
    if (v) setVoiceState(v);
    const r = readPref("dl-rate");
    if (r) setRateState(Number(r) || 0);
    fetch("/api/voices")
      .then((r) => (r.ok ? r.json() : null))
      .then((v) => v && setVoices(v))
      .catch(() => {});
  }, []);

  const setVoice = (v: string) => {
    setVoiceState(v);
    writePref("dl-voice", v);
  };
  const setRate = (r: number) => {
    setRateState(r);
    writePref("dl-rate", String(r));
  };

  const stop = useCallback(() => {
    tokenRef.current++;
    cancelAnimationFrame(rafRef.current);
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    if (typeof window !== "undefined") window.speechSynthesis?.cancel();
    setSpeakingId(null);
    setLoadingId(null);
    setPaused(false);
    setWords([]);
    setWordIndex(-1);
  }, []);

  const speakBrowser = useCallback(
    (text: string, id: string, token: number) =>
      new Promise<void>((resolve) => {
        const synth = window.speechSynthesis;
        if (!synth) return resolve();
        const u = new SpeechSynthesisUtterance(text);
        const all = synth.getVoices();
        u.voice =
          all.find((v) => /natural|online/i.test(v.name) && v.lang.startsWith("en")) ??
          all.find((v) => v.lang.startsWith("en")) ??
          null;
        u.rate = 1 + rate / 100;
        const ws = text.split(/\s+/).map((t, i) => ({ t: i, d: 0, text: t }));
        setWords(ws);
        let wi = 0;
        u.onboundary = (e) => {
          if (e.name === "word" && tokenRef.current === token) setWordIndex(wi++);
        };
        u.onend = () => {
          if (tokenRef.current === token) stop();
          resolve();
        };
        u.onerror = () => resolve();
        setLoadingId(null);
        setSpeakingId(id);
        synth.speak(u);
      }),
    [rate, stop]
  );

  const speak = useCallback(
    async (text: string, id = "default", t = "") => {
      stop();
      const token = ++tokenRef.current;
      setTitle(t);
      setLoadingId(id);
      const chunks = splitChunks(text);
      const fetchChunk = (c: string) => {
        const p = fetch("/api/tts", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: c, voice, rate: `${rate >= 0 ? "+" : ""}${rate}%` }),
        }).then((r) => {
          if (!r.ok) throw new Error("tts failed");
          return r.json();
        });
        p.catch(() => {}); // a prefetch may fail while the previous chunk is still playing
        return p;
      };
      // Synthesise sentence chunks one ahead of playback, so speech starts fast
      // and a long narration never waits on one huge request.
      let next = fetchChunk(chunks[0]);
      for (let i = 0; i < chunks.length; i++) {
        let data: { audio: string; mime: string; words: Word[] };
        try {
          data = await next;
        } catch {
          if (tokenRef.current !== token) return;
          setFallback(true);
          await speakBrowser(chunks.slice(i).join(" "), id, token);
          return;
        }
        if (tokenRef.current !== token) return;
        if (i + 1 < chunks.length) next = fetchChunk(chunks[i + 1]);
        setFallback(false);
        const audio = new Audio(`data:${data.mime};base64,${data.audio}`);
        audioRef.current = audio;
        setWords(data.words);
        setWordIndex(-1);
        setLoadingId(null);
        setSpeakingId(id);
        const tick = () => {
          if (tokenRef.current !== token || audioRef.current !== audio) return;
          const ms = audio.currentTime * 1000;
          let k = -1;
          for (let j = 0; j < data.words.length; j++) if (data.words[j].t <= ms + 60) k = j;
          setWordIndex(k);
          rafRef.current = requestAnimationFrame(tick);
        };
        const ended = new Promise<void>((resolve) => {
          audio.addEventListener("ended", () => resolve());
          audio.addEventListener("pause", () => {
            if (tokenRef.current !== token) resolve();
          });
        });
        try {
          await audio.play();
        } catch {
          if (tokenRef.current !== token) return;
          setFallback(true);
          await speakBrowser(chunks.slice(i).join(" "), id, token);
          return;
        }
        cancelAnimationFrame(rafRef.current);
        rafRef.current = requestAnimationFrame(tick);
        await ended;
        if (tokenRef.current !== token) return;
      }
      stop();
    },
    [voice, rate, stop, speakBrowser]
  );

  const togglePause = useCallback(() => {
    const a = audioRef.current;
    if (a) {
      if (a.paused) {
        a.play();
        setPaused(false);
      } else {
        a.pause();
        setPaused(true);
      }
      return;
    }
    const s = window.speechSynthesis;
    if (s?.speaking) {
      if (s.paused) {
        s.resume();
        setPaused(false);
      } else {
        s.pause();
        setPaused(true);
      }
    }
  }, []);

  return (
    <NarratorCtx.Provider
      value={{
        speak, stop, togglePause, speakingId, loadingId, paused, title, words, wordIndex,
        fallback, voices, voice, setVoice, rate, setRate,
      }}
    >
      {children}
    </NarratorCtx.Provider>
  );
}

export function useNarrator() {
  const c = useContext(NarratorCtx);
  if (!c) throw new Error("useNarrator outside provider");
  return c;
}
