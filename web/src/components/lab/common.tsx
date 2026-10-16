"use client";

import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";

/** POST `body` to /api/`path` whenever it changes (debounced). */
export function useLive<T = any>(path: string, body: unknown, delay = 350) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const key = JSON.stringify(body);
  const seq = useRef(0);
  useEffect(() => {
    const id = ++seq.current;
    setLoading(true);
    const t = setTimeout(() => {
      api<T>(path, body)
        .then((d) => {
          if (id === seq.current) {
            setData(d);
            setError(null);
          }
        })
        .catch((e) => id === seq.current && setError(e.message))
        .finally(() => id === seq.current && setLoading(false));
    }, delay);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, key, delay]);
  return { data, loading, error };
}

export function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block text-sm">
      <span className="mb-1 block text-xs font-medium text-ink-3">{label}</span>
      {children}
    </label>
  );
}

export function Panel({ title, sub, children, className = "" }: { title?: string; sub?: React.ReactNode; children: React.ReactNode; className?: string }) {
  return (
    <div className={`card p-5 ${className}`}>
      {title && <div className="font-semibold">{title}</div>}
      {sub && <div className="mb-3 text-sm text-ink-3">{sub}</div>}
      {children}
    </div>
  );
}

export type TreeNode = { label: string; tag?: string | null; children: TreeNode[] };

/** Simple top-down tree layout rendered as SVG. */
export function TreeView({ tree }: { tree: TreeNode }) {
  type P = { node: TreeNode; x: number; y: number; kids: P[] };
  let leaf = 0;
  const depthOf = (n: TreeNode): number => (n.children.length ? 1 + Math.max(...n.children.map(depthOf)) : 0);
  const D = depthOf(tree);
  const place = (n: TreeNode, d: number): P => {
    if (!n.children.length) return { node: n, x: leaf++, y: D, kids: [] };
    const kids = n.children.map((c) => place(c, d + 1));
    return { node: n, x: (kids[0].x + kids[kids.length - 1].x) / 2, y: d, kids };
  };
  const root = place(tree, 0);
  const W = Math.max(leaf, 1) * 78;
  const H = (D + 1) * 56;
  const sx = (x: number) => 39 + x * 78;
  const sy = (y: number) => 22 + y * 56;
  const els: React.ReactNode[] = [];
  const walk = (p: P, k: string) => {
    p.kids.forEach((c, i) => {
      els.push(<line key={`${k}-l${i}`} x1={sx(p.x)} y1={sy(p.y) + 10} x2={sx(c.x)} y2={sy(c.y) - 12} stroke="var(--line)" strokeWidth={1.5} />);
      walk(c, `${k}-${i}`);
    });
    const isLeaf = !p.kids.length;
    els.push(
      <g key={k}>
        <rect x={sx(p.x) - 34} y={sy(p.y) - 12} width={68} height={24} rx={8} fill={isLeaf ? "var(--surface-2)" : p.node.label === "SELF" ? "var(--accent)" : "var(--accent-soft)"} />
        <text x={sx(p.x)} y={sy(p.y) + 4} textAnchor="middle" fontSize={12} fontWeight={isLeaf ? 400 : 600} fill={p.node.label === "SELF" ? "white" : isLeaf ? "var(--ink)" : "var(--accent)"}>
          {p.node.label.length > 10 ? p.node.label.slice(0, 9) + "…" : p.node.label}
        </text>
      </g>
    );
  };
  walk(root, "r");
  return (
    <div className="overflow-x-auto">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="mx-auto block">
        {els}
      </svg>
    </div>
  );
}
