export async function api<T = any>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`/api/${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {}
    throw new Error(msg);
  }
  return res.json();
}

export const STATE_META: Record<string, { name: string; short: string; color: string; blurb: string }> = {
  O: { name: "No change", short: "O", color: "var(--st-o)", blurb: "The writer sounds like their usual self." },
  S: { name: "Switch", short: "S", color: "var(--st-s)", blurb: "A sudden shift in mood compared with recent posts." },
  E: { name: "Escalation", short: "E", color: "var(--st-e)", blurb: "A gradual build-up across several posts." },
  B: { name: "Switch + escalation", short: "B", color: "var(--st-b)", blurb: "Both a sudden shift and part of a build-up." },
};
