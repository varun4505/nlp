"""Extract the target user's own writings from the eRisk 2026 Task 2 files.

Each subject file holds whole Reddit threads (other people's comments included).
Only items flagged ``target: true`` belong to the subject, so everything else is
dropped. Output: one gzipped JSON line per subject with its label and texts.
"""
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from driftlens.paths import ERISK_T2_JSON, ERISK_T2_LABELS, ERISK_USERS  # noqa: E402

MAX_WRITINGS = 400


def target_writings(items):
    out = []
    for item in items:
        sub = item.get("submission") or {}
        if sub.get("target"):
            text = " ".join(t for t in (sub.get("title"), sub.get("body")) if t)
            out.append((sub.get("created_utc", ""), text))
        for cm in item.get("comments") or []:
            if cm.get("target") and cm.get("body"):
                out.append((cm.get("created_utc", ""), cm["body"]))
    out = [(d, t.strip()) for d, t in out if t and t.strip() not in ("[removed]", "[deleted]")]
    out.sort(key=lambda x: x[0])
    return out


def main():
    labels = {}
    for line in ERISK_T2_LABELS.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            labels[parts[0]] = int(parts[1])

    files = sorted(ERISK_T2_JSON.glob("subject_*.json"))
    t0 = time.time()
    n_written = 0
    with gzip.open(ERISK_USERS, "wt", encoding="utf-8") as fh:
        for i, f in enumerate(files):
            sid = f.stem
            if sid not in labels:
                continue
            with open(f, encoding="utf-8") as src:
                items = json.load(src)
            writings = target_writings(items)[-MAX_WRITINGS:]
            fh.write(json.dumps({"subject": sid, "label": labels[sid],
                                 "dates": [d for d, _ in writings],
                                 "texts": [t for _, t in writings]}) + "\n")
            n_written += 1
            if i % 100 == 0:
                print(f"{i}/{len(files)} files, {time.time() - t0:.0f}s", flush=True)
    print(f"wrote {n_written} subjects to {ERISK_USERS} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
