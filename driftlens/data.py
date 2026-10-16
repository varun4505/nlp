"""Loaders for the CLPsych timelines and the pre-extracted eRisk users."""
import gzip
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from .paths import CLPSYCH_TEST, CLPSYCH_TRAIN, ERISK_USERS

# Post state alphabet used throughout the project.
# O = no change, S = switch, E = escalation, B = both switch and escalation.
STATES = ["O", "S", "E", "B"]


@dataclass
class Post:
    post_id: str
    index: int
    date: Optional[datetime]
    text: str
    state: Optional[str] = None           # None for unlabelled posts
    wellbeing: Optional[int] = None
    evidence: dict = field(default_factory=dict)


@dataclass
class Timeline:
    timeline_id: str
    posts: List[Post]

    @property
    def states(self):
        return [p.state for p in self.posts]


def _parse_date(s):
    if not s:
        return None
    for fmt in ("%d-%m-%Y, %H:%M:%S", "%Y-%m-%d %H:%M:%S UTC", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _state(sw, esc):
    s, e = sw not in (None, "0", 0), esc not in (None, "0", 0)
    return "B" if s and e else "S" if s else "E" if e else "O"


def load_timeline(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    posts = []
    for p in sorted(d["posts"], key=lambda x: x["post_index"]):
        labelled = "Switch" in p
        posts.append(Post(
            post_id=p["post_id"], index=p["post_index"], date=_parse_date(p.get("date")),
            text=p.get("post") or "",
            state=_state(p.get("Switch"), p.get("Escalation")) if labelled else None,
            wellbeing=p.get("Well-being"),
            evidence=p.get("evidence") or {},
        ))
    return Timeline(d["timeline_id"], posts)


def load_clpsych(split="train"):
    folder = CLPSYCH_TRAIN if split == "train" else CLPSYCH_TEST
    return [load_timeline(f) for f in sorted(folder.glob("*.json"))]


def evidence_spans(post, polarity):
    """Return highlighted evidence strings for 'adaptive' or 'maladaptive'."""
    block = post.evidence.get(f"{polarity}-state") or {}
    spans = []
    for key, val in block.items():
        if isinstance(val, dict) and val.get("highlighted_evidence"):
            spans.append(val["highlighted_evidence"])
    return spans


def load_erisk_users():
    with gzip.open(ERISK_USERS, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]
