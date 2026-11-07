"""Spaced retrieval for this repo's questions (Leitner boxes, interleaved across labs).

Why: retrieval spread over days beats re-reading and beats open-ended AI study for long-term retention
(see LEARNING.md). This turns every questions.md you finish into flashcards that come back at 1, 3, 7, 21, 60 days.

Usage (from the repo root):
    python scripts/review.py add weeks/w1-.../d1-...     # after finishing a lab: its questions become cards due tomorrow
    python scripts/review.py                             # today's session: up to 6 due cards, mixed across labs
    python scripts/review.py --max 10                    # longer session
    python scripts/review.py due                         # list what is due, no session
    python scripts/review.py stats                       # cards per box, retention rate

In a session: read the question, answer OUT LOUD (or write it), press Enter, compare with the model answer
(read from the `solutions` branch, only after your attempt), then grade yourself:
    2 = clean and complete   -> card moves up one box (longer interval)
    1 = got there, slow/partial -> stays in its box, due again after the box interval
    0 = missed               -> back to box 0, due tomorrow
State is kept in review/cards.json (commit it: it is your learning record).
"""
import argparse
import datetime as dt
import json
import random
import re
import subprocess
import sys
from pathlib import Path

INTERVALS = [1, 3, 7, 21, 60]  # days per box
ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "review" / "cards.json"


def today(arg=None) -> dt.date:
    return dt.date.fromisoformat(arg) if arg else dt.date.today()


def load() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"cards": {}, "log": []}


def save(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")


def numbered_items(text: str) -> dict:
    """{n: text} for top-level items '1. ...' (continuation lines included until the next item or a heading)."""
    items, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"^(\d+)\.\s+(.*)$", line)
        if m:
            cur = int(m.group(1))
            items[cur] = m.group(2).strip()
        elif cur is not None and line.strip() and not line.startswith("#") and not line.startswith("<"):
            items[cur] += " " + line.strip()
        elif line.startswith("#") or line.startswith("<"):
            cur = None
    return items


def questions_of(lab: Path) -> dict:
    text = (lab / "questions.md").read_text()
    text = re.split(r"^## (Model answers|What)", text, flags=re.M)[0]
    return numbered_items(text)


def git_show(path: str):
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "show", f"solutions:{path}"], capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return out.stdout if out.returncode == 0 else None


def answers_of(lab_rel: str) -> dict:
    """Model answers from the solutions branch: answers.md, or the 'Model answers' section of questions.md."""
    txt = git_show(f"{lab_rel}/answers.md")
    if txt is None:
        txt = git_show(f"{lab_rel}/questions.md")
        if txt is None:
            return {}
        parts = re.split(r"^## (?:Model answers|What .*)$", txt, flags=re.M)
        if len(parts) < 2:
            return {}
        txt = parts[1]
    return numbered_items(txt)


def cmd_add(state: dict, labs: list, now: dt.date) -> None:
    added = 0
    for lab in labs:
        p = (ROOT / lab).resolve()
        if not (p / "questions.md").exists():
            print(f"no questions.md in {lab}")
            continue
        rel = str(p.relative_to(ROOT))
        for n in questions_of(p):
            cid = f"{rel}#{n}"
            if cid not in state["cards"]:
                state["cards"][cid] = {"box": 0, "due": str(now + dt.timedelta(days=1)), "seen": 0}
                added += 1
    print(f"added {added} cards (due tomorrow)")


def due_cards(state: dict, now: dt.date) -> list:
    return sorted(cid for cid, c in state["cards"].items() if dt.date.fromisoformat(c["due"]) <= now)


def grade(card: dict, g: int, now: dt.date) -> None:
    if g == 2:
        card["box"] = min(card["box"] + 1, len(INTERVALS) - 1)
    elif g == 0:
        card["box"] = 0
    card["due"] = str(now + dt.timedelta(days=INTERVALS[card["box"]] if g else 1))
    card["seen"] += 1


def cmd_run(state: dict, now: dt.date, max_cards: int, seed=None) -> None:
    due = due_cards(state, now)
    if not due:
        print("nothing due today. Add cards with: python scripts/review.py add <lab dir>")
        return
    rng = random.Random(seed)
    rng.shuffle(due)  # interleave across labs: mixing is good for REVIEW (not for first exposure)
    session = due[:max_cards]
    print(f"{len(due)} due, reviewing {len(session)}. Answer out loud before pressing Enter.\n")
    cache = {}
    for i, cid in enumerate(session, 1):
        lab, n = cid.rsplit("#", 1)
        n = int(n)
        qs = questions_of(ROOT / lab)
        if n not in qs:
            print(f"[{cid}] question no longer exists, removing")
            del state["cards"][cid]
            continue
        print(f"({i}/{len(session)}) {lab}\n  Q{n}. {qs[n]}")
        input("  ... your answer out loud, then Enter ")
        if lab not in cache:
            cache[lab] = answers_of(lab)
        ans = cache[lab].get(n)
        print("  MODEL: " + (ans if ans else "(no model answer found: fetch the solutions branch, or check the lab README)"))
        while True:
            g = input("  grade 2 clean / 1 partial / 0 missed: ").strip()
            if g in ("0", "1", "2"):
                break
        grade(state["cards"][cid], int(g), now)
        state["log"].append({"date": str(now), "card": cid, "grade": int(g)})
        print()
    save(state)
    print("saved. Misses come back tomorrow.")


def cmd_stats(state: dict, now: dt.date) -> None:
    boxes = [0] * len(INTERVALS)
    for c in state["cards"].values():
        boxes[c["box"]] += 1
    print("cards per box (interval days):", {f"{INTERVALS[i]}d": boxes[i] for i in range(len(INTERVALS))})
    recent = [e for e in state["log"] if dt.date.fromisoformat(e["date"]) >= now - dt.timedelta(days=30)]
    if recent:
        clean = sum(e["grade"] == 2 for e in recent) / len(recent)
        print(f"last 30 days: {len(recent)} reviews, {clean:.0%} clean")
    print(f"due today: {len(due_cards(state, now))}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="run", choices=["run", "add", "due", "stats"])
    ap.add_argument("labs", nargs="*")
    ap.add_argument("--max", type=int, default=6)
    ap.add_argument("--today", help="YYYY-MM-DD (testing)")
    a = ap.parse_args(argv)
    now = today(a.today)
    state = load()
    if a.command == "add":
        cmd_add(state, a.labs, now)
        save(state)
    elif a.command == "due":
        for cid in due_cards(state, now):
            print(cid)
    elif a.command == "stats":
        cmd_stats(state, now)
    else:
        cmd_run(state, now, a.max)


if __name__ == "__main__":
    main(sys.argv[1:])
