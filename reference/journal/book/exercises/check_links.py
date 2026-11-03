"""Check that every section, chapter and gate reference in the interview rehearsal files resolves.

    python3 journal/book/exercises/check_links.py [files...]

Default files: exercises/interview/*.md. A section reference (16.1c, 8.7, 12d.6) must match a `## <ref>` heading in
journal/book/; "chapter N" must match a book file whose name starts with N-; a gate id (G0, G11-A, W1-A, C1, REAP v2, …)
must appear in an ID cell of journal/EXPERIMENTS.md. Exits 1 and lists the misses otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

BOOK = Path(__file__).resolve().parents[1]
REPO = BOOK.parents[1]
GATE = re.compile(r"\b(G\d+[a-z]?(?:-[A-Za-z0-9′]+)*|W1-[A-Z]\d?|C1|REAP v[12]|G-flora(?:-abl)?|Judge τ)\b")
SECTION = re.compile(r"(?<![\w.$°%])(\d{1,2}[a-z]\.\d{1,2}[a-z]?|\d{1,2}\.\d{1,2}[a-z]|(?<=§)\d{1,2}\.\d{1,2})(?![\w.%°])")
SOURCE = re.compile(r"\*\(([^)]*)\)\*")  # source tags *(16.1c, chapter 6.2)*: plain N.M counts there
PLAIN = re.compile(r"(?<![\w.])([1-9]\d?\.\d{1,2}[a-z]?)(?![\w.])")
ALIAS = {"REAP v1": "REAP monitors"}
CHAPTER = re.compile(r"\bchapters? (\d{1,2}[a-z]?)\b")


def headings():
    out = set()
    for f in BOOK.glob("*.md"):
        for m in re.finditer(r"^#{2,3} (\d{1,2}[a-z]?\.\d{1,2}[a-z]?)\b", f.read_text(), re.M):
            out.add(m.group(1))
    return out


def gate_ids():
    ids = set()
    for line in (REPO / "journal/EXPERIMENTS.md").read_text().split("\n"):
        cells = re.split(r"(?<!\\)\|", line)
        if len(cells) > 3 and re.match(r"\s*\**(\d{4}-)?\d{2}-\d{2}", cells[1]):
            ids.add(re.sub(r"[*`]", "", cells[2]).strip())
    return ids


def main(files):
    hs, ids = headings(), gate_ids()
    chapters = {re.match(r"0*(\d{1,2}[a-z]?)-", f.name).group(1) for f in BOOK.glob("*.md") if re.match(r"\d{1,2}[a-z]?-", f.name)}
    misses = []
    for f in files:
        t = re.sub(r"\d+(\.\d+)? ?(ms|s|°|%|mm|cm|GB|B)\b", "", f.read_text())  # numbers are not references
        refs = [m.group(1) for m in SECTION.finditer(t)]
        for src in SOURCE.finditer(t):
            refs += [m.group(1) for m in PLAIN.finditer(src.group(1))]
        for r in refs:
            if r not in hs:
                misses.append((f.name, "section", r))
        for m in CHAPTER.finditer(t):
            if m.group(1).split(".")[0] not in chapters:
                misses.append((f.name, "chapter", m.group(1)))
        for m in GATE.finditer(t):
            g = ALIAS.get(m.group(1), m.group(1))
            if not any(g == i or i.startswith((g + " ", g + "(", g + "-")) or g in i.split() for i in ids):
                misses.append((f.name, "gate", g))
    for x in sorted(set(misses)):
        print("unresolved:", *x)
    print(f"checked {len(files)} files: {len(set(misses))} unresolved")
    return 1 if misses else 0


if __name__ == "__main__":
    args = [Path(a) for a in sys.argv[1:]] or sorted((BOOK / "exercises/interview").glob("*.md"))
    sys.exit(main(args))
