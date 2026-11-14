"""W12 D1: blocked counterbalanced schedule, blinding, record validation, drift."""
import random

from evalkit import OUTCOMES, TAXONOMY


def make_schedule(policies: list[str], n_starts: int, repeats: int, seed: int) -> list[dict]:
    """Run sheet for a session. rng = random.Random(seed).
    1. base = rng.sample(policies, len(policies))  (one random base order)
    2. for each repeat r: starts = rng.sample(range(n_starts), n_starts); each start s is one BLOCK
       (blocks are numbered 0, 1, 2, ... across repeats).
    3. inside block b the order is base rotated left by b % len(policies).
    Return one row per rollout, in execution order:
    {"trial": global index, "block": b, "repeat": r, "start_id": s, "policy": name}."""
    rng = random.Random(seed)
    base = rng.sample(policies, len(policies))
    rows, block = [], 0
    for r in range(repeats):
        for s in rng.sample(range(n_starts), n_starts):
            k = block % len(base)
            for p in base[k:] + base[:k]:
                rows.append({"trial": len(rows), "block": block, "repeat": r, "start_id": s, "policy": p})
            block += 1
    return rows


def blind_aliases(policies: list[str], seed: int) -> dict[str, str]:
    """Map each policy to an alias "A", "B", ... (as many letters as policies) in a random order:
    letters = the first len(policies) uppercase letters, shuffled with random.Random(seed).shuffle,
    then zipped with `policies` in the given order."""
    letters = [chr(ord("A") + i) for i in range(len(policies))]
    random.Random(seed).shuffle(letters)
    return dict(zip(policies, letters))


def validate_record(rec: dict) -> list[str]:
    """Return a list of error messages (empty = valid). One message per broken rule:
    - required keys: trial, alias, start_id, outcome, failure_tag, duration_s, interventions, operator
      (report each missing key; if any is missing, return just those messages);
    - trial and start_id: int >= 0; interventions: int >= 0 (bool is not an int here);
    - outcome in OUTCOMES;
    - failure_tag: in TAXONOMY if outcome == "failure", else None;
    - duration_s: a number > 0;
    - operator and alias: non-empty strings."""
    required = ("trial", "alias", "start_id", "outcome", "failure_tag", "duration_s", "interventions", "operator")
    missing = [f"missing {k}" for k in required if k not in rec]
    if missing:
        return missing
    errors = []

    def is_int(v):
        return type(v) is int and v >= 0

    for k in ("trial", "start_id", "interventions"):
        if not is_int(rec[k]):
            errors.append(f"{k} must be an int >= 0")
    if rec["outcome"] not in OUTCOMES:
        errors.append("unknown outcome")
    elif rec["outcome"] == "failure":
        if rec["failure_tag"] not in TAXONOMY:
            errors.append("failure needs a failure_tag from TAXONOMY")
    elif rec["failure_tag"] is not None:
        errors.append("failure_tag only for failures")
    d = rec["duration_s"]
    if isinstance(d, bool) or not isinstance(d, (int, float)) or d <= 0:
        errors.append("duration_s must be > 0")
    for k in ("operator", "alias"):
        if not isinstance(rec[k], str) or not rec[k]:
            errors.append(f"{k} must be a non-empty string")
    return errors


def expected_success(schedule: list[dict], p_at_trial) -> dict[str, float]:
    """Expected success rate of each policy if the success probability depends ONLY on the trial index
    (p_at_trial(i)): the mean of p_at_trial over that policy's trials. This isolates the drift."""
    acc: dict[str, list[float]] = {}
    for row in schedule:
        acc.setdefault(row["policy"], []).append(p_at_trial(row["trial"]))
    return {p: sum(v) / len(v) for p, v in acc.items()}
