"""W12 D3: rater agreement, per-cause failure rates, stage funnel.

A record here: {"policy", "outcome" in OUTCOMES, "failure_tag" (None unless failure), "stage_reached": number of
STAGES completed, 0..len(STAGES)}."""
from collections import Counter

from evalkit import STAGES, TAXONOMY, wilson


def cohen_kappa(r1: list[str], r2: list[str]) -> float:
    """kappa = (po - pe) / (1 - pe). po: observed agreement. pe: sum over labels of P(r1 = label) * P(r2 = label).
    If pe == 1 (both raters always use the same single label), return 1.0."""
    # TODO
    raise NotImplementedError


def breakdown(records: list[dict], policy: str) -> dict[str, dict]:
    """For every tag in TAXONOMY: {"count", "rate", "lo", "hi"} where rate = count / n and (lo, hi) = wilson(count, n),
    n = episodes of this policy whose outcome is "success" or "failure" (aborted episodes are excluded).
    Tags that never occur appear with count 0."""
    # TODO
    raise NotImplementedError


def stage_funnel(records: list[dict], policy: str) -> list[dict]:
    """One row per stage k (in STAGES order): {"stage", "reached", "conditional", "cumulative"}.
    reached = episodes (success or failure, of this policy) with stage_reached >= k + 1.
    conditional = reached / reached of the previous stage (the first stage divides by n); None if that is 0.
    cumulative = reached / n."""
    # TODO
    raise NotImplementedError


def biggest_drop(funnel: list[dict]) -> str:
    """The stage with the lowest conditional success (ignore None; ties: the earliest stage)."""
    # TODO
    raise NotImplementedError
