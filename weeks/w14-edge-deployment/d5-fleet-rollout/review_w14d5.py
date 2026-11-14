"""REVIEW. Automatic rollback rule of the fleet controller. A new policy was rolled out to all 50 robots after its
5-robot canary "had far fewer interventions than the fleet"; the fleet's intervention rate then doubled.
Find the bug, fix, replace:
BUG:
"""


def canary_is_better(canary_interventions: int, canary_hours: float, fleet_interventions: int, fleet_hours: float) -> bool:
    return canary_interventions < fleet_interventions
