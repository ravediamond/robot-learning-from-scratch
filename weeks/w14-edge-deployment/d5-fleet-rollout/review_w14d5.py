"""REVIEW. SOLUTION.

BUG: it compared COUNTS. Five robots for a few days produce far fewer events than 45 robots whatever their quality.
Compare rates (events per robot-hour), and require enough canary hours before deciding; better, a test of the rate
ratio (see canary_decision) so noise does not decide either way.
"""


def canary_is_better(canary_interventions: int, canary_hours: float, fleet_interventions: int, fleet_hours: float) -> bool:
    return canary_interventions / canary_hours < fleet_interventions / fleet_hours
