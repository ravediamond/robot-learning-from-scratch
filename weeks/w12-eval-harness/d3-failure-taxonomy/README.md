# W12 D3: failure taxonomy and where each policy loses

Primer: `knowledge/13-evaluation-at-scale.md` §3.

## The one idea

"62% success" tells you which policy wins. It does not tell you what to fix. Two tools do:

1. **A failure taxonomy**, written before the session: a short closed list of causes with definitions
   (perception: wrong object or position; grasp: closes on nothing or drops; transport; placement;
   collision; timeout; safety stop). Every failed rollout gets exactly one primary tag. Two raters
   label a subset independently and you report **Cohen's κ**: if they do not agree, the tags are noise.
2. **A stage funnel**: did the policy reach, grasp, lift, place? Conditional success per stage shows
   where it loses: "VLA-JEPA reaches as well as ACT but grasps 20 points worse".

Rates per cause are computed over **all** episodes of the policy (with a Wilson CI), never as a share
of its failures: the share hides how often the failure happens.

## Today's build (`w12d3_failures.py`)

`cohen_kappa`, `breakdown`, `stage_funnel`, `biggest_drop`. `wilson` and the taxonomy are in `evalkit.py`.

## Staff angle

A failure breakdown is what turns an eval into a roadmap: "grasp failures are 70% of what separates us
from ACT, they cluster on the two starts near the edge of the camera view, so the next data collection
targets those". That sentence is the job description of the evaluation side of research engineering.
