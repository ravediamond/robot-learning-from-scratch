# Chapter 14b. LeRobot from the inside: adding a policy, the plugin system, and serving policies remotely or through an API

Most robot-learning work in this book runs through LeRobot, and a research engineer at a company that builds on it, or contributes to it, is expected to know three things beyond the command line: how a policy is wired into the library (so a new architecture can be trained, evaluated and deployed with the standard tools), how to ship one as an independent package without forking the library (the plugin system), and how a policy can run somewhere other than the computer wired to the robot — on a GPU across the network with LeRobot's remote inference, or behind a vendor's hosted API. This chapter explains all three from zero.

Sources: the `lerobot` 0.6.1 release (the version installed for this chapter) and the LeRobot `main` branch as of 2026-10-23 (commit `d3793b7`), in particular `docs/source/bring_your_own_policies.mdx`, `docs/source/remote_inference.mdx` and `src/lerobot/remote_inference/README.md`; this repository's own plugin, `lerobot_policy_turbovla_so101`; and public reporting on Physical Intelligence's API beta. The minimal plugin in 14b.7 was written for this chapter, installed, discovered, trained, run and reloaded against `lerobot` 0.6.1 on 2026-10-24. LeRobot moves fast: check the version you have before relying on a detail here.

## 14b.1 Why this matters

- **Every new idea needs a home.** A world-model-conditioned policy, a new action head, a data-augmentation variant: to compare it fairly with ACT or SmolVLA it must train on the same datasets, with the same loaders, normalisation, evaluation and deployment. The policy interface is what makes that possible.
- **It is the contribution path.** Out-of-tree plugins are how most new policies reach LeRobot users first; in-tree contributions follow once they stabilise.
- **It is where reproducibility breaks.** The official guide says it plainly: processors (normalisation) are "the most common reproducibility pain point" — a wrong normalisation mode trains and evaluates without any error and silently wrecks the result. Chapter 8's "rule out first" list exists for the same reason.
- **Deployment is changing.** Policies are getting too large for the computer next to the arm; LeRobot now ships a remote inference stack, and at least one frontier lab now sells its policies as a hosted API. Both change what "evaluating a policy" means.

## 14b.2 How LeRobot finds a policy: a registry and three naming conventions

When you type `--policy.type=act`, three lookups happen, all by **name**:

1. **The configuration class** is found in a registry. `PreTrainedConfig` is a `draccus.ChoiceRegistry`; the decorator `@PreTrainedConfig.register_subclass("act")` adds a config class under the name `act` *when its module is imported*. `PreTrainedConfig.get_known_choices()` lists every registered name.
2. **The policy class** is found by convention from the config class: the factory takes the config's module path, replaces `configuration_` with `modeling_`, imports that module, and looks for a class named after the config with `Config` replaced by `Policy` (`ACTConfig` → `ACTPolicy` in `modeling_act`). Get either name wrong and you get "Policy class … not found".
3. **The processors** are found the same way: `configuration_` → `processor_`, and a function named `make_<type>_pre_post_processors` (`make_act_pre_post_processors`).

So one string is "load-bearing", in the guide's words: the name passed to `register_subclass`, the policy's `name` attribute, the processor function's name and the `--policy.type` flag must all agree.

## 14b.3 The configuration class

A dataclass that inherits from `PreTrainedConfig`. It is saved as `config.json` next to the weights and is everything needed to rebuild the model. What you must provide:

| Member | What it is for |
|---|---|
| fields (dataclass) | your hyper-parameters: chunk size, widths, learning rates; all become CLI flags (`--policy.hidden_dim=256`) |
| `observation_delta_indices` | which observation frames the dataset loader returns, as offsets in frames relative to the current one (`None` = current only; `[-1, 0]` = previous and current) |
| `action_delta_indices` | which action frames: `list(range(chunk_size))` for a policy that predicts the next `chunk_size` actions. This is the `delta_timestamps` mechanism of chapter 10b.4, expressed in frames |
| `reward_delta_indices` | the same for rewards (`None` for imitation learning) |
| `get_optimizer_preset()` | the default optimiser (`AdamWConfig` from `lerobot.optim`) |
| `get_scheduler_preset()` | the default learning-rate schedule, or `None` |
| `validate_features()` | check the dataset provides what the policy needs; **call it yourself** from the policy's `__init__` (the base class does not) |
| `normalization_mapping` | per feature type (`VISUAL`, `STATE`, `ACTION`), one of `IDENTITY`, `MEAN_STD`, `MIN_MAX`, or quantile modes |

Inherited fields you will use: `input_features` and `output_features` (filled from the dataset: names, types, shapes), `device`, `use_amp`, `pretrained_path`, `repo_id`, `push_to_hub`; and the helpers `robot_state_feature`, `image_features`, `action_feature`.

## 14b.4 The policy class

An `nn.Module` that inherits from `PreTrainedPolicy`. Two class attributes are checked the moment the class is defined (`__init_subclass__`): `config_class` and `name`. The methods the tools call:

| Method | Called by | Contract |
|---|---|---|
| `forward(batch)` | `lerobot-train` | return `(loss, info_dict)`; `info_dict` holds plain Python numbers for logging. `batch["action_is_pad"]` marks padded steps at episode ends: exclude them from the loss (chapter 10b.4) |
| `select_action(batch)` | evaluation and rollout | return **one** action `(B, action_dim)` for this tick. A chunking policy keeps a queue: predict a chunk when the queue is empty, pop one action per call |
| `predict_action_chunk(batch)` | the policy itself, remote inference, RTC | return a whole chunk `(B, chunk_size, action_dim)` |
| `reset()` | at the start of every episode | clear per-episode state (the action queue, caches). Forgetting it leaks the end of one episode into the start of the next |
| `get_optim_params()` | the optimiser | the parameters, or groups with their own learning rates (frozen backbones, smaller lr for pretrained parts) |
| `update()` *(optional)* | after each optimiser step | EMA targets, replay buffers (TD-MPC uses it). A JEPA-style policy with an EMA target encoder would put its EMA here |

Batches are flat dictionaries keyed by constants from `lerobot.utils.constants`: `OBS_STATE` (`"observation.state"`), image keys under `"observation.images.<camera>"`, `ACTION`, and `"task"` for the instruction string. Reuse the constants; do not invent keys.

**Saving and loading** come from the base class: `save_pretrained(dir)` writes `config.json` and the weights as safetensors; `from_pretrained(path_or_repo_id)` rebuilds the config and loads the weights (strictly by default: chapter 6's strict-load rule is built in). `lerobot-train` also saves the processors next to them and can push everything to the Hub.

## 14b.5 Processors: the part that silently breaks results

A policy never sees raw robot data. A **pre-processor pipeline** prepares inputs and a **post-processor pipeline** converts outputs back. Each is a list of steps; the default pair is:

- input: rename observation keys → add a batch dimension → move to the device → **normalise** (with the dataset statistics and the config's `normalization_mapping`);
- output: **un-normalise** the actions → move to CPU.

They are saved as JSON (plus the statistics) next to the checkpoint, and at inference the saved pipelines are loaded, not rebuilt. Three consequences:

1. The statistics travel with the checkpoint (chapter 10b.5): evaluating with another dataset's statistics is impossible by accident only if you load the saved processors.
2. A normalisation mode that differs between training and deployment (or a quantile mode whose `q01`/`q99` statistics do not exist) produces no error and a wrong robot.
3. If your model applies its own normalisation (an ImageNet-pretrained vision backbone does), map that feature type to `IDENTITY` so the pipeline leaves it alone and the model normalises it once. The TurboVLA plugin does exactly this for images.

If a policy must adjust pipelines loaded from a checkpoint (re-padding statistics, renaming keys), the `main` branch looks for an optional `make_<type>_pre_post_processors_from_pretrained` in the same module.

## 14b.6 The plugin system

You do not need to fork LeRobot to add a policy, a robot, a camera, a teleoperator or a simulation environment. Every LeRobot command (`lerobot-train`, `-record`, `-rollout`, `-eval`, `-calibrate`…) calls `register_third_party_plugins()` at start-up. It walks every installed Python distribution and **imports each one whose name starts with** `lerobot_policy_`, `lerobot_robot_`, `lerobot_camera_`, `lerobot_teleoperator_` or `lerobot_env_`. Importing your package runs your `register_subclass` decorators; from then on, `--policy.type=<your name>` works anywhere a built-in type does.

Gotchas, each of which costs an afternoon the first time:

- **The distribution name is imported as a module name.** Name the project with underscores in `pyproject.toml` (`name = "lerobot_policy_tinymlp"`), identical to the package directory under `src/`. The TurboVLA plugin's `pyproject.toml` carries a comment saying exactly this.
- **Import errors are logged, not raised.** If your package fails to import (a missing dependency, a syntax error), LeRobot logs an exception and continues; the symptom is "Unknown policy name". Check with `PreTrainedConfig.get_known_choices()` after `register_third_party_plugins()`.
- **Keep the import cheap.** Discovery imports every plugin in every command, including `lerobot-calibrate`. Import heavy libraries (transformers, model weights) inside the constructor, not at module top.
- **Export the config from `__init__.py`.** That import is what registers the policy.

## 14b.7 A complete minimal plugin

A policy that maps the joint state to a chunk of future actions with an MLP: useless as a robot policy, complete as an integration. Five files.

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=71"]
build-backend = "setuptools.build_meta"

[project]
name = "lerobot_policy_tinymlp"          # the prefix lerobot_policy_ is what LeRobot looks for
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["lerobot>=0.6.0"]

[tool.setuptools.packages.find]
where = ["src"]
```

`src/lerobot_policy_tinymlp/__init__.py`:

```python
try:
    import lerobot  # noqa: F401
except ImportError as e:
    raise ImportError("lerobot is not installed; install it to use this policy package.") from e

from .configuration_tinymlp import TinyMLPConfig
from .modeling_tinymlp import TinyMLPPolicy
from .processor_tinymlp import make_tinymlp_pre_post_processors

__all__ = ["TinyMLPConfig", "TinyMLPPolicy", "make_tinymlp_pre_post_processors"]
```

`configuration_tinymlp.py`:

```python
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("tinymlp")      # the name users type: --policy.type=tinymlp
@dataclass
class TinyMLPConfig(PreTrainedConfig):
    chunk_size: int = 10           # actions predicted per call
    n_action_steps: int = 10       # actions executed before predicting again
    hidden_dim: int = 128
    optimizer_lr: float = 1e-3
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
            "VISUAL": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps cannot exceed chunk_size")

    def validate_features(self) -> None:
        if self.robot_state_feature is None:
            raise ValueError("TinyMLP needs observation.state")
        if self.action_feature is None:
            raise ValueError("TinyMLP needs an action feature")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr)

    def get_scheduler_preset(self):
        return None

    @property
    def observation_delta_indices(self):
        return None                                   # current frame only

    @property
    def action_delta_indices(self):
        return list(range(self.chunk_size))           # the next chunk_size actions

    @property
    def reward_delta_indices(self):
        return None
```

`modeling_tinymlp.py`:

```python
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_tinymlp import TinyMLPConfig


class TinyMLPPolicy(PreTrainedPolicy):
    config_class = TinyMLPConfig   # both attributes are checked when the class is defined
    name = "tinymlp"               # must equal the string given to register_subclass

    def __init__(self, config: TinyMLPConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.state_dim = config.robot_state_feature.shape[0]
        self.action_dim = config.action_feature.shape[0]
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, config.hidden_dim), nn.GELU(),
            nn.Linear(config.hidden_dim, config.chunk_size * self.action_dim),
        )
        self.reset()

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        self._queue = deque(maxlen=self.config.n_action_steps)   # cleared at every episode start

    def _chunk(self, batch):
        out = self.net(batch[OBS_STATE])                          # (B, chunk*A)
        return out.view(-1, self.config.chunk_size, self.action_dim)

    @torch.no_grad()
    def predict_action_chunk(self, batch, **kwargs) -> torch.Tensor:
        self.eval()
        return self._chunk(batch)                                 # (B, chunk, A)

    @torch.no_grad()
    def select_action(self, batch, **kwargs) -> torch.Tensor:
        self.eval()
        if len(self._queue) == 0:
            chunk = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._queue.extend(chunk.transpose(0, 1))             # queue of (B, A) actions
        return self._queue.popleft()                              # (B, A)

    def forward(self, batch):
        pred = self._chunk(batch)                                 # (B, chunk, A)
        err = F.l1_loss(pred, batch[ACTION], reduction="none")    # (B, chunk, A)
        valid = ~batch["action_is_pad"].unsqueeze(-1)             # (B, chunk, 1)
        loss = (err * valid).sum() / (valid.sum() * self.action_dim).clamp_min(1)
        return loss, {"l1": loss.item()}
```

`processor_tinymlp.py`:

```python
from lerobot.processor import make_default_pre_post_processors


def make_tinymlp_pre_post_processors(config, dataset_stats=None):
    # input: rename -> add batch dim -> to device -> normalise; output: unnormalise -> to cpu
    return make_default_pre_post_processors(config, dataset_stats, normalizer_device=config.device)
```

Install it with `pip install -e .` in its directory. Then this script exercises every path the tools use — discovery, the factory's naming conventions, the processors, a training loop with padded steps, one-action-at-a-time inference, save and reload:

```python
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.import_utils import register_third_party_plugins

print('before:', 'tinymlp' in PreTrainedConfig.get_known_choices())
register_third_party_plugins()                        # what every lerobot-* script calls at start
print('after: ', 'tinymlp' in PreTrainedConfig.get_known_choices())

cfg_cls = PreTrainedConfig.get_choice_class('tinymlp')
cfg = cfg_cls(device='cpu',
              input_features={'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(6,))},
              output_features={'action': PolicyFeature(type=FeatureType.ACTION, shape=(6,))})
policy_cls = get_policy_class('tinymlp')              # resolved by naming convention
print(policy_cls.__name__, cfg.type, cfg.action_delta_indices)
policy = policy_cls(cfg)

stats = {'observation.state': {'mean': torch.zeros(6), 'std': torch.ones(6)},
         'action': {'mean': torch.zeros(6), 'std': torch.ones(6)}}
pre, post = make_pre_post_processors(cfg, dataset_stats=stats)

# a fake training batch: B=4, chunk 10, last 3 steps of each chunk padded
batch = {'observation.state': torch.randn(4, 6), 'action': torch.randn(4, 10, 6),
         'action_is_pad': torch.zeros(4, 10, dtype=torch.bool)}
batch['action_is_pad'][:, 7:] = True
opt = torch.optim.AdamW(policy.get_optim_params(), lr=1e-2)
for step in range(200):
    loss, info = policy.forward(batch)
    opt.zero_grad(); loss.backward(); opt.step()
print('loss start->end', round(info['l1'], 4))

# inference through the processors, one action per call
policy.reset()
obs = pre({'observation.state': torch.randn(6)})       # adds the batch dim, normalises
a = post(policy.select_action(obs))
print('action shape', tuple(a.shape), 'queue left', len(policy._queue))

policy.save_pretrained('/tmp/tinymlp_ckpt')
re = policy_cls.from_pretrained('/tmp/tinymlp_ckpt')
print('reloaded', type(re).__name__, torch.allclose(re.net[0].weight, policy.net[0].weight))
```

The run of 2026-10-24 with `lerobot` 0.6.1 printed: `before: False`, `after: True` (discovery works only after `register_third_party_plugins()`); `TinyMLPPolicy tinymlp [0, 1, …, 9]` (the factory found the class by convention; the loader would return 10 future actions); the masked L1 loss fell to 0.017 on the fixed batch; one inference call returned an action of shape (1, 6) and left 9 in the queue; the reloaded policy's weights were identical. With the package installed, `lerobot-train --policy.type=tinymlp --dataset.repo_id=…` trains it like any built-in policy (the training script calls the same discovery function; running it needs the `dataset` extra).

## 14b.8 A real plugin: `lerobot_policy_turbovla_so101`

This repository's TurboVLA port is a published plugin of exactly this shape (config, modeling, processor, a temporal-ensembling helper, tests, benchmarks). Design choices worth being able to explain:

- **Images mapped to `IDENTITY`** in `normalization_mapping`: the DINOv3 backbone applies its own ImageNet normalisation inside the model, so the dataset normaliser must not touch images. State and actions use `MEAN_STD`.
- **The default processor pipeline is enough**: the task string rides through untouched as `batch["task"]` and is tokenised inside the model against whichever language backbone the config names.
- **Fail loudly on a missing instruction**: `forward` raises if the batch has no `task`, because otherwise the language-conditioned policy silently trains as a worse ACT.
- **Two inference modes in `select_action`**: an action queue (predict every `n_action_steps`), or temporal ensembling (predict every step and blend overlapping chunks), which requires `n_action_steps = 1` and is validated in `__post_init__`.
- **Optimiser presets copied from the paper's recipe** (betas (0.9, 0.95), weight decay 1e-10, gradient clip 1.0), after a docstring-versus-code mismatch was found.
- **The lesson of chapter 8**: a plugin is a *port*. The two-week "the model ignores the instruction" diagnosis was run on this plugin before it was compared with the authors' code; the faithful upstream run (path A) then showed the instruction was ignored there too, so the data finding stood and the port claim was withdrawn. Whatever you port, run the parity check against the reference implementation first (chapter 6).

## 14b.9 Plugin or in-tree

The official guide recommends starting as a plugin (no review, your release cycle) and moving in-tree once the policy is stable and useful to others. An in-tree contribution adds: the files under `src/lerobot/policies/<name>/` with the same names; the config re-exported from `policies/__init__.py` (that import registers it; the modeling class stays lazily imported so `import lerobot` stays fast); optional heavy dependencies behind an extra and a `require_package` check in the constructor; a model-card template entry and documentation page; tests; and a published checkpoint with **one reproducible number** — a simulation benchmark with at least 50 episodes per suite, or for real-robot-only policies a public dataset, the training command, the checkpoint and a success rate over at least 10 real episodes with `lerobot-rollout`. Reviewers also ask that you **subclass** an existing `transformers` model to modify it, not copy its modeling file. Every item on that checklist is an evaluation rule from Part I in library form.

## 14b.10 Serving a policy over the network: LeRobot remote inference

**The problem.** A large policy needs a GPU; the computer wired to the arm may have none, or a small one (the Orin runs SmolVLA in 238 ms per chunk only through vla.cpp). And with synchronous inference the arm stops at every chunk.

**The design** (merged into LeRobot `main` on 2026-10-13 as PR #4937, after the earlier PR #4836 that this repository's owner tested on hardware; not in the 0.6.1 release): a **policy server** (`lerobot-policy-server`) loads one checkpoint with its saved processors on a GPU and serves **one robot client** at a time; the **robot computer** keeps cameras, motors, teleoperation and recording and runs `lerobot-rollout --inference.type=remote`. The client keeps a buffer of upcoming actions, asks for a new chunk when the buffer runs low, keeps moving meanwhile, and splices the new chunk into the not-yet-executed part of the plan. Transport is **Zenoh** (a publish/subscribe and query protocol) with explicit endpoints; messages are bounded **MessagePack**, with no pickle. It replaces the older `async_inference` gRPC stack, which deserialised network input with `pickle` and was the subject of an unauthenticated remote-code-execution vulnerability (CVE-2026-25874, published April 2026).

**Starting it** (from the official guide; a generic SO-101 example, not a tested profile). The server reads a YAML file declaring the checkpoint, the execution settings, the Zenoh endpoints, and the exact feature schema and units the policy was trained for:

```bash
lerobot-policy-server --config_path=server.yaml        # model.repo_or_path, model.revision (pin it), features, semantics, action_fps
```

```bash
lerobot-rollout \
  --inference.type=remote \
  --inference.endpoint=tcp/127.0.0.1:7447 \
  --inference.deployment=manipulation \
  --inference.mode=chunk \
  --inference.chunk_merge=aligned \
  --inference.semantics=so101-degrees-gripper-percent-v1 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.use_degrees=true \
  --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
  --fps=30 --task="pick up the cube" --interactive=true
```

Admission is a contract: the client and server compare features, units ("semantics"), action rate and mode, and an incompatible pair is rejected **before any motion**. The robot driver must declare **position-hold** support (the SO follower does), because while waiting the client re-sends the last applied target.

**The settings that matter**, all on the client:

- **`chunk_merge`**: `append` (play the new chunk after the old: simple, but starts from an older observation), `aligned` (the default: drop incoming steps the robot has already passed, replace only the future), alignment plus **blending** (`blend_steps`, `blend_weight`, an explicit list of joints, never the gripper by default), or **RTC** modes for checkpoints that support real-time chunking. This is chapter 11.4's chunk-boundary problem with named knobs.
- **`refill_seconds`** (default 0.5 s of remaining playback): when to ask for the next chunk. The effective value is at least the recently measured turnaround plus one tick. Ask early to react, late to follow through.
- **Deadlines**: per-request timeouts, a maximum observation age, and an **action-starvation grace** (default 1 s) during which the arm holds its last target before the session faults.

**Policy support** (main branch, 2026-10-23): ACT (temporal ensembling off), SmolVLA, XVLA, EVO1 and LaWAM have run on real robots through this path; VLA-JEPA, GR00T, π0/π0.5 and others are supported "at the contract level" without hardware evidence; history-dependent policies (Diffusion, VQ-BeT, TD-MPC) need adaptation.

**Security**: binding to a LAN interface limits exposure but does not authenticate anyone; outside a trusted lab, use the maintained Zenoh router configurations with TLS, mutual certificates and access-control lists. Operating through a router, a private hosted GPU or a Hugging Face Space was **not yet validated** at the time of writing.

**What hardware testing of the earlier PR found on this rig** (PR #4836, October 2026; findings were reported for that head and may have been addressed in the merged version — re-test before relying on any of them):

- Two raw 640×480 cameras (about 1.8 MB per observation) from the Jetson over Wi-Fi: observations never reached the server and the session faulted after the 10 s startup deadline, with no clear error; JPEG at quality 90 worked (server 75–115 ms per call on an M-series Mac, client at 29.9 Hz).
- SmolVLA served from the Mac took 1.83 s per prediction, longer than its 1.67 s action horizon; the second request failed with a protocol error ("Invalid RTC delay") instead of a clear "turnaround exceeds horizon".
- A camera unplugged mid-task froze the arm mid-trajectory; the return-to-start was skipped because it needs a camera frame, and torque was released on disconnect about 2 s later. The arm stayed in place on gearbox friction in the one pose observed, which is not a guarantee.

The general lessons: measure full turnaround (encoding, network, queueing, inference, decoding) against the chunk's playback time before anything else; the payload of uncompressed images is a network-design decision; and every remote fault path ends in a hardware shutdown path that must be tested on the arm, not read in the code.

## 14b.11 Policies as a hosted API: proprietary models

**What happened.** On 13–14 October 2026 Physical Intelligence (the company behind π0, π0-FAST and π0.5) opened a **beta of its Pi API**: you upload your robot data, fine-tune their latest model on compute they manage, and run the fine-tuned model through a **hosted inference gateway**. Access is through a waitlist. Reported early users include Almond (a dual-arm robot deburring parts after four days of iteration), Actor (an autonomous wheel loader) and Chef Robotics, among others; most showcased deployments ran with **cloud-hosted inference**, and the company reports that some faced **network-connectivity problems**. As of 2026-10-24 the public material does **not** state the data format, the inference protocol and control rates, latency figures, pricing, the robots and action spaces supported, whether weights can be exported or run on-premises, whether a model version can be pinned, or any LeRobot compatibility. The open-weight π0 and π0.5 remain available separately through the `openpi` repository and as LeRobot policies (full fine-tuning of those needs a GPU with about 70 GB of memory).

**What changes for a research engineer when the policy is behind an API:**

1. **The model is a black box you cannot pin unless the vendor lets you.** "We can update the models behind the API quickly" is a feature for a customer and a confound for an experiment: a success rate measured in October may not reproduce in November. Record the model identifier and date with every trial; ask for version pinning before running any comparison.
2. **Latency moves into the network.** Chapter 10a.5's two clocks gain a third: the round trip to a data centre, with its variance. The arm must keep moving (or hold safely) while waiting, exactly the buffering, refill and starvation problem of 14b.10, now over the internet. Measure the latency distribution from your site, not the median from a demo.
3. **Failures have new causes.** Connectivity loss, rate limits, gateway errors and silent model updates join the robot's own failure modes; the harness needs a defined behaviour for each (hold, retreat, stop), tested on the arm.
4. **Evaluation stays your job.** A vendor's success rate is a claim on their tasks (chapter 2). On your tasks you still need the baseline (your own fine-tuned SmolVLA or ACT on the same data and layouts), paired trials, pre-registered bars and a verifier you trust (chapters 3–7). The API removes the GPU cluster, not the experiment.
5. **Data leaves your site.** Uploaded demonstrations and inference observations (camera images of your workspace) are processed by a third party; read the terms on retention and use before uploading anything from a customer site.

**Three deployment options side by side:**

| | On the robot computer (vla.cpp on the Orin) | LeRobot remote inference (your GPU, your network) | Vendor API (hosted proprietary model) |
|---|---|---|---|
| Model | open weights you trained | open weights you trained | the vendor's model, fine-tuned on your data |
| Latency | fixed, measured (238 ms per SmolVLA chunk) | LAN round trip + GPU time | internet round trip + queueing + GPU time |
| Control over version | full | full (pin the Hub revision) | whatever the vendor offers |
| Failure modes added | none beyond the robot | network, server process | network, gateway, rate limits, silent updates |
| Cost | hardware already owned | a GPU machine | per use (undisclosed for the beta) |
| Reproducibility | full | full | depends on version pinning |
| Best for | small policies, demos away from a network | large open policies, lab use | trying a frontier model without a GPU cluster |

**A protocol, if you get access** (written in the book's style, bars blank for you): decision sentence ("does a hosted frontier model fine-tuned on our 60 LEGO demonstrations beat our own SmolVLA on the sealed layouts by enough to justify its latency and dependence?"); matched data (the same episodes uploaded as used for the local fine-tune); pinned model version or recorded identifier; latency measured first (distribution over 500 calls from the robot's network, against the chunk playback time); paired sealed layouts, the same verifier, Wilson intervals; failure classes including network faults; a register row either way.

## 14b.12 Questions to answer in one minute each

1. What three names must agree for `--policy.type=my_policy` to work, and what does each look up?
2. What do `observation_delta_indices` and `action_delta_indices` control, and how do they relate to `delta_timestamps`?
3. `select_action` vs `predict_action_chunk`: who calls which, and why does `reset()` matter?
4. Why are processors the most common reproducibility failure, and when do you map a feature to `IDENTITY`?
5. How does LeRobot discover a plugin, and what is the symptom when the plugin fails to import?
6. When would you contribute a policy in-tree rather than keep it as a plugin, and what must the PR prove?
7. In remote inference, what do `chunk_merge=aligned`, `refill_seconds` and the starvation grace each control?
8. Why did the old async inference server need replacing (one sentence on the security issue)?
9. A vendor offers its policy as a hosted API. Name three things you would ask for before running a comparison, and the one measurement you would make first.

## 14b.13 Exercises

1. Build and install the 14b.7 plugin, run its test script, then break each convention once (rename the policy class; rename the processor function; put a hyphen in the project name) and record the exact error each produces.
2. Add an `update()` method that maintains an EMA copy of the network, and check after training that the EMA weights differ from the online ones by the expected amount (chapter P2.15).
3. Train the plugin on your bowls dataset with `lerobot-train --policy.type=tinymlp` for 2 000 steps; compare its held-out L1 with ACT's on the same split. It should lose clearly; if it does not, the task is solvable from the joint state alone, which is chapter 8's fact 7 again.
4. On a LAN, serve an ACT checkpoint with `lerobot-policy-server` and run the SO-101 client; log the full turnaround distribution and the effective refill; then add a 5-step blend on the arm joints and compare the boundary jump with chapter 11's measurement.
5. Write the Pi API comparison protocol of 14b.11 in full in your own words, with bars, before any access, and list the vendor questions that would void it if unanswered.
