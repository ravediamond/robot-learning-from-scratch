# W9 D1: Distributed data parallel and checkpoint/resume, by hand (2 CPU processes)

Book: P2.12 (checkpoints), P2.13 (DDP, to discuss). Coding list item 24. Asked as "what breaks going
from 1 to 8 GPUs?" in almost every research-engineer loop. Today you run it for real on CPU (gloo).

## DDP in five facts
1. One process per device (`world_size` of them), each with a full model copy and its own data shard.
2. `backward()` all-reduces (averages) the gradients across processes, so every copy takes the SAME step.
3. With a mean loss per process and equal shard sizes, DDP on B/world per process == one process on B.
4. Shard the data: `DistributedSampler` (call `set_epoch(e)` each epoch, or every epoch shuffles the same).
5. Only rank 0 logs and saves; every rank must call the collective ops in the same order or it hangs.

## Resume = restore EVERYTHING that influences the next step
model, optimizer state (Adam moments!), scheduler, step/epoch counters, RNG states (torch, the
sampler's generator), and the data position. Test: train 2N steps straight vs N + save + kill + load + N:
weights must be identical.

## Today's build (`w9d1_ddp.py`)
`ddp_worker` (spawned per rank: init gloo, wrap with DDP, train on the rank's shard, rank 0 saves),
`single_process_reference`, `save_checkpoint`, `load_checkpoint`, `train_steps`.

Run: `uv run pytest weeks/w9-mock-interviews/d1-ddp-resume -s`
