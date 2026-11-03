# W5 D5: A mini VLA from your own parts (capstone)

Primers: `knowledge/03-vla.md`, `02-vlm.md`. Book: 11a.1 (SmolVLA). Parts: your W3 attention and
Block (copied in `vla_blocks.py`), your W4 flow matching, today's W5 plumbing.

## The world (`vla_world.py`, read it)
Two targets, red and blue, at random places; an instruction "go to the red / blue". The IMAGE says
where the targets are, the INSTRUCTION says which one. A policy that ignores either one fails half
the time. That makes "does it use language?" measurable: swap the instruction and see where it goes.

## Architecture (SmolVLA layout, tiny)
```
image (B,3,16,16) -> patchify p=4 -> 16 tokens x 48 -> Linear(48, D) + Linear(2, D)(patch centre xy)
tokens (B,4)      -> Embedding(6, D) + learned text positions (4, D)
state (B,2)       -> Linear(2, D)                                   one token
prefix = 2 Blocks (bidirectional) over the 21 tokens, then LayerNorm          computed ONCE per act()
action expert: x_t (B,H,2) -> Linear(2, D) + learned action positions (H, D) + Linear(32, D)(time_embedding(t))
               -> Block with cross-attention to the prefix (context=prefix)
               -> Block with causal self-attention over the H action tokens
               -> Linear(D, 2) = velocity
train: flow matching on chunks (t=0 noise, t=1 data); act: 10 Euler steps, prefix cached.
```
Positions from patch-centre coordinates (instead of a learned table) make localisation learnable in a
few thousand CPU steps: a real design trick (coordinate features).

## Today's build (`w5d5_mini_vla.py`)
`MiniVLA` with `encode_prefix`, `expert`, `forward`, `act`; `train_vla`.
Tests (~2 min): success > 85% with the given instruction AND > 85% when the instruction is swapped.

Run: `uv run pytest weeks/w5-vlm-vla/d5-mini-vla -s`
