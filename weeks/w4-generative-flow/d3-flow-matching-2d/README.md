# W4 D3: Flow matching from scratch (capstone, part 1)

Primer: `knowledge/04-diffusion-flow-matching.md`. Book: 11.3. Coding list items 11, 12.
This is the loss and sampler inside SmolVLA's and π0's action expert.

## Convention used everywhere in this course: t = 0 is noise, t = 1 is data
```
x0 ~ N(0, I) (noise),  x1 = data,  t ~ U(0, 1) per sample
x_t = (1 - t) x0 + t x1            straight line from noise to data
target velocity = x1 - x0          constant along the line
loss = || v_theta(x_t, t) - (x1 - x0) ||^2
sample: x = x0;  for i in 0..N-1:  t = i/N;  x = x + (1/N) * v_theta(x, t)
```
Papers and codebases disagree on which end is noise. Always check: at t = 0, is x_t the noise?

## The 1-step experiment (today's key insight)
With ONE Euler step from t = 0, the network must answer "where does this noise go" before it
knows anything: the best L2 answer is the data MEAN. One step = MSE regression again. More steps
let the trajectory commit to a mode. You will measure this.

## Today's build (`w4d3_flow.py`)
`fm_interpolate`, `VelocityNet`, `fm_loss`, `euler_sample`, `train_fm`.
Test: 10-step samples on the moons; 1-step samples collapse toward the mean.
Optional: `uv run python plot_flow.py` saves `flow_samples.png` (data, 1-step, 10-step).

Run: `uv run pytest weeks/w4-generative-flow/d3-flow-matching-2d`
