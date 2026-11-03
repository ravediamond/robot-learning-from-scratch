# W4 D1: Why MSE fails on demonstrations, and the VAE

Primer: `knowledge/04-diffusion-flow-matching.md`. Book: 11.3.

## MSE predicts the mean
For a fixed observation, the MSE-optimal prediction is E[action | obs]. If half the operators
go left (−1) and half right (+1), the optimum is 0: straight into the obstacle. No amount of
data or capacity fixes this: it is what the loss asks for. Fix = model the distribution
p(action | obs) and SAMPLE from it.

## VAE in one page
```
encoder(x) -> mu, logvar          q(z|x) = N(mu, exp(logvar))
z = mu + exp(0.5*logvar) * eps    eps ~ N(0, I)      "reparameterization": gradient flows to mu, logvar
decoder(z) -> x_hat
loss = recon(x_hat, x) + beta * KL(q(z|x) || N(0, I))
KL   = 0.5 * sum( exp(logvar) + mu^2 - 1 - logvar )     closed form, per sample
sample: z ~ N(0, I) -> decoder(z)
```
ACT is a conditional VAE (CVAE): the encoder sees (obs, true action chunk) during training,
the decoder sees (obs, z); at test time z = 0. Week 6.

## Today's build (`w4d1_vae.py`)
`mse_optimum` (closed form), `reparameterize`, `kl_to_standard_normal`, `VAE`, `vae_loss`,
`train_vae`. Test: the trained VAE puts samples in BOTH modes of a bimodal dataset, while the
MSE optimum sits in the empty gap between them.

Run: `uv run pytest weeks/w4-generative-flow/d1-mse-fails-vae`
