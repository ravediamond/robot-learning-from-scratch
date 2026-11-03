# Questions (out loud)

1. Why is a constant encoder a global minimum of the invariance loss? Why does gradient descent find it so fast?
2. What does each VICReg term prevent? What failure remains if you drop the covariance term (predict item 3)?
3. Effective rank vs per-dimension std: why do you need both to detect partial collapse?
4. The review bug: what does `var(dim=1)` measure? Why did it not prevent collapse?
5. VICReg needs reasonably large batches. Why? What does LeWM's SIGReg change about that?
6. Interviewer: "Your world model's latent has 192 dims but effective rank 3. Problem or not?"
