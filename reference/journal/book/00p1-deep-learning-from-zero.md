# Chapter P1. Deep learning from zero

This chapter assumes you know Python and high-school maths and nothing else. It builds every idea the coding exercises use, one at a time, each with a small example whose numbers you can check by hand. Chapter P2 (PyTorch for the coding round) then shows how each idea is written in PyTorch. Read this one first, in order; every section uses the ones before it.

Every number in the worked examples was computed in PyTorch on 2026-10-24. Redo them on paper first, then in a REPL.

## P1.1 What "learning" means

A **model** is a function with adjustable numbers inside it. Those numbers are called **parameters** (or **weights**). Learning means: change the parameters until the function's outputs match the examples you have.

The smallest possible example. You have three examples (x, y): (1, 3), (2, 5), (3, 7). The true rule is y = 2x + 1, but the model does not know it. The model is

  ŷ = w · x + b

with two parameters, w and b. ŷ ("y-hat") is the model's guess. Start with w = 0 and b = 0: the model guesses 0 for every x.

To improve it, you need a single number that says how wrong it is. That number is the **loss**. A common choice is the **mean squared error (MSE)**: for each example, take the error ŷ − y, square it (so negative and positive errors both count, and big errors count a lot), and average.

With w = 0, b = 0: errors are −3, −5, −7; squared 9, 25, 49; mean **27.667**.

Learning is now a precise problem: **find the w and b that make the loss as small as possible.** Everything else in deep learning (the network shapes, the optimisers, the tricks) is detail about how to do this when there are millions of parameters instead of two.

## P1.2 Gradients: which way is downhill

Picture the loss as a landscape: each position is a choice of (w, b), the height is the loss. You want the lowest point. You cannot see the whole landscape; you can only feel the slope where you stand.

The **derivative** of a function tells you its slope: if you nudge the input a tiny bit, how much does the output change? For a function of several inputs, you take one derivative per input, holding the others fixed; these are **partial derivatives**, written ∂L/∂w and ∂L/∂b. Together they form the **gradient**, a vector that points in the direction the loss rises fastest. So you step in the **opposite** direction.

For our loss L = mean((w·x + b − y)²), calculus gives

  ∂L/∂w = mean(2 · (ŷ − y) · x)  ∂L/∂b = mean(2 · (ŷ − y))

At w = 0, b = 0: ∂L/∂w = 2 · mean(−3·1, −5·2, −7·3) = **−22.667**; ∂L/∂b = 2 · mean(−3, −5, −7) = **−10**. Both negative: increasing w or b lowers the loss.

**Gradient descent** takes a step against the gradient:

  w ← w − η · ∂L/∂w  b ← b − η · ∂L/∂b

η (eta) is the **learning rate**: how big a step. With η = 0.05:

| Step | w | b | Loss |
|---|---|---|---|
| 0 | 0.000 | 0.000 | 27.667 |
| 1 | 1.133 | 0.500 | 5.489 |
| 2 | 1.638 | 0.723 | 1.090 |
| … | | | |
| 500 | 2.000 | 1.000 | 0.000 |

It finds w = 2, b = 1. That is all training is: compute the loss, compute the gradient, take a small step, repeat.

**The learning rate matters.** With η = 0.3 instead, the steps overshoot the bottom and land higher on the other side, each time worse: loss 27.7 → 150 → 812 → 4 403 → 23 857. This is **divergence**, and "loss goes up and becomes NaN" is the usual symptom of a learning rate that is too large. Too small, and training is just slow. Most of tuning is choosing η.

## P1.3 Backpropagation: getting gradients for free

For a two-parameter model you can derive the gradient by hand. For a network with millions of parameters you cannot. **Backpropagation** computes all of them automatically, using one rule from calculus, the **chain rule**: if a quantity flows through several steps, its derivative is the product of the derivatives of each step.

Take L = (w₁x₁ + w₂x₂ − 1)² with w = (2, −1), x = (3, 4). Break it into steps, as the computer does:

1. s = w₁x₁ + w₂x₂ = 6 − 4 = 2
2. e = s − 1 = 1
3. L = e² = 1

Now go **backwards** from L:

- ∂L/∂e = 2e = 2
- ∂e/∂s = 1, so ∂L/∂s = 2 · 1 = 2
- ∂s/∂w₁ = x₁ = 3, so ∂L/∂w₁ = 2 · 3 = **6**; likewise ∂L/∂w₂ = 2 · 4 = **8**

The forward pass records every step (this record is the **computation graph**); the backward pass walks it in reverse, multiplying local derivatives. The cost of the backward pass is about twice the forward pass, whatever the number of parameters. This is why deep learning is practical. In PyTorch it is one call, `loss.backward()` (chapter P2.3), and this example is the first code block there.

## P1.4 Neural networks: stacking simple layers

**A linear layer** generalises ŷ = w·x + b to many inputs and many outputs at once. With an input vector x of size d_in and an output of size d_out, the layer holds a matrix W of shape (d_out, d_in) and a vector b of size d_out, and computes

  y = W x + b

Each output is a weighted sum of all inputs plus a bias. It has d_out·d_in + d_out parameters.

**Why one linear layer is not enough.** Two linear layers in a row are still a linear function: W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂), one big linear layer. Straight-line functions cannot represent most things you care about (whether a grasp succeeded, where an arm will be). So between linear layers you insert a simple **nonlinear** function applied to each number separately, called an **activation**:

- **ReLU**: max(0, z). Negative numbers become 0.
- **GELU**: a smooth version of ReLU; used in transformers and in the predictors of the JEPA papers.
- **tanh**, **sigmoid**: squash numbers into (−1, 1) and (0, 1).

**A multi-layer perceptron (MLP)** alternates them: Linear → activation → Linear → activation → … → Linear. With enough width (outputs per layer) and depth (number of layers), an MLP can approximate almost any function. Every network in this book, including transformers, is made of linear layers, activations and a few more pieces from sections P1.9 and P1.10.

**Counting parameters** is a habit worth having: an MLP 4 → 16 → 3 has (4·16 + 16) + (16·3 + 3) = 131 parameters. Chapter P2.4 checks a count like this in code.

## P1.5 Tensors, batches, epochs

**A tensor** is a box of numbers with a shape. A single number is a 0-dimensional tensor; a vector of 6 joint angles is shape (6,); an image is (3, 480, 640) (colour channels, height, width); a video clip of 10 frames is (10, 3, 480, 640).

**A batch.** You do not compute the loss on one example at a time. You stack B examples into one tensor with a new first axis (the **batch** axis) and compute the average loss over them. Two reasons: the gradient averaged over many examples is less noisy, and a GPU computes B examples almost as fast as one. So inputs have shapes like (B, 6) for joint states or (B, T, D) for sequences: B examples, T time steps, D features per step. Learning to read these letters fluently is most of the coding round.

**Vocabulary of a training run:**

- **Step** (or iteration): one batch → loss → gradient → parameter update.
- **Epoch**: one full pass over the training data. With 10 000 examples and B = 100, an epoch is 100 steps.
- **Shuffling**: the order of examples is randomised each epoch. If the data were sorted (all successes first, then all failures), each batch would be unrepresentative and training would zig-zag.
- **Stochastic gradient descent (SGD)**: gradient descent where each step uses a random batch rather than all the data. "Stochastic" means random. Every optimiser in deep learning is a variant of this.

## P1.6 Loss functions: what "wrong" means

The loss must match the kind of output.

**Regression** (the output is a number or a vector of numbers: joint angles, a future latent):

- **MSE**, mean of squared errors: punishes big errors heavily. Used for predicting latents in JEPA world models, and for flow-matching velocities.
- **L1**, mean of absolute errors: less sensitive to a few huge errors (outliers). ACT trains its actions with L1.
- **Huber** (smooth L1): squared for small errors, absolute for large ones; a compromise.

**Classification** (the output is one of C categories: success / failure, which brick):

The network outputs C raw scores called **logits**, one per class, any real numbers. **Softmax** turns them into probabilities: exponentiate each, divide by the sum.

  logits (2, 1, 0) → exp (7.39, 2.72, 1.00) → sum 11.11 → probabilities **(0.665, 0.245, 0.090)**

The largest logit gets the most probability; probabilities are positive and sum to 1.

**Cross-entropy** is the loss: minus the log of the probability the model gave to the **correct** class. If the correct class is the first one: −ln(0.665) = **0.41**. If it is the third: −ln(0.090) = **2.41**. Confident and right costs almost nothing; confident and wrong costs a lot. It is a measure of surprise.

Two practical facts follow, both in chapter P2.2 and P2.6: you give the loss function the logits, not the probabilities (it does the softmax internally, in a numerically safe way); and **binary** classification (success or not) uses one logit and the **sigmoid** function instead of softmax, with **binary cross-entropy**.

**Masked losses.** If some positions are padding (an action chunk that runs past the end of an episode), you compute the loss per position, multiply by a mask that is 1 for real positions and 0 for padding, and divide by the number of real positions.

## P1.7 Generalisation: why you split the data

The goal is not a low loss on the examples you trained on. It is good predictions on examples the model has **never seen**: new episodes, new layouts. That is **generalisation**.

A large model can memorise its training examples and still be useless on new ones. This is **overfitting**: training loss keeps falling, loss on new data starts rising. The opposite, **underfitting**, is a model too small or trained too little to fit even the training data.

So you split the data:

- **Training set**: the model learns from it.
- **Validation set**: you watch the loss on it during training to choose hyper-parameters (learning rate, model size, when to stop).
- **Test set**: touched once, at the end, for the number you report. Once you tune anything on it, it is no longer a test set (chapter 4).

**Split by episode, never by frame.** Consecutive video frames are almost identical. If frame 100 of an episode is in training and frame 101 in validation, the model "predicts" validation by remembering training, and your metric is inflated. Whole episodes go to one side.

**Regularisation** means anything that reduces overfitting:

- **Weight decay**: shrink every weight a little each step, so the model prefers small weights (simpler functions).
- **Dropout**: during training, randomly zero a fraction p of a layer's outputs; the network cannot rely on any single unit. Turned off at evaluation.
- **Data augmentation**: train on modified copies (colour jitter, small crops, a camera blanked out) so the model learns what should not matter.
- **Early stopping**: keep the checkpoint with the best validation loss.
- **More data**: the strongest regulariser, and the reason for the data engine.

**Hyper-parameters** are the settings you choose rather than learn: learning rate, batch size, number of layers, dropout p, weight decay. **Parameters** are learned; **hyper-parameters** are chosen.

## P1.8 Optimisers: smarter steps

Plain gradient descent uses the same step size for every parameter and reacts to every noisy batch. Two improvements are universal.

**Momentum.** Keep a running average of recent gradients and step along it, like a ball rolling downhill that keeps its speed through small bumps: v ← μ·v + g; w ← w − η·v, with μ around 0.9. Noise between batches cancels; consistent directions accumulate.

**Adam.** Keep two running averages per parameter: of the gradient (m, the direction, like momentum) and of the squared gradient (v, how large gradients typically are for this parameter). Step by m / √v. Parameters whose gradients are always tiny get relatively bigger steps, and vice versa, so one learning rate works across very different layers. Because m and v start at zero, early averages are too small; Adam divides by (1 − β₁ᵗ) and (1 − β₂ᵗ) to correct this (**bias correction**). The full equations are in chapter P2.7. **AdamW** is Adam with weight decay applied directly to the weights rather than through the gradient; it is the default for transformers.

**Learning-rate schedules.** The best step size changes during training. At the very start, weights are random and gradients large and erratic, so a big step can wreck the model: **warm-up** raises η linearly from 0 over the first few hundred or thousand steps. Near the end, small steps settle into the minimum: **cosine decay** lowers η smoothly along half a cosine curve. Warm-up then cosine is the standard recipe; exercise 10 implements it.

**Gradient clipping.** If the total size (norm) of the gradient exceeds a threshold, scale it down to the threshold. It prevents one bad batch from throwing the model far away.

## P1.9 Making deep networks trainable: initialisation, normalisation, residuals

Stacking many layers creates a numerical problem: each layer multiplies its input by a matrix, and repeated multiplication makes numbers explode or vanish.

**Initialisation.** If a layer with 1 000 inputs starts with weights drawn from a normal distribution with standard deviation 1, each output is a sum of 1 000 random terms and is about √1000 ≈ 32 times larger than the inputs. After 10 such layers, the numbers in the PyTorch check were around 10¹⁵. With standard deviation 1/√1000 instead, the size stays about 1 through all ten layers. That is why default initialisations scale by 1/√(number of inputs) (**fan-in**).

**Normalisation layers** re-centre and re-scale activations inside the network so they stay in a sensible range during training. **LayerNorm** takes each example's vector of features, subtracts its mean, divides by its standard deviation, then applies a learned scale and shift. **RMSNorm** skips the mean subtraction. **BatchNorm** normalises each feature across the batch instead; it behaves differently in training and evaluation and is rarely used in transformers.

**Residual connections.** Instead of y = f(x), compute y = x + f(x): each block adds a correction to its input. The gradient can flow straight through the "+ x" path, so very deep networks still train. Every transformer block is residual.

## P1.10 Sequences, embeddings and attention

**Tokens and embeddings.** A transformer works on a sequence of vectors called **tokens**. A token can be a word (turned into a vector by a lookup table called an **embedding**), a 14×14-pixel patch of an image (turned into a vector by a linear layer), a joint state, or a learned vector with no input at all (a **query token**). An **embedding** in general means "a vector that represents something"; the space those vectors live in is a **latent space**, and a single such vector is a **latent**.

**Positions.** Attention (below) treats its inputs as an unordered set. To tell the model which token came first, or where a patch sits in the image, you add a **positional encoding**: a vector that depends on the position (fixed sine and cosine patterns, or learned).

**Attention** lets every token gather information from every other token, with learned, input-dependent weights. Each token produces three vectors through three linear layers:

- a **query** q: what am I looking for?
- a **key** k: what do I contain?
- a **value** v: what do I hand over if selected?

For token i, compare its query with every key by dot product (large when they point the same way), divide by √d (d is the vector size; this keeps the scores from growing with d and saturating the softmax), apply softmax to get weights that sum to 1, and output the weighted average of the values:

  output_i = Σ_j softmax_j(q_i · k_j / √d) · v_j

A worked example with three tokens, d = 2, keys equal to queries: q = k = (1, 0), (0, 1), (1, 1); values (10, 0), (0, 10), (5, 5). Token 1's scores against the three keys are (1, 0, 1)/√2 = (0.71, 0, 0.71); softmax gives weights (0.40, 0.20, 0.40); its output is 0.40·(10, 0) + 0.20·(0, 10) + 0.40·(5, 5) = **(6.02, 3.98)**. Token 1 attended mostly to itself and to token 3, whose keys resemble its query. It is a soft dictionary lookup.

**Multi-head attention** runs several attentions in parallel on smaller slices of the vectors (each slice is a **head**), so different heads can look for different things, then concatenates the results and mixes them with one more linear layer.

**Causal mask.** When predicting the future, token i must not see tokens after it. You set the scores for j > i to −∞ before the softmax, so their weights become exactly 0.

**A transformer block** is: LayerNorm → multi-head attention → add the input (residual) → LayerNorm → MLP → add the input (residual). A transformer is a stack of these. SmolVLA's language model, the vision encoders (ViT, DINOv2) and the JEPA predictors are all transformers.

## P1.11 Representations and self-supervised learning

An **encoder** is a network that turns raw input (an image) into a compact vector (a latent) that keeps what matters and drops what does not. A good encoder makes later tasks easy: a small linear layer on top (a **probe**) can read out arm pose or whether the bowl is in place.

Labels are expensive; images are cheap. **Self-supervised learning** trains encoders without labels by inventing a task from the data itself: predict a hidden part of an image from the visible part, or make two augmented views of the same image give similar latents. DINOv2 and V-JEPA are encoders trained this way on millions of images or videos. **Frozen** means its parameters are not updated when you train something on top.

**Generative vs predictive.** A generative model reconstructs the pixels (a video model draws the next frame). A **joint-embedding predictive architecture (JEPA)** predicts the **latent** of the next frame instead of its pixels. The argument (chapter 12.1): most pixels are irrelevant detail (texture, lighting), and spending model capacity reproducing them is wasted. Predicting in latent space keeps only what the encoder keeps.

## P1.12 JEPA world models and collapse

**A world model** predicts what happens next. **Action-conditioned**: given the current observation and an action, predict the next observation, or here its latent:

  z_t = encoder(image_t)  ẑ_{t+1} = predictor(z_t, a_t)  loss = ‖ẑ_{t+1} − encoder(image_{t+1})‖²

**Collapse.** If the encoder is trained by this loss too, there is a trivial solution: the encoder outputs the same vector for every image, the predictor outputs that vector, and the loss is exactly 0. Nothing has been learned. This is **representation collapse**, and every JEPA needs a defence against it. Your W1 run showed a partial version: an effective rank of 5.8 instead of 24–30 means the latents used only a handful of directions (chapter 12.2).

The defences you will be asked to write or explain:

1. **Stop-gradient on the target.** Compute encoder(image_{t+1}) without letting the gradient flow back into it (`detach`): the target is treated as a fixed answer. The encoder is then only trained through the prediction side.
2. **EMA target encoder.** Keep a second copy of the encoder for the target, whose weights are an **exponential moving average** of the online encoder's: target ← 0.99·target + 0.01·online after every step. It changes slowly, giving a stable target. (EMA in general: a running average that weights recent values more, controlled by a decay τ close to 1.)
3. **Variance and covariance penalties (VICReg).** Add terms to the loss that penalise latents whose spread across the batch is too small and whose dimensions are correlated.
4. **SIGReg (LeJEPA / LeWM).** Project the batch of latents onto many random directions; for each direction, test whether the projected values look like a standard normal distribution (mean 0, spread 1, bell-shaped); penalise the deviation. A collapsed encoder fails this test badly. Chapter 12c explains the statistics.
5. **A frozen pretrained encoder** (DINO-WM, LaLa-World): if the encoder is not trained at all, it cannot collapse; only the predictor learns.

## P1.13 Planning with a world model: rollouts, MPC, CEM

**A rollout** is the world model run forward several times: from z_t and a sequence of actions a_t, a_{t+1}, …, predict ẑ_{t+1}, then feed it back in to predict ẑ_{t+2}, and so on for a **horizon** H. Errors compound with every step, which is why short horizons work and long ones do not (Planning Limits, 12b).

**Planning** picks the action sequence whose rollout ends closest to a goal: encode a goal image to z_goal; the **cost** of a candidate sequence is ‖ẑ_{t+H} − z_goal‖².

**Model-predictive control (MPC)**: plan a sequence, execute only the first action (or first few), observe the real result, plan again. Replanning corrects the model's errors.

**The cross-entropy method (CEM)** is the usual way to search for the best sequence without gradients:

1. Keep a Gaussian (a mean and a standard deviation per action dimension and time step).
2. Sample N candidate sequences from it.
3. Roll each out, compute its cost.
4. Keep the k best (the **elites**).
5. Set the new mean and standard deviation to those of the elites.
6. Repeat a few times; return the mean.

A one-dimensional example: find the action a that minimises (a − 3)², starting from mean 0 and standard deviation 2, with 10 samples and 3 elites per round. The mean went 0 → 1.96 → 2.52 → 3.00 → 3.03 while the standard deviation shrank 2 → 1.0 → 0.49 → 0.12. Four rounds of guessing, no gradients. Exercise 23 is this with a predictor inside step 3.

## P1.14 Policies: behaviour cloning, action chunks, flow matching

**A policy** maps an observation (camera images, joint state, an instruction) to an action. **Imitation learning / behaviour cloning** trains it on demonstrations: the loss compares the policy's action with the demonstrator's. ACT, SmolVLA and TurboVLA are all trained this way.

**Compounding error.** A cloned policy makes small mistakes; each takes it slightly away from the states in the demonstrations; there it makes bigger mistakes. This drift is the central weakness of behaviour cloning and the reason recovery data and judges matter (book 16.5a).

**Action chunks.** Instead of one action per step, the policy predicts the next K actions at once (K = 50 for SmolVLA). Fewer decisions per second means fewer chances to drift, smoother motion, and time to compute the next chunk while the current one plays. The cost is the **boundary**: where one chunk ends and the next begins, the two may disagree, and the arm jumps (chapter 11.4).

**Why not just regress the action with MSE?** Demonstrations are often **multimodal**: for the same scene, the operator sometimes went left of an obstacle and sometimes right. MSE regression predicts the average of the two, straight into the obstacle. The policy has to represent a distribution over actions, not one answer.

**Flow matching** does this. Start from random noise x₀ (a sample from a standard normal distribution) and move it to a real action x₁ along a straight line, x_t = (1 − t)·x₀ + t·x₁ for t from 0 to 1. Train a network v(x_t, t) to predict the **velocity** along that line, x₁ − x₀, from x_t and t (MSE loss). At inference, start from fresh noise and follow the predicted velocity in N small steps of size 1/N (the **Euler method**: x ← x + v·(1/N)). Different starting noise lands on different modes, so the policy can go left or right but not into the middle. One-dimensional check: if every demonstration were the single value 2, the ideal velocity at t = 0 from x₀ = 0.5 is 2 − 0.5 = 1.5, and one Euler step of size 1 lands exactly on 2. **Diffusion** is the older cousin of the same idea with curved paths and more steps. SmolVLA's action expert uses flow matching; N controls how much compute each chunk costs.

## P1.15 Statistics for evaluation

The coding list includes four statistics items because every claim on the robot rests on them (chapter 7).

**A success rate** k/n from n trials is an estimate. With 7 successes in 10 trials, the true rate could plausibly be anywhere from about 40 % to 89 %. The **Wilson interval** is the formula that gives that range correctly at small n (the naive "± 1.96 × standard error" breaks near 0 % and 100 %).

**The bootstrap** estimates uncertainty for any statistic without a formula: resample your episodes with replacement many times (some episodes appear twice, some not at all), recompute the statistic on each resample, and read the 2.5th and 97.5th percentiles. Resample **episodes**, not frames, for the same reason you split by episode.

**Paired comparisons.** When two policies run on the same layouts, compare them layout by layout: the difference per layout removes the variation between layouts, so you need fewer trials to see a real difference.

**AUROC** measures how well a score separates two groups (failures vs successes) without choosing a threshold: the probability that a randomly chosen positive gets a higher score than a randomly chosen negative. Positives scored (0.9, 0.7, 0.4), negatives (0.5, 0.3): of the six positive-negative pairs, five are ordered correctly, AUROC = **0.83**. 0.5 is chance; 1.0 is perfect separation.

**A baseline** is the simplest thing that could produce the result (copy the last frame, use the commanded joints, use episode length). A model only "works" if it beats its baseline (chapter 3).

## P1.16 Numbers and hardware

**Floating-point formats.** Computers store real numbers with a fixed number of bits. **float32** (32 bits) is the default. **bfloat16** and **float16** use 16 bits: half the memory and much faster on modern GPUs, at the cost of precision. bfloat16 keeps float32's range (it does not overflow) and loses precision; float16 keeps more precision but overflows above 65 504, which is why float16 training needs **loss scaling**. **Mixed precision** runs the heavy matrix multiplications in 16 bits and keeps the weights, the loss and sensitive operations in 32.

**Memory per parameter** during training with Adam is about 16 bytes: the weights, their gradients and Adam's two running averages, plus a 32-bit master copy of the weights in mixed precision. A 450 M-parameter SmolVLA therefore needs about 7 GB before any activations, which is why fine-tuning it on the 16 GB Orin is tight.

**GPUs** run thousands of multiplications in parallel. Everything involved in an operation (model and data) must be on the same device.

**Data-parallel training** runs one copy of the model per GPU, gives each copy a different slice of every batch, and averages the gradients across GPUs after each backward pass (an **all-reduce**), so all copies stay identical. The **effective batch size** is the per-GPU batch times the number of GPUs. **Gradient accumulation** simulates a bigger batch on one GPU by summing gradients over several small batches before taking a step.

## P1.17 The vocabulary, in one place

| Term | Meaning |
|---|---|
| activation | a nonlinear function applied elementwise (ReLU, GELU); also, loosely, any intermediate output of a layer |
| Adam / AdamW | optimiser with per-parameter step sizes; AdamW decouples weight decay |
| attention | each token takes a weighted average of other tokens' values, weights from query–key similarity |
| autograd | PyTorch's automatic backpropagation |
| backpropagation | computing all gradients by applying the chain rule backwards through the computation graph |
| baseline | the cheapest thing that could explain a result |
| batch | B examples processed together |
| behaviour cloning | training a policy to copy demonstrated actions |
| bootstrap | uncertainty by resampling with replacement |
| CEM | sample, keep the best, refit, repeat |
| checkpoint | saved parameters (and optimiser state) |
| chunk | K future actions predicted at once |
| collapse | an encoder producing (almost) the same latent for every input |
| cross-entropy | −log of the probability given to the correct class |
| dropout | randomly zeroing outputs during training |
| EMA | exponential moving average; a slow-moving copy |
| embedding / latent | a vector that represents an input |
| encoder | network from raw input to a latent |
| epoch | one pass over the training data |
| evaluation mode | dropout off, normalisation uses stored statistics |
| flow matching | learning a velocity field that moves noise to data |
| frozen | parameters not updated |
| generalisation | performing well on unseen data |
| gradient | vector of partial derivatives; points uphill |
| head | one of several parallel attention computations |
| horizon | number of steps a rollout predicts |
| hyper-parameter | a setting you choose (learning rate, size), not learned |
| inference | running a trained model to get outputs |
| JEPA | predict the next latent, not the next pixels |
| LayerNorm / RMSNorm | per-example normalisation of a feature vector |
| learning rate | step size of gradient descent |
| logits | raw class scores before softmax |
| loss | one number measuring how wrong the model is |
| MLP | linear layers with activations between them |
| MPC | plan, execute the first action, replan |
| MSE / L1 / Huber | squared / absolute / mixed error losses |
| overfitting | low training loss, high loss on new data |
| parameter / weight | a learned number inside the model |
| policy | observation → action |
| positional encoding | vector added to tokens to mark their position |
| predictor | the part of a world model that maps (latent, action) to the next latent |
| probe | a small model trained on top of frozen latents to test what they contain |
| residual connection | output = input + correction |
| rollout | running a model forward several steps, feeding outputs back in |
| self-supervised | trained without human labels, on a task made from the data |
| SIGReg | penalty pushing latents to look Gaussian along random directions |
| softmax | turns scores into probabilities |
| step | one parameter update |
| stop-gradient / detach | treat a value as a constant during backpropagation |
| tensor | an n-dimensional array with a shape |
| token | one vector in a transformer's input sequence |
| transformer | stack of attention + MLP blocks with norms and residuals |
| validation set | data held out to tune choices |
| warm-up / cosine | raise the learning rate at the start / lower it smoothly to the end |
| weight decay | shrink weights slightly every step |
| world model | predicts the next state from the current state and an action |

## P1.18 Exercises

1. Redo the table of P1.2 by hand for two steps. Then write the gradient-descent loop in plain Python (no PyTorch) and reproduce step 500, and the divergence with η = 0.3.
2. Redo the backward pass of P1.3 on paper with w = (1, 1), x = (2, −1).
3. Compute softmax and cross-entropy for logits (0, 0, 0) and explain the answer without a calculator.
4. Redo the attention example of P1.10 for token 3 and check you get (5.00, 5.00). Why is it exactly the middle?
5. Explain collapse to someone who knows no maths, in three sentences, then name the defence W1 uses.
6. Explain in two sentences why MSE regression fails on a demonstration set that goes left half the time and right half the time, and how flow matching avoids it.
7. Compute AUROC for positives (0.8, 0.6) and negatives (0.7, 0.2, 0.1) by counting pairs.

Then go to chapter P2 and run its code.
