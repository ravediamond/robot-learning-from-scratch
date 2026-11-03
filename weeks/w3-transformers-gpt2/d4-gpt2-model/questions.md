# Questions (out loud)

1. Why tie input and output embeddings? When would you not?
2. Why scale the init of `c_proj` by 1/√(2L)? What happens to the residual stream's variance at depth without it?
3. Why does GPT-2 use learned absolute positions, and what does that mean for sequences longer than 1024?
4. Where does the compute go in a GPT-2 forward pass for T = 1024? (attention vs MLP vs lm_head)
5. The review bug: what did `.clone()` + `nn.Parameter` do? How would you have caught it in 10 seconds?
6. Interviewer: "Your VLA is a VLM plus an action expert. Which of today's pieces does SmolVLA keep, cut, or change?"
