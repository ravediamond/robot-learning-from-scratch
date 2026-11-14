# Questions (out loud)

1. Derive the policy gradient in three lines (the log-derivative trick). Why does no gradient flow through the environment?
2. Why does subtracting a baseline keep the gradient unbiased? What property must the baseline have?
3. GAE: what do λ = 0 and λ = 1 give, and which way do bias and variance move between them?
4. `done` because the task ended vs `done` because of a time limit: why should the value bootstrap differ?
5. Interviewer (Skild-style): "Rewards are sparse: 1 at success, else 0. Your REINFORCE agent learns nothing. Why, and what are three ways out?"
6. Why is on-policy policy gradient a poor fit for a real robot arm, and what do people do instead?
