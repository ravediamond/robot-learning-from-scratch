# Questions (out loud)

1. Your loader runs without error but the logits differ from Hugging Face by 0.3. List what you check, in order.
2. Why is checking "same logits on the same input" a stronger test than "it generates plausible text"?
3. KV cache: what exactly is cached, what is the memory for GPT-2 small at T = 1024, batch 1, fp16? Why does it make generation O(T) per token?
4. Temperature, top-k, top-p: what does each do to the distribution? For a VLA with discrete action tokens, which would you use at deployment and why?
5. Your char model on the book: train vs val loss after 3,000 steps. Overfitting? What does the sample tell you that the loss doesn't?
6. Interviewer: "You fine-tune a VLA from a checkpoint and results are random. The training loss starts at the from-scratch value." What happened and how do you prove it?
