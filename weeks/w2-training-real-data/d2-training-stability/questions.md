# Questions (out loud)

1. Step-0 CE is 9.2 with 100 classes. Good or bad? What if it is 2.1?
2. Why does Adam need warmup more than SGD? (Think about the second-moment estimate at step 1.)
3. Clip value 1.0 and your logged grad norm is always 50. What is actually happening to your training? Is that bad?
4. The review bug: what does the call to `clip_grad_norm_` do at that point in the code? Why did no error appear?
5. Loss is NaN at step 3,000. Walk through your debugging, using today's tools, in order.
6. Interviewer: "fp16 training of your policy diverges, fp32 does not." Three causes and fixes.
