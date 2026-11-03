# Questions (out loud)

1. Why several heads instead of one big one, at the same parameter count? What would you expect heads in a VLA action expert to specialise in?
2. Count the parameters of MHA(D, h). Why does h not appear in the answer?
3. The review bug: what does "head 0" actually contain? Why does the model still learn?
4. Cross-attention: which side's length can change at inference without retraining, and why?
5. Grouped-query / multi-query attention: what is shared, what does it save at inference, and why do edge VLAs care?
6. Interviewer: "Your attention output matches PyTorch to 1e-3, not 1e-6. Bug or not?"
