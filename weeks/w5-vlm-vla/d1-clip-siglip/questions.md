# Questions (out loud)

1. Why normalize the embeddings? Why then multiply by a learned scale?
2. Why do CLIP results improve with batch size? What does SigLIP change about that?
3. False negatives (predict items 2–3): how do they bias training here? How would you fix them on a robot dataset where many frames share the same instruction?
4. Your TinyCLIP gets >90% zero-shot. What would break if the test images had a NEW colour? What does that say about "zero-shot" claims?
5. The review bug: what can the encoders do to cheat the loss without normalization?
6. Interviewer: "Would you train a CLIP-style model on your teleop data (frames, instructions)? What for?"
