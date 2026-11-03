# Questions (out loud). Friday: connect it to your robot.

1. Give your four predict numbers. Which ablation proves the model uses language? Which proves it uses vision?
2. What did the "<pad>" instruction do, and what does that tell you about out-of-distribution instructions on a real VLA?
3. Why is the prefix computed once per act()? What fraction of the inference cost does that save here, and in SmolVLA?
4. Map each part of your MiniVLA to SmolVLA. What is different in scale, and what is different in kind?
5. Why did learned position tables learn too slowly here, and coordinate features fast? Where else do "coordinate features" appear in robotics models?
6. Interviewer: "Your VLA reaches 95% in sim. Design the 5 tests you would run before claiming it follows instructions on a real arm."
