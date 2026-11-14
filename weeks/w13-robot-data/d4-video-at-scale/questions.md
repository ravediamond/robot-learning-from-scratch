# Questions (out loud)

1. Estimate storage for 10,000 hours of 3-camera teleoperation, raw and compressed. What dominates the cost?
2. Why can video decoding become the training bottleneck, and what are three fixes?
3. Map-style vs iterable (streaming) datasets: when do you need the second, and what changes for shuffling and epochs?
4. How do you make a streaming loader reproducible and resumable mid-epoch?
5. What does LeRobot store per dataset (v3 format), and why video files plus parquet?
6. A shuffle buffer of 1,000 on a stream ordered by episode: what does the model see, and why can it hurt?
