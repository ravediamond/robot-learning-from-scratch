# W13 D4: video and storage at scale

Primer: `knowledge/14-robot-data-engineering.md` §4.

## The one idea

Robot datasets are mostly video. LeRobot stores each camera as compressed video (AV1/H.264) next to a
parquet table. Three numbers decide whether training is fed or starved:

1. **Storage.** Raw frames are huge (640×480×3 bytes × 30 fps ≈ 27.6 MB/s per camera); compressed video
   is 100–500× smaller. Know both before promising "we keep everything".
2. **Random access cost.** Video is coded in groups of pictures (GOP): to decode frame k you start at the
   previous **keyframe**. With keyframes every g frames, a random frame costs on average (g + 1)/2 decoded
   frames. A training loader that samples random frames across thousands of files is decode-bound;
   shorter GOPs or sampling contiguous windows fix it.
3. **Sharding.** Data is stored in shards (files). With several training processes (ranks) and several
   loader workers each, every (rank, worker) must read a **disjoint** subset of shards, and together they
   must cover all of them. A **shuffle buffer** then mixes samples from the shards being read.

## Today's build (`w13d4_video.py`)

`raw_bytes_per_hour`, `compressed_tb`, `expected_decoded_frames`, `decode_cores_needed`,
`assign_shards`, `ShuffleBuffer`.

## Staff angle

"10,000 hours of three-camera teleoperation is 1.1 TB/hour raw, ~4 TB/1000 h compressed at 1 Mbit/s per
camera; random-frame sampling with 2-second GOPs needs 30 decoded frames per sample, so we sample
contiguous windows and re-encode with 0.5-second GOPs for training": that is a design answer.
