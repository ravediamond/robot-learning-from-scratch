# Questions (out loud)

1. What must be inside the exported graph for a policy, and what usually is not?
2. How do you test export parity, on which inputs, and what tolerance do you accept in fp32, fp16, int8?
3. TorchScript vs ONNX vs TensorRT vs `torch.compile`/AOT: when do you use each on a Jetson?
4. Your exported ACT policy matches to 1e-6 but behaves differently on the robot. Where do you look?
5. What metadata do you attach to a deployed artifact?
6. A VLA with a 500M-parameter vision-language backbone and a flow-matching head: what is hard about exporting it, and what would you do?
