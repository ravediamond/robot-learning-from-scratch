# TensorRT on the Jetson Orin (reference, not runnable on a Mac)

Written from NVIDIA's documentation, not executed by the author of this file. JetPack versions change the
details; check `trtexec --help` on your device.

1. Export ONNX on any machine (today's `export_onnx`), copy it to the Orin with the parity inputs (`.npy`).
2. Build an engine on the Orin itself (engines are specific to the GPU and TensorRT version):
   `/usr/src/tensorrt/bin/trtexec --onnx=policy.onnx --saveEngine=policy_fp16.plan --fp16 --minShapes=obs:1x<D> --optShapes=obs:1x<D> --maxShapes=obs:4x<D>`
   (the shape flags are needed because the batch axis is dynamic; replace `obs` and `<D>` with your input name and size).
3. `trtexec --loadEngine=policy_fp16.plan --shapes=obs:1x<D>` prints latency percentiles: compare with your
   W12 D4 numbers for the PyTorch path (same power mode, warm).
4. Parity: run the engine on the recorded observations (Python bindings or `polygraphy run policy.onnx --trt --onnxrt`)
   and compare outputs. fp16 adds error around 1e-3 relative: check the effect on ACTIONS in joint units and on
   closed-loop success, not only on the tensor.
5. Vision backbones and attention are the usual sources of unsupported or slow ops: check the build log for layers
   falling back, and profile with `--dumpProfile`.
6. Record in the manifest: TensorRT version, precision, shapes, engine file hash, parity numbers.
