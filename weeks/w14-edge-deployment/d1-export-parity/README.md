# W14 D1: export with numerical parity

Primer: `knowledge/15-edge-deployment.md` §1.

## The one idea

Deployment runtimes (ONNX Runtime, TensorRT) run a **graph**, not your Python. Exporting is where
silent bugs enter:

- **Normalisation left behind.** The policy was trained on normalised state and outputs normalised
  actions. If the stats live in a Python dataloader, the exported graph is wrong on the robot. Put
  normalise → model → unnormalise in ONE module with the stats as **buffers**, and export that.
- **Shapes frozen at export.** Export traces with an example input; its batch size becomes a constant
  unless you declare the axis dynamic. Fine for one arm, broken for two cameras or a batched evaluator.
- **Parity.** Run the same inputs through PyTorch (eval mode) and the runtime and compare: max absolute
  error around 1e-6 in fp32. Anything larger is a bug until proven otherwise (an unsupported op, a
  different default, a train-mode layer).

## Today's build (`w14d1_export.py`)

`DeployablePolicy`, `export_onnx` (legacy exporter: `torch.onnx.export(..., dynamo=False)`), `run_onnx`,
`parity`.

## Staff angle

"Exported graph, parity 3e-7 on 1,000 recorded observations, normalisation inside the graph, dynamic
batch, fingerprint of the source checkpoint in the ONNX metadata": one line in a deployment review
that removes a whole class of field incidents.
