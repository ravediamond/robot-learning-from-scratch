import torch

from w12d5_report import config_hash, flatten, manifest_diff, results_table, run_manifest, tensor_fingerprint


def test_fingerprint():
    torch.manual_seed(0)
    sd = {"b": torch.randn(3, 4), "a": torch.randn(10_000)}
    fp = tensor_fingerprint(sd)
    assert len(fp) == 64 and fp == tensor_fingerprint({"a": sd["a"].clone(), "b": sd["b"].clone()})
    changed = {k: v.clone() for k, v in sd.items()}
    changed["a"][5_000] += 1e-6
    assert tensor_fingerprint(changed) != fp, "one value in the middle of a large tensor"
    assert tensor_fingerprint({"a": sd["a"].half(), "b": sd["b"]}) != fp
    assert tensor_fingerprint({"a": sd["a"].view(100, 100), "b": sd["b"]}) != fp, "shape is part of the identity"
    assert tensor_fingerprint({"x": sd["b"], "a": sd["a"]}) != fp, "names are part of the identity"
    tensor_fingerprint({"bf": torch.randn(4).bfloat16()})


def test_fingerprint_ignores_memory_layout():
    t = torch.randn(4, 6)
    assert tensor_fingerprint({"w": t.t().contiguous()}) == tensor_fingerprint({"w": t.t()})


def test_config_hash_and_manifest():
    assert config_hash({"a": 1, "b": {"c": 2}}) == config_hash({"b": {"c": 2}, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})
    m = run_manifest("sha", "fp", "v1", {"lr": 1e-4}, {"device": "orin"}, {"torch": "2.8"})
    assert m["config_hash"] == config_hash({"lr": 1e-4})
    assert set(m) == {"git_sha", "checkpoint", "dataset", "config", "config_hash", "hardware", "packages"}


def test_flatten_and_diff():
    assert flatten({"a": {"b": 1, "c": {"d": 2}}, "e": [1, 2]}) == {"a.b": 1, "a.c.d": 2, "e": [1, 2]}
    a = {"git_sha": "x", "config": {"chunk": 50, "steps": 10}, "packages": {"torch": "2.8"}}
    b = {"git_sha": "x", "config": {"chunk": 25, "steps": 10}, "packages": {"torch": "2.8", "lerobot": "0.6.2"}}
    assert manifest_diff(a, b) == [("config.chunk", 50, 25), ("packages.lerobot", None, "0.6.2")]
    assert manifest_diff(a, a) == []


def test_results_table():
    table = results_table([{"policy": "smolvla", "k": 31, "n": 50, "top_failure": "grasp", "p95_ms": 182.4},
                           {"policy": "act", "k": 50, "n": 50, "top_failure": None, "p95_ms": 21.6}])
    assert table == ("| Policy | n | Success (95% CI) | Top failure | p95 latency (ms) |\n"
                     "|---|---|---|---|---|\n"
                     "| smolvla | 50 | 62% (48–74) | grasp | 182 |\n"
                     "| act | 50 | 100% (93–100) | - | 22 |")
