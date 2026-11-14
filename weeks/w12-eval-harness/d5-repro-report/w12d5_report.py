"""W12 D5: fingerprints, run manifests, manifest diffs, results table."""
import hashlib
import json

import torch

from evalkit import wilson


def tensor_fingerprint(state_dict: dict[str, torch.Tensor]) -> str:
    """SHA-256 hex digest. For each key in sorted order, update the hash with: the key (utf-8), str(dtype),
    str(tuple(shape)), and the raw bytes of tensor.detach().cpu().contiguous() (use .numpy().tobytes();
    convert bfloat16 to its raw 16-bit pattern with .view(torch.int16) first, numpy has no bfloat16)."""
    h = hashlib.sha256()
    for key in sorted(state_dict):
        t = state_dict[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        if t.dtype == torch.bfloat16:
            t = t.view(torch.int16)
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def config_hash(config: dict) -> str:
    """SHA-256 hex of json.dumps(config, sort_keys=True) (utf-8): key order must not matter."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def run_manifest(git_sha: str, checkpoint_fp: str, dataset_version: str, config: dict, hardware: dict,
                 packages: dict) -> dict:
    """{"git_sha", "checkpoint": checkpoint_fp, "dataset": dataset_version, "config": config,
       "config_hash": config_hash(config), "hardware": hardware, "packages": packages}"""
    return {"git_sha": git_sha, "checkpoint": checkpoint_fp, "dataset": dataset_version, "config": config,
            "config_hash": config_hash(config), "hardware": hardware, "packages": packages}


def flatten(d: dict, prefix: str = "") -> dict:
    """Nested dicts to dotted keys: {"a": {"b": 1}, "c": 2} -> {"a.b": 1, "c": 2}. Non-dict values are leaves."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def manifest_diff(a: dict, b: dict) -> list[tuple]:
    """Sorted list of (dotted key, value in a, value in b) for every flattened key whose values differ;
    a key missing on one side has value None there."""
    fa, fb = flatten(a), flatten(b)
    return sorted((k, fa.get(k), fb.get(k)) for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))


def results_table(rows: list[dict]) -> str:
    """Markdown table, one line per row in the given order. rows: {"policy", "k", "n", "top_failure", "p95_ms"}.
    Exact format:
    | Policy | n | Success (95% CI) | Top failure | p95 latency (ms) |
    |---|---|---|---|---|
    | smolvla | 50 | 62% (48–74) | grasp | 182 |
    Percents: round(100 * x) for k/n and the two Wilson bounds; the CI dash is an en dash (–); latency rounded
    to an int; top_failure None is written as a dash (-). Lines joined with "\n", no trailing newline."""
    lines = ["| Policy | n | Success (95% CI) | Top failure | p95 latency (ms) |", "|---|---|---|---|---|"]
    for r in rows:
        lo, hi = wilson(r["k"], r["n"])
        pct = round(100 * r["k"] / r["n"])
        lines.append(f"| {r['policy']} | {r['n']} | {pct}% ({round(100 * lo)}–{round(100 * hi)}) | "
                     f"{r['top_failure'] if r['top_failure'] is not None else '-'} | {round(r['p95_ms'])} |")
    return "\n".join(lines)
