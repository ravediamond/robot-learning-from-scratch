#!/usr/bin/env python3
"""WS-076: render the 4-page edge-inference draft and its figures from the result files (no hand-typed number).

    python3 journal/papers/edge-inference/make_paper.py

Reads the gates' result files (road-to-robotics `shared/gate/data/`, `projects/*/results/`, and the paper-vs-hardware
`expected/` folders, PVH_REPO or ~/workspace/git/paper-vs-hardware), writes `numbers.json` (every value with its anchor),
`fig/*.pdf|png`, and renders `DRAFT_4p.src.md` -> `DRAFT_4p.md` and `main.src.tex` -> `main.tex` by replacing
{{key}} placeholders. A placeholder without a value, or a number typed into a template outside a placeholder in the
results / abstract sections, is an error (check_templates).
"""
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RR = HERE.parents[2]
PVH = Path(os.environ.get("PVH_REPO", Path.home() / "workspace/git/paper-vs-hardware")) / "posts"
FIG = HERE / "fig"


def J(p):
    return json.loads(Path(p).read_text())


N, A = {}, {}  # values (already formatted) and their anchors


def put(key, value, anchor, fmt="{:.1f}"):
    N[key] = fmt.format(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)
    A[key] = anchor


def collect():
    g = RR / "shared/gate/data"
    glat = J(g / "2026-10-20_glat2/glat2_gpu640_summary.json")["vla"]
    a = "G-lat-2, shared/gate/data/2026-10-20_glat2/glat2_gpu640_summary.json (mean of 400 chunks)"
    put("glat_total", glat["server_total_ms"]["mean"], a); put("glat_vision", glat["server_vision_ms"]["mean"], a)
    put("glat_rest", glat["server_inference_ms"]["mean"], a)
    put("glat_vision_pct", 100 * glat["server_vision_ms"]["mean"] / glat["server_total_ms"]["mean"], a, "{:.0f}")

    hd = PVH / "2026-10-30-paper-vs-hardware-horizon-depth/expected"
    a2 = J(hd / "a2_latency.json"); a = "horizon x depth A2, paper-vs-hardware 2026-10-30-paper-vs-hardware-horizon-depth/expected/a2_latency.json (MAXN_SUPER, median of 50 calls)"
    n10, n2 = a2["by_n"]["10"]["median"]["rtt"], a2["by_n"]["2"]["median"]["rtt"]
    put("n10_ms", n10, a); put("n2_ms", n2, a); put("n2_saved_ms", n10 - n2, a, "{:.0f}")
    put("n2_saved_pct", 100 * (n10 - n2) / n10, a, "{:.0f}"); put("n2_over_n10", a2["n2_over_n10"], a, "{:.2f}")
    put("step_ms", a2["split"]["per_step_denoise_ms"], a); put("vision_ms_a2", a2["split"]["vision_ms"], a)
    put("prefill_ms", a2["split"]["prefill_ms"], a)
    a1 = J(hd / "a1_cloud.json"); a = "horizon x depth A1, .../expected/a1_cloud.json (20 frames x 32 draws)"
    r = a1["summary"]["2"]["R_ratio"]
    put("r2_ratio", r["median"], a, "{:.2f}"); put("r2_lo", r["ci95"][0], a, "{:.2f}"); put("r2_hi", r["ci95"][1], a, "{:.2f}")
    put("r2_verdict", a1["verdict_A1"], a)

    o = J(g / "2026-10-19_g13b_offline/g13b_summary.json")["results"]["beta=5"]; pf = J(g / "2026-10-19_g13b_offline/g13b_preflight.json")
    a = "G13b, shared/gate/data/2026-10-19_g13b_offline/ (g13b_summary.json, g13b_preflight.json; PyTorch, 25 W, 254 boundaries per delay)"
    for d in ("8", "12", "20"):
        nv = o["summary"]["naive"][d]["jump_ratio"]
        put(f"ow_cut_{d}", 100 * o["decision"]["per_delay"][d]["jump_cut_vs_naive"], a, "{:.0f}")
        put(f"ex_cut_{d}", 100 * (1 - o["summary"]["exact"][d]["jump_ratio"] / nv), a, "{:.0f}")
    put("pt_naive_ms", pf["naive"], a, "{:.0f}"); put("pt_exact_ms", pf["exact_d8"], a, "{:.0f}")
    put("exact_over_naive", pf["exact_d8"] / pf["naive"], a, "{:.1f}")
    v = J(g / "2026-10-19_vlacpp_prefix_overwrite/vlacpp_overwrite_summary.json")
    a = "G13b in vla.cpp, shared/gate/data/2026-10-19_vlacpp_prefix_overwrite/vlacpp_overwrite_summary.json"
    for d in ("8", "12", "20"):
        put(f"cpp_cut_{d}", 100 * v["jump_cut_vs_naive"][d], a, "{:.0f}")
    put("cpp_ow_added_ms", v["summary"]["overwrite"]["8"]["ms"] - v["summary"]["naive"]["8"]["ms"], a, "{:.1f}")

    pa = PVH / "2026-10-30-paper-vs-hardware-pact/expected"
    ga, gb = J(pa / "gate_a.json"), J(pa / "gate_b.json")["k8"]; a = "PACT-VLA gates A / B, .../2026-10-30-paper-vs-hardware-pact/expected/gate_a.json, gate_b.json"
    put("pact_ms", ga["T_PACT_ms"], a, "{:,.0f}"); put("pact_tgen1", ga["T_gen_1_ms"], a)
    put("pact_d", ga["d_PACT"], a, "{:d}"); put("naive_d", ga["d_naive"], a, "{:d}")
    put("pact_sel", gb["pact_minus_random"]["mean"], a, "{:.2f}")
    put("pact_sel_lo", gb["pact_minus_random"]["ci95"][0], a, "{:.2f}"); put("pact_sel_hi", gb["pact_minus_random"]["ci95"][1], a, "{:.2f}")

    ac = PVH / "2026-10-30-paper-vs-hardware-adaptive-chunking/expected"
    g1, g0 = J(ac / "gate1_latency.json"), J(ac / "gate0.json"); a = "AAC gates 0 / 1, .../2026-10-30-paper-vs-hardware-adaptive-chunking/expected/gate0.json, gate1_latency.json"
    put("aac_n20_ms", g1["A4_n20_ms"], a, "{:,.0f}"); put("aac_bar_ms", g1["A4_bar_ms"], a, "{:.0f}")
    put("aac_h_transport", g0["A1"]["median_h_transport"], a, "{:.0f}"); put("aac_h_grasp", g0["A1"]["median_h_grasp"], a, "{:.0f}")
    put("aac_jump", g0["A3"]["aac+overwrite_d8"], a, "{:.3f}"); put("aac_fixed_jump", g0["A3"]["fixed+overwrite_d8"], a, "{:.3f}")

    s = J(RR / "projects/smooth-chunking/results/a_summary.json"); a = "smooth MP chunking gate A, projects/smooth-chunking/results/a_summary.json (20 episodes)"
    put("sm_err", s["A1"]["50"]["error_S2_minus_S0"]["mean"], a, "{:+.3f}")
    put("sm_jerk", 100 * (1 - s["A1"]["50"]["within_jerk_S2_over_S0"]), a, "{:.0f}")
    put("sm_seam25", 100 * (1 - s["A2"]["25"]["boundary_jerk_S3_over_S2"]), a, "{:.0f}")
    put("sm_seam50", 100 * (1 - s["A2"]["50"]["boundary_jerk_S3_over_S2"]), a, "{:.0f}")
    put("sm_cost", s["A2"]["50"]["error_S3_minus_S2"]["mean"], a, "{:+.2f}")

    for k, h in (("bar_hd", "0a79846"), ("bar_pact", "0a79846"), ("bar_aac", "581a955"), ("bar_sm", "6f58d7b"), ("bar_glat", "82fdc15")):
        put(k, h, "first commit of the gate's protocol (git log --diff-filter=A)")


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 8, "figure.dpi": 150})
    # 1. where a chunk's time goes (A2 split at N = 10)
    v, p, st = float(N["vision_ms_a2"]), float(N["prefill_ms"]), float(N["step_ms"])
    fig, ax = plt.subplots(figsize=(3.3, 1.35))
    left = 0
    for lab, w, c in (("vision", v, "0.25"), ("prefill", p, "0.55"), ("10 denoise steps", 10 * st, "0.8")):
        ax.barh(0, w, left=left, color=c, edgecolor="black", linewidth=0.4)
        if w < 25:  # narrow segment (prefill): label above the bar
            ax.annotate(f"{lab} {w:.0f} ms", (left + w / 2, 0.4), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=6,
                        arrowprops=dict(arrowstyle="-", lw=0.5))
        else:
            ax.text(left + w / 2, 0, f"{lab}\n{w:.0f} ms", ha="center", va="center", fontsize=6.5, color="white" if c == "0.25" else "black")
        left += w
    ax.set_yticks([]); ax.set_xlabel("ms per SmolVLA chunk, vla.cpp, Orin NX (MAXN_SUPER)"); ax.set_xlim(0, left * 1.02)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "latency_split.pdf"); fig.savefig(FIG / "latency_split.png"); plt.close(fig)
    # 2. boundary jump cut per method and delay
    ds = ["8", "12", "20"]
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    for i, (lab, key, c) in enumerate((("prefix overwrite (PyTorch)", "ow_cut", "0.2"), ("prefix overwrite (vla.cpp)", "cpp_cut", "0.5"),
                                         ("exact RTC guidance (PyTorch)", "ex_cut", "0.85"))):
        ax.bar([j + (i - 1) * 0.27 for j in range(3)], [float(N[f"{key}_{d}"]) for d in ds], 0.27, color=c, edgecolor="black", linewidth=0.4, label=lab)
    ax.axhline(50, ls="--", lw=0.8, color="black"); ax.text(2.45, 51, "bar 50 %", fontsize=6, ha="right")
    ax.set_xticks(range(3)); ax.set_xticklabels([f"d = {d}" for d in ds]); ax.set_ylabel("jump cut vs naive (%)")
    ax.legend(fontsize=5.5, frameon=False, loc="upper left", ncol=1); ax.set_ylim(0, 120)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "jump_cut.pdf"); fig.savefig(FIG / "jump_cut.png"); plt.close(fig)
    # 3. cost per decision vs a 238 ms chunk (log scale)
    rows = [("1 chunk (N = 10)", float(N["n10_ms"])), ("2 Euler steps", float(N["n2_ms"])),
            ("PACT cycle (4 candidates)", float(N["pact_ms"].replace(",", ""))), ("AAC decision (N = 20)", float(N["aac_n20_ms"].replace(",", "")))]
    fig, ax = plt.subplots(figsize=(3.3, 1.4))
    ax.barh(range(len(rows)), [r[1] for r in rows], color=["0.4", "0.4", "0.8", "0.8"], edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=6.5); ax.invert_yaxis(); ax.set_xscale("log")
    for i, r in enumerate(rows):
        ax.text(r[1] * 1.08, i, f"{r[1]:,.0f} ms", va="center", fontsize=6)
    ax.axvline(float(N["aac_bar_ms"]), ls="--", lw=0.8, color="black"); ax.set_xlabel("ms per decision on the Orin NX (log)")
    ax.set_xlim(100, 20000)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "decision_cost.pdf"); fig.savefig(FIG / "decision_cost.png"); plt.close(fig)


def render(src, dst):
    t = (HERE / src).read_text()
    keys = set(re.findall(r"\{\{(\w+)\}\}", t))
    missing = keys - N.keys()
    assert not missing, f"{src}: placeholders without a value: {sorted(missing)}"
    (HERE / dst).write_text(re.sub(r"\{\{(\w+)\}\}", lambda m: N[m.group(1)], t))


def bib():
    """refs.bib from the reference ids of DRAFT_4p.src.md, titles and years from the papers mirror's index.json."""
    ids = re.findall(r"(\d{4}\.\d{4,5})", (HERE / "DRAFT_4p.src.md").read_text().split("## References")[1])
    idx = {r.get("arxiv"): r for r in json.loads((Path.home() / "workspace/papers/index.json").read_text())["papers"]}
    out = []
    for i in ids:
        r = idx.get(i, {})
        title = (r.get("title") or f"arXiv {i}").replace("&", "\\&")
        out.append(f"@misc{{arxiv{i.replace('.', '')},\n  title = {{{{{title}}}}},\n  year = {{{r.get('year') or '20' + i[:2]}}},\n"
                   f"  eprint = {{{i}}}, archivePrefix = {{arXiv}},\n  note = {{arXiv:{i}}}\n}}")
    (HERE / "refs.bib").write_text("\n\n".join(out) + "\n")
    return len(out)


def main():
    collect()
    nb = bib()
    (HERE / "numbers.json").write_text(json.dumps({k: {"value": N[k], "anchor": A[k]} for k in sorted(N)}, indent=1))
    figures()
    render("DRAFT_4p.src.md", "DRAFT_4p.md")
    render("main.src.tex", "main.tex")
    print(f"{len(N)} numbers, 3 figures, {nb} references, DRAFT_4p.md and main.tex rendered")


if __name__ == "__main__":
    main()
