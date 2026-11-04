"""Paper figures, numbers and tables for V1/V2, generated from the result JSONs only (no hand-copied numbers).

Usage: MPLBACKEND=Agg python make_paper_figures_v2.py --root <AeroForge-TRIAD root> --out <new output dir>
Inputs are located by file name under <root>/results (files named *_v2partial* are ignored; every name must match
exactly one file, otherwise the script stops). Outputs (all new): fig_*.png / fig_*.pdf, numbers_v2.json (every value
used, with the sha256 of each input file) and table_v2_runlevel.md (the paper's run-level table).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

INPUTS = {
    "h1_v1_fno": "h1_test_metrics_fno.json", "h1_v1_mlp": "h1_test_metrics_mlp.json",
    "h1_v2_fno": "h1v2_test_metrics_fno.json", "h1_v2_mlp": "h1v2_test_metrics_mlp.json",
    "h2_v2_fno": "h2v2_iid_fno.json", "h2_v2_mlp": "h2v2_iid_mlp.json",
    "s2_main": "s2_h5_h6_h7.json", "s2_base": "s2_baselines_descriptive.json",
    "s2_cert": "s2_certified_fraction_descriptive.json",
}
# Reference categorical palette (light mode), fixed order; neutral and text tokens.
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a3a29d"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def find_inputs(root: Path) -> dict:
    out = {}
    for key, name in INPUTS.items():
        hits = [p for p in (root / "results").rglob(name) if "_v2partial" not in p.name]
        if len(hits) != 1:
            raise SystemExit(f"{name}: expected exactly one file under results/, found {[str(h) for h in hits]}")
        out[key] = hits[0]
    return out


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, out: Path, name: str) -> None:
    fig.patch.set_facecolor(SURFACE)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_{name}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    files = find_inputs(root)
    d = {k: json.loads(p.read_text()) for k, p in files.items()}
    nums = {"inputs": {k: {"path": str(p.relative_to(root)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                       for k, p in files.items()}}

    # Figure 1: value vs gradient error (ensemble medians, i.i.d. test sets) -- dumbbell on a log axis.
    rows = [("V1 FNO", "h1_v1_fno"), ("V1 MLP", "h1_v1_mlp"), ("V2 FNO", "h1_v2_fno"), ("V2 MLP", "h1_v2_mlp")]
    nums["h1"] = {lab: {k: d[key]["ensemble"][k] for k in ("median_rel_err_LD", "median_rel_grad_err_LD", "median_cos",
                                                          "frac_cos_below_0.98", "n_grad_points")} for lab, key in rows}
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    style(ax)
    for i, (lab, _) in enumerate(rows):
        v, g = nums["h1"][lab]["median_rel_err_LD"] * 100, nums["h1"][lab]["median_rel_grad_err_LD"] * 100
        ax.plot([v, g], [i, i], color=GRID, linewidth=2, zorder=1)
        ax.scatter([v], [i], s=46, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3, label="L/D value" if i == 0 else None)
        ax.scatter([g], [i], s=46, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3, label="L/D gradient (M-norm)" if i == 0 else None)
        ax.text(g * 1.12, i, f"{g / v:.0f}×", va="center", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("median relative error on the i.i.d. test set (%)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK2)
    save(fig, out, "value_vs_gradient_error")

    # Figure 2: whole-run coverage of every method at both alpha (20 test runs, Clopper-Pearson 95 % CI).
    methods = [("i.i.d. split conformal", "main", "H7_iid"), ("pooled trajectory points", "main", "pooled_baseline"),
               ("run-max (Thm 1)", "main", "H5_runmax"), ("Gaussian ensemble UQ*", "base", "gaussian"),
               ("Bonferroni α/15*", "base", "bonferroni"), ("weighted conformal*", "base", "weighted"),
               ("ACI γ=0.01*", "base", "aci_gamma0.01"), ("ACI γ=0.05*", "base", "aci_gamma0.05")]
    nums["whole_run"] = {}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    for ax, al in zip(axes, ("0.1", "0.2")):
        style(ax)
        tgt = 1 - float(al)
        for j, (lab, src, key) in enumerate(methods):
            r = (d["s2_main"] if src == "main" else d["s2_base"])["alphas"][al][key]
            w, (lo, hi) = r["whole_run"], r["whole_run_CI95"]
            nums["whole_run"].setdefault(al, {})[lab] = {"whole_run": w, "CI95": [lo, hi],
                                                       "per_iterate": r["per_iterate"],
                                                       "threshold": r.get("qhat", r.get("median_threshold"))}
            c = BLUE if key == "H5_runmax" else GRAY
            ax.barh(j, w, height=0.55, color=c, zorder=2)
            ax.plot([lo, hi], [j, j], color=INK2, linewidth=1.2, zorder=3)
        ax.axvline(tgt, color=INK, linewidth=1, linestyle="--", zorder=4)
        ax.axvline(tgt - 0.05, color=INK2, linewidth=0.8, linestyle=":", zorder=4)
        ax.set_xlim(0, 1.02)
        ax.set_title(f"α = {al}  (target {tgt:.1f}, dotted: rule {tgt - 0.05:.2f})", fontsize=9, color=INK)
        ax.set_xlabel("fraction of test runs with every iterate covered", color=INK2, fontsize=8)
    axes[0].set_yticks(range(len(methods)), [m[0] for m in methods])
    axes[0].invert_yaxis()
    fig.text(0.01, -0.09, "* descriptive, not pre-registered. Bars: 20 test runs; whiskers: Clopper–Pearson 95 % CI.",
             fontsize=7, color=INK2)
    save(fig, out, "whole_run_coverage")

    # Figure 3: what a certificate buys -- share of steps certified vs share of certified steps actually covered.
    cert = [("i.i.d.", "iid"), ("pooled", "pooled"), ("run-max", "runmax"), ("random iterate", "random_iterate_median")]
    nums["certified"] = {al: {lab: d["s2_cert"]["alphas"][al][k] for lab, k in cert} for al in ("0.1", "0.2")}
    nums["certified"]["frac_all_steps_true_ascent"] = d["s2_cert"]["frac_all_steps_true_ascent"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    for ax, al in zip(axes, ("0.1", "0.2")):
        style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for lab, k in cert:
            r = d["s2_cert"]["alphas"][al][k]
            x, y = r["frac_steps_certified"], r["frac_certified_steps_covered"]
            ax.scatter([x], [y], s=60, color=BLUE if k == "runmax" else GRAY, edgecolor=SURFACE, linewidth=2, zorder=3)
            dx, dy, ha = {"random iterate": (-7, -12, "right"), "pooled": (7, -9, "left")}.get(lab, (6, -3, "left"))
            ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK2, ha=ha)
        ax.set_xlim(0.3, 1.0)
        ax.set_ylim(0.5, 1.03)
        ax.set_title(f"α = {al}", fontsize=9, color=INK)
        ax.set_xlabel("share of test-run steps certified", color=INK2, fontsize=8)
    axes[0].set_ylabel("share of certified steps actually covered", color=INK2, fontsize=8)
    save(fig, out, "certified_fraction")

    # Figure 4: i.i.d. split-conformal coverage (H2'), pre-registered split, both models.
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    style(ax)
    nums["h2v2"] = {}
    for i, (lab, key) in enumerate((("FNO", "h2_v2_fno"), ("MLP", "h2_v2_mlp"))):
        for r in d[key]["preregistered_split"]:
            y = i + (0.15 if r["alpha"] == 0.2 else -0.15)
            c = BLUE if r["alpha"] == 0.1 else ORANGE
            ax.plot(r["coverage_CI95"], [y, y], color=c, linewidth=1.2)
            ax.scatter([r["coverage"]], [y], s=40, color=c, edgecolor=SURFACE, linewidth=2, zorder=3,
                       label=f"α = {r['alpha']}" if i == 0 else None)
            nums["h2v2"].setdefault(lab, {})[str(r["alpha"])] = {k: r[k] for k in ("coverage", "coverage_CI95", "qhat", "n_cal", "n_test")}
    for t in (0.9, 0.8):
        ax.axvline(t, color=INK2, linewidth=0.8, linestyle="--")
    ax.set_yticks([0, 1], ["V2 FNO", "V2 MLP"])
    ax.invert_yaxis()
    ax.set_xlabel("i.i.d. test coverage (95 % CI); dashed: targets", color=INK2, fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK2)
    save(fig, out, "iid_coverage_v2")

    # Table: run-level study (paper §5.5), every cell from the JSONs.
    lines = ["| method (test runs, n = 20) | α 0.1 whole-run (CP 95 %) | per-iterate | threshold | α 0.2 whole-run (CP 95 %) | per-iterate | threshold |",
             "|---|---|---|---|---|---|---|"]
    for lab, _, _ in methods:
        cells = []
        for al in ("0.1", "0.2"):
            r = nums["whole_run"][al][lab]
            cells += [f"{r['whole_run']:.2f} ({r['CI95'][0]:.2f}–{r['CI95'][1]:.2f})", f"{r['per_iterate']:.3f}", f"{r['threshold']:.2f}"]
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("\\* descriptive, not pre-registered. Generated by scripts/paper/make_paper_figures_v2.py from: "
                 + ", ".join(f"{v['path']} ({v['sha256'][:8]}…)" for k, v in nums["inputs"].items() if k.startswith("s2_")))
    (out / "table_v2_runlevel.md").write_text("\n".join(lines) + "\n")
    (out / "numbers_v2.json").write_text(json.dumps(nums, indent=1, ensure_ascii=False))
    print("PAPER_OUT", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
