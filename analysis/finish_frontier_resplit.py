# (1) Redraw the frontier figure without direct labels (labels collided; run-max / PAC / per-step Bonferroni get their
# own legend markers) -> fig_frontier_v3b.png (original kept). Same inputs as v3_frontier_resplit.py.
# (2) Append the frontier + H2'' re-split row to the experiment log (sha must start 44d1391c); numbers read from JSON.
import hashlib, json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = Path("."); LOG = R / "docs/experiment_log.md"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
b = LOG.read_bytes(); h = hashlib.sha256(b).hexdigest()
assert h.startswith("44d1391c"), "LOG CHANGED " + h[:8]
F = R / "results/v3_paper_figs/fig_frontier_v3b.png"; assert not F.exists(), "exists"
B = json.loads((R / "results/v3_baselines/baselines.json").read_text())["designs"]["secondary_39x38"]["alphas"]
P = json.loads((R / "results/v3_paper_figs/numbers.json").read_text())["pac"]
STY = {"run-max (Thm 1)": ("#2563eb", "o", "#2563eb"), "PAC run-max": ("#2563eb", "D", "#2563eb"),
       "per-step Bonferroni (q̂ = ∞)": ("white", "o", "#2563eb"),
       "conformal, no trajectory guarantee": ("#d97706", "s", "#d97706"),
       "Gaussian ensemble (uncalibrated)": ("#6b7280", "^", "#6b7280"), "adaptive conformal (ACI)": ("#059669", "P", "#059669")}
def style_of(k):
    if k.startswith("run-max"): return "run-max (Thm 1)"
    if k == "PAC run-max": return "PAC run-max"
    if k.startswith("B2a"): return "per-step Bonferroni (q̂ = ∞)"
    if k.startswith("B1"): return "Gaussian ensemble (uncalibrated)"
    if k.startswith("B4"): return "adaptive conformal (ACI)"
    return "conformal, no trajectory guarantee"
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), sharey=True)
for ax, a in zip(axes, ("0.1", "0.2")):
    m = dict(B[a]); pac = P[f"secondary_39x38 a{a}"]
    m["PAC run-max"] = {"whole_run": pac["whole_run"], "certified_fraction": pac["certified_fraction"]}
    for k, v in m.items():
        fc, mk, ec = STY[style_of(k)]
        ax.scatter(100 * v["certified_fraction"], v["whole_run"], s=60, facecolor=fc, marker=mk,
                   edgecolor=ec if fc == "white" else "white", linewidth=1.5, zorder=4 if style_of(k) in ("run-max (Thm 1)", "PAC run-max") else 3)
    ax.axhline(1 - float(a), color=INK, ls="--", lw=1); ax.text(99, 1 - float(a) + 0.015, f"target {1 - float(a):.1f}", ha="right", fontsize=8)
    ax.set_xlim(-3, 100); ax.set_ylim(0, 1.05); ax.set_xlabel("certified test steps (%)")
    ax.set_title(f"α = {a}, 39 calibration / 38 test runs", fontsize=10, loc="left")
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("fraction of test runs fully covered")
fig.legend(handles=[Line2D([], [], ls="", marker=mk, markerfacecolor=fc, markeredgecolor=ec, markersize=7, label=k)
                    for k, (fc, mk, ec) in STY.items()], frameon=False, ncol=3, loc="upper center", fontsize=8.5)
fig.tight_layout(rect=(0, 0, 1, 0.84)); fig.savefig(F, dpi=200); plt.close(fig)

Q = R / "results/v3_resplit_h2/h2_resplit.json"; H = json.loads(Q.read_text())["kinds"]
hp = "; ".join(f"{k.upper()} α {a}: observed {d['observed']:.3f}, exact single-split 5–95 % {d['theory_q05']:.3f}–{d['theory_q95']:.3f}, "
               f"P(pass rule) {d['theory_P_pass_rule']:.3f}, observed CDF {d['theory_observed_cdf']:.3f} (re-split percentile "
               f"{d['observed_resplit_percentile']:.3f}, re-split 5–95 % {d['resplit_q05']:.3f}–{d['resplit_q95']:.3f})"
               for k, o in H.items() for a, d in o["alphas"].items())
row = ("| 01:44 Oct 26 + this row (UTC) | Colab CPU (notebook 24; _staging/v3_frontier_resplit.py, _staging/finish_frontier_resplit.py) "
       "| **Descriptive (not pre-registered): H2″ re-split check and frontier figure.** results/v3_resplit_h2/h2_resplit.json sha "
       f"{sha(Q)[:16]}…; script asserts the recomputed original-split coverages equal results/v3_analysis/v3_analysis.json. "
       f"Cal vs test score Mann–Whitney p: FNO {H['fno']['mwu_cal_vs_test_p']:.2f}, MLP {H['mlp']['mwu_cal_vs_test_p']:.2f}. {hp}. "
       "Reading: with 128/64 points a valid split-conformal certificate meets the V2 rule in ≈ 85–89 % of splits; the FNO miss at "
       "α 0.1 (0.797) lies at ≈ the 2nd percentile of the exchangeable distribution — unusual but with no detected cal/test "
       "difference; MLP is typical. The pre-registered verdict is unchanged. Figures: fig_frontier_v3.png "
       f"{sha(R / 'results/v3_paper_figs/fig_frontier_v3.png')[:12]} (direct labels collided; superseded), fig_frontier_v3b.png "
       f"{sha(F)[:12]} (used in the paper). | Descriptive | — |\n")
LOG.write_bytes(b + row.encode("utf-8"))
print("FIG", F.name, sha(F)[:12]); print("LOG", h[:8], "->", sha(LOG)[:8], flush=True)
