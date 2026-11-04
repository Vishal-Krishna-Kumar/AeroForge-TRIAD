# (1) Redraw the split-variability figure with a legend instead of overlapping text labels (new file *_v3b.png; the
# original is kept). Inputs only results/v3_paper_figs/numbers.json. (2) Append the theory/PAC/figures row to the
# experiment log (log sha must start 26456163).
import hashlib, json, math
from pathlib import Path
import numpy as np
from scipy.stats import betabinom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = Path("."); LOG = R / "docs/experiment_log.md"; OUT = R / "results/v3_paper_figs"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
b = LOG.read_bytes(); h = hashlib.sha256(b).hexdigest()
assert h.startswith("26456163"), "LOG CHANGED " + h[:8]
NJ = OUT / "numbers.json"; N = json.loads(NJ.read_text()); T, P = N["theory"], N["pac"]

INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
for ax, name in zip(axes, ("primary_20x20", "secondary_39x38")):
    t = T[f"{name} a0.1"]; n, m, k, obs = t["n"], t["m"], t["k"], t["observed"]
    bb = betabinom(m, k, n + 1 - k); xs = np.arange(m + 1)
    ax.bar(xs / m, bb.pmf(xs), width=0.8 / m, color="#93c5fd")
    ax.axvline(obs, color="#dc2626", lw=2); ax.axvline(0.85, color=MUTED, ls=":", lw=1)
    ax.legend(handles=[Line2D([], [], color="#dc2626", lw=2, label=f"observed split ({obs:.2f})"),
                       Line2D([], [], color=MUTED, ls=":", lw=1, label="pass rule (0.85)")],
              frameon=False, loc="upper left", fontsize=8)
    ax.set_xlim(0.4, 1.02); ax.set_xlabel("whole-run coverage of one calibration split")
    ax.set_title(f"{n} calibration / {m} test runs (α = 0.1)", fontsize=10, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("probability (exchangeable runs)")
fig.tight_layout(); F = OUT / "fig_split_variability_v3b.png"
assert not F.exists(), "exists"
fig.savefig(F, dpi=200); plt.close(fig)

f = lambda x: "∞" if x is None or (isinstance(x, float) and math.isinf(x)) else f"{x:.3f}"
th = "; ".join(f"{key}: mean {v['mean']:.3f}, 5–95 % {v['q05']:.3f}–{v['q95']:.3f}, P(pass rule) {v['P_pass_rule']:.3f}, "
               f"observed {v['observed']:.3f} at CDF {v['observed_cdf']:.3f}" for key, v in T.items())
pc = "; ".join(f"{key}: k' {v['k_pac']}, q̂_PAC {f(v['qhat_pac'])} (marginal q̂ {f(v['qhat_marginal'])}), whole-run "
               f"{v['whole_run']:.3f}, certified {v['certified_fraction']:.3f}" for key, v in P.items())
figs = ", ".join(f"{p.name} {sha(p)[:12]}" for p in sorted(OUT.glob("*.png")))
row = ("| 19:34–19:39 Oct 25 + this row (UTC) | Colab CPU (notebook 24; _staging/v3_theory_pac_figs.py, "
       "_staging/finish_theory_figs.py) | **Descriptive (not pre-registered): analytic single-split variability, PAC "
       f"run-max variant, V3 paper figures.** results/v3_paper_figs/numbers.json sha {sha(NJ)[:16]}…. (1) For exchangeable "
       "runs the number of covered test runs of one calibration split is Beta-Binomial(m, k, n+1−k), k = ⌈(n+1)(1−α)⌉: "
       f"{th}. So even an exactly valid certificate fails the pre-registered pass rule (≥ 1−α−0.05) in ≈17–27 % of "
       "single splits at these sizes; the secondary observations lie in the lower tail (CDF ≈ 0.05 / 0.09) but are not "
       f"outside it. (2) PAC (training-conditional, δ = 0.1) run-max: {pc}. With 20 calibration runs a (0.1, 0.1) PAC "
       "certificate is impossible (k' > n); with 39 runs it restores coverage at the cost of informativeness. "
       f"(3) Figures: {figs}; fig_split_variability_v3b replaces v3 in the paper (legend instead of overlapping "
       "labels; same data). | Descriptive | — |\n")
LOG.write_bytes(b + row.encode("utf-8"))
print("FIG", F.name, sha(F)[:12]); print("LOG", h[:8], "->", sha(LOG)[:8], flush=True)
