"""DESCRIPTIVE (not pre-registered): (1) exact single-split distribution of whole-run coverage for exchangeable runs,
(2) a training-conditional (PAC) run-max variant, (3) V3 paper figures. Reads only saved results; writes
results/v3_paper_figs/ (refuses to overwrite numbers.json).

(1) For exchangeable runs, n calibration runs and k = ceil((n+1)(1-α)), the coverage of a fresh run given the calibration
set is Beta(k, n+1-k); the number of m test runs covered is Beta-Binomial(m, k, n+1-k). We report its mean, 5/95 % points,
P(covered fraction >= 1-α-0.05) (= the pre-registered pass rule) and the percentile of each observed split.
(2) PAC run-max: the smallest k' such that P(Beta(k', n+1-k') >= 1-α) >= 1-δ (δ = 0.1); q̂ = k'-th smallest calibration
run-max (inf if k' > n). Applied to the primary (20/20) and secondary (39/38) designs: whole-run coverage and certified
fraction (step certified iff ||ĝ||_M > q̂ σ̂, as the pre-registered H9 rule).
"""
import json, math, os
from pathlib import Path
import numpy as np
from scipy.stats import beta as B, betabinom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(os.environ.get("AF_ROOT", "."))
OUT = R / "results/v3_paper_figs"; OUT.mkdir(parents=True, exist_ok=True)
assert not (OUT / "numbers.json").exists(), "exists"
S = R / "results/v3_analysis_secondary"
D = json.loads((R / "datasets/designs/dataset_v3_design.json").read_text())
SPAN = np.asarray(D["box_hi"]) - np.asarray(D["box_lo"])
z = np.load(S / "member_gradients_traj_fno.npz", allow_pickle=True); G, cids = z["G"], [str(c) for c in z["case_id"]]
zi = np.load(S / "member_gradients_iid_fno.npz", allow_pickle=True)


def stats(G):
    gh = G.mean(0)
    sq = np.stack([np.linalg.norm((G[k] - gh) * SPAN, axis=1) ** 2 for k in range(G.shape[0])])
    return gh, np.sqrt(sq.sum(0) / (G.shape[0] - 1))


floor = 1e-6 * float(np.median(stats(zi["G_cal"])[1]))
gh, sg = stats(G)
RAW = R / "datasets/raw/dataset_v3_traj"
gt = []
for c in cids:
    g = json.loads((RAW / c / "gradients.json").read_text()); gt.append([g["objectives"]["J"]["grad"][k] for k in g["dv"]])
gt = np.asarray(gt)
s = np.linalg.norm((gt - gh) * SPAN, axis=1) / np.maximum(sg, floor)
ghn = np.linalg.norm(gh * SPAN, axis=1); sgf = np.maximum(sg, floor)
run = np.array([int(c[4:6]) for c in cids])
rmax = {r: float(s[run == r].max()) for r in sorted(set(run.tolist()))}
sec = json.loads((S / "design_secondary.json").read_text())
designs = {"primary_20x20": (list(range(20)), list(range(20, 40))), "secondary_39x38": (sec["cal_runs"], sec["test_runs"])}
A = json.loads((R / "results/v3_analysis/v3_analysis.json").read_text()); A2 = json.loads((S / "v3_analysis.json").read_text())
observed = {"primary_20x20": {a: A["alphas"][a]["H5_runmax"]["whole_run"] for a in ("0.1", "0.2")},
            "secondary_39x38": {a: A2["alphas"][a]["H5_runmax"]["whole_run"] for a in ("0.1", "0.2")}}
num = {"theory": {}, "pac": {}}
DELTA = 0.1
for name, (cal, test) in designs.items():
    n, m = len(cal), len(test)
    for a in (0.1, 0.2):
        k = math.ceil((n + 1) * (1 - a)); bb = betabinom(m, k, n + 1 - k)
        thr = 1 - a - 0.05; need = math.ceil(thr * m - 1e-9)
        obs = observed[name][str(a)]; x = round(obs * m)
        num["theory"][f"{name} a{a}"] = {"n": n, "m": m, "k": k, "mean": float(bb.mean() / m),
                                         "q05": float(bb.ppf(0.05) / m), "q95": float(bb.ppf(0.95) / m),
                                         "P_pass_rule": float(bb.sf(need - 1)), "observed": obs, "observed_cdf": float(bb.cdf(x))}
        kp = next((kk for kk in range(1, n + 1) if B.sf(1 - a, kk, n + 1 - kk) >= 1 - DELTA), None)
        cv = np.sort([rmax[r] for r in cal]); q = float(cv[kp - 1]) if kp else math.inf
        tm = np.isin(run, test)
        num["pac"][f"{name} a{a}"] = {"k_pac": kp, "qhat_pac": q, "qhat_marginal": float(cv[k - 1]) if k <= n else math.inf,
                                      "whole_run": float(np.mean([rmax[r] <= q for r in test])),
                                      "certified_fraction": float(np.mean(ghn[tm] > q * sgf[tm]))}
(OUT / "numbers.json").write_text(json.dumps(num, indent=1, default=float))
print("NUMBERS", json.dumps(num, default=float)[:2500], flush=True)

# ---- figures (dataviz: few colours, single axis per panel, labelled CIs) ----
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
C = {"run-max": "#2563eb", "i.i.d.": "#d97706", "pooled": "#6b7280", "PAC run-max": "#059669"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# (a) value vs gradient error, V3 test set
fig, ax = plt.subplots(figsize=(5.2, 3.2))
H = {m: json.loads((R / f"results/reports_v3/h1v3_test_metrics_{m}.json").read_text())["ensemble"] for m in ("fno", "mlp")}
x = np.arange(2); w = 0.36
ax.bar(x - w / 2, [H[m]["median_rel_err_J"] * 100 for m in ("fno", "mlp")], w, color="#93c5fd", label="objective J")
ax.bar(x + w / 2, [H[m]["median_rel_grad_err_J"] * 100 for m in ("fno", "mlp")], w, color="#1d4ed8", label="gradient dJ/dθ")
for i, m in enumerate(("fno", "mlp")):
    ax.text(i - w / 2, H[m]["median_rel_err_J"] * 100 + 0.4, f"{H[m]['median_rel_err_J']*100:.1f}%", ha="center", fontsize=9)
    ax.text(i + w / 2, H[m]["median_rel_grad_err_J"] * 100 + 0.4, f"{H[m]['median_rel_grad_err_J']*100:.1f}%", ha="center", fontsize=9)
ax.set_xticks(x, ["FNO ensemble", "MLP ensemble"]); ax.set_ylabel("median relative error (%)")
ax.set_title("Transonic test set (n = 64): values vs gradients", fontsize=10, loc="left"); ax.legend(frameon=False); style(ax)
fig.tight_layout(); fig.savefig(OUT / "fig_value_vs_gradient_v3.png", dpi=200); plt.close(fig)

# (b) whole-run coverage with Clopper-Pearson CIs, primary vs secondary, α = 0.1 and 0.2
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
for ax, a in zip(axes, ("0.1", "0.2")):
    labs, vals, los, his, cols = [], [], [], [], []
    for dn, J in (("primary\n(20 test runs)", A), ("secondary\n(38 test runs)", A2)):
        z_ = J["alphas"][a]
        for meth, key in (("run-max", "H5_runmax"), ("i.i.d.", "H7_iid"), ("pooled", "pooled_baseline_descriptive")):
            v = z_[key]["whole_run"]; lo, hi = z_[key]["whole_run_CI95"]
            labs.append(f"{meth}\n{dn}"); vals.append(v); los.append(v - lo); his.append(hi - v); cols.append(C[meth])
    xs = np.array([0, 1, 2, 3.6, 4.6, 5.6])
    ax.bar(xs, vals, 0.8, color=cols); ax.errorbar(xs, vals, yerr=[los, his], fmt="none", ecolor=INK, capsize=3, lw=1)
    ax.axhline(1 - float(a), color=INK, ls="--", lw=1); ax.text(5.9, 1 - float(a) + 0.02, f"target {1-float(a):.1f}", ha="right", fontsize=8)
    ax.axhline(1 - float(a) - 0.05, color=MUTED, ls=":", lw=1)
    ax.set_xticks([1, 4.6], ["primary design\n20 test runs", "secondary design\n38 test runs"]); ax.set_ylim(0, 1.05)
    ax.set_title(f"α = {a}", fontsize=10, loc="left"); style(ax)
axes[0].set_ylabel("fraction of test runs fully covered")
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=C[k], label=k) for k in ("run-max", "i.i.d.", "pooled")], frameon=False, ncol=3, loc="upper center")
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(OUT / "fig_wholerun_coverage_v3.png", dpi=200); plt.close(fig)

# (c) single-split coverage distribution (Beta-Binomial) with observed splits marked, α = 0.1
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=False)
for ax, (name, (cal, test)) in zip(axes, designs.items()):
    n, m = len(cal), len(test); a = 0.1; k = math.ceil((n + 1) * (1 - a)); bb = betabinom(m, k, n + 1 - k)
    xs = np.arange(m + 1); ax.bar(xs / m, bb.pmf(xs), width=0.8 / m, color="#93c5fd")
    ax.axvline(observed[name]["0.1"], color="#dc2626", lw=2); ax.axvline(0.85, color=MUTED, ls=":", lw=1)
    ax.text(observed[name]["0.1"], ax.get_ylim()[1] * 0.92, f" observed {observed[name]['0.1']:.2f}", color="#dc2626", fontsize=8)
    ax.set_xlim(0.4, 1.02); ax.set_xlabel("whole-run coverage of one calibration split")
    ax.set_title(f"{n} calibration / {m} test runs (α = 0.1)", fontsize=10, loc="left"); style(ax)
axes[0].set_ylabel("probability (exchangeable runs)")
fig.tight_layout(); fig.savefig(OUT / "fig_split_variability_v3.png", dpi=200); plt.close(fig)

# (d) mesh refinement
M = json.loads((R / "results/v3_checks/meshref_summary.json").read_text())["summary"]
fig, ax = plt.subplots(figsize=(5.2, 3.2))
pairs = (("E141_vs_E100", "×1/√2 vs base"), ("E200_vs_E100", "×1/2 vs base"), ("E200_vs_E141", "×1/2 vs ×1/√2"))
x = np.arange(3); w = 0.36
ax.bar(x - w / 2, [M[p]["dJ_rel"]["median"] * 100 for p, _ in pairs], w, color="#93c5fd", label="objective J")
ax.bar(x + w / 2, [M[p]["dgrad_rel_M"]["median"] * 100 for p, _ in pairs], w, color="#1d4ed8", label="gradient (M-norm)")
ax.set_xticks(x, [l for _, l in pairs]); ax.set_ylabel("median relative change (%)")
ax.set_title("Mesh refinement, 10 test designs", fontsize=10, loc="left"); ax.legend(frameon=False); style(ax)
fig.tight_layout(); fig.savefig(OUT / "fig_mesh_refinement_v3.png", dpi=200); plt.close(fig)
print("FIGS_DONE", sorted(p.name for p in OUT.glob("*.png")), flush=True)
