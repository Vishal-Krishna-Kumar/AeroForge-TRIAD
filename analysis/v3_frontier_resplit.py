"""DESCRIPTIVE (not pre-registered). (1) Frontier figure: whole-run coverage vs certified fraction of test steps for every
certificate in results/v3_baselines/baselines.json (+ PAC run-max from results/v3_paper_figs/numbers.json), secondary
design, alpha 0.1 and 0.2 -> results/v3_paper_figs/fig_frontier_v3.png. (2) H2'' re-split check: the V3 i.i.d. split
(cal 128 / test 64 after NaN removal) of FNO and MLP, (a) recomputed and asserted equal to results/v3_analysis/v3_analysis.json,
(b) exact Beta-Binomial reference distribution of single-split coverage and the observed CDF, (c) Mann-Whitney test of cal vs
test scores (the exchangeability evidence), (d) 5000 random 128/64 re-splits of the same 192 points (seed 20261041; the
mean is ~1-alpha for any fixed scores, so only the spread is informative) -> results/v3_resplit_h2/h2_resplit.json.
Refuses to overwrite either output.
"""
import json, math, os, sys
from pathlib import Path
import numpy as np
from scipy.stats import betabinom, mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = Path(os.environ.get("AF_ROOT", "."))
sys.path.insert(0, str(R / "src")); sys.path.insert(0, str(R / "scripts"))
FIG = R / "results/v3_paper_figs/fig_frontier_v3.png"; OUT = R / "results/v3_resplit_h2"
assert not FIG.exists() and not (OUT / "h2_resplit.json").exists(), "exists"
D = json.loads((R / "datasets/designs/dataset_v3_design.json").read_text())
SPAN = np.asarray(D["box_hi"]) - np.asarray(D["box_lo"])
ALPHAS = (0.1, 0.2)


def cq(scores, alpha):
    s = np.sort(np.asarray(scores, float)); k = math.ceil((len(s) + 1) * (1 - alpha))
    return math.inf if k > len(s) else float(s[k - 1])


def stats(G):
    gh = G.mean(0)
    sq = np.stack([np.linalg.norm((G[k] - gh) * SPAN, axis=1) ** 2 for k in range(G.shape[0])])
    return gh, np.sqrt(sq.sum(0) / (G.shape[0] - 1))


# ---------------- (2) H2'' re-split ----------------
from train_surrogates_v3 import load_v3
A = json.loads((R / "results/v3_analysis/v3_analysis.json").read_text())
cal, te = load_v3(R / "datasets/processed/dataset_v3/cal", with_fields=False), load_v3(R / "datasets/processed/dataset_v3/test", with_fields=False)
okc, okt = ~np.isnan(cal["g_J"]).any(1), ~np.isnan(te["g_J"]).any(1)
res = {"n_resplits": 5000, "seed": 20261041, "kinds": {}}
for kind in ("fno", "mlp"):
    z = np.load(R / f"results/v3_analysis/member_gradients_iid_{kind}.npz", allow_pickle=True)
    assert [str(c) for c in cal["case_id"][okc]] == [str(c) for c in z["cal_case_id"]], "cal order"
    assert [str(c) for c in te["case_id"][okt]] == [str(c) for c in z["test_case_id"]], "test order"
    G = np.concatenate([z["G_cal"], z["G_test"]], axis=1); gt = np.concatenate([cal["g_J"][okc], te["g_J"][okt]])
    gh, sg = stats(G); nc, nt = int(okc.sum()), int(okt.sum()); N = nc + nt
    score = lambda idx_cal: np.linalg.norm((gt - gh) * SPAN, axis=1) / np.maximum(sg, 1e-6 * float(np.median(sg[idx_cal])))
    ic, it = np.arange(nc), np.arange(nc, N)
    s0 = score(ic)
    out = {"n_cal": nc, "n_test": nt, "mwu_cal_vs_test_p": float(mannwhitneyu(s0[ic], s0[it]).pvalue),
           "median_score_cal": float(np.median(s0[ic])), "median_score_test": float(np.median(s0[it])), "alphas": {}}
    rng = np.random.default_rng(20261041)
    perms = [rng.permutation(N) for _ in range(5000)]
    for a in ALPHAS:
        cov0 = float(np.mean(s0[it] <= cq(s0[ic], a)))
        saved = A["H2pp"][kind]["alphas"][str(a)]["coverage"]
        assert abs(cov0 - saved) < 1e-9, ("consistency", kind, a, cov0, saved)
        k = math.ceil((nc + 1) * (1 - a)); bb = betabinom(nt, k, nc + 1 - k)
        need = math.ceil((1 - a - 0.05) * nt - 1e-9); x = round(cov0 * nt)
        covs = []
        for p in perms:
            pc, pt = p[:nc], p[nc:]; sp = score(pc)
            covs.append(float(np.mean(sp[pt] <= cq(sp[pc], a))))
        covs = np.asarray(covs)
        out["alphas"][str(a)] = {"observed": cov0, "theory_mean": float(bb.mean() / nt), "theory_q05": float(bb.ppf(0.05) / nt),
                                 "theory_q95": float(bb.ppf(0.95) / nt), "theory_P_pass_rule": float(bb.sf(need - 1)),
                                 "theory_observed_cdf": float(bb.cdf(x)), "resplit_mean": float(covs.mean()),
                                 "resplit_q05": float(np.quantile(covs, 0.05)), "resplit_q95": float(np.quantile(covs, 0.95)),
                                 "resplit_P_pass_rule": float(np.mean(covs >= 1 - a - 0.05 - 1e-12)),
                                 "observed_resplit_percentile": float(np.mean(covs <= cov0 + 1e-12))}
    res["kinds"][kind] = out
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "h2_resplit.json").write_text(json.dumps(res, indent=1))
for kind, o in res["kinds"].items():
    print("H2RESPLIT", kind, "mwu p", round(o["mwu_cal_vs_test_p"], 3), {a: {k: round(v, 3) for k, v in d.items()} for a, d in o["alphas"].items()}, flush=True)

# ---------------- (1) frontier figure ----------------
B = json.loads((R / "results/v3_baselines/baselines.json").read_text())["designs"]["secondary_39x38"]["alphas"]
P = json.loads((R / "results/v3_paper_figs/numbers.json").read_text())["pac"]
FAM = {"whole-run guarantee": ("#2563eb", "o"), "conformal, no trajectory guarantee": ("#d97706", "s"),
       "Gaussian ensemble (uncalibrated)": ("#6b7280", "^"), "adaptive conformal (ACI)": ("#059669", "D")}
fam = lambda k: (list(FAM)[0] if k.startswith(("run-max", "B2a")) else list(FAM)[2] if k.startswith("B1") else
                 list(FAM)[3] if k.startswith("B4") else list(FAM)[1])
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), sharey=True)
for ax, a in zip(axes, ("0.1", "0.2")):
    m = dict(B[a]); pac = P[f"secondary_39x38 a{a}"]
    m["PAC run-max"] = {"whole_run": pac["whole_run"], "certified_fraction": pac["certified_fraction"]}
    for k, v in m.items():
        f = "whole-run guarantee" if k == "PAC run-max" else fam(k); c, mk = FAM[f]
        ax.scatter(100 * v["certified_fraction"], v["whole_run"], s=46, color=c, marker=mk, edgecolor="white", linewidth=1.5, zorder=3)
    lab = {"run-max (Thm 1, reference)": "run-max", "PAC run-max": "PAC run-max", "B2a per-step conformal, Bonferroni alpha/T": "per-step Bonferroni (q̂ = ∞)"}
    for k, t in lab.items():
        v = m[k]; x, y = 100 * v["certified_fraction"], v["whole_run"]
        ax.annotate(t, (x, y), xytext=(6, -12 if k == "PAC run-max" else 6), textcoords="offset points", fontsize=8, color=INK)
    ax.axhline(1 - float(a), color=INK, ls="--", lw=1); ax.text(99, 1 - float(a) + 0.015, f"target {1 - float(a):.1f}", ha="right", fontsize=8)
    ax.set_xlim(-3, 100); ax.set_ylim(0, 1.05); ax.set_xlabel("certified test steps (%)")
    ax.set_title(f"α = {a}, 39 calibration / 38 test runs", fontsize=10, loc="left")
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("fraction of test runs fully covered")
fig.legend(handles=[Line2D([], [], ls="", marker=mk, color=c, markersize=7, label=f) for f, (c, mk) in FAM.items()],
           frameon=False, ncol=2, loc="upper center", fontsize=8.5)
fig.tight_layout(rect=(0, 0, 1, 0.86)); fig.savefig(FIG, dpi=200); plt.close(fig)
print("FIG", FIG.name, "DONE", flush=True)
