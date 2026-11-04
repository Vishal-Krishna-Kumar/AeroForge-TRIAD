"""DESCRIPTIVE investigation (not pre-registered) of why run-max whole-run coverage in the secondary (77-run) design fell
below nominal. Uses only saved files: member gradients of the secondary run (results/v3_analysis_secondary/
member_gradients_traj_fno.npz, member_gradients_iid_fno.npz) and the CFD J gradients from the raw records. Same score as
scripts/run_v3_analysis.py: s = ||g - ĝ||_M / max(σ̂, 1e-6 median σ̂_cal_iid).
Checks: (1) per-run max scores by group (orig cal / orig test / ext cal / ext test) + rank tests; (2) 5000 random
39/38 re-splits of the 77 runs (exchangeability check: if runs are exchangeable, mean whole-run coverage over splits is
>= 1-α); (3) cross-group calibration (orig cal -> ext test etc.); (4) which step attains each run's max; (5) uncovered
test runs vs capped adjoints. Output results/v3_analysis_secondary/investigation.json (refuses to overwrite).
"""
import json, math, os
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu, kruskal

R = Path(os.environ.get("AF_ROOT", "."))
S = R / "results/v3_analysis_secondary"; OUT = S / "investigation.json"
assert not OUT.exists(), "exists"
D = json.loads((R / "datasets/designs/dataset_v3_design.json").read_text())
SPAN = np.asarray(D["box_hi"]) - np.asarray(D["box_lo"])
z = np.load(S / "member_gradients_traj_fno.npz", allow_pickle=True); G, cids = z["G"], [str(c) for c in z["case_id"]]
zi = np.load(S / "member_gradients_iid_fno.npz", allow_pickle=True); Gc = zi["G_cal"]


def stats(G):
    gh = G.mean(0)
    sq = np.stack([np.linalg.norm((G[k] - gh) * SPAN, axis=1) ** 2 for k in range(G.shape[0])])
    return gh, np.sqrt(sq.sum(0) / (G.shape[0] - 1))


floor = 1e-6 * float(np.median(stats(Gc)[1]))
gh, sg = stats(G)
RAW = R / "datasets/raw/dataset_v3_traj"
gt, capped = [], []
for c in cids:
    g = json.loads((RAW / c / "gradients.json").read_text()); J = g["objectives"]["J"]
    gt.append([J["grad"][k] for k in g["dv"]]); capped.append((J.get("iterations") or 0) >= 8000)
gt, capped = np.asarray(gt), np.asarray(capped)
s = np.linalg.norm((gt - gh) * SPAN, axis=1) / np.maximum(sg, floor)
run = np.array([int(c[4:6]) for c in cids]); step = np.array([int(c[8:10]) for c in cids])
design = json.loads((S / "design_secondary.json").read_text())
runs = sorted(set(design["cal_runs"]) | set(design["test_runs"]))
keep = np.isin(run, runs); s, run, step, capped = s[keep], run[keep], step[keep], capped[keep]
rmax = {r: float(s[run == r].max()) for r in runs}
argmax_step = {r: int(step[run == r][np.argmax(s[run == r])]) for r in runs}
cap_run = {r: bool(capped[run == r].any()) for r in runs}
grp = lambda r: ("orig" if r < 40 else "ext") + ("_cal" if (r % 40) < 20 else "_test")
groups = {g: [rmax[r] for r in runs if grp(r) == g] for g in ("orig_cal", "orig_test", "ext_cal", "ext_test")}


def cq(x, a):
    x = np.sort(np.asarray(x)); k = math.ceil((len(x) + 1) * (1 - a))
    return math.inf if k > len(x) else float(x[k - 1])


res = {"n_runs": len(runs), "floor": floor,
       "group_runmax": {g: {"n": len(v), "median": float(np.median(v)), "q90": float(np.quantile(v, .9)), "max": float(max(v))} for g, v in groups.items()},
       "kruskal_4groups_p": float(kruskal(*groups.values()).pvalue),
       "mwu_orig_vs_ext_p": float(mannwhitneyu(groups["orig_cal"] + groups["orig_test"], groups["ext_cal"] + groups["ext_test"]).pvalue),
       "mwu_cal_vs_test_p": float(mannwhitneyu(groups["orig_cal"] + groups["ext_cal"], groups["orig_test"] + groups["ext_test"]).pvalue)}
rng = np.random.default_rng(20261036); vals = np.array([rmax[r] for r in runs]); nc = len(design["cal_runs"])
split = {}
for a in (0.1, 0.2):
    cov = []
    for _ in range(5000):
        p = rng.permutation(len(vals)); q = cq(vals[p[:nc]], a); cov.append(float((vals[p[nc:]] <= q).mean()))
    cov = np.asarray(cov)
    obs = float(np.mean([rmax[r] <= cq([rmax[c] for c in design["cal_runs"]], a) for r in design["test_runs"]]))
    split[str(a)] = {"mean_cov_random_splits": float(cov.mean()), "q05": float(np.quantile(cov, .05)), "q95": float(np.quantile(cov, .95)),
                     "observed": obs, "observed_percentile": float((cov <= obs).mean())}
res["random_resplits"] = split
cross = {}
for cal_g, test_g in (("orig_cal", "orig_test"), ("orig_cal", "ext_test"), ("ext_cal", "ext_test"), ("ext_cal", "orig_test")):
    for a in (0.1, 0.2):
        q = cq(groups[cal_g], a)
        cross[f"{cal_g}->{test_g} a{a}"] = {"qhat": q, "whole_run": float(np.mean([v <= q for v in groups[test_g]]))}
res["cross_group"] = cross
qs = {str(a): cq([rmax[c] for c in design["cal_runs"]], a) for a in (0.1, 0.2)}
unc = {a: [r for r in design["test_runs"] if rmax[r] > q] for a, q in qs.items()}
res["uncovered_test_runs"] = {a: [{"run": r, "runmax": rmax[r], "argmax_step": argmax_step[r], "has_capped_adjoint": cap_run[r]} for r in v] for a, v in unc.items()}
res["capped_runs"] = [r for r in runs if cap_run[r]]
res["argmax_step_hist"] = {str(k): int(v) for k, v in zip(*np.unique(list(argmax_step.values()), return_counts=True))}
res["argmax_at_capped_point"] = int(sum(capped[(run == r)][np.argmax(s[run == r])] for r in runs))
res["runmax_capped_vs_not_median"] = [float(np.median([rmax[r] for r in runs if cap_run[r]] or [np.nan])), float(np.median([rmax[r] for r in runs if not cap_run[r]]))]
OUT.write_text(json.dumps(res, indent=1))
print("INVESTIGATION", json.dumps(res)[:3000], flush=True)
