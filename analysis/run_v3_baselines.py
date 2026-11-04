"""DESCRIPTIVE (not pre-registered): V3 comparison baselines for whole-run gradient certificates, on the primary (20/20)
and secondary (39/38) designs, FNO ensemble, alpha in {0.1, 0.2}. Reads only saved results; same score as
scripts/run_v3_analysis.py: s = ||g - g_hat||_M / max(sigma_hat, 1e-6 x median sigma_hat_cal), T = 12 steps per run.

  B1 Gaussian ensemble UQ, no calibration: (a) isotropic, s^2 ~ chi2_13/13; (b) ensemble covariance, s^2 ~ sum_j l_j chi2_1
     with l_j the eigenvalues of the member covariance / its trace (Monte Carlo, 20000 draws, seed 20261040). Per-iterate
     level alpha and whole-run level alpha/T (Bonferroni over steps).
  B2 Bonferroni over steps, split conformal at alpha/T: (a) per step index t, calibrated on the calibration runs' step-t
     scores (valid by the union bound; infinite if ceil((n+1)(1-alpha/T)) > n); (b) i.i.d. V3 cal scores; (c) pooled
     calibration-run trajectory points.
  B3 Weighted split conformal under covariate shift (Tibshirani et al. 2019): i.i.d. V3 cal scores re-weighted by
     w(theta) = P(traj | theta) / P(iid | theta) from a logistic classifier (box-scaled theta and theta^2) fitted on i.i.d.
     cal points vs CALIBRATION-run trajectory points only; per-point level alpha and alpha/T. Median effective sample size
     reported.
  B4 Adaptive conformal inference (Gibbs & Candes 2021) along each test run from the i.i.d. cal scores, alpha_1 = alpha,
     alpha_{t+1} = alpha_t + gamma (alpha - err_t), gamma in {0.005, 0.05}. NB: needs the true gradient after every step.
  Reference rows: run-max (Thm 1) and i.i.d. copied from the saved analyses (and recomputed here as a consistency check).
For each: whole-run coverage of the test runs (Clopper-Pearson 95 %), per-iterate coverage, and certified fraction of
test steps (||g_hat||_M > q sigma_hat, the H9 rule). Output results/v3_baselines/baselines.json (refuses to overwrite).
"""
import json, math, os, sys
from pathlib import Path
import numpy as np
from scipy.stats import beta as beta_dist, chi2
from sklearn.linear_model import LogisticRegression

R = Path(os.environ.get("AF_ROOT", "."))
sys.path.insert(0, str(R / "src")); sys.path.insert(0, str(R / "scripts"))
OUT = R / "results/v3_baselines"
assert not (OUT / "baselines.json").exists(), "exists"
S = R / "results/v3_analysis_secondary"
D = json.loads((R / "datasets/designs/dataset_v3_design.json").read_text())
LO, HI = np.asarray(D["box_lo"]), np.asarray(D["box_hi"]); SPAN = HI - LO
T, ALPHAS = 12, (0.1, 0.2)


def cq(scores, alpha):
    if alpha >= 1:
        return 0.0
    s = np.sort(np.asarray(scores, float)); k = math.ceil((len(s) + 1) * (1 - alpha))
    return math.inf if k > len(s) else float(s[k - 1])


def cp(k, n, a=0.05):
    return [0.0 if k == 0 else float(beta_dist.ppf(a / 2, k, n - k + 1)), 1.0 if k == n else float(beta_dist.ppf(1 - a / 2, k + 1, n - k))]


def stats(G):
    gh = G.mean(0)
    sq = np.stack([np.linalg.norm((G[k] - gh) * SPAN, axis=1) ** 2 for k in range(G.shape[0])])
    return gh, np.sqrt(sq.sum(0) / (G.shape[0] - 1))


# ---- i.i.d. V3 calibration scores (as run_v3_analysis.h2, FNO) ----
from train_surrogates_v3 import load_v3
zi = np.load(S / "member_gradients_iid_fno.npz", allow_pickle=True)
cal = load_v3(R / "datasets/processed/dataset_v3/cal", with_fields=False)
okc = ~np.isnan(cal["g_J"]).any(1)
assert [str(c) for c in cal["case_id"][okc]] == [str(c) for c in zi["cal_case_id"]], "cal order"
ghc, sgc = stats(zi["G_cal"]); floor = 1e-6 * float(np.median(sgc))
s_cal = np.linalg.norm((cal["g_J"][okc] - ghc) * SPAN, axis=1) / np.maximum(sgc, floor)
th_cal = (cal["theta"][okc] - LO) / SPAN

# ---- trajectory points (as v3_theory_pac_figs.py) ----
z = np.load(S / "member_gradients_traj_fno.npz", allow_pickle=True); G, cids = z["G"], [str(c) for c in z["case_id"]]
gh, sg = stats(G); sgf = np.maximum(sg, floor)
RAW = R / "datasets/raw/dataset_v3_traj"
gt = []
for c in cids:
    g = json.loads((RAW / c / "gradients.json").read_text()); gt.append([g["objectives"]["J"]["grad"][k] for k in g["dv"]])
gt = np.asarray(gt)
s = np.linalg.norm((gt - gh) * SPAN, axis=1) / sgf
ghn = np.linalg.norm(gh * SPAN, axis=1)
run = np.array([int(c[4:6]) for c in cids]); step = np.array([int(c[8:10]) for c in cids])
pts = {}
for p in ("results/v3_traj/trajectory_points.json", "results/v3_traj_ext/trajectory_points.json"):
    pts.update({q["case_id"]: q["theta"] for q in json.loads((R / p).read_text())["points"]})
th = (np.asarray([pts[c] for c in cids]) - LO) / SPAN
sec = json.loads((S / "design_secondary.json").read_text())
DESIGNS = {"primary_20x20": (list(range(20)), list(range(20, 40))), "secondary_39x38": (sec["cal_runs"], sec["test_runs"])}
A = {"primary_20x20": json.loads((R / "results/v3_analysis/v3_analysis.json").read_text()),
     "secondary_39x38": json.loads((S / "v3_analysis.json").read_text())}

# ---- per-point Gaussian (ensemble covariance) quantiles, shared draws ----
rng = np.random.default_rng(20261040)
Z2 = rng.standard_normal((20000, G.shape[0] - 1)) ** 2


def gauss_cov_q(i, levels):
    Dm = (G[:, i, :] - gh[i]) * SPAN
    lam = np.clip(np.linalg.eigvalsh(Dm @ Dm.T / (G.shape[0] - 1)), 0, None)[::-1][: G.shape[0] - 1]
    if lam.sum() <= 0:
        return [math.inf for _ in levels]
    r = np.sqrt(Z2 @ (lam / lam.sum()))
    return [float(np.quantile(r, 1 - a)) for a in levels]


GQ = {a: np.array([gauss_cov_q(i, (a, a / T)) for i in range(len(cids))]) for a in ALPHAS}


def weighted_q(sc, wc, wt, alpha):
    o = np.argsort(sc); sc, wc = sc[o], wc[o]
    p = np.cumsum(wc) / (wc.sum() + wt)
    j = np.searchsorted(p, 1 - alpha - 1e-12)
    return math.inf if j >= len(sc) else float(sc[j])


def summarise(qv, test):
    """qv: per-point radius array over all points (or scalar); whole-run/per-iterate/certified over the test runs."""
    qv = np.broadcast_to(np.asarray(qv, float), s.shape)
    tm = np.isin(run, test); cov = s <= qv
    whole = np.array([cov[run == r].all() for r in test]); point = np.array([cov[run == r].mean() for r in test])
    return {"whole_run": float(whole.mean()), "whole_run_CI95": cp(int(whole.sum()), len(whole)), "per_iterate": float(point.mean()),
            "certified_fraction": float(np.mean(ghn[tm] > qv[tm] * sgf[tm])),
            "median_radius": float(np.median(qv[tm])) if np.isfinite(qv[tm]).any() else math.inf}


res = {"n_iid_cal": int(okc.sum()), "T": T, "designs": {}}
for name, (calr, test) in DESIGNS.items():
    n = len(calr); cm = np.isin(run, calr)
    # density-ratio classifier: i.i.d. cal (0) vs calibration-run trajectory points (1)
    feat = lambda x: np.hstack([x, x ** 2])
    X = np.vstack([feat(th_cal), feat(th[cm])]); y = np.r_[np.zeros(len(th_cal)), np.ones(cm.sum())]
    clf = LogisticRegression(C=1.0, max_iter=5000).fit(X, y)
    odds = lambda x: (lambda p: p / (1 - p))(np.clip(clf.predict_proba(feat(x))[:, 1], 1e-6, 1 - 1e-6))
    wc, wt_all = odds(th_cal), odds(th)
    ess = float(np.median([(wc.sum() + w) ** 2 / ((wc ** 2).sum() + w ** 2) for w in wt_all[np.isin(run, test)]]))
    d = {"n_cal_runs": n, "n_test_runs": len(test), "weighted_classifier_train_acc": float(clf.score(X, y)),
         "weighted_median_ESS": ess, "alphas": {}}
    for a in ALPHAS:
        rmax = [s[run == r].max() for r in calr]
        chk = {"runmax_qhat": cq(rmax, a), "iid_qhat": cq(s_cal, a)}
        ref = A[name]["alphas"][str(a)]
        assert abs(chk["runmax_qhat"] - ref["H5_runmax"]["qhat"]) < 1e-4 and abs(chk["iid_qhat"] - ref["H7_iid"]["qhat"]) < 1e-4, ("consistency", name, a, chk)
        m = {"run-max (Thm 1, reference)": summarise(chk["runmax_qhat"], test), "i.i.d. split conformal (reference)": summarise(chk["iid_qhat"], test)}
        iso = lambda lev: math.sqrt(chi2.ppf(1 - lev, len(SPAN)) / len(SPAN))
        m["B1a Gaussian isotropic, level alpha"] = summarise(iso(a), test)
        m["B1a Gaussian isotropic, Bonferroni alpha/T"] = summarise(iso(a / T), test)
        m["B1b Gaussian ensemble-cov, level alpha"] = summarise(GQ[a][:, 0], test)
        m["B1b Gaussian ensemble-cov, Bonferroni alpha/T"] = summarise(GQ[a][:, 1], test)
        qt = {t: cq(s[cm & (step == t)], a / T) for t in range(T)}
        m["B2a per-step conformal, Bonferroni alpha/T"] = dict(summarise(np.array([qt[t] for t in step]), test),
                                                               finite=bool(all(np.isfinite(list(qt.values())))),
                                                               min_cal_runs_for_finite=int(math.ceil(T / a) - 1))
        m["B2b i.i.d. conformal, Bonferroni alpha/T"] = summarise(cq(s_cal, a / T), test)
        m["B2c pooled trajectory points, Bonferroni alpha/T"] = summarise(cq(s[cm], a / T), test)
        m["B3 weighted conformal, level alpha"] = summarise(np.array([weighted_q(s_cal, wc, w, a) for w in wt_all]), test)
        m["B3 weighted conformal, Bonferroni alpha/T"] = summarise(np.array([weighted_q(s_cal, wc, w, a / T) for w in wt_all]), test)
        for gam in (0.005, 0.05):
            qa = np.full(s.shape, np.nan)
            for r in test:
                at = a
                for i in np.where(run == r)[0][np.argsort(step[run == r])]:
                    qa[i] = cq(s_cal, at); at = at + gam * (a - float(s[i] > qa[i]))
            m[f"B4 ACI gamma={gam}"] = summarise(np.where(np.isnan(qa), math.inf, qa), test)
        d["alphas"][str(a)] = m
    res["designs"][name] = d
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "baselines.json").write_text(json.dumps(res, indent=1, default=float))
for name, d in res["designs"].items():
    print("DESIGN", name, "ESS", round(d["weighted_median_ESS"], 1), "clf_acc", round(d["weighted_classifier_train_acc"], 3))
    for a, m in d["alphas"].items():
        for k, v in m.items():
            print(f"  a{a} {k:52s} whole {v['whole_run']:.3f} it {v['per_iterate']:.3f} cert {v['certified_fraction']:.3f} rad {v['median_radius']:.2f}")
print("BASELINES_DONE", flush=True)
