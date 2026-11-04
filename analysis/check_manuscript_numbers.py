"""Re-pull every number quoted in docs/.../manuscript/manuscript_v1.tex from the saved result files and compare.

For each quoted number: the string printed in the manuscript (EXPECTED), the value recomputed from the JSON and formatted
the same way (round-half-up, FROM_JSON), and STATUS = MATCH / MISMATCH / MISSING (key or file absent) / NEW (number not yet
in the manuscript, reported for inclusion). Also checks that each EXPECTED string occurs in the manuscript text.
Outputs (refuses to overwrite): results/paper_numbers/numbers_check_v1.{json,txt} and numbers_v1.tex (\pnum{key} macros, so
the final manuscript can use script-generated numbers only). Appends one row to docs/experiment_log.md (sha must start
005aefa5). Reads only saved results; computes nothing new except ratios/ranges of saved numbers.
"""
import hashlib, json, math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

R = Path("."); OUT = R / "results/paper_numbers"; LOG = R / "docs/experiment_log.md"
assert not (OUT / "numbers_check_v1.json").exists(), "exists"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
b = LOG.read_bytes(); h = hashlib.sha256(b).hexdigest()
assert h.startswith("005aefa5"), "LOG CHANGED " + h[:8]


def find(name):
    hits = [p for p in (R / "results").rglob(name) if "partial" not in p.name]
    assert len(hits) == 1, (name, [str(p) for p in hits])
    return hits[0]


FILES = {"v1fno": find("h1_test_metrics_fno.json"), "v1mlp": find("h1_test_metrics_mlp.json"),
         "v2fno": find("h1v2_test_metrics_fno.json"), "v2mlp": find("h1v2_test_metrics_mlp.json"),
         "v3fno": R / "results/reports_v3/h1v3_test_metrics_fno.json", "v3mlp": R / "results/reports_v3/h1v3_test_metrics_mlp.json",
         "i1fno": find("h2_iid_fno.json"), "i1mlp": find("h2_iid_mlp.json"), "i2fno": find("h2v2_iid_fno.json"),
         "i2mlp": find("h2v2_iid_mlp.json"), "s2": find("s2_h5_h6_h7.json"), "sb": find("s2_baselines_descriptive.json"),
         "sc": find("s2_certified_fraction_descriptive.json"), "h4": find("h4_trajectory_coverage.json"),
         "P": R / "results/v3_analysis/v3_analysis.json", "S": R / "results/v3_analysis_secondary/v3_analysis.json",
         "C": R / "results/v3_analysis_sensitivity_capped/v3_analysis.json",
         "INV": R / "results/v3_analysis_secondary/investigation.json", "MESH": R / "results/v3_checks/meshref_summary.json",
         "NUM": R / "results/v3_paper_figs/numbers.json", "BL": R / "results/v3_baselines/baselines.json",
         "RS": R / "results/v3_resplit_h2/h2_resplit.json"}
D = {k: json.loads(p.read_text()) for k, p in FILES.items()}
TEX = next(iter(sorted((R / "docs").rglob("manuscript_v1.tex"))), None)
tex = TEX.read_text() if TEX else ""


def rnd(x, n):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-n), rounding=ROUND_HALF_UP))


def fmt(v, f):
    if v is None:
        return "none"
    if f == "ci2":
        return f"{rnd(v[0], 2)}--{rnd(v[1], 2)}"
    if f == "ci3":
        return f"{rnd(v[0], 3)}--{rnd(v[1], 3)}"
    if isinstance(v, float) and math.isinf(v):
        return "inf"
    if f == "i":
        return str(int(round(v)))
    return rnd(v * 100, int(f[1:])) if f[0] == "p" else rnd(v, int(f[1:]))


CHECKS = []  # (key, expected or None for NEW, format, getter)
add = lambda key, exp, f, get: CHECKS.append((key, exp, f, get))
pre = lambda d, a: d["preregistered_split"][0 if a == "0.1" else 1]

# ---- Table 2: gradients vs values ----
for tag, val, grad, cos, med, vkey, gkey in (
        ("v1fno", "0.66", "5.8", "1.6", "0.9992", "median_rel_err_LD", "median_rel_grad_err_LD"),
        ("v1mlp", "0.33", "1.25", "0", None, "median_rel_err_LD", "median_rel_grad_err_LD"),
        ("v2fno", "1.65", "9.0", "1.6", "0.9990", "median_rel_err_LD", "median_rel_grad_err_LD"),
        ("v2mlp", "1.22", "3.1", "0", "0.9997", "median_rel_err_LD", "median_rel_grad_err_LD"),
        ("v3fno", "2.10", "20.4", "42.2", "0.9843", "median_rel_err_J", "median_rel_grad_err_J"),
        ("v3mlp", "1.76", "18.3", "35.9", "0.9906", "median_rel_err_J", "median_rel_grad_err_J")):
    e = lambda t=tag: D[t]["ensemble"]
    add(f"h1.{tag}.value", val, "p2", lambda e=e, k=vkey: e()[k])
    add(f"h1.{tag}.grad", grad, "p1" if len(grad.split(".")[-1]) == 1 and "." in grad else ("p2" if "." in grad else "p0"), lambda e=e, k=gkey: e()[k])
    add(f"h1.{tag}.cosfrac", cos, "p0" if cos == "0" else "p1", lambda e=e: e()["frac_cos_below_0.98"])
    add(f"h1.{tag}.mediancos", med, "f4", lambda e=e: e()["median_cos"])
add("h1.v1.ntest", "128", "i", lambda: D["v1fno"]["meta"]["n_test"]); add("h1.v2.ntest", "127", "i", lambda: D["v2fno"]["meta"]["n_test"])
add("h1.v3.ntest", "64", "i", lambda: D["P"]["H2pp"]["fno"]["n_test"])

# ---- i.i.d. coverage (Section 5.2) ----
for tag, a, cov, ci in (("i1fno", "0.1", "0.891", "0.823--0.939"), ("i1fno", "0.2", "0.688", "0.600--0.766"),
                        ("i1mlp", "0.1", "0.938", None), ("i1mlp", "0.2", "0.883", None),
                        ("i2fno", "0.1", "0.906", "0.841--0.950"), ("i2fno", "0.2", "0.811", "0.732--0.875"),
                        ("i2mlp", "0.1", "0.858", "0.785--0.914"), ("i2mlp", "0.2", "0.803", "0.723--0.868")):
    add(f"iid.{tag}.a{a}.cov", cov, "f3", lambda t=tag, a=a: pre(D[t], a)["coverage"])
    add(f"iid.{tag}.a{a}.ci", ci, "ci3", lambda t=tag, a=a: pre(D[t], a)["coverage_CI95"])
    add(f"iid.{tag}.a{a}.resplitmean", None, "f3", lambda t=tag, a=a: D[t]["resample_mean_coverage"][a])
add("iid.v1.ncal", "128", "i", lambda: D["i1fno"]["splits"]["cal"]["n"])
add("iid.v2.ncal", "256", "i", lambda: D["i2fno"]["splits"]["cal"]["n"]); add("iid.v2.ntest", "127", "i", lambda: D["i2fno"]["splits"]["test"]["n"])
for kind, a, cov, ci in (("fno", "0.1", "0.797", "0.678--0.887"), ("fno", "0.2", "0.750", "0.626--0.850"),
                         ("mlp", "0.1", "0.891", "0.788--0.955"), ("mlp", "0.2", "0.797", "0.678--0.887")):
    add(f"iid.v3{kind}.a{a}.cov", cov, "f3", lambda k=kind, a=a: D["P"]["H2pp"][k]["alphas"][a]["coverage"])
    add(f"iid.v3{kind}.a{a}.ci", ci, "ci3", lambda k=kind, a=a: D["P"]["H2pp"][k]["alphas"][a]["coverage_CI95"])
for a in ("0.1", "0.2"):  # V1 i.i.d. calibration along optimiser trajectories (H2 trajectory part, H4) -- not yet in the manuscript
    add(f"h4.v1.a{a}.iidtraj", None, "f3", lambda a=a: D["h4"]["primary_LD"]["alphas"][a]["iid_preregistered"]["coverage"])
    for i in (0, 1):
        add(f"h4.v1.a{a}.fold{i}.traj", None, "f3", lambda a=a, i=i: D["h4"]["primary_LD"]["alphas"][a]["H4_run_level_split"][i]["coverage_trajectory_calibrated"])
        add(f"h4.v1.a{a}.fold{i}.iid", None, "f3", lambda a=a, i=i: D["h4"]["primary_LD"]["alphas"][a]["H4_run_level_split"][i]["coverage_iid_calibrated"])
rs = lambda k, a: D["RS"]["kinds"][k]["alphas"][a]
add("rs.a0.1.q05", "0.83", "f2", lambda: rs("fno", "0.1")["theory_q05"]); add("rs.a0.1.q95", "0.97", "f2", lambda: rs("fno", "0.1")["theory_q95"])
add("rs.a0.2.q05", "0.70", "f2", lambda: rs("fno", "0.2")["theory_q05"]); add("rs.a0.2.q95", "0.91", "f2", lambda: rs("fno", "0.2")["theory_q95"])
add("rs.a0.1.ppass", "89", "p0", lambda: rs("fno", "0.1")["theory_P_pass_rule"]); add("rs.a0.2.ppass", "85", "p0", lambda: rs("fno", "0.2")["theory_P_pass_rule"])
add("rs.fno.a0.1.resplitpct", "2.1", "p1", lambda: rs("fno", "0.1")["observed_resplit_percentile"])
add("rs.fno.a0.1.theorycdf", "2", "p0", lambda: rs("fno", "0.1")["theory_observed_cdf"])
add("rs.mwu.fno", "0.22", "f2", lambda: D["RS"]["kinds"]["fno"]["mwu_cal_vs_test_p"]); add("rs.mwu.mlp", "0.46", "f2", lambda: D["RS"]["kinds"]["mlp"]["mwu_cal_vs_test_p"])
add("rs.pct.min", "21", "p0", lambda: min(rs("fno", "0.2")["observed_resplit_percentile"], rs("mlp", "0.1")["observed_resplit_percentile"], rs("mlp", "0.2")["observed_resplit_percentile"]))
add("rs.pct.max", "47", "p0", lambda: max(rs("fno", "0.2")["observed_resplit_percentile"], rs("mlp", "0.1")["observed_resplit_percentile"], rs("mlp", "0.2")["observed_resplit_percentile"]))

# ---- Table 3: V2 run-level ----
V2 = {"0.1": dict(iid=("0.20", "0.06--0.44", "0.653", "2.41"), pooled=("0.60", "0.36--0.81", "0.833", "4.33"), runmax=("1.00", "0.83--1.00", "1.000", "8.69"),
                  h6=("0.877", "4.64"), gauss=("0.00", "0.397", "1.23"), bonferroni=("1.00", "1.000", "9.41"), weighted=("0.25", "0.720", "3.00"),
                  aci_gamma0_01=("0.20", "0.693", "2.41"), aci_gamma0_05=("0.10", "0.837", "2.41"), cert="44.7", ratio="3.6"),
      "0.2": dict(iid=("0.05", "0.00--0.25", "0.547", "1.81"), pooled=("0.35", "0.15--0.59", "0.710", "3.06"), runmax=("0.85", "0.62--0.97", "0.987", "7.66"),
                  h6=("0.742", "3.33"), gauss=("0.00", "0.347", "1.14"), bonferroni=("0.70", "0.887", "5.03"), weighted=("0.10", "0.590", "2.05"),
                  aci_gamma0_01=("0.00", "0.547", "1.81"), aci_gamma0_05=("0.00", "0.687", "1.81"), cert="51.3", ratio="4.2")}
s2 = lambda a: D["s2"]["alphas"][a]
for a, E in V2.items():
    for lab, key in (("iid", "H7_iid"), ("pooled", "pooled_baseline"), ("runmax", "H5_runmax")):
        w, ci, it, q = E[lab]
        add(f"v2.{lab}.a{a}.whole", w, "f2", lambda a=a, k=key: s2(a)[k]["whole_run"]); add(f"v2.{lab}.a{a}.ci", ci, "ci2", lambda a=a, k=key: s2(a)[k]["whole_run_CI95"])
        add(f"v2.{lab}.a{a}.periter", it, "f3", lambda a=a, k=key: s2(a)[k]["per_iterate"]); add(f"v2.{lab}.a{a}.qhat", q, "f2", lambda a=a, k=key: s2(a)[k]["qhat"])
    add(f"v2.h6.a{a}.periter", E["h6"][0], "f3", lambda a=a: s2(a)["H6_subsample"]["per_iterate_mean_over_draws"])
    add(f"v2.h6.a{a}.qhat", E["h6"][1], "f2", lambda a=a: s2(a)["H6_subsample"]["qhat_median"])
    for lab in ("gauss", "bonferroni", "weighted", "aci_gamma0_01", "aci_gamma0_05"):
        key = {"gauss": "gaussian", "aci_gamma0_01": "aci_gamma0.01", "aci_gamma0_05": "aci_gamma0.05"}.get(lab, lab); w, it, q = E[lab]
        add(f"v2.{lab}.a{a}.whole", w, "f2", lambda a=a, k=key: D["sb"]["alphas"][a][k]["whole_run"])
        add(f"v2.{lab}.a{a}.periter", it, "f3", lambda a=a, k=key: D["sb"]["alphas"][a][k]["per_iterate"])
        add(f"v2.{lab}.a{a}.thr", q, "f2", lambda a=a, k=key: D["sb"]["alphas"][a][k]["median_threshold"])
    add(f"v2.cert.runmax.a{a}", E["cert"], "p1", lambda a=a: D["sc"]["alphas"][a]["runmax"]["frac_steps_certified"])
    add(f"v2.ratio.a{a}", E["ratio"], "f1", lambda a=a: s2(a)["H5_runmax"]["qhat"] / s2(a)["H7_iid"]["qhat"])
add("v2.nonascent.steps", "1", "i", lambda: (1 - D["sc"]["frac_all_steps_true_ascent"]) * D["sc"]["n_test_points"])
add("v2.test.steps", "300", "i", lambda: D["sc"]["n_test_points"])

# ---- Table 4: V3 primary and secondary ----
V3 = {("P", "0.1"): dict(iid=("0.65", "0.41--0.85", "0.933", "2.31"), pooled=("0.55", "0.32--0.77", "0.917", "2.18"), runmax=("0.95", "0.75--1.00", "0.996", "4.90"), h6=("0.926", "2.44"), loo="0.90"),
      ("P", "0.2"): dict(iid=("0.35", "0.15--0.59", "0.879", "1.97"), pooled=("0.20", "0.06--0.44", "0.813", "1.72"), runmax=("0.70", "0.46--0.88", "0.975", "3.23"), h6=("0.834", "1.82"), loo="0.80"),
      ("S", "0.1"): dict(iid=("0.61", "0.43--0.76", "0.932", "2.31"), pooled=("0.53", "0.36--0.69", "0.917", "2.17"), runmax=("0.76", "0.60--0.89", "0.980", "3.44"), h6=("0.915", "2.18"), loo="0.91"),
      ("S", "0.2"): dict(iid=("0.39", "0.24--0.57", "0.875", "1.97"), pooled=("0.18", "0.08--0.34", "0.798", "1.72"), runmax=("0.66", "0.49--0.80", "0.961", "2.79"), h6=("0.806", "1.73"), loo="0.81")}
for (d, a), E in V3.items():
    al = lambda d=d, a=a: D[d]["alphas"][a]
    for lab, key in (("iid", "H7_iid"), ("pooled", "pooled_baseline_descriptive"), ("runmax", "H5_runmax")):
        w, ci, it, q = E[lab]
        add(f"v3{d}.{lab}.a{a}.whole", w, "f2", lambda al=al, k=key: al()[k]["whole_run"]); add(f"v3{d}.{lab}.a{a}.ci", ci, "ci2", lambda al=al, k=key: al()[k]["whole_run_CI95"])
        add(f"v3{d}.{lab}.a{a}.periter", it, "f3", lambda al=al, k=key: al()[k]["per_iterate"]); add(f"v3{d}.{lab}.a{a}.qhat", q, "f2", lambda al=al, k=key: al()[k]["qhat"])
    add(f"v3{d}.h6.a{a}.periter", E["h6"][0], "f3", lambda al=al: al()["H6_random_iterate"]["per_iterate_mean_over_draws"])
    add(f"v3{d}.h6.a{a}.qhat", E["h6"][1], "f2", lambda al=al: al()["H6_random_iterate"]["qhat_median"])
    add(f"v3{d}.loo.a{a}", E["loo"], "f2", lambda al=al: al()["LOO_runmax_whole_run_descriptive"])
add("v3C.runmax.a0.2.whole", "0.75", "f2", lambda: D["C"]["alphas"]["0.2"]["H5_runmax"]["whole_run"])
add("v3C.runmax.a0.2.ci", None, "ci2", lambda: D["C"]["alphas"]["0.2"]["H5_runmax"]["whole_run_CI95"])
add("v3P.ratio.a0.1", "2.1", "f1", lambda: D["P"]["alphas"]["0.1"]["H5_runmax"]["qhat"] / D["P"]["alphas"]["0.1"]["H7_iid"]["qhat"])
add("v3P.ratio.a0.2", "1.6", "f1", lambda: D["P"]["alphas"]["0.2"]["H5_runmax"]["qhat"] / D["P"]["alphas"]["0.2"]["H7_iid"]["qhat"])
add("v3S.ratio.a0.1", None, "f1", lambda: D["S"]["alphas"]["0.1"]["H5_runmax"]["qhat"] / D["S"]["alphas"]["0.1"]["H7_iid"]["qhat"])
add("v3S.ratio.a0.2", "1.4", "f1", lambda: D["S"]["alphas"]["0.2"]["H5_runmax"]["qhat"] / D["S"]["alphas"]["0.2"]["H7_iid"]["qhat"])
add("v3P.cert.runmax", "24.6", "p1", lambda: D["P"]["certified_fraction_descriptive"]["runmax"])
add("v3S.cert.runmax", "41", "p0", lambda: D["S"]["certified_fraction_descriptive"]["runmax"])
add("v3S.cert.iid", "62", "p0", lambda: D["S"]["certified_fraction_descriptive"]["iid"])
add("v3P.h9.steps", "240", "i", lambda: D["P"]["H9"]["n_steps"]); add("v3P.h9.nonascent", "10", "i", lambda: D["P"]["H9"]["n_steps"] * D["P"]["H9"]["non_ascent_fraction"])
add("v3P.h9.frac", "4.2", "p1", lambda: D["P"]["H9"]["non_ascent_fraction"]); add("v3S.h9.frac", "4.4", "p1", lambda: D["S"]["H9"]["non_ascent_fraction"])
add("v3P.npoints", "480", "i", lambda: D["P"]["n_traj_points"]); add("v3S.npoints", None, "i", lambda: D["S"]["n_traj_points"])

# ---- Table 5: exact single-split distribution ----
for name, a, mean, q05, q95, pp, obs, cdf in (("primary_20x20", "0.1", "0.905", "0.75", "1.00", "0.83", "0.95", "0.76"),
                                               ("primary_20x20", "0.2", "0.810", "0.60", "1.00", "0.78", "0.70", "0.23"),
                                               ("secondary_39x38", "0.1", "0.900", "0.76", "1.00", "0.77", "0.76", "0.05"),
                                               ("secondary_39x38", "0.2", "0.800", "0.63", "0.92", "0.73", "0.66", "0.09")):
    t = lambda n=name, a=a: D["NUM"]["theory"][f"{n} a{a}"]
    for lab, exp, f, k in (("mean", mean, "f3", "mean"), ("q05", q05, "f2", "q05"), ("q95", q95, "f2", "q95"), ("ppass", pp, "f2", "P_pass_rule"),
                           ("observed", obs, "f2", "observed"), ("cdf", cdf, "f2", "observed_cdf")):
        add(f"beta.{name}.a{a}.{lab}", exp, f, lambda t=t, k=k: t()[k])
add("beta.failrate.min", "17", "p0", lambda: 1 - max(v["P_pass_rule"] for v in D["NUM"]["theory"].values()))
add("beta.failrate.max", "27", "p0", lambda: 1 - min(v["P_pass_rule"] for v in D["NUM"]["theory"].values()))

# ---- Table 6: PAC ----
for name, a, qm, kp, qp, w, c in (("primary_20x20", "0.1", "4.90", "none", "inf", "1.00", "0.0"), ("primary_20x20", "0.2", "3.23", "19", "4.90", "0.95", "24.6"),
                                  ("secondary_39x38", "0.1", "3.44", "38", "4.90", "0.92", "25.2"), ("secondary_39x38", "0.2", "2.79", "35", "3.23", "0.71", "43.6")):
    p = lambda n=name, a=a: D["NUM"]["pac"][f"{n} a{a}"]
    add(f"pac.{name}.a{a}.qmarg", qm, "f2", lambda p=p: p()["qhat_marginal"]); add(f"pac.{name}.a{a}.kpac", kp, "i", lambda p=p: p()["k_pac"])
    add(f"pac.{name}.a{a}.qpac", qp, "f2", lambda p=p: p()["qhat_pac"]); add(f"pac.{name}.a{a}.whole", w, "f2", lambda p=p: p()["whole_run"])
    add(f"pac.{name}.a{a}.cert", c, "p1", lambda p=p: p()["certified_fraction"])

# ---- Table 7: baselines ----
BL = {"run-max (Thm 1, reference)": ["0.95/25", "0.70/42.5", "0.76/41", "0.66/51"],
      "i.i.d. split conformal (reference)": ["0.65/61", "0.35/68", "0.61/62", "0.39/68"],
      "B1a Gaussian isotropic, level alpha": ["0.05/82", "0.05/83", "0.03/80", "0.03/82"],
      "B1b Gaussian ensemble-cov, level alpha": ["0.10/76", "0.05/82", "0.08/75", "0.03/80"],
      "B1b Gaussian ensemble-cov, Bonferroni alpha/T": ["0.60/59", "0.40/65", "0.55/60", "0.45/64"],
      "B2b i.i.d. conformal, Bonferroni alpha/T": ["0.95/21", "0.90/30", "0.95/21", "0.89/31"],
      "B2c pooled trajectory points, Bonferroni alpha/T": ["0.95/20", "0.95/25", "0.92/25", "0.82/37"],
      "B3 weighted conformal, level alpha": ["0.55/65", "0.20/72", "0.53/65", "0.29/70"],
      "B3 weighted conformal, Bonferroni alpha/T": ["1.00/20", "1.00/31", "1.00/22", "0.92/32"],
      "B4 ACI gamma=0.005": ["0.60/63", "0.35/68", "0.55/63", "0.39/68"],
      "B4 ACI gamma=0.05": ["0.55/63", "0.30/70", "0.53/62", "0.24/69"]}
cells = [("primary_20x20", "0.1"), ("primary_20x20", "0.2"), ("secondary_39x38", "0.1"), ("secondary_39x38", "0.2")]
bl = lambda d, a, m: D["BL"]["designs"][d]["alphas"][a][m]
for m, vals in BL.items():
    for (d, a), v in zip(cells, vals):
        w, c = v.split("/"); slug = m.split(" ")[0] + ("" if "level" in m or "reference" in m or "ACI" in m else ".bonf") + ("." + m.split("=")[-1] if "ACI" in m else "")
        if "Gaussian" in m:
            slug += ".iso" if "isotropic" in m else ".cov"
        add(f"bl.{slug}.{d}.a{a}.whole", w, "f2", lambda d=d, a=a, m=m: bl(d, a, m)["whole_run"])
        add(f"bl.{slug}.{d}.a{a}.cert", c, "p1" if "." in c else "p0", lambda d=d, a=a, m=m: bl(d, a, m)["certified_fraction"])
add("bl.perstep.mincal.a0.1", "119", "i", lambda: bl("secondary_39x38", "0.1", "B2a per-step conformal, Bonferroni alpha/T")["min_cal_runs_for_finite"])
add("bl.perstep.mincal.a0.2", "59", "i", lambda: bl("secondary_39x38", "0.2", "B2a per-step conformal, Bonferroni alpha/T")["min_cal_runs_for_finite"])
add("bl.perstep.finite.any", "0", "i", lambda: sum(bl(d, a, "B2a per-step conformal, Bonferroni alpha/T")["finite"] for d, a in cells))
add("bl.ess.primary", "96", "f0", lambda: D["BL"]["designs"]["primary_20x20"]["weighted_median_ESS"])
add("bl.ess.secondary", "92", "f0", lambda: D["BL"]["designs"]["secondary_39x38"]["weighted_median_ESS"])
add("bl.acc.primary", "0.70", "f2", lambda: D["BL"]["designs"]["primary_20x20"]["weighted_classifier_train_acc"])
add("bl.acc.secondary", "0.80", "f2", lambda: D["BL"]["designs"]["secondary_39x38"]["weighted_classifier_train_acc"])
G = lambda: [bl(d, a, m) for d, a in cells for m in ("B1a Gaussian isotropic, level alpha", "B1b Gaussian ensemble-cov, level alpha")]
add("bl.gauss.periter.min", "46", "p0", lambda: min(x["per_iterate"] for x in G())); add("bl.gauss.periter.max", "73", "p0", lambda: max(x["per_iterate"] for x in G()))
add("bl.gauss.whole.min", "3", "p0", lambda: min(x["whole_run"] for x in G())); add("bl.gauss.whole.max", "10", "p0", lambda: max(x["whole_run"] for x in G()))
W = lambda: [bl(d, a, m)["whole_run"] for d, a in cells for m in ("B3 weighted conformal, level alpha", "B4 ACI gamma=0.005", "B4 ACI gamma=0.05")]
add("bl.wtaci.whole.min", "20", "p0", lambda: min(W())); add("bl.wtaci.whole.max", "60", "p0", lambda: max(W()))
RAT = lambda: [bl(d, a, m)["median_radius"] / bl(d, a, "run-max (Thm 1, reference)")["median_radius"] for d, a in cells
               for m in ("B2b i.i.d. conformal, Bonferroni alpha/T", "B2c pooled trajectory points, Bonferroni alpha/T")]
add("bl.bonf.radiusratio.min", "1.2", "f1", lambda: min(RAT())); add("bl.bonf.radiusratio.max", "1.7", "f1", lambda: max(RAT()))
CB = lambda: [bl(d, a, m)["certified_fraction"] for d, a in cells for m in ("B2b i.i.d. conformal, Bonferroni alpha/T", "B2c pooled trajectory points, Bonferroni alpha/T")]
add("bl.bonf.cert.min", "20", "p0", lambda: min(CB())); add("bl.bonf.cert.max", "37", "p0", lambda: max(CB()))

# ---- mesh refinement ----
ms = lambda p: D["MESH"]["summary"][p]
add("mesh.141.dJ.median", "5.5", "p1", lambda: ms("E141_vs_E100")["dJ_rel"]["median"]); add("mesh.141.dJ.max", "8.9", "p1", lambda: ms("E141_vs_E100")["dJ_rel"]["max"])
add("mesh.200.dJ.median", "10.1", "p1", lambda: ms("E200_vs_E100")["dJ_rel"]["median"]); add("mesh.200.dJ.max", "14.8", "p1", lambda: ms("E200_vs_E100")["dJ_rel"]["max"])
add("mesh.200v141.dJ.median", "5.0", "p1", lambda: ms("E200_vs_E141")["dJ_rel"]["median"])
add("mesh.141.dgrad.median", "10", "p0", lambda: ms("E141_vs_E100")["dgrad_rel_M"]["median"]); add("mesh.200.dgrad.median", "19", "p0", lambda: ms("E200_vs_E100")["dgrad_rel_M"]["median"])
add("mesh.141.cos.median", "0.998", "f3", lambda: ms("E141_vs_E100")["cos_M"]["median"]); add("mesh.200.cos.median", "0.995", "f3", lambda: ms("E200_vs_E100")["cos_M"]["median"])
add("mesh.capped.141", "8", "i", lambda: D["MESH"]["summary"]["capped_adjoints"]["E141"]); add("mesh.capped.200", "10", "i", lambda: D["MESH"]["summary"]["capped_adjoints"]["E200"])
for e, exp in (("E100", "28"), ("E141", "56"), ("E200", "110")):
    add(f"mesh.cells.{e}", exp, "f0", lambda e=e: D["MESH"]["summary"]["n_cells_median"][e] / 1000)

# ---- investigation of the secondary shortfall ----
inv = D["INV"]
add("inv.kruskal", "0.59", "f2", lambda: inv["kruskal_4groups_p"]); add("inv.mwu.origext", "0.52", "f2", lambda: inv["mwu_orig_vs_ext_p"])
add("inv.mwu.caltest", "0.35", "f2", lambda: inv["mwu_cal_vs_test_p"])
add("inv.max.extcal", "3.76", "f2", lambda: inv["group_runmax"]["ext_cal"]["max"]); add("inv.max.origcal", "8.05", "f2", lambda: inv["group_runmax"]["orig_cal"]["max"])
add("inv.max.exttest", "5.89", "f2", lambda: inv["group_runmax"]["ext_test"]["max"])
add("inv.uncovered.a0.1", "9", "i", lambda: len(inv["uncovered_test_runs"]["0.1"]))
add("inv.uncovered.capped.a0.1", "0", "i", lambda: sum(x["has_capped_adjoint"] for x in inv["uncovered_test_runs"]["0.1"]))
add("inv.resplit.a0.1.q05", "0.79", "f2", lambda: inv["random_resplits"]["0.1"]["q05"]); add("inv.resplit.a0.1.q95", "1.00", "f2", lambda: inv["random_resplits"]["0.1"]["q95"])
add("inv.resplit.a0.1.pct", "5", "p0", lambda: inv["random_resplits"]["0.1"]["observed_percentile"])
add("inv.ncal.ext", None, "i", lambda: inv["group_runmax"]["ext_cal"]["n"]); add("inv.ntest.ext", None, "i", lambda: inv["group_runmax"]["ext_test"]["n"])

# ---- abstract-level ranges ----
IID = lambda: [D["s2"]["alphas"]["0.1"]["H7_iid"]["whole_run"], D["P"]["alphas"]["0.1"]["H7_iid"]["whole_run"], D["S"]["alphas"]["0.1"]["H7_iid"]["whole_run"]]
add("abs.iid.a0.1.min", "20", "p0", lambda: min(IID())); add("abs.iid.a0.1.max", "65", "p0", lambda: max(IID()))
add("abs.v3.gradvsval.grad", "20", "p0", lambda: D["v3fno"]["ensemble"]["median_rel_grad_err_J"]); add("abs.v3.gradvsval.val", "2", "p0", lambda: D["v3fno"]["ensemble"]["median_rel_err_J"])

# ---- evaluate ----
rows = []
for key, exp, f, get in CHECKS:
    try:
        got = fmt(get(), f)
    except Exception as e:  # noqa: BLE001  (a wrong key must show up in the report, not stop it)
        rows.append({"key": key, "expected": exp, "from_json": None, "status": "MISSING", "error": repr(e)[:200]}); continue
    st = "NEW" if exp is None else ("MATCH" if got == exp else "MISMATCH")
    rows.append({"key": key, "expected": exp, "from_json": got, "status": st,
                 "in_manuscript": (exp in tex) if (exp is not None and tex) else None})
OUT.mkdir(parents=True, exist_ok=True)
cnt = {s: sum(r["status"] == s for r in rows) for s in ("MATCH", "MISMATCH", "MISSING", "NEW")}
res = {"manuscript": str(TEX) if TEX else None, "manuscript_sha": sha(TEX) if TEX else None,
       "files": {k: {"path": str(p.relative_to(R)), "sha": sha(p)[:16]} for k, p in FILES.items()}, "counts": cnt, "rows": rows}
(OUT / "numbers_check_v1.json").write_text(json.dumps(res, indent=1))
order = {"MISMATCH": 0, "MISSING": 1, "NEW": 2, "MATCH": 3}
lines = [f"numbers check v1: {cnt}", f"manuscript: {res['manuscript']}", ""]
for r in sorted(rows, key=lambda r: order[r["status"]]):
    lines.append(f"{r['status']:8s} {r['key']:52s} manuscript={r['expected']!s:14s} json={r['from_json']!s:14s} "
                 + (f"in_tex={r['in_manuscript']}" if r.get("in_manuscript") is not None else "") + (f" {r.get('error', '')}" if r["status"] == "MISSING" else ""))
(OUT / "numbers_check_v1.txt").write_text("\n".join(lines) + "\n")
mac = ["% generated by check_manuscript_numbers.py from saved results; use \\pnum{key}", "\\newcommand{\\pnum}[1]{\\csname pn@#1\\endcsname}"]
INF = "\\infty"
for r in rows:
    if r["from_json"] is not None:
        val = INF if r["from_json"] == "inf" else r["from_json"]
        mac.append("\\expandafter\\def\\csname pn@" + r["key"] + "\\endcsname{" + val + "}")
(OUT / "numbers_v1.tex").write_text("\n".join(mac) + "\n")
row = ("| " + __import__("datetime").datetime.utcnow().strftime("%H:%M %b %d") + " (UTC) | Colab CPU (_staging/check_manuscript_numbers.py) | "
       f"**Manuscript number check (descriptive).** {len(rows)} quoted numbers re-pulled from saved results and compared with "
       f"manuscript_v1.tex (sha {res['manuscript_sha'][:16] if TEX else 'n/a'}…): MATCH {cnt['MATCH']}, MISMATCH {cnt['MISMATCH']}, MISSING key "
       f"{cnt['MISSING']}, NEW (not yet in text) {cnt['NEW']}. Outputs results/paper_numbers/numbers_check_v1.{{json,txt}} "
       f"(sha {sha(OUT / 'numbers_check_v1.json')[:16]}…) and numbers_v1.tex (\\pnum macros). Mismatches are fixed in the manuscript, "
       "never in the results. | Descriptive | — |\n")
LOG.write_bytes(b + row.encode("utf-8"))
print("\n".join(lines[:60])); print("LOG", h[:8], "->", sha(LOG)[:8], flush=True)
