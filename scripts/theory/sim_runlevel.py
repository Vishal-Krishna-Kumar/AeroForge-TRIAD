"""Monte Carlo sanity check of the run-level certificate theory (docs/theory/run_level_certificates_v1.md).

Toy problem, NOT CFD and not evidence about airfoils: J(u) = smooth function on [0,1]^d, true gradient g, a fixed
"surrogate" ĝ = g + structured error e(u) (smooth, fixed across runs), σ̂(u) = a fixed positive scale that is only
partly informative. Deterministic normalised gradient ascent from i.i.d. uniform starts (Assumption A holds:
everything is fixed given D0, only the start is random). For each repetition: R calibration runs + 1 test run;
methods: run-max (Thm 1), one-random-iterate (Thm 2), pooled (baseline, no guarantee).
Checks: P(all test-run iterates covered) for run-max ≥ 1-α; P(random test iterate covered) for subsample ≥ 1-α;
pooled whole-run coverage (expected to fall below 1-α). Also checks the ascent inequality of Prop. 3 on E.
Usage: python scripts/theory/sim_runlevel.py [reps]
"""
import math
import sys

import numpy as np

D, T, H = 4, 15, 0.05
rng0 = np.random.default_rng(12345)
A = rng0.normal(size=(6, D)); B = rng0.uniform(0, 2 * np.pi, 6); W = rng0.normal(size=6)
EA = rng0.normal(size=(5, D)) * 3; EB = rng0.uniform(0, 2 * np.pi, 5); EW = rng0.normal(size=(5, D)) * 0.15


def grad(u):          # J(u) = sum_k W_k sin(A_k·u + B_k) - 0.5||u-0.5||^2
    return (W * np.cos(A @ u + B)) @ A - (u - 0.5)


def err(u):           # smooth, input-dependent surrogate error (fixed function)
    return np.sin(EA @ u + EB) @ EW


def sig(u):           # partially informative scale
    return 0.05 + 0.1 * np.abs(np.sin(EA[0] @ u))


def run(u0):
    u, pts = u0.copy(), []
    for _ in range(T):
        g, gh = grad(u), grad(u) + err(u)
        s = np.linalg.norm(g - gh) / sig(u)
        pts.append((s, g, gh, sig(u)))
        u = np.clip(u + H * gh / np.linalg.norm(gh), 0, 1)
    return pts


def cq(x, a):
    x = np.sort(x); k = math.ceil((len(x) + 1) * (1 - a))
    return np.inf if k > len(x) else x[k - 1]


def main(reps=2000, R=40, alpha=0.1):
    rng = np.random.default_rng(7)
    whole, sub, pooled_whole, pooled_pt, asc_ok, n_acc = [], [], [], [], 0, 0
    for _ in range(reps):
        runs = [run(rng.uniform(0, 1, D)) for _ in range(R + 1)]
        S = [np.array([p[0] for p in r]) for r in runs]
        cal, te = S[:R], S[R]
        q1 = cq([s.max() for s in cal], alpha)
        whole.append(float((te <= q1).all()))
        q2 = cq([s[rng.integers(len(s))] for s in cal], alpha)
        sub.append(float(te[rng.integers(len(te))] <= q2))
        qp = cq(np.concatenate(cal), alpha)
        pooled_whole.append(float((te <= qp).all())); pooled_pt.append(float((te <= qp).mean()))
        if (te <= q1).all():          # Prop. 3 on E: gate with c = 0.5
            for s, g, gh, sg in runs[R]:
                if q1 * sg <= 0.5 * np.linalg.norm(gh):
                    n_acc += 1
                    asc_ok += float(g @ gh >= (1 - 0.5) * gh @ gh - 1e-12)
    print(f"alpha={alpha} R={R} reps={reps}")
    print(f"Thm1 run-max: P(all iterates covered) = {np.mean(whole):.3f}  (target >= {1-alpha})")
    print(f"Thm2 subsample: P(random iterate covered) = {np.mean(sub):.3f}  (target >= {1-alpha})")
    print(f"pooled baseline: whole-run {np.mean(pooled_whole):.3f}, pointwise {np.mean(pooled_pt):.3f}")
    print(f"Prop3 on E: accepted steps {n_acc}, ascent inequality held on {asc_ok}/{n_acc}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
