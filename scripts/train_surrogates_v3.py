"""Train surrogate ensembles on Dataset V3 (transonic Euler, CST-13) and evaluate H1'' on the V3 test split.

Protocol: research/protocol/experiment_protocol_v3.md §4 and §6 (H1''): exactly the V2 architectures and training
settings (scripts/train_surrogates_v2.py hyper-parameters, unchanged here; FNO 500 epochs, MLP 3000 epochs), 5 seeds.
* Fit set = first floor(0.875 N_train) train cases in case-id order, validation = the rest (monitoring only). cal and
  test are never used for fitting or selection; test is evaluated once, after training.
* Value targets CL and CD (+ fields for the FNO); the objective J = CL / (CD + CD0), CD0 = 0.008, and its design
  gradient are computed from the surrogate's (CL, CD) and their autograd Jacobian:
      dJ = dCL / (CD + CD0) - CL dCD / (CD + CD0)^2,
  compared with the CFD adjoint gradient dJ_dtheta stored in the V3 npz files (cal/test).
* Metric M = diag(span^2) of the V3 box (datasets/designs/dataset_v3_design.json), as in V2.
* H1'' is reported with the V2 rule: supported iff median relative gradient error > 3 x median relative J error and
  more than 5 % of test points have cosine < 0.98; falsified iff >= 95 % have cosine >= 0.98.
* The snapshot seed (FNO seed 0) records frames for the training-visualisation video (as V2, with J instead of L/D).

Usage:
  python scripts/train_surrogates_v3.py --model fno --seeds 0 1 2 3 4 --epochs 500 --out checkpoints/v3/fno
  python scripts/train_surrogates_v3.py --model mlp --seeds 0 1 2 3 4 --epochs 3000 --out checkpoints/v3/mlp
Environment: AF_ROOT (project root), AF_DATA (optional local copy of datasets/processed/dataset_v3), AF_REPORTS (report
folder; default results/reports_v3 — smoke tests point it elsewhere), AF_N_FIT (smoke tests only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("AF_ROOT", "."))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data.postprocess import GridSpec  # noqa: E402
from gradient_certification.cdgc import cosine, m_norm  # noqa: E402
from models.training import TrainConfig, build_model, load_split, predict_coefficients, predict_gradients, train  # noqa: E402

PROC = Path(os.environ.get("AF_DATA", ROOT / "datasets" / "processed" / "dataset_v3"))
CD0 = 0.008
_D = json.loads((ROOT / "datasets" / "designs" / "dataset_v3_design.json").read_text())
SPAN = np.asarray(_D["box_hi"]) - np.asarray(_D["box_lo"])
M = np.diag(SPAN ** 2)
HP = {"fno": dict(batch=16, lr=1e-3, width=32, modes=(16, 24), layers=4),  # = scripts/train_surrogates_v2.HP
      "mlp": dict(batch=64, lr=3e-3)}


def data_hash(folder: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(folder.glob("*.npz")):
        h.update(f.name.encode()); h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()


def load_v3(folder: Path, with_fields: bool) -> dict:
    """load_split (V1/V2 loader, unchanged) plus the V3 adjoint J gradient as 'g_J' (NaN where absent)."""
    d = load_split(folder, with_fields=with_fields)
    g = []
    for cid in d["case_id"]:
        z = np.load(folder / f"{cid}.npz")
        g.append(z["dJ_dtheta"] if "dJ_dtheta" in z else np.full(d["theta"].shape[1], np.nan))
    d["g_J"] = np.asarray(g, dtype=float)
    return d


def subset(d: dict, idx) -> dict:
    return {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(d["theta"]) else v) for k, v in d.items()}


def j_and_grad(model, theta: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Surrogate J = CL/(CD+CD0) and dJ/dtheta from the autograd Jacobian of (CL, CD)."""
    c = predict_coefficients(model, theta, device=device)
    _, _, jac = predict_gradients(model, theta, device=device)  # jac: (N, 2, d)
    den = c[:, 1] + CD0
    J = c[:, 0] / den
    g = jac[:, 0, :] / den[:, None] - (c[:, 0] / den ** 2)[:, None] * jac[:, 1, :]
    return J, g


class Recorder:
    """Training-video frames for one member (same content as V2, J instead of L/D)."""

    def __init__(self, path: Path, val: dict, cal: dict, device: str, n_fields: int = 4, n_grad: int = 16):
        self.path, self.device = path, device
        jv = val["CL"] / (val["CD"] + CD0)
        order = np.argsort(jv)
        self.fi = order[np.linspace(0, len(order) - 1, min(n_fields, len(order))).round().astype(int)]  # low -> high J
        self.val_theta = val["theta"]
        self.true_fields = np.asarray(val["p"])[self.fi].astype(np.float32)
        self.mask = np.asarray(val["mask_fluid"])[self.fi]
        ok = ~np.isnan(cal["g_J"]).any(axis=1)
        gi = np.flatnonzero(ok)[:n_grad]
        self.cal_theta, self.g_true = cal["theta"][gi], cal["g_J"][gi]
        self.frames: dict[str, list] = {k: [] for k in ("epoch", "p_pred", "val_CL", "val_CD", "grad_cos", "loss",
                                                         "val_rel_err_LD", "val_field_mse", "elapsed_s")}
        self.meta = {"field_cases": [val["case_id"][i] for i in self.fi], "grad_cases": [cal["case_id"][i] for i in gi],
                     "objective": "J = CL/(CD+0.008)", "note": "val_rel_err_LD is the training loop's L/D monitor"}

    def __call__(self, ep: int, model, rec: dict) -> None:
        dev = self.device
        with torch.no_grad():
            th = torch.tensor(self.val_theta[self.fi], dtype=torch.float32, device=dev)
            p = model.physical_fields(th)[:, 0].cpu().numpy() if hasattr(model, "physical_fields") else None
        c = predict_coefficients(model, self.val_theta, device=dev)
        _, g = j_and_grad(model, self.cal_theta, dev) if len(self.cal_theta) else (None, np.zeros((0, len(SPAN))))
        model.train()
        f = self.frames
        f["epoch"].append(ep); f["p_pred"].append(p.astype(np.float16) if p is not None else np.zeros(1, np.float16))
        f["val_CL"].append(c[:, 0]); f["val_CD"].append(c[:, 1])
        f["grad_cos"].append(cosine(g, self.g_true, M) if len(g) else np.zeros(0))
        for k in ("loss", "val_rel_err_LD", "val_field_mse", "elapsed_s"):
            f[k].append(float(rec.get(k, np.nan)))
        if len(f["epoch"]) % 25 == 0:
            self.flush()

    def flush(self) -> None:
        f = {k: np.asarray(v) for k, v in self.frames.items()}
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **f, p_true=self.true_fields, mask=self.mask, val_theta=self.val_theta,
                            field_index=self.fi, cal_theta=self.cal_theta, g_true=self.g_true, meta=json.dumps(self.meta))
        os.replace(tmp, self.path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["fno", "mlp"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--snapshot-seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    gx, gy = GridSpec().coords()
    t_load = time.time()
    tr_all = load_v3(PROC / "train", with_fields=True)
    cal = load_v3(PROC / "cal", with_fields=False)
    te = load_v3(PROC / "test", with_fields=False)
    assert list(tr_all["case_id"]) == sorted(tr_all["case_id"])
    n_tr = len(tr_all["theta"])
    n_fit = int(os.environ.get("AF_N_FIT", int(np.floor(0.875 * n_tr))))  # protocol v3 §4
    fit, val = subset(tr_all, np.arange(n_fit)), subset(tr_all, np.arange(n_fit, n_tr))
    meta = {"train_hash": data_hash(PROC / "train"), "test_hash": data_hash(PROC / "test"), "cal_hash": data_hash(PROC / "cal"),
            "n_train": n_tr, "n_fit": n_fit, "n_val": int(len(val["theta"])), "n_cal": int(len(cal["theta"])),
            "n_test": int(len(te["theta"])), "args": vars(args), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "load_s": round(time.time() - t_load, 1)}
    print("META", json.dumps(meta), flush=True)
    preds_J, preds_g = [], []
    for s in args.seeds:
        cfg = TrainConfig(model=args.model, seed=s, epochs=args.epochs, **HP[args.model])
        final = out / f"seed{s}.pt"
        if final.exists() and (out / f"seed{s}.json").exists():
            print(f"seed{s} already trained; loading", flush=True)
            model = build_model(cfg, gx, gy, fit).to(cfg.device)
            model.load_state_dict(torch.load(final, map_location=cfg.device)); model.eval()
        else:
            rec = Recorder(out / f"video_frames_seed{s}.npz", val, cal, cfg.device) if s == args.snapshot_seed and len(val["theta"]) else None
            snap_every = max(1, args.epochs // args.frames) if rec else 0
            model, info = train(cfg, fit, gx, gy, log=lambda line, s=s: print(f"seed{s}", line, flush=True),
                                val=val if len(val["theta"]) else None, ckpt=out / f"seed{s}.ckpt", snapshot=rec,
                                snap_every=snap_every)
            if rec:
                rec.flush()
            torch.save(model.state_dict(), final)
            (out / f"seed{s}.json").write_text(json.dumps({**info, **meta}, default=str))
            print(f"seed{s} DONE", json.dumps(info["history"][-1]), flush=True)
        J, g = j_and_grad(model, te["theta"], cfg.device)
        preds_J.append(J); preds_g.append(g)
    PJ, G = np.stack(preds_J), np.stack(preds_g)  # (S, N), (S, N, 13)
    j_true = te["CL"] / (te["CD"] + CD0)
    ok = ~np.isnan(te["g_J"]).any(axis=1)
    g_true = te["g_J"]

    def metrics(Jp, g):
        cos = cosine(g[ok], g_true[ok], M)
        return {"median_rel_err_J": float(np.median(np.abs(Jp - j_true) / np.maximum(np.abs(j_true), 1e-2))),
                "median_rel_grad_err_J": float(np.median(m_norm(g[ok] - g_true[ok], M) / m_norm(g_true[ok], M))),
                "frac_cos_below_0.98": float(np.mean(cos < 0.98)), "median_cos": float(np.median(cos)),
                "n_grad_points": int(ok.sum())}

    res = {"meta": meta, "members": [metrics(PJ[k], G[k]) for k in range(len(args.seeds))],
           "ensemble": metrics(PJ.mean(0), G.mean(0))}
    e = res["ensemble"]
    res["H1pp_supported"] = bool(e["median_rel_grad_err_J"] > 3 * e["median_rel_err_J"] and e["frac_cos_below_0.98"] > 0.05)
    res["H1pp_falsified"] = bool(1 - e["frac_cos_below_0.98"] >= 0.95)
    np.savez(out / "test_predictions.npz", J=PJ, G=G, theta=te["theta"], case_id=te["case_id"])
    (out / "h1v3_test_metrics.json").write_text(json.dumps(res, indent=1))
    rep = Path(os.environ.get("AF_REPORTS", ROOT / "results" / "reports_v3")); rep.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out / "h1v3_test_metrics.json", rep / f"h1v3_test_metrics_{args.model}.json")
    print("H1pp", json.dumps(e), "supported" if res["H1pp_supported"] else "", "falsified" if res["H1pp_falsified"] else "", flush=True)
    print("AF_TRAIN_V3_DONE", args.model, flush=True)


if __name__ == "__main__":
    main()
