"""Train surrogate ensembles on Dataset V1 and evaluate H1 (gradient gap) on the test split.

Protocol: research/protocol/experiment_protocol_v1.md §2–3.
* Fit on train cases 0–447; the last 64 Sobol points of train (448–511) are the validation fold, used only
  for monitoring (hyper-parameters are fixed in advance in this file). cal and test are never used for
  fitting or selection. The test split is evaluated once, after training.
* Each member writes a resumable checkpoint (model, optimiser, scheduler, RNG, history) every 25 epochs, so a
  run interrupted on one machine continues on another from the last checkpoint.
* The snapshot seed (FNO seed 0 by default) records frames for the training-visualisation video:
  predicted vs CFD pressure fields on 4 fixed validation airfoils, validation CL/CD predictions, and the
  cosine between surrogate and adjoint L/D gradients on 16 fixed calibration cases (monitoring only).

Usage:
  python scripts/train_surrogates.py --model fno --seeds 0 1 2 3 4 --epochs 400 --out checkpoints/v1/fno
  python scripts/train_surrogates.py --model mlp --seeds 0 1 2 3 4 --epochs 3000 --out checkpoints/v1/mlp
Environment: AF_ROOT (project root on Drive), AF_DATA (optional local copy of datasets/processed/dataset_v1).
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
from gradient_certification.cdgc import cosine, ld_gradient, m_norm  # noqa: E402
from models.training import THETA_SPAN, TrainConfig, load_split, predict_coefficients, predict_gradients, train  # noqa: E402

PROC = Path(os.environ.get("AF_DATA", ROOT / "datasets" / "processed" / "dataset_v1"))
N_FIT = int(os.environ.get("AF_N_FIT", "448"))  # smoke tests only override this
HP = {"fno": dict(batch=16, lr=1e-3, width=32, modes=(16, 24), layers=4),
      "mlp": dict(batch=64, lr=3e-3)}


def data_hash(folder: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(folder.glob("*.npz")):
        h.update(f.name.encode()); h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()


def subset(d: dict, idx) -> dict:
    return {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(d["theta"]) else v) for k, v in d.items()}


class Recorder:
    """Training-video frames for one member (kept in memory, flushed to .npz)."""

    def __init__(self, path: Path, val: dict, cal: dict, gx, gy, device: str, n_fields: int = 4, n_grad: int = 16):
        self.path, self.device = path, device
        ld = val["CL"] / val["CD"]
        order = np.argsort(ld)
        self.fi = order[np.linspace(0, len(order) - 1, n_fields).round().astype(int)]  # low → high L/D
        self.val_theta = val["theta"]
        self.true_fields = np.stack([val["p"][self.fi]]).squeeze(0).astype(np.float32)
        self.mask = np.asarray(val["mask_fluid"])[self.fi]
        ok = ~np.isnan(cal["jac"]).any(axis=(1, 2))
        gi = np.flatnonzero(ok)[:n_grad]
        self.cal_theta = cal["theta"][gi]
        self.g_true = ld_gradient(cal["CL"][gi], cal["CD"][gi], cal["jac"][gi, 0], cal["jac"][gi, 1])
        self.frames: dict[str, list] = {k: [] for k in ("epoch", "p_pred", "val_CL", "val_CD", "grad_cos", "loss",
                                                         "val_rel_err_LD", "val_field_mse", "elapsed_s")}
        self.meta = {"field_cases": [val["case_id"][i] for i in self.fi], "grad_cases": [cal["case_id"][i] for i in gi]}

    def __call__(self, ep: int, model, rec: dict) -> None:
        dev = self.device
        with torch.no_grad():
            th = torch.tensor(self.val_theta[self.fi], dtype=torch.float32, device=dev)
            p = model.physical_fields(th)[:, 0].cpu().numpy() if hasattr(model, "physical_fields") else None
        c = predict_coefficients(model, self.val_theta, device=dev)
        _, g, _ = predict_gradients(model, self.cal_theta, device=dev)
        model.train()
        cos = cosine(g, self.g_true, np.diag(THETA_SPAN ** 2))
        f = self.frames
        f["epoch"].append(ep); f["p_pred"].append(p.astype(np.float16) if p is not None else np.zeros(1, np.float16))
        f["val_CL"].append(c[:, 0]); f["val_CD"].append(c[:, 1]); f["grad_cos"].append(cos)
        for k in ("loss", "val_rel_err_LD", "val_field_mse", "elapsed_s"):
            f[k].append(float(rec.get(k, np.nan)))
        if len(f["epoch"]) % 25 == 0:
            self.flush()

    def flush(self) -> None:
        f = {k: np.asarray(v) for k, v in self.frames.items()}
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **f, p_true=self.true_fields, mask=self.mask, val_theta=self.val_theta,
                            field_index=self.fi, cal_theta=self.cal_theta, g_true=self.g_true,
                            meta=json.dumps(self.meta))
        os.replace(tmp, self.path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["fno", "mlp"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--w-sobolev", type=float, default=0.0)
    ap.add_argument("--snapshot-seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    gx, gy = GridSpec().coords()
    t_load = time.time()
    tr_all = load_split(PROC / "train", with_fields=True)
    cal = load_split(PROC / "cal", with_fields=False)
    te = load_split(PROC / "test", with_fields=False)
    assert list(tr_all["case_id"][:3]) == ["D1_train_0000", "D1_train_0001", "D1_train_0002"], tr_all["case_id"][:3]
    fit, val = subset(tr_all, np.arange(N_FIT)), subset(tr_all, np.arange(N_FIT, len(tr_all["theta"])))
    meta = {"train_hash": data_hash(PROC / "train"), "test_hash": data_hash(PROC / "test"), "cal_hash": data_hash(PROC / "cal"),
            "n_fit": N_FIT, "n_val": int(len(val["theta"])), "n_test": int(len(te["theta"])), "args": vars(args),
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "load_s": round(time.time() - t_load, 1)}
    print("META", json.dumps(meta), flush=True)
    preds_c, preds_g = [], []
    for s in args.seeds:
        cfg = TrainConfig(model=args.model, seed=s, epochs=args.epochs, w_sobolev=args.w_sobolev, **HP[args.model])
        final = out / f"seed{s}.pt"
        if final.exists() and (out / f"seed{s}.json").exists():
            print(f"seed{s} already trained; loading", flush=True)
            from models.training import build_model
            model = build_model(cfg, gx, gy, fit).to(cfg.device)
            model.load_state_dict(torch.load(final, map_location=cfg.device)); model.eval()
        else:
            rec = None
            if s == args.snapshot_seed:
                rec = Recorder(out / f"video_frames_seed{s}.npz", val, cal, gx, gy, cfg.device)
            snap_every = max(1, args.epochs // args.frames) if rec else 0
            model, info = train(cfg, fit, gx, gy, log=lambda line, s=s: print(f"seed{s}", line, flush=True), val=val,
                                ckpt=out / f"seed{s}.ckpt", snapshot=rec, snap_every=snap_every)
            if rec:
                rec.flush()
            torch.save(model.state_dict(), final)
            (out / f"seed{s}.json").write_text(json.dumps({**info, **meta}, default=str))
            print(f"seed{s} DONE", json.dumps(info["history"][-1]), flush=True)
        dev = cfg.device
        preds_c.append(predict_coefficients(model, te["theta"], device=dev))
        _, g, _ = predict_gradients(model, te["theta"], device=dev)
        preds_g.append(g)
    C = np.stack(preds_c); G = np.stack(preds_g)          # (S, N, 2), (S, N, 4)
    cl, cd = te["CL"], te["CD"]
    ld_true = cl / cd
    ok = ~np.isnan(te["jac"]).any(axis=(1, 2))
    g_true = ld_gradient(cl, cd, te["jac"][:, 0], te["jac"][:, 1])
    M = np.diag(THETA_SPAN ** 2)
    res = {"meta": meta, "members": [], "ensemble": {}}

    def metrics(c, g):
        ld = c[:, 0] / c[:, 1]
        e_ld = np.abs(ld - ld_true) / np.maximum(np.abs(ld_true), 1e-2)
        cos = cosine(g[ok], g_true[ok], M)
        rel_g = m_norm(g[ok] - g_true[ok], M) / m_norm(g_true[ok], M)
        return {"median_rel_err_CL": float(np.median(np.abs(c[:, 0] - cl) / np.maximum(np.abs(cl), 1e-3))),
                "median_rel_err_CD": float(np.median(np.abs(c[:, 1] - cd) / cd)),
                "median_rel_err_LD": float(np.median(e_ld)), "median_rel_grad_err_LD": float(np.median(rel_g)),
                "frac_cos_below_0.98": float(np.mean(cos < 0.98)), "median_cos": float(np.median(cos)),
                "n_grad_points": int(ok.sum())}
    for k in range(len(args.seeds)):
        res["members"].append(metrics(C[k], G[k]))
    res["ensemble"] = metrics(C.mean(0), G.mean(0))
    e = res["ensemble"]
    res["H1_supported"] = bool(e["median_rel_grad_err_LD"] > 3 * e["median_rel_err_LD"] and e["frac_cos_below_0.98"] > 0.05)
    res["H1_falsified"] = bool(1 - e["frac_cos_below_0.98"] >= 0.95)
    np.savez(out / "test_predictions.npz", C=C, G=G, theta=te["theta"], case_id=te["case_id"])
    (out / "h1_test_metrics.json").write_text(json.dumps(res, indent=1))
    rep = ROOT / "results" / "reports"; rep.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out / "h1_test_metrics.json", rep / f"h1_test_metrics_{args.model}.json")
    print("H1", json.dumps(res["ensemble"]), "supported" if res["H1_supported"] else "", "falsified" if res["H1_falsified"] else "", flush=True)
    print("AF_TRAIN_DONE", args.model, flush=True)


if __name__ == "__main__":
    main()
