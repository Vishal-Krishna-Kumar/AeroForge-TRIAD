"""Training and evaluation of Dataset V1 surrogates (AeroForge-TRIAD).

Data: one .npz per case (data.postprocess / scripts/run_dataset_v1.py) with theta, CL, CD, fields p/u/v
on the 192x96 wind-frame grid and, for cal/test cases, dCL_dtheta and dCD_dtheta from the adjoint.

Losses (all on standardised quantities):
  field  : mean squared error of (p, u, v) over fluid grid points        (FNO only)
  coeff  : mean squared error of (CL, CD)
  sobolev: mean squared error of d(CL, CD)/d(theta), scaled per variable  (optional; derivative-informed
           baseline in the spirit of Czarnecki et al. 2017 / O'Leary-Roseberry et al. 2022)
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from models.geometry_torch import sdf_grid
from models.surrogates import FNOSurrogate, MLPSurrogate, lift_to_drag_and_grad

DV = ("m", "p", "t", "alpha_deg")
THETA_SPAN = np.array([0.04, 0.4, 0.06, 6.0])  # final Dataset V1 box widths (ADR-003), used to scale gradients


def load_split(folder: str | Path, with_fields: bool = True) -> dict[str, np.ndarray]:
    files = sorted(Path(folder).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz files in {folder}")
    out: dict[str, list] = {"case_id": [], "theta": [], "CL": [], "CD": []}
    if with_fields:
        out.update({"p": [], "u": [], "v": [], "mask_fluid": []})
    grads, gld = [], []
    for f in files:
        z = np.load(f)
        out["case_id"].append(f.stem)
        for k in ("theta", "CL", "CD") + (("p", "u", "v", "mask_fluid") if with_fields else ()):
            out[k].append(z[k])
        d = len(z["theta"])
        grads.append(np.stack([z["dCL_dtheta"], z["dCD_dtheta"]]) if "dCL_dtheta" in z else np.full((2, d), np.nan))
        gld.append(z["dLD_dtheta"] if "dLD_dtheta" in z else np.full(d, np.nan))   # V2: L/D gradient only
    res = {k: np.asarray(v) for k, v in out.items()}
    res["jac"] = np.asarray(grads)
    res["g_ld"] = np.asarray(gld)
    return res


@dataclass
class TrainConfig:
    model: str = "fno"           # fno | mlp
    seed: int = 0
    epochs: int = 300
    batch: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-5
    w_field: float = 1.0
    w_coeff: float = 1.0
    w_sobolev: float = 0.0
    width: int = 32
    modes: tuple[int, int] = (16, 24)
    layers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(cfg: TrainConfig, gx: np.ndarray, gy: np.ndarray, data: dict) -> torch.nn.Module:
    if cfg.model == "fno":
        m = FNOSurrogate(torch.tensor(gx), torch.tensor(gy), cfg.width, cfg.modes, cfg.layers,
                         n_theta=np.asarray(data["theta"]).shape[1])
    elif cfg.model == "mlp":
        m = MLPSurrogate(n_theta=np.asarray(data["theta"]).shape[1])
    else:
        raise ValueError(cfg.model)
    th = torch.tensor(data["theta"], dtype=torch.float32)
    y = torch.tensor(np.stack([data["CL"], data["CD"]], 1), dtype=torch.float32)
    m.theta_mu.copy_(th.mean(0)); m.theta_sd.copy_(th.std(0))
    m.y_mu.copy_(y.mean(0)); m.y_sd.copy_(y.std(0))
    return m


def _fields_tensor(data: dict) -> tuple[torch.Tensor, torch.Tensor]:
    fields = torch.tensor(np.stack([data["p"], data["u"], data["v"]], 1), dtype=torch.float32)
    mask = torch.tensor(np.asarray(data["mask_fluid"])[:, None], dtype=torch.float32)
    return fields, mask


def precompute_sdf(theta: np.ndarray, gx: np.ndarray, gy: np.ndarray, device: str, batch: int = 16) -> torch.Tensor:
    th = torch.tensor(theta, dtype=torch.float64, device=device)
    gxt, gyt = torch.tensor(gx, device=device), torch.tensor(gy, device=device)
    with torch.no_grad():
        return torch.cat([sdf_grid(th[i:i + batch], gxt, gyt).float() for i in range(0, len(th), batch)])


@torch.no_grad()
def evaluate(model: torch.nn.Module, cfg: TrainConfig, data: dict, sdf: torch.Tensor | None, field_stats,
             batch: int = 32) -> dict:
    """Validation metrics: field MSE (standardised, fluid points) and median relative CL/CD/L-D errors."""
    dev = torch.device(cfg.device)
    model.eval()
    th = torch.tensor(data["theta"], dtype=torch.float32)
    preds, lf_sum, w_sum = [], 0.0, 0.0
    if cfg.model == "fno":
        fields, mask = _fields_tensor(data)
        f_mu, f_sd = field_stats
        fields_n = (fields - f_mu[None, :, None, None]) / f_sd[None, :, None, None] * mask
    for i in range(0, len(th), batch):
        tb = th[i:i + batch].to(dev)
        if cfg.model == "fno":
            pf, pc = model(tb, sdf[i:i + batch].to(dev))
            mk = mask[i:i + batch].to(dev)
            lf_sum += float(((pf - fields_n[i:i + batch].to(dev)) ** 2 * mk).sum()); w_sum += float(mk.sum() * 3)
        else:
            pc = model(tb)
        preds.append(pc.cpu().numpy())
    model.train()
    c = np.concatenate(preds)
    cl, cd = np.asarray(data["CL"]), np.asarray(data["CD"])
    out = {"val_rel_err_CL": float(np.median(np.abs(c[:, 0] - cl) / np.maximum(np.abs(cl), 1e-3))),
           "val_rel_err_CD": float(np.median(np.abs(c[:, 1] - cd) / cd)),
           "val_rel_err_LD": float(np.median(np.abs(c[:, 0] / c[:, 1] - cl / cd) / np.maximum(np.abs(cl / cd), 1e-2)))}
    if cfg.model == "fno":
        out["val_field_mse"] = lf_sum / max(w_sum, 1.0)
    return out


def train(cfg: TrainConfig, data: dict, gx: np.ndarray, gy: np.ndarray, log=print, val: dict | None = None,
          ckpt: Path | None = None, ckpt_every: int = 25, snapshot=None, snap_every: int = 0) -> tuple[torch.nn.Module, dict]:
    """Train one surrogate. `val` (optional) is evaluated every epoch (monitoring only, never used for
    selection beyond what the protocol allows). `ckpt` (optional path) enables save/resume: the full state
    (model, optimiser, scheduler, RNG, history, epoch) is written every `ckpt_every` epochs and at the end,
    and an existing checkpoint is resumed. `snapshot(ep, model, rec)` is called every `snap_every` epochs
    (and at epoch 0 and the last epoch) for the training-visualisation frames."""
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = torch.device(cfg.device)
    model = build_model(cfg, gx, gy, data).to(dev)
    th = torch.tensor(data["theta"], dtype=torch.float32)
    y = torch.tensor(np.stack([data["CL"], data["CD"]], 1), dtype=torch.float32)
    n = len(th)
    sdf = sdf_val = None
    field_stats = None
    if cfg.model == "fno":
        sdf = precompute_sdf(data["theta"], gx, gy, cfg.device).cpu()
        fields, mask = _fields_tensor(data)
        f_mu = (fields * mask).sum((0, 2, 3)) / mask.sum((0, 2, 3))
        f_sd = torch.sqrt((((fields - f_mu[None, :, None, None]) * mask) ** 2).sum((0, 2, 3)) / mask.sum((0, 2, 3)))
        fields_n = (fields - f_mu[None, :, None, None]) / f_sd[None, :, None, None] * mask
        field_stats = (f_mu, f_sd)
        model.field_mu.copy_(f_mu); model.field_sd.copy_(f_sd)
        if val is not None:
            sdf_val = precompute_sdf(val["theta"], gx, gy, cfg.device).cpu()
    jac = torch.tensor(data["jac"], dtype=torch.float32)
    has_jac = ~torch.isnan(jac).any(dim=(1, 2))
    if cfg.w_sobolev > 0 and not bool(has_jac.any()):
        raise ValueError("w_sobolev > 0 but the training data carry no adjoint gradients")
    jac_scale = torch.tensor(THETA_SPAN, dtype=torch.float32)[None, None, :] / model.y_sd.cpu()[None, :, None]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * math.ceil(n / cfg.batch))
    hist, start = [], 0
    if ckpt is not None and Path(ckpt).exists():
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        if st.get("config") != asdict(cfg):
            raise ValueError(f"checkpoint {ckpt} was written with a different config; refusing to resume")
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["rng_cpu"]); np.random.set_state(st["rng_np"])
        if torch.cuda.is_available() and st.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(st["rng_cuda"])
        hist, start = st["history"], st["epoch"] + 1
        log(json.dumps({"resumed_from_epoch": st["epoch"]}))

    def save_ckpt(ep: int) -> None:
        if ckpt is None:
            return
        tmp = Path(str(ckpt) + ".tmp")
        torch.save({"config": asdict(cfg), "epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "rng_cpu": torch.get_rng_state(), "rng_np": np.random.get_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "history": hist}, tmp)
        os.replace(tmp, ckpt)

    t0 = time.time() - (hist[-1]["elapsed_s"] if hist else 0.0)
    if snapshot is not None and start == 0:
        snapshot(-1, model, {"epoch": -1})  # untrained network (frame 0 of the video)
    for ep in range(start, cfg.epochs):
        perm = torch.randperm(n)
        tot = {"loss": 0.0, "field": 0.0, "coeff": 0.0, "sob": 0.0}
        for i in range(0, n, cfg.batch):
            idx = perm[i:i + cfg.batch]
            tb = th[idx].to(dev)
            loss = torch.zeros((), device=dev)
            if cfg.model == "fno":
                pf, pc = model(tb, sdf[idx].to(dev))
                mk = mask[idx].to(dev)
                lf = ((pf - fields_n[idx].to(dev)) ** 2 * mk).sum() / (mk.sum() * 3)
                loss = loss + cfg.w_field * lf
                tot["field"] += lf.item() * len(idx)
            else:
                pc = model(tb)
            lc = (((pc - y[idx].to(dev)) / model.y_sd) ** 2).mean()
            loss = loss + cfg.w_coeff * lc
            tot["coeff"] += lc.item() * len(idx)
            if cfg.w_sobolev > 0 and has_jac[idx].any():
                sel = idx[has_jac[idx]]
                ts = th[sel].to(dev).requires_grad_(True)
                c = model.coefficients(ts)
                J = torch.stack([torch.autograd.grad(c[:, k].sum(), ts, create_graph=True)[0] for k in range(2)], 1)
                ls = (((J - jac[sel].to(dev)) * jac_scale.to(dev)) ** 2).mean()
                loss = loss + cfg.w_sobolev * ls
                tot["sob"] += ls.item() * len(sel)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot["loss"] += loss.item() * len(idx)
        rec = {k: v / n for k, v in tot.items()} | {"epoch": ep, "elapsed_s": round(time.time() - t0, 1),
                                                     "lr": sched.get_last_lr()[0]}
        if val is not None:
            rec |= evaluate(model, cfg, val, sdf_val, field_stats)
        hist.append(rec)
        last = ep == cfg.epochs - 1
        if ep % max(1, cfg.epochs // 20) == 0 or last:
            log(json.dumps(rec))
        if snapshot is not None and snap_every and (ep % snap_every == 0 or last):
            snapshot(ep, model, rec)
        if (ep + 1) % ckpt_every == 0 or last:
            save_ckpt(ep)
    model.eval()
    return model, {"config": asdict(cfg), "history": hist}


@torch.no_grad()
def predict_coefficients(model: torch.nn.Module, theta: np.ndarray, device: str = "cpu", batch: int = 8) -> np.ndarray:
    model.eval().to(device)
    out = []
    for i in range(0, len(theta), batch):
        out.append(model.coefficients(torch.tensor(theta[i:i + batch], dtype=torch.float32, device=device)).cpu().numpy())
    return np.concatenate(out)


def predict_gradients(model: torch.nn.Module, theta: np.ndarray, device: str = "cpu", batch: int = 4) -> tuple[np.ndarray, ...]:
    """(L/D, dLD/dtheta, Jacobian of (CL, CD)) by autograd, batched."""
    model.eval().to(device)
    lds, gs, js = [], [], []
    for i in range(0, len(theta), batch):
        ld, g, J = lift_to_drag_and_grad(model, torch.tensor(theta[i:i + batch], dtype=torch.float32, device=device))
        lds.append(ld.cpu().numpy()); gs.append(g.cpu().numpy()); js.append(J.cpu().numpy())
    return np.concatenate(lds), np.concatenate(gs), np.concatenate(js)
