"""DESCRIPTIVE figure (no new analysis): median relative error of the objective and of its design gradient on the i.i.d.
test sets of all three studies (V1, V2 laminar: L/D; V3 transonic: J), FNO and MLP ensembles, log scale. Reads the same
result files as _staging/check_manuscript_numbers.py and asserts the plotted values round to the Table 2 entries before
drawing. Output results/v3_paper_figs/fig_value_vs_gradient_all.png (refuses to overwrite); replaces Figure 1
(fig_value_vs_gradient_v3.png, transonic only) in manuscript v7.
"""
import json, os
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

R = Path(os.environ.get("AF_ROOT", "."))
OUT = R / "results/v3_paper_figs/fig_value_vs_gradient_all.png"
assert not OUT.exists(), "exists"


def find(name):
    hits = [p for p in (R / "results").rglob(name) if "partial" not in p.name]
    assert len(hits) == 1, (name, [str(p) for p in hits])
    return hits[0]


SRC = {("V1", "FNO"): find("h1_test_metrics_fno.json"), ("V1", "MLP"): find("h1_test_metrics_mlp.json"),
       ("V2", "FNO"): find("h1v2_test_metrics_fno.json"), ("V2", "MLP"): find("h1v2_test_metrics_mlp.json"),
       ("V3", "FNO"): R / "results/reports_v3/h1v3_test_metrics_fno.json",
       ("V3", "MLP"): R / "results/reports_v3/h1v3_test_metrics_mlp.json"}
# Table 2 entries (percent), verified by check_manuscript_numbers.py (log d50cb863)
TABLE2 = {("V1", "FNO"): ("0.66", "5.8"), ("V1", "MLP"): ("0.33", "1.25"), ("V2", "FNO"): ("1.65", "9.0"),
          ("V2", "MLP"): ("1.22", "3.1"), ("V3", "FNO"): ("2.10", "20.4"), ("V3", "MLP"): ("1.76", "18.3")}
rnd = lambda x, s: str(Decimal(repr(100 * float(x))).quantize(Decimal(1).scaleb(-(len(s.split(".")[1]) if "." in s else 0)),
                                                              rounding=ROUND_HALF_UP))
vals = {}
for key, p in SRC.items():
    e = json.loads(p.read_text())["ensemble"]
    suf = "J" if key[0] == "V3" else "LD"
    v, g = e[f"median_rel_err_{suf}"], e[f"median_rel_grad_err_{suf}"]
    assert (rnd(v, TABLE2[key][0]), rnd(g, TABLE2[key][1])) == TABLE2[key], (key, v, g)
    vals[key] = (100 * v, 100 * g)

INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
C_VAL, C_GRAD = "#94a3b8", "#2563eb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
fig, ax = plt.subplots(figsize=(7.2, 3.4))
keys = list(SRC)
xs, w = [], 0.36
for i, key in enumerate(keys):
    x = i + (i // 2) * 0.6
    xs.append(x)
    ax.bar(x - w / 2 - 0.01, vals[key][0], w, color=C_VAL, edgecolor="white", linewidth=1.5, zorder=3)
    ax.bar(x + w / 2 + 0.01, vals[key][1], w, color=C_GRAD, edgecolor="white", linewidth=1.5, zorder=3)
ax.set_yscale("log"); ax.set_ylim(0.2, 40)
ax.set_yticks([0.3, 1, 3, 10, 30]); ax.set_yticklabels(["0.3", "1", "3", "10", "30"])
ax.set_ylabel("median relative error (%)")
ax.set_xticks(xs); ax.set_xticklabels([k[1] for k in keys])
for j, lab in enumerate(("V1 laminar, 4-D\n(n = 128)", "V2 laminar, 13-D\n(n = 127)", "V3 transonic, 13-D\n(n = 64)")):
    ax.text((xs[2 * j] + xs[2 * j + 1]) / 2, -0.2, lab, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9, color=INK)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(handles=[Patch(color=C_VAL, label="objective value (L/D or J)"), Patch(color=C_GRAD, label="design gradient (M-norm)")],
          frameon=False, loc="upper left", fontsize=8.5)
fig.tight_layout(); fig.subplots_adjust(bottom=0.27)
fig.savefig(OUT, dpi=200); plt.close(fig)
print("FIG_DONE", OUT.name, {f"{a}-{b}": tuple(round(x, 3) for x in v) for (a, b), v in vals.items()}, flush=True)
