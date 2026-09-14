"""inlp_r2_by_iteration_<run>.png per run + the combined panel — the style of
experiments/adjacent_flip_ablation/outputs/inlp_r2_by_iteration.png (light = early point, dark = late; dashed grey = random directions removed instead, point 4)."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
EXP = Path(__file__).resolve().parents[1]; SC = EXP / "scores"; OUT = EXP / "outputs"; OUT.mkdir(exist_ok=True)
ORDER = [("L-dw-noiseless-20m", ""), ("L-dw-8ray-20m", ""), ("L-dw-smooth-20m", ""), ("L-dw-8ray-tok-20m", ""), ("L-dw-noiseless-20m", "_fac"), ("L-dw-8ray-20m", "_fac")]
LABEL = {"L-dw-noiseless-20m": "dw-noiseless", "L-dw-8ray-20m": "dw-8ray", "L-dw-smooth-20m": "dw-smooth", "L-dw-8ray-tok-20m": "dw-8ray, token model"}
def panel(ax, d, title):
    P = d["points"]; cm = plt.get_cmap("Blues")
    for k in sorted(P, key=int):
        r = P[k]["mean_r2_curve"]; ax.plot(np.arange(1, len(r) + 1), r, color=cm(0.3 + 0.7 * int(k) / 8), lw=1.2 + 1.2 * int(k) / 8, label=f"point {k}")
    if "4" in P:
        r = P["4"]["mean_r2_random_curve"]; ax.plot(np.arange(1, len(r) + 1), r, "--", color="grey", lw=1.5, label="random directions removed instead, point 4 (stops where the fitted cascade exhausted)")
    n = {k: P[k]["k_exhaust_mean"] for k in ("1", "4", "8") if k in P}
    ax.set_title(f"{title}\ncopies per variable: " + ", ".join(f"{v:.0f} at point {k}" for k, v in n.items()), loc="left", fontsize=11)
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256]); ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 128, 256]); ax.set_ylim(0, 1); ax.set_xlim(1, 400)
    ax.grid(alpha=.3); ax.set_xlabel("orthogonal deflation iteration (log scale)")
have = [(n, s) for n, s in ORDER if (SC / f"inlp_{n}{s}.json").exists()]
YL = {"": "held-out R² of a state variable (frustum position / velocity), mean over 8 variables", "_fac": "held-out R² of an appearance factor (run centre / length), mean over 4 variables"}
for n, s in have:
    d = json.load(open(SC / f"inlp_{n}{s}.json")); fig, ax = plt.subplots(figsize=(7.5, 5.4)); panel(ax, d, LABEL[n] + (" — appearance-fac" if s else ""))
    ax.set_ylabel(YL[s], fontsize=9); ax.legend(fontsize=7.5, ncol=2, loc="upper right")
    fig.suptitle("INLP on the state: how many orthogonal linear copies of one state variable the residual holds, per residual point", fontsize=10.5)
    fig.tight_layout(); fig.savefig(OUT / f"inlp_r2_by_iteration_{n}{s}.png", dpi=150); plt.close(fig)
if have:
    fig, axes = plt.subplots(1, len(have), figsize=(5.6 * len(have), 5.8), sharey=True); axes = np.atleast_1d(axes)
    for ax, (n, s) in zip(axes, have):
        panel(ax, json.load(open(SC / f"inlp_{n}{s}.json")), LABEL[n] + (" — appearance-fac" if s else ""))
    axes[0].set_ylabel("held-out R² of one target variable, mean over variables", fontsize=9)
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, 0.93), frameon=False)
    fig.suptitle("INLP on the state: how many orthogonal linear copies of one state variable the residual holds, per residual point (light = early layer, dark = late)", fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(OUT / "inlp_r2_by_iteration.png", dpi=150); plt.close(fig)
print("plots:", [p.name for p in sorted(OUT.glob("inlp_r2_by_iteration*.png"))])
