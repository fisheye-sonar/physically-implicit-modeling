"""PILOT (scratch, CPU only): Bayes floor of next-frame MSE on a noiseless flat-shaded discworld
instance by posterior sampling over the initial state x0 = (p0, v) of both discs.

posterior(x0 | frames 0..t)  ∝  generator prior(x0) · 1[generator accepts the 40-frame trajectory]
                                 · 1[x0 renders exactly frames 0..t]
SMC over t with MCMC rejuvenation; floor_t = E[ Var(frame t+1 | frames 0..t) ] (mean over rays).
Checks: (a) renderer parity with the stored frames, (b) exact population floor at t = 0, 1 from
grouped prior samples, (c) posterior-mean MSE vs posterior variance, (d) the trained model per position.
"""
import json, sys, time, math
import h5py, numpy as np, torch

sys.path.insert(0, "/home/sevan/research/PIM/physically-implicit-modeling")
from pim.environments import layout

torch.set_num_threads(12)
torch.manual_seed(0)
D = torch.float64
INST = sys.argv[1] if len(sys.argv) > 1 else "dw-8ray"
S = int(sys.argv[2]) if len(sys.argv) > 2 else 300          # test sequences
P = int(sys.argv[3]) if len(sys.argv) > 3 else 128          # particles per sequence
K = int(sys.argv[4]) if len(sys.argv) > 4 else 8            # MCMC sweeps per step
INIT = int(sys.argv[5]) if len(sys.argv) > 5 else 150

with h5py.File(layout.eval_file("discworld", INST), "r") as f:
    sim = json.loads(f.attrs["config_json"])["dataset"]["sim"]
    obs = f["obs_intensity"][:S].astype(np.float64)
    pos = f["positions"][:S].astype(np.float64)
    vel = f["velocities"][:S].astype(np.float64)
    print("stored dtypes", f["positions"].dtype, f["obs_intensity"].dtype)
R_CAST, DROP = int(sim["obs_res"]), bool(sim.get("drop_edge_rays", False))
YN, YF, XN, XF, RAD, DT = (float(sim[k]) for k in ("y_near", "y_far", "x_near", "x_far", "radius", "dt"))
SMIN, SMAX = float(sim["speed_min"]), float(sim["speed_max"])
SEP = float(sim["collision_margin"]) * 2 * RAD
T = obs.shape[1]
LV = torch.tensor([0.0, 0.4, 0.8], dtype=D)
s_ = torch.linspace(-1, 1, R_CAST, dtype=D) * (XF / YF)
nrm = torch.sqrt(s_ ** 2 + 1)
DX, DY = s_ / nrm, 1 / nrm
FR = torch.arange(T, dtype=D)


def xlim(y):
    return XN + (XF - XN) * (y - YN) / (YF - YN) - RAD


def render(p):
    """p (..., 2 objects, 2) → (..., R) codes 0 / 1 / 2 (background / disc 0 / disc 1)."""
    cx, cy = p[..., 0], p[..., 1]                                  # (..., 2)
    b = DX[:, None] * cx[..., None, :] + DY[:, None] * cy[..., None, :]   # (..., R, 2)
    disc = b ** 2 - (cx ** 2 + cy ** 2 - RAD ** 2)[..., None, :]
    hit = disc >= 0
    t = torch.where(hit, b - torch.sqrt(disc.clamp_min(0)), torch.full_like(b, math.inf))
    h0, h1 = hit[..., 0], hit[..., 1]
    code = torch.where(h0 & (~h1 | (t[..., 0] <= t[..., 1])), 1, torch.where(h1, 2, 0))
    return code[..., 1:-1] if DROP else code


def traj(p0, v):
    return p0[..., None, :, :] + FR[:, None, None] * DT * v[..., None, :, :]   # (..., T, 2, 2)


def accepted(tr):
    x, y = tr[..., 0], tr[..., 1]
    inside = ((y - RAD >= YN) & (y + RAD <= YF) & (x.abs() <= xlim(y))).all(-1).all(-1)
    sep = (tr[..., 0, :] - tr[..., 1, :]).norm(dim=-1) >= SEP
    return inside & sep.all(-1)


# ── (a) renderer parity on the stored test frames ───────────────────────────────────────
code_obs = torch.from_numpy(np.rint(obs / 0.4).astype(np.int64))              # (S, T, R)
code_re = render(torch.from_numpy(pos))
print(f"(a) parity: re-render of stored positions == stored frames on "
      f"{(code_re == code_obs).all(-1).float().mean():.6f} of frames")
p0_true, v_true = torch.from_numpy(pos[:, 0]), torch.from_numpy(vel[:, 0])
tr_true = traj(p0_true, v_true)
ok_true = (render(tr_true) == code_obs).all(-1).all(-1) & accepted(tr_true)
print(f"    linear trajectory from stored (p0, v0) reproduces all 40 frames + acceptance: {ok_true.float().mean():.4f}")

# ── (b) exact population floor at t = 0, 1 from grouped prior samples ───────────────────
def prior(n):
    y = YN + RAD + (YF - YN - 2 * RAD) * torch.rand(n, 2, dtype=D)
    x = (2 * torch.rand(n, 2, dtype=D) - 1) * xlim(y)
    sp = SMIN + (SMAX - SMIN) * torch.rand(n, 2, dtype=D)
    an = 2 * math.pi * torch.rand(n, 2, dtype=D)
    return torch.stack([x, y], -1), torch.stack([sp * an.cos(), sp * an.sin()], -1)


t0 = time.time()
fr = []
for _ in range(40):
    p0, v = prior(500_000)
    tr = traj(p0, v)
    keep = accepted(tr)
    fr.append(render(tr[keep][:, :3]))
fr = torch.cat(fr)                                                            # (N, 3, R)
B = 3 ** fr.shape[-1]
codes = (fr * (3 ** torch.arange(fr.shape[-1]))).sum(-1)                      # (N, 3)
vals = LV[fr]


def grouped_var(key, y):
    """E[Var(y | key)]: unbiased within-group variance, weighted by group size (n >= 2)."""
    u, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
    sm = torch.zeros(len(u), y.shape[1], dtype=D).index_add_(0, inv, y)
    sq = torch.zeros(len(u), y.shape[1], dtype=D).index_add_(0, inv, y ** 2)
    n = cnt[:, None].to(D)
    ss = (sq - sm ** 2 / n)                                                   # within-group SS
    m = cnt >= 2
    return float(ss[m].sum() / ((n[m] - 1).sum() * 1.0) / y.shape[1] * ((n[m] - 1).sum() / (n[m] - 1).sum())), int(m.sum()), float(cnt[m].sum() / len(key))


# pooled within-group variance: SS / (N - G), per ray
def pooled(key, y):
    u, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
    sm = torch.zeros(len(u), y.shape[1], dtype=D).index_add_(0, inv, y)
    sq = torch.zeros(len(u), y.shape[1], dtype=D).index_add_(0, inv, y ** 2)
    ss = (sq - sm ** 2 / cnt[:, None].to(D)).sum()
    return float(ss / (len(key) - len(u)) / y.shape[1]), len(u)


ex0, g0 = pooled(codes[:, 0], vals[:, 1])
ex1, g1 = pooled(codes[:, 0] * B + codes[:, 1], vals[:, 2])
print(f"(b) exact (grouped prior, N={len(fr):,} accepted of 20M, {time.time() - t0:.0f}s): "
      f"floor_0 = {ex0:.5f} ({g0} groups)   floor_1 = {ex1:.5f} ({g1} groups)   "
      f"unconditional Var(frame) = {float(vals[:, 0].var(0).mean()):.5f}")

# ── the sampler ────────────────────────────────────────────────────────────────────────
def log_prior(p0, v):
    return -(torch.log(xlim(p0[..., 1])).sum(-1)) - torch.log(v.norm(dim=-1)).sum(-1)


def valid(p0, v, t):
    sp = v.norm(dim=-1)
    ok = ((sp >= SMIN) & (sp <= SMAX)).all(-1)
    tr = traj(p0, v)
    ok &= accepted(tr)
    ok &= (render(tr[..., : t + 1, :, :]) == code_obs[:, None, : t + 1]).all(-1).all(-1)
    return ok


def sweep(p0, v, t):
    """One MH move per particle: a random disc, a log-uniform scale, a position move or a
    velocity move that pivots about a random observed frame (keeps that frame's position)."""
    shp = p0.shape[:2]
    j = torch.randint(0, 2, shp)
    sc = 10 ** (-3.5 + 3.5 * torch.rand(shp, dtype=D))
    kind = torch.rand(shp) < 0.5
    k = (torch.rand(shp, dtype=D) * (t + 1)).floor()
    dv = torch.randn(*shp, 2, dtype=D) * (sc * 0.07)[..., None] * (~kind)[..., None]
    dp = torch.randn(*shp, 2, dtype=D) * (sc * 2.0)[..., None] * kind[..., None] - (k * DT)[..., None] * dv
    oh = torch.nn.functional.one_hot(j, 2).to(D)[..., None]
    p0n, vn = p0 + oh * dp[..., None, :], v + oh * dv[..., None, :]
    ok = valid(p0n, vn, t)
    lr = log_prior(p0n, vn) - log_prior(p0, v)
    acc = ok & (torch.rand(shp, dtype=D).log() < lr.nan_to_num(nan=-math.inf))
    a = acc[..., None, None]
    return torch.where(a, p0n, p0), torch.where(a, vn, v), acc.float().mean().item()


t0 = time.time()
p0 = p0_true[:, None].repeat(1, P, 1, 1)
v = v_true[:, None].repeat(1, P, 1, 1)
for _ in range(INIT):
    p0, v, ar = sweep(p0, v, 0)
print(f"init {INIT} sweeps {time.time() - t0:.0f}s (last acceptance {ar:.2f})")
fl_var, fl_mse, surv, resets, accs = [], [], [], 0, []
for t in range(T - 1):
    nxt = render(traj(p0, v)[..., t + 1, :, :])                               # (S, P, R) codes
    y = LV[nxt]
    fl_var.append(y.var(1, unbiased=True).mean(-1))                           # (S,)
    fl_mse.append(((y.mean(1) - LV[code_obs[:, t + 1]]) ** 2).mean(-1))
    m = (nxt == code_obs[:, None, t + 1]).all(-1)                             # survivors
    surv.append(m.float().mean().item())
    dead = ~m.any(1)
    resets += int(dead.sum())
    w = m.to(D) + dead[:, None].to(D) * 1e-30
    idx = torch.multinomial(w, P, replacement=True)
    g = idx[..., None, None].expand(-1, -1, 2, 2)
    p0, v = p0.gather(1, g), v.gather(1, g)
    if dead.any():
        p0[dead], v[dead] = p0_true[dead][:, None], v_true[dead][:, None]
    for _ in range(K):
        p0, v, ar = sweep(p0, v, t + 1)
    accs.append(ar)
el = time.time() - t0
fl_var, fl_mse = torch.stack(fl_var, 1), torch.stack(fl_mse, 1)               # (S, T-1)
se = lambda a: float(a.mean(1).std() / math.sqrt(S))
print(f"sampler: S={S} P={P} K={K}  {el:.0f}s CPU  resets {resets}/{S * (T - 1)}  "
      f"mean survivors {np.mean(surv):.3f}  MH acceptance {np.mean(accs):.2f}")
print(f"(b) sampler floor_0 = {fl_var[:, 0].mean():.5f} ± {fl_var[:, 0].std() / math.sqrt(S):.5f}   "
      f"floor_1 = {fl_var[:, 1].mean():.5f} ± {fl_var[:, 1].std() / math.sqrt(S):.5f}")
print(f"(c) FLOOR (mean over 39 positions × rays): posterior variance {fl_var.mean():.5f} ± {se(fl_var):.5f}   "
      f"posterior-mean MSE vs truth {fl_mse.mean():.5f} ± {se(fl_mse):.5f}")

# ── (d) the trained model on the same sequences, per position ───────────────────────────
RUN = {"dw-8ray": "ray_ablation/L-dw-8ray-20m", "dw-5ray": "ray_ablation/L-dw-5ray-20m",
       "dw-16ray": "ray_ablation/L-dw-16ray-20m", "dw-noiseless": "noise_ablation/L-dw-noiseless-20m"}.get(INST)
if RUN:
    from pim.models import load_checkpoint
    model, _ = load_checkpoint(layout.REPO / "runs" / RUN / "best_model.pt", device="cpu")
    model.eval()
    x = torch.from_numpy(obs).float()
    with torch.no_grad():
        pr = torch.cat([model(x[i: i + 100, :-1]) for i in range(0, S, 100)])
    mp = ((pr - x[:, 1:]) ** 2).mean(-1).double()                             # (S, T-1)
    print(f"(d) model {RUN}: MSE {mp.mean():.5f} ± {se(mp):.5f} on the same {S} sequences   "
          f"excess over floor {(mp - fl_mse).mean():+.5f} (vs post-mean MSE) / {(mp - fl_var).mean():+.5f} (vs variance)")
    print("    t :  " + "  ".join(f"{t:>6d}" for t in (0, 1, 2, 3, 5, 10, 20, 38)))
    for nm, a in (("floor var", fl_var), ("floor mse", fl_mse), ("model", mp)):
        print(f"    {nm:<10}" + "  ".join(f"{a[:, t].mean():.4f}" for t in (0, 1, 2, 3, 5, 10, 20, 38)))
    print(f"    positions where the model beats the floor by > 2 SE: "
          f"{[t for t in range(T - 1) if (mp[:, t] - fl_mse[:, t]).mean() < -2 * (mp[:, t] - fl_mse[:, t]).std() / math.sqrt(S)]}")
