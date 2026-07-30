"""Batched, device-agnostic (CPU **or GPU**) version of :class:`InteractiveWorld`.

Why this exists
---------------
The scalar :class:`~pim.simulator.interactive.InteractiveWorld` holds one Python object per
world and is stepped in a Python ``for`` loop, so environment cost is **linear in the number
of worlds**.  Profiling the endogenous-action training loop (batch 64) gave: simulator ~39 %,
model forward during collection ~45 %, gradient update ~16 %.  The model forward is
*latency*-bound and therefore nearly **flat** in batch size (16× the batch for ~1.6× the
time), while the Python-loop simulator is strictly **linear**.  So the simulator is what
prevents us from using the large batches the GPU is otherwise idle-waiting for; removing the
loop is worth ~10× environment-frames/second, not the ~1.6× the direct saving suggests.

This module keeps the world state as ``(B, n_obj, 2)`` tensors and vectorises the physics,
the wall handling, the collision test, the death→rebirth state machine and the ray-casting
renderer over the batch.  ``device="cuda"`` keeps observations on the GPU so they never make
a round trip for the model.

Fidelity to the scalar world (this is the contract — see ``tests/test_interactive_batched.py``)
----------------------------------------------------------------------------------------------
With **noise disabled** (``drift_force_std=0`` and ``SimConfig.obs_noise_std=0``) this class
reproduces the scalar world **bit-for-bit** in float64, given the same initial state and the
same actions: positions, velocities, observations, and the collision / wall / death / rebirth
event flags.  Two subtleties are preserved deliberately rather than "cleaned up":

* **shift-mode acceptance is sequential over objects.**  The scalar code mutates ``new_pos``
  while looping over objects, so object 1's collision check sees object 0's *already shifted*
  position.  We keep an inner loop over ``n_obj`` (small and fixed) so that ordering is identical.
* **wall handling resolves y before x.**  The x half-width is computed from the *updated* y.

With noise **enabled** the streams cannot match bit-for-bit, because the scalar world draws
from one ``numpy`` generator per world in a specific order whereas this class draws a single
batched tensor.  Parity is then **statistical** (matched noise σ, matched death rates), which
the test module checks explicitly.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import SimConfig
from .interactive import InteractiveConfig
from .sim import OBJECT_COLORS


def _frustum_half_width_t(y: torch.Tensor, sim: SimConfig) -> torch.Tensor:
    """Torch port of :func:`pim.simulator.sim.frustum_half_width`."""
    t = (y - sim.y_near) / (sim.y_far - sim.y_near)
    return sim.x_near + (sim.x_far - sim.x_near) * t


class BatchedInteractiveWorld:
    """``B`` independent interactive worlds advanced together as tensors."""

    def __init__(
        self,
        sim: SimConfig,
        cfg: InteractiveConfig | None = None,
        *,
        batch: int = 64,
        seed: int = 0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.sim = sim
        self.cfg = cfg or InteractiveConfig()
        self.B = int(batch)
        self.n = int(sim.n_objects if sim.n_objects is not None else sim.n_objects_max)
        self.device = torch.device(device)
        self.dtype = dtype
        self.gen = torch.Generator(device=self.device).manual_seed(int(seed))

        self._contact = 2.0 * sim.radius * self.cfg.collision_slack
        self._spawn_sep = 2.0 * sim.radius * self.cfg.spawn_clearance

        # static per-object identity (shared across the batch, as in the scalar world)
        self._radii = torch.full((self.n,), sim.radius, device=self.device, dtype=dtype)
        self._colors = np.array(OBJECT_COLORS[: self.n], dtype=float)
        if sim.fixed_reflectivities:
            refl = np.linspace(sim.refl_min, sim.refl_max, self.n)
        else:  # one draw shared by the batch, so objects keep a stable identity
            refl = np.random.default_rng(seed).uniform(
                sim.refl_min, sim.refl_max, self.n
            )
        self._refl = torch.tensor(refl, device=self.device, dtype=dtype)

        # ray geometry (constant): unit directions, matching renderer.render_frame exactly
        R = sim.obs_res
        s = torch.linspace(-1.0, 1.0, R, device=self.device, dtype=dtype)
        scale = sim.x_far / sim.y_far
        dx = s * scale
        dy = torch.ones(R, device=self.device, dtype=dtype)
        norm = torch.hypot(dx, dy)
        self._dx = dx / norm
        self._dy = dy / norm

        self.reset()

    # ── read-only views ──────────────────────────────────────────────────────
    @property
    def positions(self) -> torch.Tensor:
        return self._pos.clone()

    @property
    def velocities(self) -> torch.Tensor:
        return self._vel.clone()

    @property
    def reflectivities(self) -> torch.Tensor:
        return self._refl

    @property
    def obs_res(self) -> int:
        return self.sim.obs_res

    @property
    def n_objects(self) -> int:
        return self.n

    # ── episode control ──────────────────────────────────────────────────────
    def reset(self, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            self.gen = torch.Generator(device=self.device).manual_seed(int(seed))
        self.t = 0
        B = self.B
        self.alive = torch.ones(B, dtype=torch.bool, device=self.device)
        self.deaths = torch.zeros(B, dtype=torch.long, device=self.device)
        self.frames_survived = torch.zeros(B, dtype=torch.long, device=self.device)
        self._reset_after = torch.zeros(B, dtype=torch.long, device=self.device)
        self._pos = torch.zeros(B, self.n, 2, device=self.device, dtype=self.dtype)
        self._vel = torch.zeros_like(self._pos)
        self._last_action = torch.zeros_like(self._pos)
        self._sample_ic(torch.ones(B, dtype=torch.bool, device=self.device))
        return self._render()

    def _sample_ic(self, mask: torch.Tensor) -> None:
        """(Re)sample initial conditions for the worlds selected by ``mask``.

        Batched rejection sampling: resample only the worlds that still overlap.
        """
        if not bool(mask.any()):
            return
        sim, r = self.sim, self.sim.radius
        idx = mask.nonzero(as_tuple=True)[0]
        need = idx.clone()
        speed = (
            self.cfg.init_speed
            if self.cfg.dynamics == "force"
            else self.cfg.base_drift_speed
        )
        for _ in range(self.cfg.max_reset_attempts):
            m = need.numel()
            if m == 0:
                break
            y = torch.rand(
                m, self.n, generator=self.gen, device=self.device, dtype=self.dtype
            ) * ((sim.y_far - r) - (sim.y_near + r)) + (sim.y_near + r)
            x_lim = _frustum_half_width_t(y, sim) - r
            x = (
                torch.rand(
                    m, self.n, generator=self.gen, device=self.device, dtype=self.dtype
                )
                * 2
                - 1
            ) * x_lim
            pos = torch.stack([x, y], dim=-1)  # (m, n, 2)
            d = torch.cdist(pos, pos)  # (m, n, n)
            eye = torch.eye(self.n, dtype=torch.bool, device=self.device)
            ok = ((d >= self._spawn_sep) | eye).all(dim=2).all(dim=1)  # (m,)
            ang = torch.rand(
                m, self.n, generator=self.gen, device=self.device, dtype=self.dtype
            ) * (2 * np.pi)
            vel = speed * torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)
            good = need[ok]
            if good.numel():
                self._pos[good] = pos[ok]
                self._vel[good] = vel[ok]
            need = need[~ok]
        if need.numel():
            raise RuntimeError(
                f"BatchedInteractiveWorld: could not sample non-overlapping initial conditions "
                f"for {need.numel()} world(s) after {self.cfg.max_reset_attempts} attempts."
            )

    # ── the one-step transition ──────────────────────────────────────────────
    def step(self, actions=None) -> tuple[torch.Tensor, dict]:
        """Advance every world one frame. ``actions``: ``(B, n, 2)`` (or ``(B, n, 3)``/None)."""
        a = self._coerce(actions)
        self._last_action = a
        B = self.B
        z = lambda: torch.zeros(B, dtype=torch.bool, device=self.device)  # noqa: E731
        blocked = torch.zeros(B, self.n, dtype=torch.bool, device=self.device)
        wall = torch.zeros(B, self.n, dtype=torch.bool, device=self.device)
        dying, rebirth, died = z(), z(), z()

        self.t += 1
        dead = ~self.alive
        if self.cfg.reset_on_death and bool(dead.any()):
            # worlds mid-death: either emit a noise frame, or rebirth this frame
            noise_now = dead & (self._reset_after > 0)
            born_now = dead & (self._reset_after == 0)
            self._reset_after = torch.where(
                noise_now, self._reset_after - 1, self._reset_after
            )
            dying |= noise_now
            if bool(born_now.any()):
                self._sample_ic(born_now)
                self.alive |= born_now
                self.frames_survived = torch.where(
                    born_now,
                    torch.zeros_like(self.frames_survived),
                    self.frames_survived,
                )
                rebirth |= born_now

        step_mask = self.alive & ~rebirth  # worlds that take a physical step now
        if bool(step_mask.any()):
            if self.cfg.dynamics == "shift":
                new_pos = self._pos + self._vel * self.sim.dt  # base drift
                # sequential over objects, exactly as the scalar world does
                for i in range(self.n):
                    cand = new_pos[:, i] + a[:, i] * self.cfg.shift_scale
                    ok = self._accept(cand, i, new_pos)
                    take = ok & step_mask
                    new_pos[:, i] = torch.where(take[:, None], cand, new_pos[:, i])
                    blocked[:, i] = (~ok) & step_mask
            else:  # "force"
                drift = (
                    torch.randn(
                        B,
                        self.n,
                        2,
                        generator=self.gen,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    * self.cfg.drift_force_std
                )
                accel = (a * self.cfg.force_scale + drift) / self.cfg.mass
                vel = (self._vel + accel) * (1.0 - self.cfg.friction)
                spd = vel.norm(dim=-1, keepdim=True)
                vel = torch.where(
                    spd > self.cfg.max_speed,
                    vel / spd.clamp(min=1e-12) * self.cfg.max_speed,
                    vel,
                )
                self._vel = torch.where(step_mask[:, None, None], vel, self._vel)
                new_pos = self._pos + self._vel * self.sim.dt

            pos_w, vel_w, hit = self._apply_wall(new_pos, self._vel)
            self._pos = torch.where(step_mask[:, None, None], pos_w, self._pos)
            self._vel = torch.where(step_mask[:, None, None], vel_w, self._vel)
            wall = hit & step_mask[:, None]

            # pairwise contact
            d = torch.cdist(self._pos, self._pos)
            eye = torch.eye(self.n, dtype=torch.bool, device=self.device)
            collision = ((d < self._contact) & ~eye).any(dim=2).any(dim=1) & step_mask
        else:
            collision = z()

        died = (
            (collision & self.cfg.death_on_collision)
            | (wall.any(dim=1) & self.cfg.death_on_wall)
        ) & step_mask
        if bool(died.any()):
            self.deaths = self.deaths + died.long()
            self.alive &= ~died
            self._reset_after = torch.where(
                died,
                torch.full_like(self._reset_after, self.cfg.reset_noise_frames),
                self._reset_after,
            )
        grew = step_mask & ~died
        self.frames_survived = self.frames_survived + grew.long()

        obs = self._render()
        if bool(dying.any()):  # death frames are pure noise (SMiRL-style surprise)
            noise = torch.rand(
                B,
                self.sim.obs_res,
                generator=self.gen,
                device=self.device,
                dtype=self.dtype,
            )
            obs = torch.where(dying[:, None], noise, obs)
            self._last_obs = obs
        return obs, {
            "t": self.t,
            "alive": self.alive.clone(),
            "died": died,
            "dying": dying,
            "rebirth": rebirth,
            "collision": collision,
            "wall": wall,
            "blocked": blocked,
            "deaths": self.deaths.clone(),
            "frames_survived": self.frames_survived.clone(),
            "action": a,
            "positions": self._pos.clone(),
            "velocities": self._vel.clone(),
        }

    # ── helpers ──────────────────────────────────────────────────────────────
    def _coerce(self, actions) -> torch.Tensor:
        if actions is None:
            return torch.zeros(self.B, self.n, 2, device=self.device, dtype=self.dtype)
        a = torch.as_tensor(actions, device=self.device, dtype=self.dtype)
        if a.shape == (self.B, self.n, 3):  # [active, a1, a2] schema
            a = a[..., 0:1] * a[..., 1:3]
        return a.reshape(self.B, self.n, 2).clamp(-1.0, 1.0)

    def _accept(
        self, cand: torch.Tensor, i: int, positions: torch.Tensor
    ) -> torch.Tensor:
        """Shift-mode guard: in-frustum and non-overlapping. Returns (B,) bool."""
        r = self.sim.radius
        x, y = cand[:, 0], cand[:, 1]
        ok = (y - r >= self.sim.y_near) & (y + r <= self.sim.y_far)
        x_lim = (
            _frustum_half_width_t(y.clamp(self.sim.y_near, self.sim.y_far), self.sim)
            - r
        )
        ok &= x.abs() <= x_lim
        for j in range(self.n):
            if j == i:
                continue
            ok &= (cand - positions[:, j]).norm(dim=-1) >= self._contact
        return ok

    def _apply_wall(self, p: torch.Tensor, v: torch.Tensor):
        """Vectorised wall handling. y is resolved first; x uses the UPDATED y."""
        sim, r, mode = self.sim, self.sim.radius, self.cfg.wall_mode
        x, y = p[..., 0].clone(), p[..., 1].clone()
        vx, vy = v[..., 0].clone(), v[..., 1].clone()
        y_lo, y_hi = sim.y_near + r, sim.y_far - r

        lo, hi = y < y_lo, y > y_hi
        if mode == "bounce":
            y = torch.where(lo, 2.0 * y_lo - y, torch.where(hi, 2.0 * y_hi - y, y))
            vy = torch.where(lo, vy.abs(), torch.where(hi, -vy.abs(), vy))
        else:  # clamp
            y = torch.where(
                lo,
                torch.full_like(y, y_lo),
                torch.where(hi, torch.full_like(y, y_hi), y),
            )
            vy = torch.where(
                lo, vy.clamp(min=0.0), torch.where(hi, vy.clamp(max=0.0), vy)
            )
        hit = lo | hi
        y = y.clamp(min=y_lo, max=y_hi)

        x_lim = _frustum_half_width_t(y, sim) - r
        lo_x, hi_x = x < -x_lim, x > x_lim
        if mode == "bounce":
            x = torch.where(
                lo_x, -2.0 * x_lim - x, torch.where(hi_x, 2.0 * x_lim - x, x)
            )
            vx = torch.where(lo_x, vx.abs(), torch.where(hi_x, -vx.abs(), vx))
        else:
            x = torch.where(lo_x, -x_lim, torch.where(hi_x, x_lim, x))
            vx = torch.where(
                lo_x, vx.clamp(min=0.0), torch.where(hi_x, vx.clamp(max=0.0), vx)
            )
        hit = hit | lo_x | hi_x
        x = torch.maximum(torch.minimum(x, x_lim), -x_lim)
        return torch.stack([x, y], dim=-1), torch.stack([vx, vy], dim=-1), hit

    def _render(self) -> torch.Tensor:
        """Batched ray-cast — a faithful port of ``renderer.render_frame``."""
        sim = self.sim
        cx = self._pos[..., 0].unsqueeze(1)  # (B,1,n)
        cy = self._pos[..., 1].unsqueeze(1)
        dx = self._dx.view(1, -1, 1)  # (1,R,1)
        dy = self._dy.view(1, -1, 1)

        b = dx * cx + dy * cy  # (B,R,n)
        C = cx**2 + cy**2 - self._radii.view(1, 1, -1) ** 2  # (B,1,n)
        disc = b**2 - C
        sq = disc.clamp(min=0.0).sqrt()
        t_front, t_back = b - sq, b + sq
        hy_f, hy_b = dy * t_front, dy * t_back

        valid_front = (
            (disc >= 0) & (t_front > 1e-9) & (hy_f >= sim.y_near) & (hy_f <= sim.y_far)
        )
        t_at_near = sim.y_near / dy
        clamp_near = (disc >= 0) & (hy_f < sim.y_near) & (hy_b >= sim.y_near)

        inf = torch.full_like(t_front, float("inf"))
        t_eff = torch.where(
            valid_front,
            t_front,
            torch.where(clamp_near, t_at_near.expand_as(t_front), inf),
        )
        best_t, best_j = t_eff.min(dim=2)  # (B,R)
        hit = torch.isfinite(best_t)

        inten = torch.where(hit, self._refl[best_j], torch.zeros_like(best_t))
        self._last_depth = torch.where(
            hit, self._dy.view(1, -1) * best_t, torch.zeros_like(best_t)
        )
        self._last_id = torch.where(hit, best_j, torch.full_like(best_j, -1))
        if sim.obs_noise_std > 0:
            inten = (
                inten
                + torch.randn(
                    inten.shape,
                    generator=self.gen,
                    device=self.device,
                    dtype=self.dtype,
                )
                * sim.obs_noise_std
            )
            inten = inten.clamp(0.0, 1.0)
        self._last_obs = inten
        return inten
