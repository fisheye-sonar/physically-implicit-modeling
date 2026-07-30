#!/usr/bin/env python3
"""Interactive emulator for the endogenous-action toy world.

Play the world yourself with the keyboard, or watch a driver (random / heuristic /
— later — a trained world model) play in the SAME view.  Left panel = the 2D world
(latent state); right panel = the 1D observation waterfall (what a model sees);
bottom = a keyboard overlay showing which keys are pressed + a per-object action
arrow.  Runs indefinitely.  This is the same emulator we will reuse to visualise a
trained actor "playing" — its continuous action vectors shown on the same overlay.

Controls (god's-hand — you move every object)
---------------------------------------------
    object 0 :  W/A/S/D           (W = +y "far", S = -y "near", A = -x, D = +x)
    object 1 :  arrow keys        (also I/J/K/L if the arrows are captured)
    no key   :  no-op
    R : reset   M : toggle shift/force dynamics   C : toggle death-on-collision
    B : toggle death-on-wall (walls still bounce)   SPACE : pause   Q/Esc : quit

Usage
-----
    python scripts/play.py                        # live, force dynamics, keyboard
    python scripts/play.py --dynamics shift       # live, L1 position-shift dynamics
    python scripts/play.py --driver avoid         # watch a collision-avoiding heuristic
    python scripts/play.py --driver random --save outputs/play_demo.gif --frames 150
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from pim.simulator.config import SimConfig
from pim.simulator.interactive import InteractiveConfig, InteractiveWorld

# ── Keyboard layout (matplotlib key names) ────────────────────────────────────
# object 0 = WASD, object 1 = arrow keys (IJKL accepted as an alternative).
WASD = {"w": (1, 2, "W"), "a": (0, 1, "A"), "s": (1, 1, "S"), "d": (2, 1, "D")}
ARROWS = {
    "up": (5, 2, "↑"),
    "left": (4, 1, "←"),
    "down": (5, 1, "↓"),
    "right": (6, 1, "→"),
}
IJKL_ALIAS = {"i": "up", "k": "down", "j": "left", "l": "right"}
SPECIAL = {"r", "m", " ", "q", "escape"}


def keys_to_action(pressed: set[str], n: int) -> np.ndarray:
    """Map the set of currently-held keys to a god's-hand action ``(n, 2)``.

    object 0 ← WASD, object 1 ← arrows (or IJKL).  Objects ≥ 2 get no-op.
    """
    p = set(pressed)
    for k, tgt in IJKL_ALIAS.items():  # fold IJKL onto the arrow keys
        if k in p:
            p.add(tgt)
    a = np.zeros((n, 2))
    if n >= 1:
        a[0] = [("d" in p) - ("a" in p), ("w" in p) - ("s" in p)]
    if n >= 2:
        a[1] = [("right" in p) - ("left" in p), ("up" in p) - ("down" in p)]
    return a


# ── Non-human drivers (for demos / headless self-test) ────────────────────────
class RandomDriver:
    """Smoothed (Ornstein–Uhlenbeck) random actions — wandering, not pure jitter."""

    def __init__(self, n: int, seed: int = 0, theta: float = 0.15, sigma: float = 0.5):
        self.rng = np.random.default_rng(seed)
        self.a = np.zeros((n, 2))
        self.theta, self.sigma = theta, sigma

    def act(self, world: InteractiveWorld) -> np.ndarray:
        self.a += -self.theta * self.a + self.sigma * self.rng.normal(
            0, 1, self.a.shape
        )
        return np.clip(self.a, -1, 1)


class HeuristicAvoidDriver:
    """God's-hand collision avoider: push each object away from its nearest
    neighbour and from the frustum walls.  Makes the saved demo look purposeful."""

    def act(self, world: InteractiveWorld) -> np.ndarray:
        pos = world.positions
        n = world.n
        sim = world.sim
        a = np.zeros((n, 2))
        for i in range(n):
            f = np.zeros(2)
            for j in range(n):
                if j == i:
                    continue
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d) + 1e-6
                f += d / dist**2 * 2.0  # inverse-square repulsion from others
            # wall repulsion (near/far in y, frustum edges in x)
            f[1] += 1.0 / (pos[i, 1] - sim.y_near + 0.3) - 1.0 / (
                sim.y_far - pos[i, 1] + 0.3
            )
            a[i] = f
        m = np.abs(a).max() + 1e-6
        return np.clip(a / m, -1, 1)


class ModelDriver:
    """A trained EndogenousActorGRU drives the world, exactly like a human driver would.

    Also exposes ``last_pred`` — the model's predicted NEXT observation given the action it
    just chose — so the emulator can show a second waterfall of what the model *thinks* will
    happen next, beside what actually happens.
    """

    def __init__(self, model, device=None, deterministic=True):
        import torch

        self.torch = torch
        self.model = model.eval()
        # infer the device from the model so the caller cannot mismatch it
        self.device = device or next(model.parameters()).device
        self.deterministic = deterministic
        self.state = None
        self.last_pred = None
        c = model.cfg
        self.prev_a = torch.zeros(1, c.n_obj, c.n_axes, device=self.device)

    def act(self, world):
        torch = self.torch
        with torch.no_grad():
            obs = (
                torch.from_numpy(world._last_intensity)
                .float()
                .to(self.device)
                .unsqueeze(0)
            )
            # the PREVIOUS action drives the transition into this state (ignored unless the
            # model was trained with action_in_transition)
            h, self.state = self.model.gru_step(
                obs, self.state, prev_action=self.prev_a
            )
            action, _, _, _ = self.model.act(h, deterministic=self.deterministic)
            self.prev_a = action
            self.last_pred = (
                self.model.decode_action(h, action).squeeze(0).cpu().numpy()
            )
        return action.squeeze(0).cpu().numpy()


class AutoregressiveModelDriver:
    """Closed-loop ("dreaming") driver: after a short warm-up the model **stops seeing the real
    observation** and instead consumes its OWN predicted observation each step, while the actions
    it chooses from that imagined state are still applied to the real world.

    This is the honest stress test of the world model: prediction error compounds, so the imagined
    waterfall drifts away from reality and — if the latent has degraded — the actions start to fail
    (collisions, deaths) in the real world.  ``last_pred`` is the imagined observation stream.
    """

    def __init__(self, model, warmup: int = 15, device=None, deterministic=True):
        import torch

        self.torch = torch
        self.model = model.eval()
        self.device = device or next(model.parameters()).device
        self.deterministic = deterministic
        self.warmup = warmup
        self.state = None
        self.last_pred = None
        self.t = 0
        c = model.cfg
        self.prev_a = torch.zeros(1, c.n_obj, c.n_axes, device=self.device)

    def act(self, world):
        torch = self.torch
        with torch.no_grad():
            if self.t < self.warmup or self.last_pred is None:
                obs_in = world._last_intensity  # warm-up: real observations
            else:
                obs_in = self.last_pred  # afterwards: its OWN prediction
            o = (
                torch.from_numpy(np.asarray(obs_in, dtype=np.float32))
                .to(self.device)
                .unsqueeze(0)
            )
            h, self.state = self.model.gru_step(o, self.state, prev_action=self.prev_a)
            action, _, _, _ = self.model.act(h, deterministic=self.deterministic)
            self.prev_a = action
            self.last_pred = (
                self.model.decode_action(h, action).squeeze(0).cpu().numpy()
            )
        self.t += 1
        return action.squeeze(0).cpu().numpy()


class AutoregressivePredictor:
    """A predictor that also runs CLOSED LOOP: after ``warmup`` frames it consumes its own
    previous prediction instead of the real observation, but is told the action that was
    actually applied.  It never influences the world.

    Used to put the observer's *imagination* beside the actor's, on the same action sequence
    and the same warm-up, so their drift can be compared directly.
    """

    def __init__(self, model, warmup: int = 15):
        import torch

        self.torch = torch
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.warmup = warmup
        self.state = None
        self.last = None
        self.t = 0
        c = model.cfg
        self.prev_a = torch.zeros(1, c.n_obj, c.n_axes, device=self.device)

    def predict(self, obs_now, action):
        torch = self.torch
        with torch.no_grad():
            obs_in = (
                obs_now if (self.t < self.warmup or self.last is None) else self.last
            )
            o = (
                torch.from_numpy(np.asarray(obs_in, dtype=np.float32))
                .to(self.device)
                .unsqueeze(0)
            )
            h, self.state = self.model.gru_step(o, self.state)
            a = (
                torch.from_numpy(np.asarray(action, dtype=np.float32))
                .to(self.device)
                .unsqueeze(0)
            )
            self.last = self.model.decode_action(h, a).squeeze(0).cpu().numpy()
        self.t += 1
        return self.last


# ── The emulator ──────────────────────────────────────────────────────────────
class Emulator:
    def __init__(
        self,
        world: InteractiveWorld,
        driver=None,
        wf_rows: int = 100,
        predictors=None,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.patches import Polygon, Rectangle

        # Free the keyboard: matplotlib binds s/f/q/g/l/k/… to toolbar actions — e.g. 's'
        # opens the save dialog, whose modal eats the key-release event → stuck keys.
        for _km in [k for k in plt.rcParams if k.startswith("keymap.")]:
            plt.rcParams[_km] = []

        self.world = world
        self.driver = driver  # None → human keyboard
        self.pressed: set[str] = set()
        self.paused = False
        self._flash = ""  # transient status message (mode / death-toggle changes)
        self._flash_t = 0
        self.wf_rows = wf_rows
        self.wf = np.zeros((wf_rows, world.obs_res), dtype=np.float32)
        # predictors: [(title, model)] — each watches the SAME observation stream and
        # predicts the next obs given the action actually applied.  They never act.
        self.title = "endogenous-action emulator"
        self.predictors = list(predictors or [])
        self.pred_states = [None] * len(self.predictors)
        self.wf_preds = [
            np.zeros((wf_rows, world.obs_res), dtype=np.float32)
            for _ in self.predictors
        ]
        self.plt = plt

        BG, EDGE, TICK, TEXT = "#0a0a14", "#5c677f", "#808a9d", "#a3adc2"
        self.TEXT = TEXT
        sim = world.sim
        ncol = 2 + len(self.predictors)
        fig = plt.figure(figsize=(6.5 + 4.2 * ncol, 6.6), facecolor=BG)
        gs = GridSpec(
            2,
            ncol,
            height_ratios=[5, 1.5],
            width_ratios=[1] * ncol,
            hspace=0.24,
            wspace=0.14,
            left=0.055,
            right=0.975,
            top=0.92,
            bottom=0.06,
        )
        self.fig = fig
        axw = fig.add_subplot(gs[0, 0])
        axf = fig.add_subplot(gs[0, 1])
        axk = fig.add_subplot(gs[1, 0])
        axs = fig.add_subplot(gs[1, 1])
        axps = [fig.add_subplot(gs[0, 2 + i]) for i in range(len(self.predictors))]
        self.axw, self.axf, self.axk, self.axs, self.axps = axw, axf, axk, axs, axps
        for ax in [axw, axf, axk, axs] + axps:
            ax.set_facecolor(BG)
            for sp in ax.spines.values():
                sp.set_edgecolor(TICK)
            ax.tick_params(colors=TICK, labelsize=8)

        # ---- 2D world panel ----
        mx, my = 0.7, 0.7
        axw.set_xlim(-sim.x_far - mx, sim.x_far + mx)
        axw.set_ylim(sim.y_near - my, sim.y_far + my)
        axw.set_aspect("equal")
        axw.set_xlabel("x", color=TEXT, fontsize=9)
        axw.set_ylabel("depth  y", color=TEXT, fontsize=9)
        axw.set_title("2D world  (latent state)", color=TEXT, fontsize=11, pad=6)
        corners = np.array(
            [
                [-sim.x_near, sim.y_near],
                [sim.x_near, sim.y_near],
                [sim.x_far, sim.y_far],
                [-sim.x_far, sim.y_far],
            ]
        )
        axw.add_patch(
            Polygon(
                corners,
                closed=True,
                fill=False,
                edgecolor=EDGE,
                linewidth=1.8,
                zorder=1,
            )
        )
        self.circles, self.refl_labels, self.vel_lines, self.act_lines = [], [], [], []
        for i in range(world.n):
            c = plt.Circle(
                (0, 0),
                world.radii[i],
                facecolor=world.colors[i],
                zorder=3,
                linewidth=1.2,
                edgecolor="white",
                alpha=0.95,
            )
            axw.add_patch(c)
            self.circles.append(c)
            (vl,) = axw.plot([], [], color=world.colors[i], lw=1.6, alpha=0.5, zorder=2)
            self.vel_lines.append(vl)
            (al,) = axw.plot(
                [],
                [],
                color="white",
                lw=2.4,
                alpha=0.9,
                zorder=5,
                solid_capstyle="round",
            )
            self.act_lines.append(al)
            lbl = axw.text(
                0,
                0,
                f"{world.reflectivities[i]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=7,
                fontweight="bold",
                zorder=6,
                fontfamily="monospace",
            )
            self.refl_labels.append(lbl)
        self.event_text = axw.text(
            0.5,
            0.965,
            "",
            transform=axw.transAxes,
            ha="center",
            va="top",
            color="#FF5252",
            fontsize=13,
            fontweight="bold",
        )

        # ---- waterfall panel ----
        axf.set_title(
            "1D observation waterfall  (what the model sees)",
            color=TEXT,
            fontsize=11,
            pad=6,
        )
        axf.set_xlabel("scan position", color=TEXT, fontsize=9)
        axf.set_ylabel("time  (newest at bottom)", color=TEXT, fontsize=9)
        self.wf_img = axf.imshow(
            self.wf,
            aspect="auto",
            origin="upper",
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
            extent=[0, world.obs_res, wf_rows, 0],
        )
        axf.set_yticks([])
        self.wf_pred_imgs = []
        for i, (ttl, _m) in enumerate(self.predictors):
            axps[i].set_title(ttl, color=TEXT, fontsize=11, pad=6)
            axps[i].set_xlabel("scan position", color=TEXT, fontsize=9)
            self.wf_pred_imgs.append(
                axps[i].imshow(
                    self.wf_preds[i],
                    aspect="auto",
                    origin="upper",
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                    extent=[0, world.obs_res, wf_rows, 0],
                )
            )
            axps[i].set_yticks([])

        # ---- keyboard overlay ----
        axk.set_xlim(-0.7, 7.7)
        axk.set_ylim(0.3, 3.0)
        axk.set_aspect("equal")
        axk.set_xticks([])
        axk.set_yticks([])
        axk.set_title("keys pressed", color=TEXT, fontsize=10, pad=4)
        self.key_patches = {}
        for keymap, col in (
            (WASD, world.colors[0]),
            (ARROWS, world.colors[1] if world.n > 1 else world.colors[0]),
        ):
            for key, (cx, cy, label) in keymap.items():
                rect = Rectangle(
                    (cx - 0.44, cy - 0.44),
                    0.88,
                    0.88,
                    facecolor="#20242e",
                    edgecolor=col,
                    linewidth=2.0,
                    zorder=2,
                )
                axk.add_patch(rect)
                axk.text(
                    cx,
                    cy,
                    label,
                    ha="center",
                    va="center",
                    color=TEXT,
                    fontsize=11,
                    fontweight="bold",
                    zorder=3,
                )
                self.key_patches[key] = (rect, np.array(col))
        axk.text(
            1.0,
            0.15,
            "object 0",
            ha="center",
            color=world.colors[0],
            fontsize=8,
            transform=axk.transData,
        )
        if world.n > 1:
            axk.text(
                5.0,
                0.15,
                "object 1  (or IJKL)",
                ha="center",
                color=world.colors[1],
                fontsize=8,
            )

        # ---- status panel ----
        axs.set_xticks([])
        axs.set_yticks([])
        axs.set_title("status", color=TEXT, fontsize=10, pad=4)
        self.status_text = axs.text(
            0.03,
            0.92,
            "",
            transform=axs.transAxes,
            ha="left",
            va="top",
            color=TEXT,
            fontsize=9.5,
            fontfamily="monospace",
            linespacing=1.5,
        )

        fig.canvas.mpl_connect("key_press_event", self._on_press)
        fig.canvas.mpl_connect("key_release_event", self._on_release)

    # ---- keyboard events ----
    def _flash_msg(self, msg: str) -> None:
        self._flash, self._flash_t = msg, 25  # show for ~25 frames

    def _on_press(self, event):
        k = event.key
        if k is None:
            return
        if k in ("q", "escape"):
            self.plt.close(self.fig)
            return
        if k == " ":
            self.paused = not self.paused
            return
        if k == "r":
            self.world.reset()
            self.wf[:] = 0
            self.pressed.clear()
            return
        if k == "m":  # toggle dynamics IN PLACE (keep positions) — distinct from reset
            self.world.cfg.dynamics = (
                "shift" if self.world.cfg.dynamics == "force" else "force"
            )
            self.pressed.clear()
            self._flash_msg(f"mode → {self.world.cfg.dynamics}")
            return
        if k == "c":  # toggle death-on-collision
            self.world.cfg.death_on_collision = not self.world.cfg.death_on_collision
            self._flash_msg(f"death-on-collision: {self.world.cfg.death_on_collision}")
            return
        if k == "b":  # toggle death-on-wall (walls still bounce)
            self.world.cfg.death_on_wall = not self.world.cfg.death_on_wall
            self._flash_msg(f"death-on-wall: {self.world.cfg.death_on_wall}")
            return
        self.pressed.add(k)

    def _on_release(self, event):
        self.pressed.discard(event.key)

    def _run_predictors(self, obs_now, action):
        """Each predictor consumes the CURRENT obs + the action actually applied and
        outputs its predicted NEXT observation.  Predictors never influence the world.
        """
        import torch

        if not hasattr(self, "_pred_prev_a"):
            self._pred_prev_a = [None] * len(self.predictors)
        outs = []
        for i, (_ttl, model) in enumerate(self.predictors):
            if model == "driver":  # show the driver's own (imagined) prediction stream
                lp = getattr(self.driver, "last_pred", None)
                outs.append(
                    np.zeros(self.world.obs_res, np.float32) if lp is None else lp
                )
                continue
            if hasattr(model, "predict"):  # closed-loop predictor (keeps its own state)
                outs.append(model.predict(obs_now, action))
                continue
            dev = next(model.parameters()).device
            with torch.no_grad():
                o = torch.from_numpy(obs_now).float().to(dev).unsqueeze(0)
                c = model.cfg
                prev = self._pred_prev_a[i]
                if prev is None:
                    prev = torch.zeros(1, c.n_obj, c.n_axes, device=dev)
                h, self.pred_states[i] = model.gru_step(
                    o, self.pred_states[i], prev_action=prev
                )
                a = (
                    torch.from_numpy(np.asarray(action, dtype=np.float32))
                    .to(dev)
                    .unsqueeze(0)
                )
                self._pred_prev_a[i] = a
                outs.append(model.decode_action(h, a).squeeze(0).cpu().numpy())
        return outs

    # ---- one animation frame ----
    def update(self, _frame):
        world = self.world
        if not self.paused:
            action = (
                keys_to_action(self.pressed, world.n)
                if self.driver is None
                else self.driver.act(world)
            )
            preds = self._run_predictors(world._last_intensity, action)
            obs, info = world.step(action)
            self.wf[:-1] = self.wf[1:]
            self.wf[-1] = obs
            for i, p in enumerate(preds):
                self.wf_preds[i][:-1] = self.wf_preds[i][1:]
                self.wf_preds[i][-1] = np.clip(p, 0, 1)
        else:
            info = world._info()
            obs = world._last_intensity

        self.wf_img.set_data(self.wf)
        for i, img in enumerate(self.wf_pred_imgs):
            img.set_data(self.wf_preds[i])
        act = info["action"]
        for i in range(world.n):
            p = info["positions"][i]
            self.circles[i].center = (p[0], p[1])
            self.refl_labels[i].set_position((p[0], p[1]))
            v = info["velocities"][i]
            self.vel_lines[i].set_data(
                [p[0], p[0] + v[0] * 8.0], [p[1], p[1] + v[1] * 8.0]
            )
            self.act_lines[i].set_data(
                [p[0], p[0] + act[i, 0] * 1.4], [p[1], p[1] + act[i, 1] * 1.4]
            )

        # key overlay highlight (human) or the model/driver's discretised action
        held = (
            self.pressed if self.driver is None else self._action_to_keys(act, world.n)
        )
        folded = set(held)
        for kk, tgt in IJKL_ALIAS.items():
            if kk in folded:
                folded.add(tgt)
        for key, (rect, col) in self.key_patches.items():
            on = key in folded
            rect.set_facecolor(tuple(col) if on else "#20242e")

        msg = ""
        if info.get("dying"):
            msg = "• • •"
        elif info.get("rebirth"):
            msg = "rebirth"
        elif info.get("died"):
            msg = "✖ DEATH"
        elif info.get("collision"):
            msg = "collision"
        if self._flash_t > 0:  # a transient mode/toggle message takes precedence
            self._flash_t -= 1
            msg = self._flash
        self.event_text.set_text(msg)

        dc = "on" if world.cfg.death_on_collision else "off"
        db = "on" if world.cfg.death_on_wall else "off"
        self.status_text.set_text(
            f"dynamics : {world.cfg.dynamics}\n"
            f"frame    : {info['t']}\n"
            f"alive    : {info['alive']}\n"
            f"survived : {info['frames_survived']}\n"
            f"deaths   : {info['deaths']}\n"
            f"death: coll {dc} · wall {db}\n"
            f"{'PAUSED' if self.paused else ''}"
        )
        return []

    @staticmethod
    def _action_to_keys(action: np.ndarray, n: int) -> set[str]:
        """Discretise a continuous action back to key names (to show a model 'as key-presses')."""
        keys: set[str] = set()
        thr = 0.25
        if n >= 1:
            if action[0, 0] > thr:
                keys.add("d")
            if action[0, 0] < -thr:
                keys.add("a")
            if action[0, 1] > thr:
                keys.add("w")
            if action[0, 1] < -thr:
                keys.add("s")
        if n >= 2:
            if action[1, 0] > thr:
                keys.add("right")
            if action[1, 0] < -thr:
                keys.add("left")
            if action[1, 1] > thr:
                keys.add("up")
            if action[1, 1] < -thr:
                keys.add("down")
        return keys

    # ---- run / save ----
    def run(self, interval: int = 60):
        from matplotlib.animation import FuncAnimation

        self.fig.suptitle(
            "endogenous-action emulator  —  R reset · M mode · C coll-death · B wall-death · SPACE pause · Q quit",
            color=self.TEXT,
            fontsize=12,
            y=0.975,
        )
        self._anim = FuncAnimation(
            self.fig,
            self.update,
            frames=itertools.count(),
            interval=interval,
            blit=False,
            cache_frame_data=False,
        )
        self.plt.show()

    def save(self, path: str, frames: int = 150, fps: int = 15, dpi: int = 110):
        from matplotlib.animation import FuncAnimation, PillowWriter

        self.fig.suptitle(self.title, color=self.TEXT, fontsize=12, y=0.995)
        anim = FuncAnimation(
            self.fig,
            self.update,
            frames=range(frames),
            interval=1000 // fps,
            blit=False,
            cache_frame_data=False,
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
        print(f"saved → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def build_world(args) -> InteractiveWorld:
    sim = SimConfig(
        seed=args.seed,
        n_objects=args.n_objects,
        radius=0.5,
        obs_res=args.obs_res,
        obs_noise_std=args.obs_noise,
        fixed_reflectivities=True,
        boundary="bounce",
    )
    icfg = InteractiveConfig(
        dynamics=args.dynamics,
        death_on_collision=args.death_on_collision,
        death_on_wall=args.death_on_wall,
        reset_on_death=True,
        reset_noise_frames=args.reset_noise_frames,
        wall_mode="bounce",
    )
    return InteractiveWorld(sim, icfg, seed=args.seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive endogenous-action world emulator"
    )
    p.add_argument("--dynamics", choices=["shift", "force"], default="force")
    p.add_argument("--driver", choices=["human", "random", "avoid"], default="human")
    p.add_argument("--n-objects", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--obs-res", type=int, default=128)
    p.add_argument("--obs-noise", type=float, default=0.05)
    p.add_argument(
        "--death-on-collision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="object–object contact ends the episode (default on; --no-death-on-collision to disable)",
    )
    p.add_argument(
        "--death-on-wall",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="touching a frustum wall ends the episode (walls still bounce; --no-death-on-wall to disable)",
    )
    p.add_argument("--reset-noise-frames", type=int, default=3)
    p.add_argument("--interval", type=int, default=60, help="ms between live frames")
    p.add_argument(
        "--save",
        type=str,
        default=None,
        help="render a demo GIF here (non-human driver)",
    )
    p.add_argument("--frames", type=int, default=150, help="frames for --save")
    p.add_argument("--fps", type=int, default=15)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    world = build_world(args)
    driver = {
        "human": None,
        "random": RandomDriver(world.n, seed=args.seed),
        "avoid": HeuristicAvoidDriver(),
    }[args.driver]
    emu = Emulator(world, driver=driver)
    if args.save:
        if driver is None:
            driver = HeuristicAvoidDriver()
            emu.driver = driver  # need a non-human driver to save
        emu.save(args.save, frames=args.frames, fps=args.fps)
    else:
        emu.run(interval=args.interval)


if __name__ == "__main__":
    main()
