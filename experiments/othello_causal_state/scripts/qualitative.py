#!/usr/bin/env python
"""Qualitative boards: the true position with its dead / irrelevant squares shaded, the
probe's decoded board with the same shading and its errors marked, and the probe's
per-square entropy.

Examples: the held-out positions with the most dead squares (natural, in-distribution)
and one engineered game played with a "fill the bottom-left first" policy — legal moves
only, but far from the uniform-random play the model was trained on, so it is a
qualitative illustration, not evidence. One figure per example, rows = LIN / MLP-128.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from label_relevance import label_game  # noqa: E402
from probe_by_relevance import FAM_LABEL, FAMS, cached_grid, probe_games  # noqa: E402

from pim.environments.othello.data import T_MODEL, canonical_vocab, harvest_point  # noqa: E402
from pim.environments.othello.vendor.othello import OthelloBoardState  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402

EXP = REPO / "experiments" / "othello_causal_state"
DEAD_RGBA, IRR_RGBA = (0.90, 0.10, 0.10, 0.62), (1.0, 0.88, 0.25, 0.50)   # red / yellow


def engineered_game(max_moves: int = 59) -> list[int]:
    """Legal play that always takes the legal square nearest the bottom-left corner."""
    b, moves = OthelloBoardState(), []
    for _ in range(max_moves):
        legal = b.get_valid_moves()
        if not legal:
            break
        mv = min(legal, key=lambda m: ((7 - m // 8) + (m % 8), 7 - m // 8))
        b.umpire(mv)
        moves.append(int(mv))
    return moves


def draw_board(ax, state_abs, dead, irr, title, errors=None, last=None):
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(7.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for q in range(64):
        r, c = divmod(q, 8)
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="#3a8f4a", edgecolor="#1f4d28",
                               lw=0.8))
        if dead[q]:
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=DEAD_RGBA, edgecolor="none"))
        elif irr[q]:
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=IRR_RGBA, edgecolor="none"))
        if state_abs[q] != 0:
            ax.add_patch(Circle((c, r), 0.38, facecolor="black" if state_abs[q] > 0 else "white",
                                edgecolor="#222222", lw=0.8))
        if errors is not None and errors[q]:
            ax.plot(c, r, marker="x", color="#d62728", ms=11, mew=2.4)
    if last is not None:
        ax.plot(last % 8, last // 8, marker=".", color="#1f77b4", ms=6)
    ax.set_title(title, fontsize=9, loc="left", pad=4)


def example_figure(name, moves, t, flags, probs, nxt_color, out_png):
    """flags: label_game output at position t; probs: {fam: (64, 3) softmax}."""
    dead = flags["dead"][t].astype(bool)
    occ_t = flags["occupied"][t].astype(bool)
    # yellow = UNREAD now (on no gap-free run from any empty square; colour-blind, so free
    # of the single-flip test's redundancy problem) but not dead; red = dead
    irr = occ_t & ~flags["traversable"][t].astype(bool)
    read_inert = occ_t & flags["traversable"][t].astype(bool) & flags["irrelevant"][t].astype(bool)
    mine = flags["mine"][t]
    # mine/theirs -> absolute colour (+1 black, -1 white) for display
    def to_abs(lab):
        return np.where(lab == 0, 0, np.where(lab == 1, nxt_color, -nxt_color))

    true_abs = to_abs(mine)
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.4), gridspec_kw=dict(wspace=0.08, hspace=0.18))
    meta = {"name": name, "move_number": t + 1, "n_dead": int(dead.sum()),
            "n_unread_not_dead": int((irr & ~dead).sum()), "n_read_inert": int(read_inert.sum()),
            "n_occupied": int(occ_t.sum()), "dead_squares": np.where(dead)[0].tolist(),
            "families": {}}
    for i, fam in enumerate(FAMS):
        p = probs[fam]
        pred = p.argmax(-1)
        wrong = (pred != mine) & (mine != 0)
        ent = -(p * np.log(np.clip(p, 1e-12, None))).sum(-1)
        draw_board(axes[i, 0], true_abs, dead, irr, f"true board · after move {t + 1}",
                   last=moves[t])
        draw_board(axes[i, 1], to_abs(pred), dead, irr,
                   f"{FAM_LABEL[fam]} decoded board · × = wrong", errors=wrong)
        ax = axes[i, 2]
        im = ax.imshow(ent.reshape(8, 8), cmap="magma", vmin=0, vmax=np.log(3))
        for q in np.where(dead)[0]:                     # dead squares outlined in red
            r, c = divmod(int(q), 8)
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="#ff4d4d", lw=2.2))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{FAM_LABEL[fam]} entropy · 0 … ln 3", fontsize=9, loc="left", pad=4)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        alive = occ_t & ~irr & ~read_inert                 # read AND the legal set depends on it
        unread = irr & ~dead
        meta["families"][fam] = {
            "acc_dead": float((~wrong)[dead].mean()) if dead.any() else None,
            "acc_unread": float((~wrong)[unread].mean()) if unread.any() else None,
            "acc_read_inert": float((~wrong)[read_inert].mean()) if read_inert.any() else None,
            "acc_alive": float((~wrong)[alive].mean()) if alive.any() else None,
            "entropy_dead": float(ent[dead].mean()) if dead.any() else None,
            "entropy_unread": float(ent[unread].mean()) if unread.any() else None,
            "entropy_read_inert": float(ent[read_inert].mean()) if read_inert.any() else None,
            "entropy_alive": float(ent[alive].mean()) if alive.any() else None}
    fig.suptitle(f"{name} — move {t + 1}: {int(dead.sum())} dead squares (red), "
                 f"{int((irr & ~dead).sum())} unread now (yellow); unshaded = on a gap-free run "
                 "from an empty square, i.e. read by some legality computation", fontsize=11, y=0.99)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return meta


@torch.no_grad()
def probe_position(model, probes, point, moves, t, dev):
    stoi = canonical_vocab()
    tok = np.zeros((1, T_MODEL), np.int64)
    tok[0, : len(moves[:T_MODEL])] = [stoi[m] for m in moves[:T_MODEL]]
    acts = harvest_point(model, tok, point)                      # (1, 59, d)
    x = torch.from_numpy(acts[0, t : t + 1]).to(dev)
    return {fam: torch.softmax(probes[("mine", fam, "sequence", point)].to(dev).eval()(x).float(), -1)[0]
            .cpu().numpy() for fam in FAMS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="runs/initial_othello_comparison/L-oth-20m")
    ap.add_argument("--relevance", default="scores/relevance_test5000.npz")
    ap.add_argument("--point", type=int, default=None, help="residual point (default: MLP best)")
    ap.add_argument("--n-natural", type=int, default=3)
    ap.add_argument("--max-move", type=int, default=54,
                    help="latest move number for examples, so alive squares remain")
    a = ap.parse_args()
    run_dir = (REPO / a.run).resolve()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    S = json.loads((run_dir / "scores.json").read_text())
    point = a.point if a.point is not None else int(np.argmax(S["probe_skill"]["mine|mlp|sequence"]))
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=dev)
    model.eval()
    probes, _ = cached_grid(model, probe_games(S["settings"]["oth_probe_games"]), run_dir / "probes", dev)
    (EXP / "outputs").mkdir(parents=True, exist_ok=True)
    itos = {v: k for k, v in canonical_vocab().items()}

    R = np.load(EXP / a.relevance)
    dead_count = R["dead"].astype(bool).sum(2) * R["mask"]          # (G, T)
    order = np.argsort(-dead_count.reshape(-1))
    metas, used = [], set()
    for flat in order:
        g, t = divmod(int(flat), dead_count.shape[1])
        if g in used or dead_count[g, t] == 0 or t + 1 > a.max_move:
            continue
        used.add(g)
        moves = [int(itos[int(x)]) for x in R["tokens"][g][: R["lengths"][g]]]
        flags = label_game(moves)
        b = OthelloBoardState()
        b.update(moves[: t + 1], prt=False)
        probs = probe_position(model, probes, point, moves, t, dev)
        name = f"natural game {int(R['game_ids'][g])}"
        metas.append(example_figure(name, moves, t, flags, probs, 1 if b.next_hand_color > 0 else -1,
                                    EXP / "outputs" / f"qual_natural_{len(metas) + 1}.png"))
        print(f"{name} move {t + 1}: dead {metas[-1]['n_dead']}  "
              + "  ".join(f"{FAM_LABEL[f]} acc dead/unread/inert/alive "
                          f"{metas[-1]['families'][f]['acc_dead']}/{metas[-1]['families'][f]['acc_unread']}"
                          f"/{metas[-1]['families'][f]['acc_read_inert']}"
                          f"/{metas[-1]['families'][f]['acc_alive']}" for f in FAMS), flush=True)
        if len(metas) >= a.n_natural:
            break

    moves = engineered_game()
    flags = label_game(moves)
    dc = flags["dead"].astype(bool).sum(1) * flags["mask"]
    t = int(np.argmax(dc[: a.max_move]))
    b = OthelloBoardState()
    b.update(moves[: t + 1], prt=False)
    probs = probe_position(model, probes, point, moves, t, dev)
    m = example_figure("engineered: fill the bottom-left first (off-distribution play)", moves, t,
                       flags, probs, 1 if b.next_hand_color > 0 else -1,
                       EXP / "outputs" / "qual_engineered.png")
    m["moves"] = moves
    m["dead_count_by_move"] = dc[: len(moves)].tolist()
    metas.append(m)
    print(f"engineered game: {len(moves)} moves, max dead {int(dc.max())} at move {t + 1}", flush=True)
    (EXP / "scores" / "qualitative_examples.json").write_text(
        json.dumps({"point": point, "examples": metas}, indent=1, default=float))
    print("done qualitative")


if __name__ == "__main__":
    main()
