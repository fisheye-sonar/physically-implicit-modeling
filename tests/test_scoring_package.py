"""pim.scoring (2026-09-19) — the contracts the scoring queue leans on, pinned without a GPU or
data: which runs are scanned (the two environment hooks, the still-training guard), which blocks
a run is asked for (seed replicates inherit their parent's extra targets), what the driver
decides is missing, the arm-selection rule as it stands, and the scores.json block's shape.
The numbers themselves are gated on real runs by experiments/master_eval_refactor."""
import json

import pytest

from pim.scoring import blocks, driver, runs

S = {"dw_bases": ("frustum", "cartesian"), "dw_bases_by_instance": {"dw-8ray-obs5": ("cartesian",)},
     "dw_target": "full", "dw_edit_dims": ("all",), "oth_extra_targets": ("mine_signed",),
     "dw_extra_targets": {"ray_ablation/L-dw-5ray-20m": ("appearance-fac", "appearance")},
     "dw_alpha_nd": (1.0,), "dw_alpha_pi": (1.0,), "dw_alpha_gs": (0.1,),
     "dw_grid_alpha_nd": (2.0,), "dw_grid_alpha_pi": (2.0,), "dw_grid_alpha_gs": (0.2,)}


def _run(root, topic, name, *, steps=100, logged=100, instance="dw-5ray", env="discworld", replicate=None):
    d = root / topic / name
    d.mkdir(parents=True)
    cfg = {"arch": "transformer_l", "train": {"steps": steps}, "data": {"env": env, "instance": instance}}
    if replicate:
        cfg["replicate"] = replicate
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "best_model.pt").write_bytes(b"")
    (d / "metrics.jsonl").write_text(json.dumps({"step": logged}) + "\n")
    return d


@pytest.fixture
def tree(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    _run(root, "ray_ablation", "L-dw-5ray-20m")
    _run(root, "ray_ablation", "L-dw-5ray-20m__seed1", replicate={"of": "ray_ablation/L-dw-5ray-20m", "seed": 1})
    _run(root, "ray_ablation", "L-dw-5ray-20m__seed2", steps=512000, logged=300000)      # still training
    _run(root, "training_curve", "L-dw-8ray-20m_s032000", instance="dw-8ray")
    _run(root, "archive", "old-run")
    _run(root, "_baselines", "not-a-run")
    monkeypatch.setattr(blocks, "REPO", tmp_path)
    for k in ("PIM_ONLY_RUNS", "PIM_SKIP_TOPICS"):
        monkeypatch.delenv(k, raising=False)
    return root


def test_scan_skips_archive_underscore_topics_and_runs_still_training(tree, capsys):
    names = [r["run"] for r in runs.scan_runs(tree)]
    assert names == ["L-dw-5ray-20m", "L-dw-5ray-20m__seed1", "L-dw-8ray-20m_s032000"]
    assert "L-dw-5ray-20m__seed2  (still training)" in capsys.readouterr().out


def test_scan_honours_both_environment_hooks(tree, monkeypatch):
    monkeypatch.setenv("PIM_SKIP_TOPICS", "training_curve")
    assert [r["run"] for r in runs.scan_runs(tree)] == ["L-dw-5ray-20m", "L-dw-5ray-20m__seed1"]
    monkeypatch.setenv("PIM_ONLY_RUNS", "L-dw-5ray-20m__seed1,nonexistent")
    assert [r["run"] for r in runs.scan_runs(tree)] == ["L-dw-5ray-20m__seed1"]


def test_a_seed_replicate_inherits_its_parents_extra_targets(tree):
    want = [("frustum", "full", "frustum"), ("cartesian", "full", "cartesian"),
            ("appearance-fac", "appearance-fac", "frustum"), ("appearance", "appearance", "frustum")]
    assert blocks.discworld_blocks("ray_ablation/L-dw-5ray-20m", S) == want
    assert blocks.discworld_blocks("ray_ablation/L-dw-5ray-20m__seed1", S) == want
    assert blocks.discworld_blocks("training_curve/L-dw-8ray-20m_s032000", S) == want[:2]     # no entry, no parent
    assert blocks.dw_bases_for("dw-8ray-obs5", S) == ("cartesian",)
    assert blocks.othello_blocks(S) == ["mine/theirs", "mine_signed"]


def test_the_driver_asks_only_for_what_a_current_file_lacks(tree):
    r = {"topic": "ray_ablation", "run": "L-dw-5ray-20m__seed1", "env": "discworld"}
    im = [{"editor": "PI"}, {"editor": "IM"}]
    prev = {"bases": {"frustum": {"arms": im}, "cartesian": {"arms": im}, "appearance-fac": {"arms": [{"editor": "PI"}]}}}
    assert driver.missing_blocks(r, prev, S) == ["appearance"]
    assert driver.missing_inverse(r, prev) == ["appearance-fac"]
    oth = {"topic": "t", "run": "L-oth", "env": "othello"}
    assert driver.missing_blocks(oth, {"arms": im, "bases": {}}, S) == ["mine_signed"]
    assert driver.missing_inverse(oth, {"arms": [{"editor": "PI"}], "bases": {"mine_signed": {"arms": im}}}) == ["mine/theirs"]


def test_scorer_dispatch_is_by_what_the_model_emits():
    from pim.scoring.discworld import score_discworld, score_discworld_tokens
    from pim.scoring.othello import score_othello
    assert driver.scorer_for({"arch": "transformer_l", "env": "discworld"}) == (score_discworld, "discworld")
    assert driver.scorer_for({"arch": "transformer_l_tokens", "env": "discworld"}) == (score_discworld_tokens, "discworld/tokens")
    assert driver.scorer_for({"arch": "transformer_l_tokens", "env": "othello"}) == (score_othello, "othello")


def test_block_schema_and_the_arm_selection_rule_as_it_stands():
    """`best` = the highest Edit Index among an editor's arms, UNGUARDED (the fidelity ratio rides
    along, it does not select) — and "IM" never owns "IM-NN". Moved as is; a guard is a later change."""
    arms = [{"editor": "PI[zspace]", "point": 1, "alpha": 5.0, "edit_index": 0.35, "fidelity_ratio": 0.8, "dims": "all"},
            {"editor": "PI[zspace]", "point": 1, "alpha": 10.0, "edit_index": 0.40, "fidelity_ratio": 2.1, "dims": "all"},
            {"editor": "IM", "point": 3, "alpha": 1.0, "edit_index": 0.6, "fidelity_ratio": 0.3, "dims": "all", "curve": [1, 2]},
            {"editor": "IM-NN", "point": 3, "alpha": 1.0, "edit_index": 0.9, "fidelity_ratio": 4.0, "dims": "all"}]
    st = {"kind": "regression", "r2": 0.5, "per_dim_r2": [0.5]}
    blk = blocks.probe_block({0: (None, st)}, {0: (None, st)}, {"n_violations": 0}, {"edit_index": -0.9, "x": [1]},
                             arms, ("all",), target="full", basis="frustum", kind="regression", n_classes=None,
                             recipe={"n_seq": 3}, alphas=((1.0,), (5.0, 10.0), (0.1,)), selection=None)
    assert list(blk) == ["target", "basis", "kind", "n_classes", "probe_recipe", "alphas", "bench_selection", "unedited",
                         "probe_skill_linear", "probe_skill_mlp", "probe_perdim_linear", "probe_perdim_mlp",
                         "probe_sanity", "best", "best_by_dims", "arms"]
    assert blk["best"]["PI"]["alpha"] == 10.0 and blk["best"]["PI"]["fidelity_ratio"] == 2.1
    assert blk["best"]["IM"]["edit_index"] == 0.6 and blk["best"]["IM-NN"]["edit_index"] == 0.9
    assert blk["best"]["ND"] is None and "curve" not in blk["arms"][2] and blk["unedited"] == {"edit_index": -0.9}
    assert blk["alphas"] == {"ND": [1.0], "PI": [5.0, 10.0], "GS": [0.1]}
