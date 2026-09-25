import ast, sys
from pathlib import Path
R = Path("/home/sevan/research/PIM/physically-implicit-modeling")
want = {
 "pim/environments/discworld/arms.py": ["free_rollout","counterfactual_history","overwrite_oracle_rollout","freeze_oracle_rollout","oracle_arm","nanda_rollout","nanda_arm","categorical_direction","_roll_hook"],
 "pim/environments/discworld/token_bench.py": ["nanda_arm"],
 "pim/environments/othello/arms.py": ["observation_probes","linear_arm","grad_steer_arm","_split","gates"],
 "pim/editors/pinv.py": ["pinv_maps","pinv_step","PinvMap"],
 "pim/figures/tables.py": ["image_table","_mark","_rand_perdim","tables_components","table_arms","table_seed_variance","fig_training_curve","set_basis","reg_key","basis_star","_blank_regression_row","table_gridified","_collect"],
 "pim/models/registry.py": ["_build_s","_build_s_tokens","_build_recurrent","_infer_arch","load_run"],
 "pim/scoring/baselines.py": ["_dw_regression_floors","score_baseline_targets","score_baselines_arch","score_baseline_bases","score_all_baselines"],
 "pim/scoring/driver.py": ["missing_inverse","add_inverse","score_all"],
 "pim/scoring/othello.py": ["othello_arms","score_othello"],
 "pim/scoring/discworld.py": ["inverse_discworld","score_discworld","score_discworld_tokens"],
 "pim/environments/discworld/soft_render.py": ["render_frame_torch","render_frame_soft","blur_matrix","_profile"],
 "pim/environments/discworld/loading.py": ["Dataset","DatasetBundle","_load_h5_dataset","load_dataset"],
 "pim/environments/discworld/dataset.py": ["load_sample","generate_dataset"],
 "pim/environments/layout.py": ["_has_v1_files","ensure_marker","parse_dataset_path","legacy_probe_key","legacy_edits_instance","unused_dir"],
 "pim/environments/discworld/grid_target.py": ["AppearanceTarget","_view_cells","_n_views","_view_poses","_single_view","_grid_factors","_grid_factor_sizes","_app_factors","_app_factor_sizes","SnappedTarget","FactorisedTarget"],
 "pim/environments/othello/vendor/mingpt_model.py": ["GPTforProbing","GPTforIntervention","GPTforProbeIA"],
 "pim/training/train.py": ["mse_next_move_onehot","train"],
 "pim/environments/othello/data.py": ["signed_mine","flatten_rows","move_probs","synthetic_games"],
 "pim/environments/othello/counterfactual.py": ["mine_board","search_cf"],
 "pim/environments/othello/bench.py": ["load_li_benchmark","shipped_length_distribution"],
}
for f, names in want.items():
    t = ast.parse((R/f).read_text())
    out=[]
    for node in ast.walk(t):
        if isinstance(node,(ast.FunctionDef,ast.ClassDef,ast.AsyncFunctionDef)) and node.name in names:
            out.append(f"{node.name} L{node.lineno}-{node.end_lineno}({node.end_lineno-node.lineno+1})")
    print(f, "::", "; ".join(out))
