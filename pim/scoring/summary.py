"""Per-run summaries: the headline block of each scores.json, human-readable (moved verbatim
from master_eval cell [7], 2026-09-19)."""
# [7] Per-run summaries: the headline block of each scores.json, human-readable.
#     Discworld prints one block per PROBE TARGET (a basis of the regression target, or
#     a categorical target); each editor's reported arm names the dim set that won it, and
#     the losing dim set is shown beside it so the choice is visible rather than buried in
#     scores.json. A frames-as-tokens run (`ei_construction` "frame-set") has no zone
#     RMSEs — those columns read "—" and the construction is named. Othello prints its
#     canonical block and then every extra-target block.
import json


def print_summaries(runs) -> None:
    for r in runs:
        sp = r["dir"] / "scores.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        print(f"\n{'=' * 86}\n{s['run']}   ({s['arch']} on {s['env']}/{s['instance']})   "
              f"val {s['val_loss']:.5f}\n{'=' * 86}")
        if s["env"] == "discworld":
            cons = s.get("ei_construction", "ray-zone")
            for key, T in s["bases"].items():
                u = T["unedited"]
                print(f"  block={key}  target={T.get('target', s.get('target', 'full'))}  "
                      f"basis={T.get('basis', key)}  kind={T.get('kind', 'regression')}  "
                      f"EI construction={cons}  UNEDITED EI {u['edit_index']:+.4f}   "
                      f"probe skill (linear, best point) {max(T['probe_skill_linear']):+.4f}  "
                      f"(mlp) {max(T['probe_skill_mlp']):+.4f}  "
                      f"tripwire violations {T['probe_sanity']['n_violations']}")
                print(f"  {'editor':<8}{'dims':>6}{'pt':>4}{'alpha':>8}{'EI':>9}{'fid':>7}"
                      f"{'target':>9}{'collat':>9}   | EI by dim set")
                for ed, bst in T["best"].items():
                    if not bst:
                        continue
                    byd = "  ".join(
                        f"{d}={T['best_by_dims'][d][ed]['edit_index']:+.4f}"
                        for d in T["best_by_dims"] if T["best_by_dims"][d].get(ed))
                    fmt = lambda v: f"{v:>9.4f}" if isinstance(v, (int, float)) else f"{'—':>9}"
                    print(f"  {ed:<8}{bst.get('dims', '—'):>6}{bst['point']:>4}"
                          f"{bst['alpha']:>8}{bst['edit_index']:>+9.4f}"
                          f"{bst['fidelity_ratio']:>7.3f}{fmt(bst.get('target_rmse'))}"
                          f"{fmt(bst.get('collateral_rmse'))}   | {byd}")
        else:
            g = s["gates"]
            print(f"  gates: legal mass {g['legal_mass']:.4f}  top-1 legal {g['top1_legal']:.4f}  "
                  f"CE {g['ce']:.4f} (Bayes {g['bayes_ce']:.4f}, excess {g['ce'] - g['bayes_ce']:+.4f})")
            sk = s["probe_skill"]
            for key in ("mine|linear|sequence", "mine|mlp|sequence"):
                if key in sk:
                    print(f"  probe skill [{key}]: best point {max(sk[key]):+.4f}")
            u = s["unedited"]
            print(f"  UNEDITED EI(union) {u['edit_index_union']:+.4f}   li vs post {u['li_error_vs_post']:.3f}")
            print(f"  {'editor':<12}{'pt':>4}{'alpha':>8}{'EI(un)':>9}{'li post':>9}{'li pre':>9}{'legal':>8}")
            for ed, bst in s["best"].items():
                if bst:
                    print(f"  {ed:<12}{bst['point']:>4}{bst['alpha']:>8}"
                          f"{bst['edit_index_union']:>+9.4f}{bst['li_error_vs_post']:>9.3f}"
                          f"{bst['li_error_vs_pre']:>9.3f}{bst['legal_mass']:>8.4f}")
            for key, T in s.get("bases", {}).items():
                print(f"  block={key}  kind={T.get('kind')}  probe skill (linear, best point) "
                      f"{max(T['probe_skill_linear']):+.4f}  (mlp) {max(T['probe_skill_mlp']):+.4f}  "
                      f"tripwire violations {T['probe_sanity']['n_violations']}")
                for ed, bst in T["best"].items():
                    if bst:
                        print(f"  {ed:<12}{bst['point']:>4}{bst['alpha']:>8}"
                              f"{bst['edit_index_union']:>+9.4f}{bst['li_error_vs_post']:>9.3f}"
                              f"{bst['li_error_vs_pre']:>9.3f}{bst['legal_mass']:>8.4f}"
                              f"   fid {bst['fidelity_ratio']:.3f}")
