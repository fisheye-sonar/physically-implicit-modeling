# qualitative_edits_othello — real boards, the model's next-move distribution under each edit

`make_figure.py --seed <k> --layout rows|cols --games <n>` → `othello_edits_seed<k>_<layout>.{pdf,png,json}`.

**Variants** (`VARIANTS`): Standard, Adjacent Flip, Adjacent NoFlip, Standard NoFlip — one run each.
**Boards** are cases of each run's canonical edit bench (`load_benchmark`; real games, every case at move
20 in the shipped set), `--games` of them per variant chosen by `--seed`; the variants do not share a
scenario since their rules differ. Boards are replayed under the instance's own rules
(`tokens_and_labels`, absolute colours). **Conditions**: Unedited (pre-edit board, the model's
distribution), Ground truth (post-edit board, uniform over its legal moves — the reference the Edit
Index is scored against), PI / GS / IM (post-edit board, the distribution after the write at the run's
scored best arm under the symmetric-difference construction, called exactly as the scorer calls them:
`linear_arm` pinv, `grad_steer_arm`, `inverse_arms`). Squares are tinted yellow by probability mass
(`--gamma`, tint = p^gamma); the edited tile is outlined pink (`--no-locator` drops it).

**Layouts.** `rows` (default): variants down, the five conditions across, repeated per game — 5·n boards
wide by 4 tall, the paper-width shape. `cols`: the sketch's arrangement, variants across and conditions
down (n games per variant side by side). Every write is computed once for all 1000 cases per run and
cached in `.scratch/othello_edits_cache.pkl`, so a new seed, layout or game count is a redraw
(`--recompute` refreshes it). The sidecar JSON records the cases, move numbers and arms used.
