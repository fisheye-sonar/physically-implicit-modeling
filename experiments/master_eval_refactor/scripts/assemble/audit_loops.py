"""Diff the three notebook top-level loops against the functions they became."""
import ast
import difflib
import sys
import textwrap
from pathlib import Path

CELLS, PKG = Path(sys.argv[1]), Path(sys.argv[2]) / "pim" / "scoring"


def fn_body(path: Path, fn: str) -> list[str]:
    src = path.read_text()
    f = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == fn)
    first = f.body[0]
    skip = first.end_lineno if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) else f.lineno
    return textwrap.dedent("\n".join(src.splitlines()[skip:f.end_lineno])).splitlines()


def cell_tail(i: int, start: str, end: str | None) -> list[str]:
    s = (CELLS / f"cell{i}.py").read_text()
    a = s.index(start)
    b = len(s) if end is None else s.index(end)
    return s[a:b].rstrip("\n").splitlines()


for name, new, old in (
        ("score_all_baselines", fn_body(PKG / "baselines.py", "score_all_baselines"),
         cell_tail(6, "# every (instance, arch) pair", 'print("\\nall baselines present")')),
        ("score_all", fn_body(PKG / "driver.py", "score_all"),
         cell_tail(7, "for r in RUNS:", 'print("\\nall runs scored")')),
        ("print_summaries", fn_body(PKG / "summary.py", "print_summaries"),
         cell_tail(8, "for r in RUNS:", None))):
    d = [ln for ln in difflib.unified_diff(old, new, lineterm="", n=0)
         if ln[:1] in "+-" and ln[:3] not in ("+++", "---")]
    print(f"\n=== {name}: {len(old)} original lines; {sum(x[0] == '-' for x in d)} removed / "
          f"{sum(x[0] == '+' for x in d)} added ===")
    for ln in d:
        print("   " + ln[:160])
