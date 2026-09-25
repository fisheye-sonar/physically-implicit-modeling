"""Static import closure over pim/ from the release entry points (read-only)."""
import ast, json, sys, re
from pathlib import Path
from collections import defaultdict

REPO = Path("/home/sevan/research/PIM/physically-implicit-modeling")
PIM = REPO / "pim"

ENTRY = {
    "nb:master_eval": REPO / "notebooks/master_eval.ipynb",
    "nb:paper_tables": REPO / "notebooks/build_paper_tables_and_figs.ipynb",
    "nb:appendix_tables": REPO / "notebooks/build_appendix_tables_and_figs.ipynb",
}
for s in ["train", "generate_dataset", "make_othello_corpus", "make_othello_edits", "make_edit_selection",
          "make_discworld_tokens", "bayes_floor", "score_prediction", "reachability_table",
          "two_flip_editability", "fit_probes", "layout_checkpoint_replicate", "othello_corpus_stats"]:
    ENTRY[f"sc:{s}"] = REPO / f"scripts/{s}.py"
ENTRY["sc:demos/demo"] = REPO / "scripts/demos/demo.py"
ENTRY["sc:demos/play"] = REPO / "scripts/demos/play.py"
ENTRY["fig:qe_rw"] = REPO / "paper/figs/qualitative_edits/make_figure.py"
ENTRY["fig:qe_oth"] = REPO / "paper/figs/qualitative_edits_othello/make_figure.py"
for p in sorted((REPO / "paper/figs/qualitative_main").glob("*.py")):
    ENTRY[f"fig:qm/{p.stem}"] = p
ENTRY["fig:paper_style"] = REPO / "paper/figs/paper_style.py"
EXTRA = {}
if "--extra" in sys.argv:
    for p in ["paper/figs/predictive_quality/othello.py", "paper/figs/predictive_quality/rayworld.py",
              "paper/figs/predictive_quality/compute_rayworld.py", "paper/figs/editability_trends/by_point.py",
              "paper/figs/editability_trends/by_rays.py", "paper/figs/history_rewrite/make_figure.py",
              "paper/figs/history_rewrite/draw_paper.py"]:
        EXTRA["x:" + p.split("figs/")[1]] = REPO / p
    for p in sorted((REPO / "paper/figs/environments_overview").rglob("*.py")):
        EXTRA["x:" + str(p.relative_to(REPO / "paper/figs"))] = p
    ENTRY.update(EXTRA)


def source_of(path: Path) -> str:
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text())
        cells = []
        for c in nb["cells"]:
            if c["cell_type"] != "code":
                continue
            src = "".join(c["source"])
            src = "\n".join(("#" + l) if l.lstrip().startswith(("%", "!")) else l for l in src.splitlines())
            cells.append(src)
        return "\n\n".join(cells)
    return path.read_text()


def mod_path(mod: str):
    p = REPO / Path(*mod.split("."))
    if (p / "__init__.py").exists():
        return p / "__init__.py"
    if p.with_suffix(".py").exists():
        return p.with_suffix(".py")
    return None


def mod_name(path: Path) -> str:
    rel = path.relative_to(REPO).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def imports_of(path: Path):
    """(module, lazy) pairs; lazy = import inside a function/method body."""
    tree = ast.parse(source_of(path))
    this = mod_name(path) if path.suffix == ".py" and str(path).startswith(str(PIM)) else None
    is_pkg = path.name == "__init__.py"
    out = []

    def visit(node, lazy):
        for ch in ast.iter_child_nodes(node):
            l2 = lazy or isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            if isinstance(ch, ast.Import):
                for a in ch.names:
                    out.append((a.name, l2))
            elif isinstance(ch, ast.ImportFrom):
                if ch.level:
                    base = this.split(".")
                    if not is_pkg:
                        base = base[:-1]
                    base = base[: len(base) - (ch.level - 1)]
                    m = ".".join(base + ([ch.module] if ch.module else []))
                else:
                    m = ch.module
                out.append((m, l2))
                for a in ch.names:
                    out.append((m + "." + a.name, l2))  # submodule import possibility
            visit(ch, l2)

    visit(tree, False)
    return out


def pim_targets(path):
    res = []
    for m, lazy in imports_of(path):
        if not m or not m.startswith("pim"):
            continue
        # every package prefix gets imported too
        parts = m.split(".")
        for i in range(1, len(parts) + 1):
            mm = ".".join(parts[:i])
            mp = mod_path(mm)
            if mp is not None:
                res.append((mm, lazy))
    return res


if __name__ == "__main__":
    all_mods = sorted(mod_name(p) for p in PIM.rglob("*.py") if "__pycache__" not in str(p))
    reach = defaultdict(set)      # module -> entries
    lazy_only = defaultdict(set)  # module -> entries where reached only via lazy edges
    for ename, epath in ENTRY.items():
        if not epath.exists():
            print("MISSING", ename, epath)
            continue
        seen = {}
        stack = [(m, lz) for m, lz in pim_targets(epath)]
        while stack:
            m, lz = stack.pop()
            if m in seen and (seen[m] is False or lz):
                continue
            seen[m] = lz if m not in seen else (seen[m] and lz)
            for m2, lz2 in pim_targets(mod_path(m)):
                stack.append((m2, lz or lz2))
        for m, lz in seen.items():
            reach[m].add(ename)
            if lz:
                lazy_only[m].add(ename)

    print("== reachable ==")
    for m in all_mods:
        if m in reach:
            e = sorted(reach[m])
            lz = sorted(lazy_only[m])
            print(f"{m:45s} n={len(e):2d} {'LAZYVIA=' + ','.join(lz) if lz else ''}\n      {', '.join(e)}")
    print("== unreachable ==")
    for m in all_mods:
        if m not in reach:
            print(m)

    # per-module direct pim imports (for reference)
    if "--edges" in sys.argv:
        print("== edges ==")
        for m in all_mods:
            t = sorted({(x, lz) for x, lz in pim_targets(mod_path(m)) if x != m})
            print(m, "->", ", ".join(x + ("*" if lz else "") for x, lz in t))
