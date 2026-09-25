"""Symbol-level (top-level def/class/assign) reachability over pim/ from release entry points.

Conservative: reaching a class reaches all its methods; reaching a module runs its non-def top-level
statements; package __init__ re-exports are followed to the defining module. Names referenced
anywhere in a reached body (including attribute chains on imported modules) are followed.
Dynamic dispatch (getattr, dict-of-strings) is NOT followed -> checked by hand.
"""
import ast, json, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from closure import ENTRY, source_of, mod_path, mod_name, REPO, PIM  # noqa

MODS = {}


def _pkg_rebinds(pkg, name):
    """True if package __init__ binds `name` to a non-module symbol (e.g. pim.training.train)."""
    p = mod_path(pkg)
    if p is None or p.name != "__init__.py":
        return False
    t = ast.parse(p.read_text())
    for node in t.body:
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if (a.asname or a.name) == name and mod_path((node.module or "") + "." + a.name) is None:
                    return True
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return True
    return False


class Mod:
    def __init__(self, name, path):
        self.name, self.path = name, path
        self.tree = ast.parse(source_of(path))
        self.is_pkg = path.name == "__init__.py"
        self.defs = {}       # name -> node(s)
        self.lines = {}      # name -> (start, end)
        self.alias = {}      # local name -> ("mod", modname) | ("sym", modname, symname)
        self.toplevel_other = []
        for node in self.tree.body:
            self._top(node)

    def _resolve_from(self, node):
        if node.level:
            base = self.name.split(".")
            if not self.is_pkg:
                base = base[:-1]
            base = base[: len(base) - (node.level - 1)]
            return ".".join(base + ([node.module] if node.module else []))
        return node.module

    def _imports(self, node, into):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    into[a.asname] = ("mod", a.name)
                else:
                    into[a.name.split(".")[0]] = ("mod", a.name.split(".")[0])
                    # also note the full dotted path so attr chains resolve
                    into["__dotted__" + a.name] = ("mod", a.name)
        elif isinstance(node, ast.ImportFrom):
            m = self._resolve_from(node)
            for a in node.names:
                local = a.asname or a.name
                if m and mod_path(m + "." + a.name) is not None and m.startswith("pim") and not _pkg_rebinds(m, a.name):
                    into[local] = ("mod", m + "." + a.name)
                else:
                    into[local] = ("sym", m, a.name)

    def _top(self, node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self.defs.setdefault(node.name, []).append(node)
            self.lines[node.name] = (node.lineno, node.end_lineno)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            self._imports(node, self.alias)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = []
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        names.append(n.id)
            if names:
                for n in names:
                    self.defs.setdefault(n, []).append(node)
                    self.lines[n] = (node.lineno, node.end_lineno)
            else:
                self.toplevel_other.append(node)
        elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For)):
            # e.g. `if __name__ == "__main__"`, try/except import; treat as always-run
            self.toplevel_other.append(node)
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    self._imports(sub, self.alias)
        else:
            self.toplevel_other.append(node)


def get_mod(name):
    if name not in MODS:
        p = mod_path(name)
        if p is None:
            return None
        MODS[name] = Mod(name, p)
    return MODS[name]


def refs_in(node, mod, local_alias=None):
    """Yield (modname, symname) or ("MOD", modname) references in node."""
    alias = dict(mod.alias)
    # in-function imports
    for sub in ast.walk(node):
        if isinstance(sub, (ast.Import, ast.ImportFrom)):
            mod._imports(sub, alias)
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Import):
            for a in sub.names:
                if a.name.startswith("pim"):
                    out.add(("MOD", a.name))
        if isinstance(sub, ast.ImportFrom):
            m = mod._resolve_from(sub)
            if m and m.startswith("pim"):
                out.add(("MOD", m))
                for a in sub.names:
                    if mod_path(m + "." + a.name) is not None and not _pkg_rebinds(m, a.name):
                        out.add(("MOD", m + "." + a.name))
                    else:
                        out.add((m, a.name))
        if isinstance(sub, ast.Attribute):
            # resolve chains a.b.c
            chain = []
            cur = sub
            while isinstance(cur, ast.Attribute):
                chain.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                chain.append(cur.id)
                chain = chain[::-1]
                a = alias.get(chain[0])
                if a and a[0] == "mod":
                    base = a[1]
                    # walk down submodules
                    i = 1
                    while i < len(chain) and mod_path(base + "." + chain[i]) is not None:
                        base = base + "." + chain[i]
                        i += 1
                    if base.startswith("pim"):
                        out.add(("MOD", base))
                        if i < len(chain):
                            out.add((base, chain[i]))
        if isinstance(sub, ast.Name):
            nm = sub.id
            if nm in mod.defs:
                out.add((mod.name, nm))
            a = alias.get(nm)
            if a:
                if a[0] == "sym" and a[1] and a[1].startswith("pim"):
                    out.add((a[1], a[2]))
                elif a[0] == "mod" and a[1].startswith("pim"):
                    out.add(("MOD", a[1]))
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and sub.value in mod.defs:
            # __all__ strings etc.; skip (would over-reach)
            pass
    return out


def resolve(modname, sym, depth=0):
    """Follow re-exports to the defining module."""
    m = get_mod(modname)
    if m is None or depth > 10:
        return None
    if sym in m.defs:
        return (modname, sym)
    a = m.alias.get(sym)
    if a:
        if a[0] == "sym" and a[1] and a[1].startswith("pim"):
            return resolve(a[1], a[2], depth + 1)
        if a[0] == "mod":
            return ("MOD", a[1])
    if mod_path(modname + "." + sym) is not None:
        return ("MOD", modname + "." + sym)
    return None


def run(entries, follow_init_reexports=False):
    reached_mods = set()
    reached = set()
    reach_by = defaultdict(set)
    stack = []

    def add_refs(refs, src):
        for r in refs:
            stack.append((r, src))

    for ename, epath in entries.items():
        # entry treated as pseudo-module (for scripts inside pim-less dirs)
        pseudo = Mod.__new__(Mod)
        pseudo.name = "__entry__"
        pseudo.path = epath
        pseudo.tree = ast.parse(source_of(epath))
        pseudo.is_pkg = False
        pseudo.defs, pseudo.lines, pseudo.alias, pseudo.toplevel_other = {}, {}, {}, []
        for node in pseudo.tree.body:
            pseudo._top(node)
        # everything in an entry file is reached
        add_refs(refs_in(pseudo.tree, pseudo), ename)
        while stack:
            r, src = stack.pop()
            if r[0] == "MOD":
                mn = r[1]
                key = ("MOD", mn)
                if ename in reach_by[key]:
                    continue
                reach_by[key].add(ename)
                m = get_mod(mn)
                if m is None:
                    continue
                reached_mods.add(mn)
                # parent packages run their __init__
                parts = mn.split(".")
                for i in range(1, len(parts)):
                    stack.append((("MOD", ".".join(parts[:i])), src))
                for node in m.toplevel_other:
                    add_refs(refs_in(node, m), src)
                if follow_init_reexports and m.is_pkg:
                    for node in m.tree.body:
                        if isinstance(node, ast.ImportFrom):
                            add_refs(refs_in(node, m), src)
                continue
            res = resolve(*r)
            if res is None:
                continue
            if res[0] == "MOD":
                stack.append((res, src))
                continue
            if ename in reach_by[res]:
                continue
            reach_by[res].add(ename)
            reached.add(res)
            stack.append((("MOD", res[0]), src))
            m = get_mod(res[0])
            for node in m.defs[res[1]]:
                add_refs(refs_in(node, m), src)
    return reached, reach_by, reached_mods


if __name__ == "__main__":
    reached, reach_by, reached_mods = run(ENTRY, follow_init_reexports="--init" in sys.argv)
    all_mods = sorted(mod_name(p) for p in PIM.rglob("*.py") if "__pycache__" not in str(p))
    tot_dead = 0
    for mn in all_mods:
        m = get_mod(mn)
        n_lines = len(m.path.read_text().splitlines())
        if mn not in reached_mods:
            print(f"### {mn}  [{n_lines} lines]  MODULE NOT REACHED (symbol-level)")
            continue
        dead = []
        for name, (a, b) in sorted(m.lines.items(), key=lambda kv: kv[1][0]):
            if (mn, name) not in reached and not name.startswith("__"):
                dead.append((name, a, b))
        dl = sum(b - a + 1 for _, a, b in dead)
        tot_dead += dl
        print(f"### {mn}  [{n_lines} lines]  unreached defs: {len(dead)} (~{dl} lines)")
        for name, a, b in dead:
            print(f"    - {name}  L{a}-{b} ({b - a + 1})")
    print("TOTAL unreached def lines:", tot_dead)
    if "--by" in sys.argv:
        for k in sorted(reach_by, key=str):
            if k[0] != "MOD":
                print(k, sorted(reach_by[k]))
