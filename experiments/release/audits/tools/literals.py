import io, os, re, json, tokenize, sys
from collections import Counter, defaultdict
ROOT="/home/sevan/research/PIM/physically-implicit-modeling"
SCOPE=["pim","scripts","notebooks/master_eval.ipynb","notebooks/build_paper_tables_and_figs.ipynb","notebooks/build_appendix_tables_and_figs.ipynb","paper/figs/qualitative_edits","paper/figs/qualitative_edits_othello","paper/figs/qualitative_main","paper/figs/paper_style.py"]
RX=re.compile(r"discworld|Discworld|DISCWORLD|(?<![A-Za-z0-9])dw[-_]|L-dw")
IDRX=re.compile(r"discworld|(?<![A-Za-z0-9])dw_|_dw\b|\bdw[ab]\b|DW_")
lits=defaultdict(Counter); idents=defaultdict(Counter); imports=defaultdict(list)
def srcs():
    for s in SCOPE:
        p=os.path.join(ROOT,s)
        files=[p] if os.path.isfile(p) else [os.path.join(dp,f) for dp,dn,fn in os.walk(p) if "__pycache__" not in dp for f in fn if f.endswith(".py") or f.endswith(".ipynb")]
        for f in files:
            if f.endswith(".ipynb"):
                nb=json.load(open(f))
                code="\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
                code="\n".join(l for l in code.splitlines() if not l.lstrip().startswith(("%","!")))
                yield os.path.relpath(f,ROOT), code
            else:
                yield os.path.relpath(f,ROOT), open(f).read()
for rel,code in srcs():
    try:
        toks=list(tokenize.generate_tokens(io.StringIO(code).readline))
    except Exception as e:
        print("TOKERR",rel,e); continue
    prev=None
    for t in toks:
        if t.type==tokenize.STRING:
            s=t.string
            if RX.search(s) and len(s)<90 and "\n" not in s:
                lits[rel][s]+=1
        elif t.type==tokenize.NAME and IDRX.search(t.string):
            idents[rel][t.string]+=1
    for line in code.splitlines():
        if re.match(r"\s*(from|import)\s+\S*discworld", line): imports[rel].append(line.strip())
mode=sys.argv[1]
if mode=="lits":
    for f in sorted(lits):
        print(f"\n## {f}"); print("   "+"  ".join(f"{s}×{n}" for s,n in lits[f].most_common()))
elif mode=="idents":
    for f in sorted(idents):
        print(f"## {f}: "+"  ".join(f"{s}×{n}" for s,n in idents[f].most_common()))
elif mode=="imports":
    n=0
    for f in sorted(imports):
        n+=len(imports[f]); print(f"## {f}"); [print("   ",l[:150]) for l in imports[f]]
    print("TOTAL import lines",n, "files",len(imports))
