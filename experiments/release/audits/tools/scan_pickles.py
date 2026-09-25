import io, os, re, sys, zipfile, pickletools, json
from collections import Counter, defaultdict
ROOT="/home/sevan/research/PIM/physically-implicit-modeling"
S=os.path.dirname(os.path.abspath(__file__))
RUNS=[l.strip() for l in open(f"{S}/runs.txt") if l.strip()]
BAD=re.compile(r"/home/|sevan|brodjian|wsl|tailscale|@|caltech", re.I)
PATHLIKE=re.compile(r"/")
def ustrs(data):
    out=[]; glob=[]; prev=[]
    for op,arg,pos in pickletools.genops(io.BytesIO(data)):
        if op.name in ("SHORT_BINUNICODE","BINUNICODE","BINUNICODE8","UNICODE"):
            out.append(arg); prev.append(arg)
        elif op.name=="GLOBAL": glob.append(arg.replace(" ","."))
        elif op.name=="STACK_GLOBAL" and len(prev)>=2: glob.append(prev[-2]+"."+prev[-1])
    return out, glob
def pk(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            n=[x for x in z.namelist() if x.endswith("data.pkl")][0]
            return "zip", z.read(n), z.namelist()[0].split("/")[0]
    # legacy torch format: several pickles concatenated; read the first few
    data=open(path,"rb").read()
    return "legacy", data, None
res=defaultdict(Counter); bad=defaultdict(set); pathlike=defaultdict(Counter); globs=defaultdict(Counter); fmt=Counter(); archname=Counter()
for r in RUNS:
    base=os.path.join(ROOT,"runs",r)
    for dp,dn,fn in os.walk(base):
        for f in fn:
            if not f.endswith(".pt"): continue
            p=os.path.join(dp,f); rel=os.path.relpath(p,base)
            kind = "probes" if rel.startswith("probes/") else ("ckpt" if rel.startswith("ckpt/") else f)
            try:
                ft,data,an=pk(p)
            except Exception as e:
                fmt[(kind,"ERR "+str(e)[:40])]+=1; continue
            fmt[(kind,ft)]+=1
            if an: archname[(kind, "stem==filename" if an==os.path.splitext(f)[0] else f"other:{an}")]+=1
            try:
                if ft=="legacy":
                    # parse successive pickles
                    bio=io.BytesIO(data); strs=[]; gl=[]
                    for _ in range(3):
                        start=bio.tell()
                        try:
                            ops=list(pickletools.genops(bio))
                        except Exception: break
                        end=bio.tell(); s_,g_=ustrs(data[start:end]); strs+=s_; gl+=g_
                else:
                    strs,gl=ustrs(data)
            except Exception as e:
                fmt[(kind,"PARSEERR")]+=1; continue
            for g in gl: globs[kind][g]+=1
            for s in strs:
                if BAD.search(s): bad[kind].add((r, rel, s[:160]))
                if PATHLIKE.search(s) and len(s)<200: pathlike[kind][re.sub(r"(seed\d|s\d{6}|[0-9a-f]{12,16})","*",s)]+=1
print("formats:",dict(fmt)); print("zip archive inner name:",dict(archname))
for k in globs: print("\nGLOBALS",k,dict(globs[k]))
for k in pathlike: print("\nPATHLIKE strings",k); [print("   ",n,s) for s,n in pathlike[k].most_common(40)]
for k in bad: print("\nBAD",k,len(bad[k])); [print("   ",x) for x in sorted(bad[k])[:40]]
