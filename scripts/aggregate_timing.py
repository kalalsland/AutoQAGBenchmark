import json, os
BASE="outputs"
PARSE=json.load(open(f"{BASE}/timing_base/pipeline_timing.json", encoding="utf-8"))
parse_sec=next(s["seconds"] for s in PARSE["stages"] if s["stage"]=="parse")
norm_sec=next(s["seconds"] for s in PARSE["stages"] if s["stage"]=="normalize")
models=[("gpt-4o-mini","timing_gpt"),("gemini-2.5-flash","timing_gemini"),("claude-haiku-4-5","timing_claude")]
report={"shared":{"parse_PDF抽取_sec":parse_sec,"normalize_sec":norm_sec,"note":"parse/normalize 与模型无关，只跑一次(5篇PDF, MinerU+PyMuPDF回退)"},"models":{}}
def stage(d,name):
    for s in d["stages"]:
        if s["stage"]==name: return s
    return None
print("="*92)
print(f"{'环节':<14}{'gpt-4o-mini':>22}{'gemini-2.5-flash':>22}{'claude-haiku-4-5':>22}")
print("-"*92)
data={}
for m,wd in models:
    p=json.load(open(f"{BASE}/{wd}/pipeline_timing.json", encoding="utf-8")); data[m]=p
def row(label,key,statkey="seconds"):
    vals=[]
    for m,_ in models:
        s=stage(data[m],key); vals.append(s[statkey] if s else None)
    print(f"{label:<14}"+"".join(f"{(f'{v:.2f}s' if v is not None else '-'):>22}" for v in vals))
    return vals
print(f"{'PDF抽取(parse)':<14}"+"".join(f"{parse_sec:>21.2f}s" for _ in models)+"  ← 共享")
report["shared"]["parse_PDF抽取_sec"]=parse_sec
g=row("物理图建立(graph)","graph")
s=row("子图构建(sample)","sample")
ge=row("问题生成(generate)","generate")
print("-"*92)
# points / qa
def stat(key,field):
    out=[]
    for m,_ in models:
        st=stage(data[m],key); out.append(st["stats"].get(field) if st else None)
    return out
pts=stat("graph","points"); plans=stat("sample","plans"); qa=stat("generate","qa")
print(f"{'(points)':<14}"+"".join(f"{str(v):>22}" for v in pts))
print(f"{'(plans)':<14}"+"".join(f"{str(v):>22}" for v in plans))
print(f"{'(QA保留)':<14}"+"".join(f"{str(v):>22}" for v in qa))
print("="*92)
# per-type generate
print("\n分题型 平均单题生成时间 (墙钟, 并发=8):  avg_sec  (n)")
print("-"*92)
types=["atomic","numerical","condition","comparative","table","formula","multi_hop","summary"]
gt={}
for m,wd in models:
    f=f"{BASE}/{wd}/generate_timing.json"
    gt[m]=json.load(open(f)) if os.path.exists(f) else {"by_type":{}}
print(f"{'题型':<14}{'gpt-4o-mini':>22}{'gemini-2.5-flash':>22}{'claude-haiku-4-5':>22}")
for t in types:
    cells=[]
    for m,_ in models:
        bt=gt[m]["by_type"].get(t)
        cells.append(f"{bt['avg_sec']:.1f}s (n={bt['count']})" if bt else "-")
    print(f"{t:<14}"+"".join(f"{c:>22}" for c in cells))
print("="*92)
# build report json
for m,wd in models:
    d=data[m]
    report["models"][m]={
        "graph_物理图建立_sec":(stage(d,"graph") or {}).get("seconds"),
        "sample_子图构建_sec":(stage(d,"sample") or {}).get("seconds"),
        "generate_问题生成_sec":(stage(d,"generate") or {}).get("seconds"),
        "total_sec":d.get("total_seconds"),
        "graph_points":(stage(d,"graph") or {}).get("stats",{}).get("points"),
        "sample_plans":(stage(d,"sample") or {}).get("stats",{}).get("plans"),
        "generate_qa":(stage(d,"generate") or {}).get("stats",{}).get("qa"),
        "generate_by_type":gt[m].get("by_type"),
    }
json.dump(report,open(f"{BASE}/timing_summary.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print("已写入 outputs/timing_summary.json")
