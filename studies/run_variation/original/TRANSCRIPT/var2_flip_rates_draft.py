import re,json,itertools
t=open('docs/supervisor_artifact.html').read(); dec=json.JSONDecoder()
runs={}
for m in re.finditer(r'"run": "([^"]+)", "label"', t):
    j=m.start(); depth=0
    while j>0:
        j-=1
        if t[j]=='}': depth+=1
        elif t[j]=='{':
            if depth==0: break
            depth-=1
    try: o,_=dec.raw_decode(t[j:])
    except Exception: continue
    if not o.get('questions'): continue
    cfg={c['k']:c['v'] for c in o['config']}
    key=(cfg.get('model actually served'),cfg.get('reasoning effort'),cfg.get('lean prologue'),cfg.get('delegation-free system prompt'),cfg.get('orchestrator framing'),cfg.get('model chooses k'))
    runs[o['run']]=(key,{q['qid']:bool(q.get('correct')) for q in o['questions']})
names=list(runs)
for a,b in itertools.combinations(names,2):
    ka,A=runs[a]; kb,B=runs[b]
    if ka!=kb or ka[1]!='high': continue
    common=set(A)&set(B)
    if len(common)<40: continue
    ra=[q for q in common if A[q]]; wa=[q for q in common if not A[q]]
    brk=sum(not B[q] for q in ra); fix=sum(B[q] for q in wa)
    print(f"{a:28s} vs {b:28s} n={len(common):3d} acc {sum(A[q] for q in common)/len(common):.2f}->{sum(B[q] for q in common)/len(common):.2f}  break {brk}/{len(ra)} ({brk/max(1,len(ra)):.2f})  fix {fix}/{len(wa)} ({fix/max(1,len(wa)):.2f})")
