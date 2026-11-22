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
    runs[o['run']]=(cfg.get('reasoning effort'),cfg.get('model actually served'),{q['qid']:bool(q.get('correct')) for q in o['questions']})
print(len(runs))
for a,b in itertools.combinations(runs,2):
    ea,ma,A=runs[a]; eb,mb,B=runs[b]
    common=set(A)&set(B)
    if len(common)<40 or ea!=eb or ma!=mb or ma!='gpt-5-2025-08-07': continue
    ra=[q for q in common if A[q]]; wa=[q for q in common if not A[q]]
    brk=sum(not B[q] for q in ra); fix=sum(B[q] for q in wa)
    print(f"{ea:6s} {a:24s} vs {b:24s} n={len(common):3d} break {brk}/{len(ra)} ({brk/max(1,len(ra)):.2f}) fix {fix}/{len(wa)} ({fix/max(1,len(wa)):.2f})")
