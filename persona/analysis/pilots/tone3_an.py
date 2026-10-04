"""tone3 analysis: live restyle arms rs_para / rs_lowA / rs_lowAc (tone3) plus rs_para / rs_lowA from tone2 (round 1, same 62 cases).
score (recovered_judge), questions, stop hazard (MH, strata as runs/full1/agentside/mh.py), words/reply, fallbacks, lowAc remark retries."""
import json, os, sys, glob, random, collections, statistics as st
B = os.environ.get('PILOT_RUNS', 'runs')
SRC = {'para3': (B + '/tone3/foam/run', 'rs_para'), 'lowA3': (B + '/tone3/foam/run', 'rs_lowA'),
       'lowAc3': (B + '/tone3/foam/run', 'rs_lowAc'),
       'para2': (B + '/tone2/foam/run', 'rs_para'), 'lowA2': (B + '/tone2/foam/run', 'rs_lowA')}

def load(label):
    run, arm = SRC[label]
    out = {}
    for d in glob.glob(f"{run}/{arm}/rep1/foam/bedrock_glm5_Disambiguation/case_*"):
        try:
            ev = json.load(open(d + '/persona_eval.json')); at = json.load(open(d + '/persona_attribution.json'))
            cl = json.load(open(d + '/clarifications.json'))
        except Exception:
            continue
        rl = [json.loads(l) for l in open(d + '/restyle_log.jsonl')] if os.path.exists(d + '/restyle_log.jsonl') else []
        out[ev['case_id']] = dict(n=ev['n_items'], T=ev['turns'], q=ev['questions'], cap=ev['cap_hit'],
                                  done=ev['completed_by_agent'], rj=ev['item_rates'].get('recovered_judge'),
                                  delivered=sum(i['delivered'] for i in ev['items']) / max(ev['n_items'], 1),
                                  att=[{k: v['level'] for k, v in t['conveyed'].items()} for t in at['turns']],
                                  words=[len(c['answer'].split()) for c in cl],
                                  fb=sum(r.get('fallback', False) for r in rl), nrl=len(rl),
                                  retry=sum(any(h for h in (r.get('remark_hits') or [])) for r in rl))
    return out

D = {a: load(a) for a in SRC}
cases = sorted(c for c in set.intersection(*(set(v) for v in D.values()))
               if all(D[a][c]['rj'] is not None for a in SRC))
print('cases with all five labels (rj not None):', len(cases), {a: len(v) for a, v in D.items()})
if not cases:
    print('no common graded cases yet'); sys.exit(0)
random.seed(0)

def boot(f, n=2000):
    vals = sorted(f([random.choice(cases) for _ in cases]) for _ in range(n))
    return vals[int(.025 * n)], vals[int(.975 * n) - 1]

print('\n## per label (same cases)')
for a in SRC:
    v = [D[a][c] for c in cases]
    words = [w for x in v for w in x['words']]
    nrl = sum(x['nrl'] for x in v)
    print(f"{a:7s} rj {st.mean(x['rj'] for x in v):.3f}  questions {st.mean(x['q'] for x in v):.2f}  cap {sum(x['cap'] for x in v)}/{len(v)}"
          f"  delivered {st.mean(x['delivered'] for x in v):.3f}  words/reply {st.mean(words) if words else float('nan'):.1f}"
          f"  fallback {sum(x['fb'] for x in v)}/{nrl}")
    if a == 'lowAc3' and nrl:
        print(f"{'':7s} remark retry {sum(x['retry'] for x in v)}/{nrl} = {sum(x['retry'] for x in v) / nrl:.3f}"
              f"  fallback share {sum(x['fb'] for x in v) / nrl:.3f}")

print('\n## paired differences (case bootstrap 95% CI, 2000)')
PAIRS = (('lowA3', 'para3'), ('lowAc3', 'para3'), ('lowAc3', 'lowA3'), ('lowA2', 'para2'), ('para3', 'para2'), ('lowA3', 'lowA2'))
for a, b in PAIRS:
    for m in ('rj', 'q', 'delivered'):
        f = lambda cs, a=a, b=b, m=m: st.mean(D[a][c][m] - D[b][c][m] for c in cs)
        lo, hi = boot(f)
        print(f"{a} - {b}  {m:9s} {f(cases):+.3f} [{lo:+.3f}, {hi:+.3f}]")
for m in ('rj', 'q', 'delivered'):
    f = lambda cs, m=m: st.mean(((D['lowA2'][c][m] - D['para2'][c][m]) + (D['lowA3'][c][m] - D['para3'][c][m])) / 2 for c in cs)
    lo, hi = boot(f)
    print(f"pooled lowA - para  {m:9s} {f(cases):+.3f} [{lo:+.3f}, {hi:+.3f}]")
    f = lambda cs, m=m: st.mean((D['lowA3'][c][m] - D['para3'][c][m]) - (D['lowA2'][c][m] - D['para2'][c][m]) for c in cs)
    lo, hi = boot(f)
    print(f"round 2 minus round 1 of (lowA - para)  {m:9s} {f(cases):+.3f} [{lo:+.3f}, {hi:+.3f}]")

def strata(arm):
    S = collections.defaultdict(lambda: [0, 0])
    for c in cases:
        x = D[arm][c]; full = set()
        for k in range(1, x['T'] + 1):
            stop = (k == x['T'] and x['done'] and not x['cap']); forced = (k == x['T'] and x['cap'])
            if not forced:
                key = (c, min(x['n'] - len(full), 4), min(k, 8) // 3)
                S[key][0] += stop; S[key][1] += 1
            if k <= len(x['att']):
                full |= {i for i, l in x['att'][k - 1].items() if l == 'full'}
    return S

def mh(Sa, Sb, w=None):
    num = den = 0
    for key, (a, na) in Sa.items():
        if key not in Sb: continue
        k = (w or {}).get(key[0], 1) if w is not None else 1
        if not k: continue
        b, nb = Sb[key]; N = na + nb
        num += k * a * nb / N; den += k * b * na / N
    return num / den if den else float('nan')

print('\n## stop hazard, Mantel-Haenszel ratio (strata: case x facts not yet fully conveyed x turn bin), case bootstrap 2000')
def merged(*arms):
    S = {}
    for i, arm in enumerate(arms):
        S.update({(k[0], i) + k[1:]: v for k, v in strata(arm).items()})
    return S

for a, b in (('lowA3', 'para3'), ('lowAc3', 'para3'), ('lowAc3', 'lowA3'), ('lowA2', 'para2'), ('lowA2+lowA3', 'para2+para3')):
    if '+' in a:
        Sa, Sb = merged(*a.split('+')), merged(*b.split('+'))
    else:
        Sa, Sb = strata(a), strata(b)
    bs = []
    for _ in range(2000):
        cnt = collections.Counter(random.choice(cases) for _ in cases)
        r = mh(Sa, Sb, cnt)
        if r == r: bs.append(r)
    bs.sort()
    ci = f"[{bs[int(.025*len(bs))]:.2f}, {bs[int(.975*len(bs))-1]:.2f}]" if bs else '[n/a]'
    print(f"{a} vs {b}: {mh(Sa, Sb):.2f} {ci}")
