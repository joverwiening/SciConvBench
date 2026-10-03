"""vol1 analysis: behaviour cards bv_terse / bv_volunteer vs bf_neutral, all run the same day from one shared first agent reply.
score (recovered_judge), questions, unasked_delivered, stop hazard (MH, strata as runs/full1/agentside/mh.py), words/reply.
usage: python vol1_an.py [run_dir]  (default runs/vol1/foam/run; runs/vol1_smoke/run for the smoke)"""
import json, os, sys, glob, random, collections, statistics as st
RUNS = os.environ.get('PILOT_RUNS', 'runs')
B = RUNS
RUN = sys.argv[1] if len(sys.argv) > 1 else B + '/vol1/foam/run'
SRC = {a: RUN for a in ('bf_neutral', 'bv_terse', 'bv_volunteer')}

def load(arm):
    out = {}
    for d in glob.glob(f"{SRC[arm]}/{arm}/rep1/foam/bedrock_glm5_Disambiguation/case_*"):
        try:
            ev = json.load(open(d + '/persona_eval.json')); at = json.load(open(d + '/persona_attribution.json'))
            cl = json.load(open(d + '/clarifications.json'))
        except Exception:
            continue
        rl = [json.loads(l) for l in open(d + '/restyle_log.jsonl')] if os.path.exists(d + '/restyle_log.jsonl') else []
        out[ev['case_id']] = dict(n=ev['n_items'], T=ev['turns'], q=ev['questions'], cap=ev['cap_hit'],
                                  done=ev['completed_by_agent'], rj=ev['item_rates'].get('recovered_judge'),
                                  delivered=sum(i['delivered'] for i in ev['items']) / max(ev['n_items'], 1),
                                  unasked=sum(i['unasked_delivered'] for i in ev['items']) / max(ev['n_items'], 1),
                                  att=[{k: v['level'] for k, v in t['conveyed'].items()} for t in at['turns']],
                                  words=[len(c['answer'].split()) for c in cl],
                                  fb=sum(r.get('fallback', False) for r in rl), nrl=len(rl))
    return out

D = {a: load(a) for a in SRC}
cases = sorted(set.intersection(*(set(v) for v in D.values())))
print('cases with all three arms:', len(cases), {a: len(v) for a, v in D.items()})
random.seed(0)

def boot(f, n=2000):
    vals = sorted(f([random.choice(cases) for _ in cases]) for _ in range(n))
    return vals[int(.025 * n)], vals[int(.975 * n) - 1]

print('\n## per arm (same cases)')
for a in SRC:
    v = [D[a][c] for c in cases]
    words = [w for x in v for w in x['words']]
    print(f"{a:10s} rj {st.mean(x['rj'] for x in v):.3f}  questions {st.mean(x['q'] for x in v):.2f}  cap {sum(x['cap'] for x in v)}/{len(v)}"
          f"  delivered {st.mean(x['delivered'] for x in v):.3f}  unasked {st.mean(x['unasked'] for x in v):.3f}  words/reply {st.mean(words):.1f}")

print('\n## paired differences (case bootstrap 95% CI, 2000)')
for a, b in (('bv_terse', 'bf_neutral'), ('bv_volunteer', 'bf_neutral'), ('bv_volunteer', 'bv_terse')):
    for m in ('rj', 'q', 'delivered', 'unasked'):
        f = lambda cs, m=m: st.mean(D[a][c][m] - D[b][c][m] for c in cs)
        lo, hi = boot(f)
        print(f"{a} - {b}  {m:9s} {f(cases):+.3f} [{lo:+.3f}, {hi:+.3f}]")

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
for a, b in (('bv_terse', 'bf_neutral'), ('bv_volunteer', 'bf_neutral'), ('bv_volunteer', 'bv_terse')):
    Sa, Sb = strata(a), strata(b)
    bs = []
    for _ in range(2000):
        cnt = collections.Counter(random.choice(cases) for _ in cases)
        r = mh(Sa, Sb, cnt)
        if r == r: bs.append(r)
    bs.sort()
    print(f"{a} vs {b}: {mh(Sa, Sb):.2f} [{bs[int(.025*len(bs))]:.2f}, {bs[int(.975*len(bs))-1]:.2f}]")
