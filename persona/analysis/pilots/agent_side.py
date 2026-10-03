"""mx1 score + agent side: each MatrAIx persona (and all pooled) vs the neutral family (bf_neutral, _b.._f) on the same 30 cases.
score recovered_judge, questions/chat, cap hits, delivered share: paired by case (persona − neutral-family mean), case bootstrap 2000.
stop hazard: MH ratio, strata case × facts not yet fully conveyed (cap 4) × turn bin, as runs/full1/agentside/mh.py."""
import json, glob, random, collections, statistics as st
import os
RUNS = os.environ.get('PILOT_RUNS', 'runs')
RUN = RUNS + '/mx1/foam/run'
D = collections.defaultdict(dict)
for f in glob.glob(f"{RUN}/*/rep1/foam/bedrock_glm5_Disambiguation/case_*/persona_eval.json"):
    d = f.rsplit('/', 1)[0]; arm = d.split('/')[-5]
    ev = json.load(open(f)); at = json.load(open(d + '/persona_attribution.json'))
    D[arm][ev['case_id']] = dict(n=ev['n_items'], T=ev['turns'], q=ev['questions'], cap=ev['cap_hit'], done=ev['completed_by_agent'],
                                 rj=ev['item_rates'].get('recovered_judge'),
                                 delivered=sum(i['delivered'] for i in ev['items']) / max(ev['n_items'], 1),
                                 att=[{k: v['level'] for k, v in t['conveyed'].items()} for t in at['turns']])
NEU = [a for a in D if a.startswith('bf_neutral')]; MX = sorted(a for a in D if a.startswith('mx_'))
cases = sorted(set.intersection(*[set(D[a]) for a in NEU + MX]))
print('cases', len(cases), 'neutral arms', len(NEU), 'mx arms', len(MX), {a: len(D[a]) for a in MX})
random.seed(0)
def nmean(c, m): return st.mean(D[a][c][m] for a in NEU)
def pmean(arms, c, m): return st.mean(D[a][c][m] for a in arms)
def boot(f):
    v = sorted(f([random.choice(cases) for _ in cases]) for _ in range(2000)); return v[50], v[1949]
def strata(arms):
    S = collections.defaultdict(lambda: [0, 0])
    for a in arms:
        for c in cases:
            x = D[a][c]; full = set()
            for k in range(1, x['T'] + 1):
                stop = (k == x['T'] and x['done'] and not x['cap']); forced = (k == x['T'] and x['cap'])
                if not forced:
                    key = (c, min(x['n'] - len(full), 4), min(k, 8) // 3); S[key][0] += stop; S[key][1] += 1
                if k <= len(x['att']): full |= {i for i, l in x['att'][k - 1].items() if l == 'full'}
    return S
def mh(Sa, Sb, w=None):
    num = den = 0
    for key, (a, na) in Sa.items():
        if key not in Sb: continue
        k = 1 if w is None else w.get(key[0], 0)
        if not k: continue
        b, nb = Sb[key]; N = na + nb; num += k * a * nb / N; den += k * b * na / N
    return num / den if den else float('nan')
SN = strata(NEU)
print('\narm            rj     Δrj vs neutral           q     Δq                       cap/30  delivered  MH stop ratio')
for name, arms in [('neutral fam', NEU)] + [(a.replace('mx_candidate_', 'mx_'), [a]) for a in MX] + [('ALL mx', MX)]:
    rj = st.mean(pmean(arms, c, 'rj') for c in cases); q = st.mean(pmean(arms, c, 'q') for c in cases)
    cap = st.mean(pmean(arms, c, 'cap') for c in cases) * len(cases); dl = st.mean(pmean(arms, c, 'delivered') for c in cases)
    if name == 'neutral fam':
        print(f"{name:12s} {rj:.3f}  {'':22s} {q:5.2f}  {'':22s} {cap:5.1f}  {dl:.3f}"); continue
    out = []
    for m in ('rj', 'q'):
        f = lambda cs, m=m: st.mean(pmean(arms, c, m) - nmean(c, m) for c in cs); lo, hi = boot(f)
        out.append(f"{f(cases):+.3f} [{lo:+.3f},{hi:+.3f}]")
    Sa = strata(arms); bs = []
    for _ in range(2000):
        cnt = collections.Counter(random.choice(cases) for _ in cases); r = mh(Sa, SN, cnt)
        if r == r: bs.append(r)
    bs.sort()
    print(f"{name:12s} {rj:.3f}  {out[0]:22s} {q:5.2f}  {out[1]:22s} {cap:5.1f}  {dl:.3f}      {mh(Sa, SN):.2f} [{bs[int(.025*len(bs))]:.2f},{bs[int(.975*len(bs))-1]:.2f}]")
print('\nnoise reference: each neutral arm vs the other five (Δrj, Δq)')
for a in NEU:
    rest = [b for b in NEU if b != a]
    print(f"{a:14s} Δrj {st.mean(D[a][c]['rj'] - pmean(rest, c, 'rj') for c in cases):+.3f}  Δq {st.mean(D[a][c]['q'] - pmean(rest, c, 'q') for c in cases):+.2f}")
