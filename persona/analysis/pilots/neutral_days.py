"""day/batch check: neutral scientist score on the same cases across runs (full1 09-28, tone2 rs_para 09-30, tone3 rs_para 10-03, vol1 rep1 10-01, rep2 10-02, reps 3-6 10-04, mx1 personas 09-30)."""
import json, glob, random, statistics as st
B = '/n/holylabs/kempner_sompolinsky_lab/Everyone/jorin/benchmarks/runs'
def load(run, arm, rep=1):
    out = {}
    for f in glob.glob(f'{B}/{run}/{arm}/rep{rep}/foam/bedrock_glm5_Disambiguation/case_*/persona_eval.json'):
        ev = json.load(open(f)); r = ev['item_rates'].get('recovered_judge')
        if r is not None: out[ev['case_id']] = r
    return out
full1_arms = sorted(a.split('/')[-1] for a in glob.glob(f'{B}/full1/foam/run/bf_neutral*'))
S = {'full1 ' + a: load('full1/foam/run', a) for a in full1_arms}
S['tone2 rs_para 09-30'] = load('tone2/foam/run', 'rs_para')
S['vol1 neutral r1 10-01'] = load('vol1/foam/run', 'bf_neutral', 1)
S['vol1 neutral r2 10-02'] = load('vol1/foam/run', 'bf_neutral', 2)
S['tone3 rs_para 10-03'] = load('tone3/foam/run', 'rs_para')
for _r in (3, 4, 5, 6):
    S[f'vol1 neutral r{_r} 10-04'] = load('vol1/foam/run', 'bf_neutral', _r)
_r36 = [S[f'vol1 neutral r{_r} 10-04'] for _r in (3, 4, 5, 6)]
S['vol1 neutral r3-6 mean 10-04'] = {c: st.mean(d[c] for d in _r36) for c in set.intersection(*map(set, _r36))}
mx = [a.split('/')[-1] for a in glob.glob(f'{B}/mx1/foam/run/mx_*')]
mxd = [load('mx1/foam/run', a) for a in mx]
mx30 = [l.strip() for l in open(f'{B}/mx1/cases_mx1.txt') if l.strip()]
S['mx1 personas pooled 09-30'] = {c: st.mean(d[c] for d in mxd if c in d) for c in mx30 if any(c in d for d in mxd)}
fam = {c: st.mean(S['full1 ' + a][c] for a in full1_arms if c in S['full1 ' + a]) for c in mx30}
print('full1 neutral arms:', full1_arms, ' mx personas:', len(mx))
for name, cs in (('30 mx1 cases', mx30),):
    print(f'\n## {name}; full1 neutral-family mean {st.mean(fam[c] for c in cs):.3f}')
    random.seed(0)
    for k, d in S.items():
        cc = [c for c in cs if c in d]
        diffs = [d[c] - fam[c] for c in cc]
        bs = sorted(st.mean(random.choice(diffs) for _ in diffs) for _ in range(2000))
        print(f'{k:28s} n={len(cc):2d} rj {st.mean(d[c] for c in cc):.3f}  vs full1 family {st.mean(diffs):+.3f} [{bs[50]:+.3f}, {bs[1949]:+.3f}]')
