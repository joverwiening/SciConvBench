"""vol1 reps 1-6 + hard extras, as fixed in runs/vol1/PREREG_rep36.md.
usage: python vol1_an3.py [min_reps]  (min_reps default 6; lower only for testing on partial data)"""
import json, sys, collections
import numpy as np

B = '/n/holylabs/kempner_sompolinsky_lab/Everyone/jorin/benchmarks/runs'
MIN_REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
ARMS = ('bf_neutral', 'bv_terse', 'bv_volunteer')
SHORT = {'bf_neutral': 'neutral', 'bv_terse': 'terse', 'bv_volunteer': 'volunteer'}
CONTRASTS = [('bv_volunteer', 'bf_neutral'), ('bv_terse', 'bf_neutral'), ('bv_volunteer', 'bv_terse')]
METRICS = ('rj', 'delivered', 'questions')
rng = np.random.default_rng(20261004)
NB = 4000


def load(path):
    d = collections.defaultdict(dict)  # (arm, case) -> rep -> metrics
    for line in open(path):
        r = json.loads(line)
        rj = r['item_rates'].get('recovered_judge')
        if r['arm'] not in ARMS or rj is None:
            continue
        d[(r['arm'], r['case_id'])][r['rep']] = dict(rj=rj, delivered=r['item_rates']['delivered'],
                                                     questions=r['questions'], cap=r['cap_hit'])
    return d


main = load(f'{B}/vol1/foam/run/results.jsonl')
hard = load(f'{B}/vol1/foam/hard/results.jsonl')
full62 = open(f'{B}/full1/cases_foam_full1.txt').read().split()
extra14 = open(f'{B}/vol1/cases_hard_extra.txt').read().split()

# hard rule (prereg): full1 neutral-family mean <= 0.667
fam = collections.defaultdict(list)
for line in open(f'{B}/full1/foam/run/results.jsonl'):
    r = json.loads(line)
    v = r['item_rates'].get('recovered_judge')
    if r['arm'].startswith('bf_neutral') and v is not None:
        fam[r['case_id']].append(v)
hard32 = [c for c in full62 if c in fam and np.mean(fam[c]) <= 0.667]


def reps_common(d, c):
    s = set.intersection(*(set(d.get((a, c), {})) for a in ARMS))
    return sorted(s)


def table(d, cases, reps=None):
    """per case: list over common reps of paired arm values -> dict metric -> arm -> array[case][rep]"""
    keep, out = [], {m: {a: [] for a in ARMS} for m in METRICS}
    for c in cases:
        rs = reps_common(d, c)
        if reps is not None:
            rs = [r for r in rs if r in reps]
        if len(rs) < (MIN_REPS if reps is None else len(reps)):
            continue
        keep.append(c)
        for m in METRICS:
            for a in ARMS:
                out[m][a].append([d[(a, c)][r][m] for r in rs])
    return keep, out


def summarise(diff_per_case_rep):
    """diff_per_case_rep: list of arrays (reps) per case. case bootstrap of case means + fixed-case (reps within case)."""
    cm = np.array([np.mean(x) for x in diff_per_case_rep])
    n = len(cm)
    boot = np.array([cm[rng.integers(0, n, n)].mean() for _ in range(NB)])
    fixed = np.array([np.mean([x[rng.integers(0, len(x), len(x))].mean() for x in map(np.asarray, diff_per_case_rep)])
                      for _ in range(NB)])
    z = cm.mean() / (cm.std(ddof=1) / np.sqrt(n))
    return cm.mean(), np.percentile(boot, [2.5, 97.5]), np.percentile(fixed, [2.5, 97.5]), z, n


def report(title, d, cases, reps=None):
    keep, t = table(d, cases, reps)
    print(f'\n## {title}: {len(keep)} cases' + (f', reps {reps}' if reps else ''))
    for m in METRICS:
        means = {SHORT[a]: np.mean([np.mean(x) for x in t[m][a]]) for a in ARMS}
        print(f'{m}: ' + ', '.join(f'{k} {v:.3f}' for k, v in means.items()))
        for a, b in CONTRASTS:
            diffs = [np.array(x) - np.array(y) for x, y in zip(t[m][a], t[m][b])]
            mu, ci, fx, z, n = summarise(diffs)
            print(f'  {SHORT[a]} - {SHORT[b]}: {mu:+.3f} case CI [{ci[0]:+.3f}, {ci[1]:+.3f}] '
                  f'fixed-case CI [{fx[0]:+.3f}, {fx[1]:+.3f}] z {z:.2f}')
    return keep


both = dict(main)
both.update(hard)
print(f'hard rule: {len(hard32)} of 62 full1 cases + {len(extra14)} extras')
k62 = report('PRIMARY all 62', main, full62)
report('SECONDARY hard subset (32 + 14)', both, hard32 + extra14)
report('hard 32 of the 62 only', main, hard32)
report('14 extras only', hard, extra14)
report('easy 30 of the 62', main, [c for c in full62 if c not in hard32])

print('\n## per rep, all 62 (rj)')
for r in range(1, 7):
    keep, t = table(main, full62, reps=[r])
    if not keep:
        continue
    s = []
    for a, b in CONTRASTS:
        diffs = [np.array(x) - np.array(y) for x, y in zip(t['rj'][a], t['rj'][b])]
        s.append(f'{SHORT[a]}-{SHORT[b]} {np.mean([x.mean() for x in diffs]):+.3f}')
    arms = ' '.join(f"{SHORT[a]} {np.mean([np.mean(x) for x in t['rj'][a]]):.3f}" for a in ARMS)
    print(f'rep{r} n={len(keep)}: {arms} | ' + ', '.join(s))

print('\n## day check (62): reps 3-6 minus reps 1-2, per arm and per contrast (rj)')
_, t12 = table(main, full62, reps=[1, 2])
_, t36 = table(main, full62, reps=[3, 4, 5, 6])
for a in ARMS:
    dd = [np.mean(x) - np.mean(y) for x, y in zip(t36['rj'][a], t12['rj'][a])]
    mu, ci, _, z, n = summarise([np.array([v]) for v in dd])
    print(f'{SHORT[a]}: {mu:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]')
for a, b in CONTRASTS:
    dd = [(np.mean(x1) - np.mean(y1)) - (np.mean(x0) - np.mean(y0))
          for x1, y1, x0, y0 in zip(t36['rj'][a], t36['rj'][b], t12['rj'][a], t12['rj'][b])]
    mu, ci, _, z, n = summarise([np.array([v]) for v in dd])
    print(f'{SHORT[a]}-{SHORT[b]} effect, 3-6 minus 1-2: {mu:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]')

caps = {SHORT[a]: sum(main[(a, c)][r]['cap'] for c in k62 for r in main[(a, c)]) for a in ARMS}
print('\ncap hits over all reps, 62 cases:', caps)
