"""per item: asked rate, and among asked items delivered (full) / partial / neither, per arm. tests whether bv_terse withholds asked facts."""
import json, glob, collections
import os
RUNS = os.environ.get('PILOT_RUNS', 'runs')
RUN = RUNS + '/vol1/foam/run'
for arm in ('bf_neutral', 'bv_terse', 'bv_volunteer'):
    c = collections.Counter(); ex = []
    for f in glob.glob(f'{RUN}/{arm}/rep1/foam/*/case_*/persona_eval.json'):
        ev = json.load(open(f))
        for i in ev['items']:
            c['n'] += 1; c['asked'] += i['asked']
            if i['asked']:
                k = 'del' if i['delivered'] else 'part' if i['partial'] else 'none'
                c['a_' + k] += 1
                if k != 'del' and arm == 'bv_terse': ex.append((ev['case_id'], i['index'], k, i['item'][:90]))
            c['rec'] += i['recovered']
    a = c['asked']
    print(f"{arm:12s} items {c['n']}  asked {a/c['n']:.3f}  | of asked: delivered {c['a_del']/a:.3f} partial {c['a_part']/a:.3f} neither {c['a_none']/a:.3f}")
    if ex: print('  terse non-full examples:', len(ex)); [print('   ', e) for e in ex[:8]]
