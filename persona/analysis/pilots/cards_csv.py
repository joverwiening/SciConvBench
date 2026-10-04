"""One row per behaviour-card chat (vol1): 62 cases x 6 rounds, the 14 harder extra cases x 6 rounds, the 20-turn run.
usage: python cards_csv.py OUT.csv"""
import csv, glob, json, os, sys
import numpy as np

B = '/n/holylabs/kempner_sompolinsky_lab/Everyone/jorin/benchmarks/runs/vol1/foam'
# (run dir, turn cap, case set, date per rep); dates from the launch logs
SOURCES = [('run', 10, 'main62', {1: '2026-10-01', 2: '2026-10-02', 3: '2026-10-04', 4: '2026-10-04', 5: '2026-10-04', 6: '2026-10-04'}),
           ('cap20', 20, 'main62', {1: '2026-10-02'}),
           ('hard', 10, 'hard_extra14', {r: '2026-10-04' for r in range(1, 7)})]
COLS = ['turn_cap', 'case_set', 'arm', 'rep', 'case_id', 'n_deleted', 'recovered_judge', 'questions', 'turns', 'cap_hit',
        'completed_by_agent', 'asked', 'delivered', 'partial', 'volunteered', 'words_per_reply', 'date']

rows = []
for sub, cap, cset, dates in SOURCES:
    for line in open(f'{B}/{sub}/results.jsonl'):
        r = json.loads(line)
        d = glob.glob(f"{B}/{sub}/{r['arm']}/rep{r['rep']}/foam/*/{r['case_id']}")[0]
        cl = json.load(open(os.path.join(d, 'clarifications.json')))
        ir = r['item_rates']
        rows.append(dict(turn_cap=cap, case_set=cset, arm=r['arm'], rep=r['rep'], case_id=r['case_id'], n_deleted=r['n_items'],
                         recovered_judge=ir.get('recovered_judge'), questions=r['questions'], turns=r['turns'],
                         cap_hit=r['cap_hit'], completed_by_agent=r['completed_by_agent'], asked=ir['asked'],
                         delivered=ir['delivered'], partial=ir['partial'], volunteered=ir['unasked_delivered'],
                         words_per_reply=round(float(np.mean([len(c['answer'].split()) for c in cl])), 1) if cl else '',
                         date=dates[r['rep']]))
rows.sort(key=lambda x: (x['turn_cap'], x['case_set'], x['rep'], x['arm'], x['case_id']))
with open(sys.argv[1], 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=COLS, lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
print(len(rows), 'rows')
