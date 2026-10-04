"""tone_live_chats.csv rows for a restyle run; --check compares against the existing tone2 rows."""
import csv, json, os, sys

RUNS = os.environ.get('PILOT_RUNS', 'runs')
COLS = ['arm', 'rep', 'case_id', 'n_deleted', 'recovered_judge', 'questions', 'turns', 'cap_hit', 'completed_by_agent',
        'asked', 'delivered', 'partial', 'volunteered', 'words_per_reply', 'date', 'restyle_fallbacks', 'restyled_replies']


def rows(run, arms, date):
    for arm in arms:
        base = f'{run}/{arm}/rep1/foam/bedrock_glm5_Disambiguation'
        for case in sorted(os.listdir(base)):
            d = f'{base}/{case}'
            if not os.path.exists(f'{d}/persona_eval.json'):
                continue
            e = json.load(open(f'{d}/persona_eval.json'))
            r = e['item_rates']
            cl = json.load(open(f'{d}/clarifications.json'))
            wpr = round(sum(len(c['answer'].split()) for c in cl) / len(cl), 1) if cl else ''
            log = [json.loads(l) for l in open(f'{d}/restyle_log.jsonl')] if os.path.exists(f'{d}/restyle_log.jsonl') else []
            yield dict(arm=arm, rep=1, case_id=case, n_deleted=e['n_items'], recovered_judge=r['recovered_judge'],
                       questions=e['questions'], turns=e['turns'], cap_hit=e['cap_hit'],
                       completed_by_agent=e['completed_by_agent'], asked=r['asked'], delivered=r['delivered'],
                       partial=r['partial'], volunteered=r['unasked_delivered'], words_per_reply=wpr, date=date,
                       restyle_fallbacks=sum(bool(x.get('fallback')) for x in log) if log else '',
                       restyled_replies=len(log) if log else '')


if __name__ == '__main__':
    if sys.argv[1] == '--check':
        old = {(r['arm'], r['case_id']): r for r in csv.DictReader(open(sys.argv[2])) if r['arm'].startswith('rs_')}
        bad = 0
        for r in rows(f'{RUNS}/tone2/foam/run', ['rs_para', 'rs_lowA'], '2026-09-30'):
            o = old[(r['arm'], r['case_id'])]
            diff = [k for k in COLS if str(r[k]) != o[k] and not (o[k] and str(r[k]) and float(o[k]) == float(r[k])
                                                               if k not in ('arm', 'case_id', 'date', 'cap_hit', 'completed_by_agent') else False)]
            if diff:
                bad += 1
                if bad < 4:
                    print(r['arm'], r['case_id'], {k: (o[k], r[k]) for k in diff})
        print('mismatching rows:', bad, 'of', len(old))
    else:  # append: build_csv.py <csv> <run> <date> <arm,arm>
        path, run, date, arms = sys.argv[1:5]
        with open(path, 'a', newline='') as f:
            w = csv.DictWriter(f, COLS, lineterminator='\n')
            n = 0
            for r in rows(run, arms.split(','), date):
                w.writerow(r); n += 1
        print('appended', n)
