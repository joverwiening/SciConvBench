"""Remark labels from tone3_judge.jsonl, dropping quotes that are only interjections (Ugh, *sigh*, There, Fine ...)."""
import json, re, collections, sys
INTERJ = r"\*?\s*(sigh|scoffs?|groan|ugh|there|done|fine|whatever|take it|write it down|here|great|just great|wonderful|that works|okay|ok)\s*\.?\s*\*?"
def substantive(q):
    t = re.sub(r"[\.\!…,]+", " ", q.lower())
    t = re.sub(INTERJ, " ", t)
    return len(re.sub(r"[\s\*]", "", t)) > 0
n, k, chats = collections.Counter(), collections.Counter(), collections.defaultdict(set)
allc = collections.defaultdict(set); ex = collections.defaultdict(list)
for l in open(sys.argv[1] if len(sys.argv) > 1 else 'tone3_judge.jsonl'):
    r = json.loads(l)
    if r.get('kind') != 'remark' or not r.get('labels'): continue
    a = r['arm']; allc[a].add(r['case'])
    for x in r['labels']:
        n[a] += 1
        if x['remark'] and substantive(x['quote']):
            k[a] += 1; chats[a].add(r['case']); ex[a].append(x['quote'])
for a in sorted(n):
    print(f"{a}: replies with a non-interjection remark {k[a]}/{n[a]} = {k[a]/n[a]:.3f}; chats >=1 {len(chats[a])}/{len(allc[a])}")
    for e in ex[a][:12]: print('   ', e[:120])
