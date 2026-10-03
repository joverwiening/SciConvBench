"""mx1 first order: does a MatrAIx persona change how the simulated scientist talks?
per arm: words/reply, replies with non-Latin script, replies in Spanish/Swahili (stopword heuristic);
then 11-way decoding (10 mx personas + bf_neutral) from sim replies / agent questions / final spec,
bag of words, case mean removed, 5-fold grouped by case. chance 1/11."""
import json, glob, re, collections, statistics as st, numpy as np, sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
import os
RUNS = os.environ.get('PILOT_RUNS', 'runs')
RUN = RUNS + '/mx1/foam/run'
cases = [l.strip() for l in open(RUN + '/../../cases_mx1.txt') if l.strip()]
arms = ['bf_neutral'] + sorted(a.split('/')[-1] for a in glob.glob(RUN + '/mx_candidate_*'))
T = {}
for a in arms:
    for c in cases:
        d = f"{RUN}/{a}/rep1/foam/bedrock_glm5_Disambiguation/{c if c.startswith('case_') else 'case_' + c}"
        try:
            cl = json.load(open(d + '/clarifications.json')); spec = open(d + '/generated_prompt.txt').read()
        except FileNotFoundError:
            continue
        T[(a, c)] = dict(A=[x['answer'] for x in cl], Q=[x['question'] for x in cl], spec=spec)
nonlatin = re.compile(r'[؀-ۿ぀-ヿ一-鿿ऀ-ॿЀ-ӿ]')
es = re.compile(r'\b(el|la|los|las|que|para|con|una|por|del)\b', re.I)
sw = re.compile(r'\b(na|ya|kwa|ni|wa|katika|hii|cha)\b', re.I)
print('arm                chats  words/reply  nonLatin  es-like  sw-like  replies')
for a in arms:
    A = [x for (aa, c), v in T.items() if aa == a for x in v['A']]
    n = len(A) or 1
    print(f"{a:18s} {sum(1 for k in T if k[0]==a):5d}  {st.mean(len(x.split()) for x in A) if A else 0:11.1f}  "
          f"{sum(bool(nonlatin.search(x)) for x in A)/n:8.2f}  {sum(len(es.findall(x))>=3 for x in A)/n:7.2f}  {sum(len(sw.findall(x))>=3 for x in A)/n:7.2f}  {len(A)}")
keys = [k for k in T if T[k]['A']]
y = np.array([a for a, c in keys]); g = np.array([c for a, c in keys])
for stg in ('sim', 'agentQ', 'spec'):
    docs = [' '.join(T[k]['A']) if stg == 'sim' else ' '.join(T[k]['Q']) if stg == 'agentQ' else T[k]['spec'] for k in keys]
    X = CountVectorizer(token_pattern=r"[^\W\d_]+|\d+(?:\.\d+)?", lowercase=True, min_df=5).fit_transform(docs).toarray().astype(float)
    X = np.log1p(X); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    for c in set(g): X[g == c] -= X[g == c].mean(0)
    pred = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(5).split(X, y, g):
        pred[te] = LogisticRegression(C=1.0, max_iter=3000).fit(X[tr], y[tr]).predict(X[te])
    acc = np.mean(pred == y)
    rec = {a: np.mean(pred[y == a] == a) for a in arms}
    print(f"\n{stg}: held-out-case acc {acc:.3f} (chance {1/len(arms):.3f}, n={len(y)})")
    print('  recall ' + ' '.join(f"{a.replace('mx_candidate_','')}:{r:.2f}" for a, r in rec.items()))
