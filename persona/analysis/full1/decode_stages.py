# Produces decode_stages.json (9-trait bag-of-words decoding of scientist replies, agent questions, final spec).
import argparse, os
import json,numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--analysis-dir", required=True)
ap.add_argument("--dataset", default="dataset/foam/disambiguation_foam.json")
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
R=os.path.join(args.run_dir,'results.jsonl')
rows=[json.loads(l) for l in open(R)]
poles=['high_A','low_A','high_C','low_C','high_E','low_E','high_N','high_O','low_O']
arms=[]
for p in poles:
    b='bf_'+p; arms+= [b,b+'_w2',b+'_w3']
neut=['bf_neutral']+['bf_neutral_'+s for s in 'bcdef']
ARMS=arms+neut
idx={}
for r in rows:
    idx[(r['arm'],r['case_id'])]=r
import re,collections
base=args.run_dir
F={}
for (a,c),r in idx.items():
    if a not in ARMS+['mute','oracle']: continue
    d=f'{base}/{a}/rep1/foam/bedrock_glm5_Disambiguation/{c}'
    try:
        log=open(d+'/conversation_log.txt').read(); spec=open(d+'/generated_prompt.txt').read()
    except Exception as e: continue
    A=[l[3:] for l in log.split('\n') if l.startswith('A: ')]; Q=[l[3:] for l in log.split('\n') if l.startswith('Q: ')]
    F[(a,c)]=dict(spec=spec, simw=np.mean([len(x.split()) for x in A]) if A else 0, qw=np.mean([len(x.split()) for x in Q]) if Q else 0,
      nq=r['questions'], turns=r['turns'], cap=float(r['cap_hit']), speclen=len(spec.split()),
      **{k:r['item_rates'][k] for k in ['recovered','delivered','partial','unasked_delivered','asked','silent']})
T={}
for (a,c) in F:
    if a not in ARMS: continue
    log=open(f'{base}/{a}/rep1/foam/bedrock_glm5_Disambiguation/{c}/conversation_log.txt').read()
    # split into Q / A blocks (multi-line)
    q=[];an=[];cur=None
    for l in log.split('\n'):
        if l.startswith('Q: '): cur=q; q.append(l[3:])
        elif l.startswith('A: '): cur=an; an.append(l[3:])
        elif l.startswith('==='): cur=None
        elif cur is not None and cur: cur[-1]+=' '+l
    T[(a,c)]={'sim':' '.join(an),'agentQ':' '.join(q),'spec':F[(a,c)]['spec'],'q1':q[0] if q else ''}
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
lab={a:(a.replace('_w2','').replace('_w3','') if not a.startswith('bf_neutral') else 'neutral') for a in ARMS}
wd={a:('w2' if a.endswith('_w2') else 'w3' if a.endswith('_w3') else 'w1') for a in ARMS}
keys=[k for k in T if lab[k[0]]!='neutral']
y=np.array([lab[a] for a,c in keys]); w=np.array([wd[a] for a,c in keys]); cs=np.array([c for a,c in keys])
for st in ['sim','agentQ','spec']:
    cv=CountVectorizer(token_pattern=r"[a-z]+|\d+(?:\.\d+)?",min_df=5,sublinear_tf=False) if False else CountVectorizer(token_pattern=r"[A-Za-z]+|\d+(?:\.\d+)?",lowercase=True,min_df=5)
    X=cv.fit_transform([T[k][st] for k in keys]).toarray().astype(float); X=np.log1p(X); X/=np.linalg.norm(X,axis=1,keepdims=True)+1e-12
    # remove case mean (origin convention)
    for c in set(cs): X[cs==c]-=X[cs==c].mean(0)
    accs=[];C=np.zeros((9,9))
    OUT=globals().setdefault('OUT',{})
    classes=sorted(set(y))
    for hold in ['w1','w2','w3']:
        tr=(w!=hold); te=(w==hold)
        # also disjoint cases: split cases in half
        uc=sorted(set(cs)); half=set(uc[::2])
        for A,B in [(half,set(uc)-half),(set(uc)-half,half)]:
            trm=tr&np.isin(cs,list(A)); tem=te&np.isin(cs,list(B))
            m=LogisticRegression(C=1.0,max_iter=2000).fit(X[trm],y[trm]); p=m.predict(X[tem])
            accs.append(np.mean(p==y[tem])); C+=confusion_matrix(y[tem],p,labels=classes)
    P=C/C.sum(); px=P.sum(1,keepdims=True); py=P.sum(0,keepdims=True); MI=np.nansum(P*np.log2(P/(px*py)+1e-300))
    OUT[st]={'acc':float(np.mean(accs)),'mi_bits':float(MI),'recall':{cl:float(C[i,i]/C[i].sum()) for i,cl in enumerate(classes)}}
    print(st,'held-out-wording x held-out-case acc %.3f (chance .111) MI %.2f bits -> 2^I=%.2f classes'%(np.mean(accs),MI,2**MI))
    print('  per-pole recall',' '.join('%s:%.2f'%(cl.replace('bf_',''),C[i,i]/C[i].sum()) for i,cl in enumerate(classes)))
import json; json.dump(OUT, open(os.path.join(args.out,'decode_stages.json'),'w'), indent=1)
