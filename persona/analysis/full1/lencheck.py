# Prints Spearman correlation between log mean words/reply and per-wording effect (recovered_judge), plus per-arm word counts.
import argparse
import re,glob,os,csv,statistics as st
from scipy.stats import spearmanr
ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--analysis-dir", required=True)
ap.add_argument("--dataset", default="dataset/foam/disambiguation_foam.json")
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
words={}
for d in sorted(glob.glob(os.path.join(args.run_dir,'*/'))):
    arm=os.path.basename(d.rstrip('/')); ws=[]
    for f in glob.glob(d+'rep1/foam/*/case_*/conversation_log.txt'):
        t=re.sub(r'^=== .* ===$','',open(f).read(),flags=re.M)
        for m in re.finditer(r'^A:(.*?)(?=^Q:|\Z)',t,flags=re.M|re.S):
            ws.append(len(m.group(1).split()))
    if ws: words[arm]=(st.mean(ws),st.median(ws),len(ws))
eff={}
for r in csv.DictReader(open(os.path.join(args.analysis_dir,'pole_wordings.csv'))):
    if r['metric']=='recovered_judge': eff[r['arm']]=float(r['effect'])
arms=sorted(eff); import math
x=[math.log(words[a][0]) for a in arms]; y=[eff[a] for a in arms]
print(len(arms), spearmanr(x,y))
for a in sorted(words): print(a, round(words[a][0],1), words[a][1], words[a][2], round(eff.get(a,float('nan')),3))
