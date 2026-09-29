# Produces pole_mechanism.json and plots/mechanism.{png,pdf}.
import argparse
import json, os, re, math, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--analysis-dir", required=True)
ap.add_argument("--dataset", default="dataset/foam/disambiguation_foam.json")
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure_style.py")).read())
R = args.run_dir; OUT = args.out
NAME = {"bf_high_N": "anxious", "bf_high_E": "outgoing", "bf_high_O": "curious", "bf_high_C": "organised",
        "bf_high_A": "agreeable", "bf_low_C": "disorganised", "bf_low_O": "conventional", "bf_low_E": "reserved",
        "bf_low_A": "disagreeable", "neutral": "neutral", "mute": "never answers"}
COUNTS = {"bf_high_N", "bf_high_E", "bf_high_O"}
pole = lambda a: "neutral" if a.startswith("bf_neutral") else re.sub(r"_w[23]$", "", a)
rows = [json.loads(l) for l in open(f"{R}/results.jsonl")]
def chat_rate(r, k):
    v = [it[k] for it in r["items"] if it[k] is not None]
    return np.mean(v) if v else np.nan
def words(r):
    f = f"{R}/{r['arm']}/rep1/foam/bedrock_glm5_Disambiguation/{r['case_id']}/clarifications.json"
    if not os.path.exists(f): return np.nan
    w = [len(x["answer"].split()) for x in json.load(open(f)) if x.get("answer")]
    return np.mean(w) if w else np.nan
arm = {}
for r in rows:
    if r["arm"] == "oracle": continue
    arm.setdefault(r["arm"], []).append((chat_rate(r, "grounded"), chat_rate(r, "silent"), chat_rate(r, "recovered"),
                                         words(r), r["questions"]))
A = {a: np.nanmean(np.array(v, float), axis=0) for a, v in arm.items()}
print("check bf_neutral grounded/silent/recovered", A["bf_neutral"][:3].round(3))
P = {}
for a, v in A.items(): P.setdefault(pole(a), []).append(v)
P = {p: np.mean(v, axis=0) for p, v in P.items()}
json.dump({p: dict(zip(["grounded", "silent", "recovered", "words", "questions"], map(float, v))) for p, v in P.items()},
          open(f"{OUT}/pole_mechanism.json", "w"), indent=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize(7, 0.45), constrained_layout=True)
order = sorted(P, key=lambda p: P[p][2])
for i, p in enumerate(order):
    g, s = 100 * P[p][0], 100 * P[p][1]
    c = COLORS["primary"] if p in COUNTS else COLORS["gray"]
    ax1.barh(i, g, color=c, height=0.7)
    ax1.barh(i, s, left=g, color=c, alpha=0.35, height=0.7)
nb = 100 * P["neutral"][2]
ax1.axvline(nb, color=COLORS["gray"], ls="--", lw=0.8)
ax1.set_yticks(range(len(order)), [NAME[p] for p in order])
ax1.set_xlabel("missing facts recovered (%)"); ax1.set_xlim(0, 100)
from matplotlib.patches import Patch
ax1.legend(handles=[Patch(color=COLORS["gray"], label="scientist gave it"),
                    Patch(color=COLORS["gray"], alpha=0.35, label="agent got it anyway")],
           frameon=False, loc="lower right", fontsize=8)

nm = np.mean([A[a][2] for a in A if pole(a) == "neutral"])
pts = [(a, math.log(A[a][3]), 100 * (A[a][2] - nm)) for a in A if a != "mute"]
x = np.array([p[1] for p in pts]); y = np.array([p[2] for p in pts])
fitix = [i for i, t in enumerate(pts) if pole(t[0]) not in COUNTS]
b = np.polyfit(x[fitix], y[fitix], 1)
xx = np.linspace(x.min() - 0.1, x.max() + 0.1, 50)
fitix2 = [i for i in fitix if pole(pts[i][0]) != "bf_low_A"]
b2 = np.polyfit(x[fitix2], y[fitix2], 1)
ax2.fill_between(np.exp(xx), np.polyval(b, xx), np.polyval(b2, xx), color=COLORS["gray"], alpha=0.2, lw=0)
col = lambda p: COLORS["primary"] if p in COUNTS else COLORS["secondary"] if p == "bf_low_A" else COLORS["gray"]
for a, lx, dy in pts:
    ax2.plot(math.exp(lx), dy, "o", color=col(pole(a)), alpha=0.35, ms=3, mew=0)
OFF = {"bf_high_E": (-7, 4, "right"), "bf_high_N": (0, -12, "center"), "bf_high_O": (-7, -4, "right"),
       "bf_high_C": (0, 6, "center"), "bf_high_A": (6, -3, "left"), "bf_low_C": (-6, 3, "right"),
       "neutral": (6, -8, "left"), "bf_low_O": (-6, -2, "right"), "bf_low_E": (0, -11, "center"),
       "bf_low_A": (6, -3, "left")}
for p in P:
    if p == "mute": continue
    ix = [i for i, t in enumerate(pts) if pole(t[0]) == p]
    mx, my = np.exp(x[ix].mean()), y[ix].mean()
    ax2.plot(mx, my, "o", color=col(p), mfc="white" if p == "neutral" else col(p), ms=6, zorder=3)
    dx, dy, ha = OFF[p]
    ax2.annotate(NAME[p], (mx, my), textcoords="offset points", xytext=(dx, dy), ha=ha, va="center", fontsize=8, color=col(p))
ax2.set_xscale("log"); ax2.set_xticks([7, 10, 20, 40, 80], ["7", "10", "20", "40", "80"])
ax2.set_xlabel("words per reply"); ax2.set_ylabel("Δ facts recovered vs neutral (pts)")
ax2.axhline(0, color=COLORS["light_gray"], lw=0.6)
ax2.fill_between([], [], [], color=COLORS["gray"], alpha=0.2, lw=0, label="length-only fit,\nother traits")
ax2.legend(frameon=False, loc="upper left", fontsize=8)
for axx, L in ((ax1, "A"), (ax2, "B")):
    axx.text(-0.02, 1.04, L, transform=axx.transAxes, fontweight="bold", ha="right")
save_fig(fig, "mechanism", directory=f"{OUT}/plots")
res = y - np.polyval(b, x)
print("slope %.1f pts per log-word" % b[0])
for p in P:
    if p == "mute": continue
    ix = [i for i, t in enumerate(pts) if pole(t[0]) == p]
    print("%-12s words %5.1f dq %.2f grounded %.3f silent %.3f rec %.3f resid %+.1f" % (NAME[p], P[p][3], P[p][4], P[p][0], P[p][1], P[p][2], res[ix].mean()))
