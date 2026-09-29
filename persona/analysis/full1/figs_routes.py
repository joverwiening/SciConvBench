# Produces plots/routes.{png,pdf} (needs pairs_summary.csv and decode_stages.json in --out).
import argparse, os
import json, csv, re, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--analysis-dir", required=True)
ap.add_argument("--dataset", default="dataset/foam/disambiguation_foam.json")
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure_style.py")).read())
R = args.run_dir; OUT = args.out
NAME = {"high_N": "anxious", "high_E": "outgoing", "high_O": "curious", "high_C": "organised", "high_A": "agreeable",
        "low_C": "disorganised", "low_O": "conventional", "low_E": "reserved", "low_A": "disagreeable", "neutral": "neutral"}
COUNTS = {"high_N", "high_E", "high_O"}
col = lambda p: COLORS["primary"] if p in COUNTS else COLORS["secondary"] if p == "low_A" else COLORS["gray"]
pole = lambda a: "neutral" if a.startswith("bf_neutral") else re.sub(r"_w[23]$", "", a)[3:]
S = {r["pole"]: r for r in csv.DictReader(open(f"{OUT}/pairs_summary.csv"))}
rows = [json.loads(l) for l in open(f"{R}/results.jsonl")]
U = {}
for r in rows:
    if r["arm"] in ("mute", "oracle"): continue
    v = [it["unasked_delivered"] for it in r["items"] if it["unasked_delivered"] is not None]
    if v: U.setdefault(pole(r["arm"]), []).append(np.mean(v))
U = {p: 100 * np.mean(v) for p, v in U.items()}
D = json.load(open(f"{OUT}/decode_stages.json"))

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=figsize(7, 0.42), constrained_layout=True,
                                  gridspec_kw={"width_ratios": [1.15, 1, 0.8]})
order = sorted(NAME, key=lambda p: float(S[p]["mean_diff"]) if p != "neutral" else 0)
order = [p for p in order if p != "neutral"]
nl, nh = float(S["neutral_vs_neutral"]["ci_lo"]), float(S["neutral_vs_neutral"]["ci_hi"])
a1.axvspan(nl, nh, color=COLORS["light_gray"], alpha=0.5, lw=0)
a1.axvline(0, color=COLORS["gray"], ls="--", lw=0.8)
for i, p in enumerate(order):
    m, lo, hi = (float(S[p][k]) for k in ("mean_diff", "ci_lo", "ci_hi"))
    a1.plot([lo, hi], [i, i], color=col(p), lw=1.5); a1.plot(m, i, "o", color=col(p), ms=5)
a1.set_yticks(range(len(order)), [NAME[p] for p in order])
a1.set_xlabel("extra hidden values in\nfirst answer vs neutral")

order2 = sorted(U, key=lambda p: U[p])
for i, p in enumerate(order2):
    a2.barh(i, U[p], color=col(p) if p != "neutral" else COLORS["light_gray"], height=0.7)
a2.set_yticks(range(len(order2)), [NAME[p] for p in order2])
a2.set_xlabel("facts volunteered\nunasked (%)")

st = [("sim", "scientist\nreplies"), ("agentQ", "agent\nasks"), ("spec", "agent\nspec")]
for i, (k, lab) in enumerate(st):
    a3.bar(i, 100 * D[k]["acc"], color=COLORS["primary"] if i == 0 else COLORS["gray"], width=0.65)
    a3.text(i, 100 * D[k]["acc"] + 2, f"{100 * D[k]['acc']:.0f}%", ha="center", fontsize=8)
a3.axhline(100 / 9, color=COLORS["gray"], ls="--", lw=0.8, label="chance")
a3.legend(frameon=False, loc="upper right", fontsize=8)
a3.set_xticks(range(3), [l for _, l in st], fontsize=8); a3.set_ylim(0, 100)
a3.set_ylabel("trait decoded, 9-way (%)")
for ax, L in ((a1, "A"), (a2, "B"), (a3, "C")):
    ax.text(-0.02, 1.04, L, transform=ax.transAxes, fontweight="bold", ha="right")
save_fig(fig, "routes", directory=f"{OUT}/plots")
print({p: round(U[p], 1) for p in order2})
print({k: round(D[k]["acc"], 3) for k, _ in st})
