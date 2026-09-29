# Produces plots/trait_effects.{png,pdf}.
import argparse
import csv, os
ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--analysis-dir", required=True)
ap.add_argument("--dataset", default="dataset/foam/disambiguation_foam.json")
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure_style.py")).read())
A = args.analysis_dir
NAME = {"high_neuroticism": "anxious", "high_extraversion": "outgoing", "high_openness": "curious",
        "high_conscientiousness": "organised", "high_agreeableness": "agreeable", "low_conscientiousness": "disorganised",
        "low_openness": "conventional", "low_extraversion": "reserved", "low_agreeableness": "disagreeable"}
poles = [r for r in csv.DictReader(open(f"{A}/pole_effects.csv")) if r["metric"] == "recovered_judge"]
words = [r for r in csv.DictReader(open(f"{A}/pole_wordings.csv")) if r["metric"] == "recovered_judge"]
poles.sort(key=lambda r: float(r["effect"]))
fig, ax = plt.subplots(figsize=figsize(4.5, 0.85))
for i, r in enumerate(poles):
    e, lo, hi = (100 * float(r[k]) for k in ("effect", "ci_lo", "ci_hi"))
    gate = all(r[k] == "True" for k in ("gate_target", "gate_opposite", "gate_all_wordings"))
    counts = float(r["p_holm"]) < 0.05 and gate
    c = COLORS["primary"] if counts else COLORS["gray"]
    ax.plot([lo, hi], [i, i], color=c, lw=1.5)
    ax.plot(e, i, "o", color=c, mfc=c if counts else "white", ms=6, zorder=3)
    for w in words:
        if w["target"] == r["target"]:
            ax.plot(100 * float(w["effect"]), i, "|", color=COLORS["light_gray"], ms=7, zorder=2)
    # low_openness w3: only 9/4 of 62 pairs judged before the pairwise budget stop, so its gate is incomplete, not failed
    why = ", check incomplete" if r["target"] == "low_openness" else ", not visible"
    lab = f"p={float(r['p_holm']):.2f}" + ("" if gate else why)
    ax.text(28, i, lab, va="center", fontsize=8, color=c)
ax.axvline(0, color=COLORS["gray"], ls="--", lw=0.8)
ax.set_yticks(range(len(poles)), [NAME[r["target"]] for r in poles])
ax.set_xlim(-20, 27)
ax.set_xlabel("Δ missing facts recovered vs neutral (pts)")
ax.plot([], [], "o", color=COLORS["primary"], label="counts")
ax.plot([], [], "o", color=COLORS["gray"], mfc="white", label="doesn't count")
ax.plot([], [], "|", color=COLORS["light_gray"], ms=7, label="one wording")
ax.legend(loc="center left", fontsize=8)
save_fig(fig, "trait_effects", directory=f"{args.out}/plots")
