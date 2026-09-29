# Produces trait_effects.csv, arms.csv, chats.csv, personas.json, decoding.json, first_answers.csv in --out.
import argparse, csv, glob, json, os, re
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--analysis-dir", required=True)
ap.add_argument("--deep-dir", required=True)
ap.add_argument("--persona-dir", default="persona/bigfive")
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

POLE = {"high_A": "agreeable", "low_A": "disagreeable", "high_C": "organised", "low_C": "disorganised",
        "high_E": "outgoing", "low_E": "reserved", "high_N": "anxious", "high_O": "curious", "low_O": "conventional",
        "neutral": "neutral"}
TARGET = {"high_agreeableness": "high_A", "low_agreeableness": "low_A", "high_conscientiousness": "high_C",
          "low_conscientiousness": "low_C", "high_extraversion": "high_E", "low_extraversion": "low_E",
          "high_neuroticism": "high_N", "high_openness": "high_O", "low_openness": "low_O"}
SECONDARY = ["asked", "delivered", "partial", "unasked_delivered"]


def trait_wording(arm):
    if arm == "mute":
        return "never_answers", ""
    if arm == "oracle":
        return "full_answers", ""
    if arm.startswith("bf_neutral"):
        return "neutral", 1 if arm == "bf_neutral" else "bcdef".index(arm[-1]) + 2
    m = re.fullmatch(r"bf_(high|low)_([A-Z])(?:_w([23]))?", arm)
    return POLE[f"{m.group(1)}_{m.group(2)}"], int(m.group(3) or 1)


def read_csv(path):
    return list(csv.DictReader(open(path, newline="")))


def write_csv(path, recs):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)


def r4(v):
    return "" if v in ("", None) else round(float(v), 4)


pe = [r for r in read_csv(os.path.join(args.analysis_dir, "pole_effects.csv")) if r["k_wordings"] == "3"]
words_pole = {r["pole"]: r["mean_words"] for r in read_csv(os.path.join(args.deep_dir, "pole_content.csv"))}
recs = []
for r in pe:
    if r["metric"] != "recovered_judge":
        continue
    rec = {"trait": POLE[TARGET[r["target"]]]}
    for k in ["n_cases", "effect", "ci_lo", "ci_hi", "p", "p_holm", "gate_target", "gate_opposite",
              "gate_all_wordings", "gate", "counts"]:
        rec[k] = r[k]
    for m in SECONDARY:
        (s,) = [x for x in pe if x["target"] == r["target"] and x["metric"] == m]
        rec[f"effect_{m}"] = s["effect"]
    rec["words_per_reply"] = words_pole[TARGET[r["target"]]]
    recs.append(rec)
write_csv(os.path.join(args.out, "trait_effects.csv"), recs)

words_arm = {r["arm"]: r["mean_words"] for r in read_csv(os.path.join(args.deep_dir, "arm_content.csv"))}
recs = []
for r in read_csv(os.path.join(args.analysis_dir, "arm_means.csv")):
    t, w = trait_wording(r["arm"])
    rec = {"arm": r["arm"], "trait": t, "wording": w, "n_chats": r["n_rows"]}
    for k in ["recovered_judge", "asked", "delivered", "partial", "unasked_delivered", "grounded", "silent",
              "questions", "turns", "cap_hit"]:
        rec[k] = r4(r[k])
    rec["words_per_reply"] = r4(words_arm.get(r["arm"]))
    recs.append(rec)
write_csv(os.path.join(args.out, "arms.csv"), recs)

reply_words = defaultdict(list)
for r in csv.DictReader(open(os.path.join(args.deep_dir, "replies.csv"), newline="")):
    reply_words[(r["arm"], r["case_id"])].append(int(r["words"]))
recs = []
for line in open(os.path.join(args.run_dir, "results.jsonl")):
    r = json.loads(line)
    t, w = trait_wording(r["arm"])
    run_json = os.path.join(args.run_dir, r["arm"], f"rep{r['rep']}", r["domain"], r["agent_label"], r["case_id"],
                            "persona_run.json")
    fv = json.load(open(run_json))["final_valid"] if os.path.exists(run_json) else None
    ws = reply_words.get((r["arm"], r["case_id"]))
    rec = {"case_id": r["case_id"], "arm": r["arm"], "trait": t, "wording": w}
    for k in ["recovered_judge"] + SECONDARY:
        rec[k] = r["item_rates"].get(k)
    rec.update(questions=r["questions"], turns=r["turns"], cap_hit=r["cap_hit"], final_valid=fv,
               words_per_reply=sum(ws) / len(ws) if ws else None)
    recs.append(rec)
write_csv(os.path.join(args.out, "chats.csv"), recs)

out = {}
for f in sorted(glob.glob(os.path.join(args.persona_dir, "personas", "*.json"))):
    arm = os.path.basename(f)[:-5]
    if arm.endswith("_noex") or arm.startswith("bf_low_N"):
        continue
    p = json.load(open(f))
    t, w = trait_wording(arm)
    out[arm] = {"trait": t, "wording": w, **{k: p[k] for k in ["sheet", "appraisal", "reminder", "examples"]}}
out["shared"] = json.load(open(os.path.join(args.persona_dir, "shared.json")))
json.dump(out, open(os.path.join(args.out, "personas.json"), "w"), indent=1, ensure_ascii=False)

D = json.load(open(os.path.join(args.deep_dir, "decode_stages.json")))
out = {}
for old, new in [("sim", "scientist_replies"), ("agentQ", "agent_questions"), ("spec", "final_spec")]:
    d = dict(D[old])
    d["recall"] = {POLE[k[3:]]: v for k, v in d["recall"].items()}
    out[new] = d
out["chance"] = 1 / 9
json.dump(out, open(os.path.join(args.out, "decoding.json"), "w"), indent=1)

recs = read_csv(os.path.join(args.deep_dir, "pairs_summary.csv"))
for r in recs:
    r["pole"] = POLE.get(r["pole"], r["pole"])
recs = [{("trait" if k == "pole" else k): v for k, v in r.items()} for r in recs]
write_csv(os.path.join(args.out, "first_answers.csv"), recs)
