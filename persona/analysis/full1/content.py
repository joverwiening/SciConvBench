# Produces replies.csv, arm_content.csv, pole_content.csv, pairs.csv, pairs_null.csv, pairs_summary.csv, examples.md.
import argparse
import json, re, os, math, csv, itertools
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--analysis-dir", required=True)
ap.add_argument("--dataset", default="dataset/foam/disambiguation_foam.json")
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
R = args.run_dir
DS = args.dataset
OUT = args.out
AG = "bedrock_glm5_Disambiguation"

TRAITS = ["high_A", "low_A", "high_C", "low_C", "high_E", "low_E", "high_N", "high_O", "low_O"]
TRAIT_ARMS = [f"bf_{t}{s}" for t in TRAITS for s in ["", "_w2", "_w3"]]
NEUTRAL_ARMS = ["bf_neutral"] + [f"bf_neutral_{x}" for x in "bcdef"]
ARMS = TRAIT_ARMS + NEUTRAL_ARMS

def pole_of(arm):
    if arm.startswith("bf_neutral"):
        return "neutral"
    return re.sub(r"_w[23]$", "", arm)[3:]

SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
def presub(t):
    t = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+", lambda m: "^" + m.group(0).translate(SUP), t)
    return t.replace("−", "-")

MANT = r"(?:\d+(?:\.\d+)?|\.\d+)"
EXP = r"\(?\s*([-+]?\d+)\s*\)?"
NUM_RE = re.compile(
    r"(?<![A-Za-wyzA-WYZ_\d.^])(?:"
    rf"(?P<a>{MANT})\s*[×xX\*·]\s*10\s*\^\s*(?P<ae>[-+]?\d+|\(\s*[-+]?\d+\s*\))"
    rf"|10\s*\^\s*(?P<be>[-+]?\d+|\(\s*[-+]?\d+\s*\))"
    rf"|(?P<c>{MANT})[eE](?P<ce>[-+]?\d+)"
    r"|(?P<d>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r")")

def parse_numbers(text):
    t = presub(text)
    vals = []
    for m in NUM_RE.finditer(t):
        try:
            if m.group("a") is not None:
                v = float(m.group("a")) * 10.0 ** int(m.group("ae").strip("() "))
            elif m.group("be") is not None:
                v = 10.0 ** int(m.group("be").strip("() "))
            elif m.group("c") is not None:
                v = float(m.group("c") + "e" + m.group("ce"))
            else:
                v = float(m.group("d").replace(",", ""))
        except (ValueError, OverflowError):
            continue
        vals.append(v)
    return uniq(vals)

def isin(v, lst):
    return any(math.isclose(v, x, rel_tol=1e-9, abs_tol=0.0) for x in lst)

def uniq(vals):
    out = []
    for v in vals:
        if not isin(v, out):
            out.append(v)
    return out

def minus(a, b):
    return [v for v in a if not isin(v, b)]

def fmt(vs):
    return "[" + ", ".join(f"{v:g}" for v in vs) + "]"

cases = {c["id"]: c for c in json.load(open(DS))}
hidden = {cid: minus(parse_numbers(c["complete_user_req"]), parse_numbers(c["incomplete_user_req"])) for cid, c in cases.items()}

def norm_ws(s):
    return re.sub(r"\s+", " ", s).strip()

rows = []
chats = {}
missing = []
mismatches = []
for arm in ARMS:
    base = os.path.join(R, arm, "rep1", "foam", AG)
    for cid in sorted(os.listdir(base)):
        cf = os.path.join(base, cid, "clarifications.json")
        if not os.path.exists(cf):
            missing.append((arm, cid))
            continue
        cl = json.load(open(cf))
        logf = os.path.join(base, cid, "conversation_log.txt")
        nq = sum(1 for ln in open(logf) if ln.startswith("Q:")) if os.path.exists(logf) else None
        if nq != len(cl):
            mismatches.append((arm, cid, len(cl), nq))
        hid = hidden[cid]
        seen = []
        chat_rows = []
        for e in sorted(cl, key=lambda e: e["turn"]):
            ans = e["answer"] or ""
            nums = parse_numbers(ans)
            qn = parse_numbers(e["question"] or "")
            given = [v for v in nums if isin(v, hid) and not isin(v, qn)]
            new = minus(given, seen)
            seen += new
            r = dict(arm=arm, pole=pole_of(arm), case_id=cid, turn=e["turn"], words=len(ans.split()),
                     n_numbers=len(nums), n_hidden_given=len(given), n_hidden_new=len(new),
                     numbers=fmt(nums), hidden_given=fmt(given), question=e["question"], answer=ans)
            rows.append(r)
            chat_rows.append(r)
        chats[(arm, cid)] = dict(rows=chat_rows, distinct=len(seen), n_hidden=len(hid), cl=cl, given_lists=[])

# sanity
rng = np.random.default_rng(0)
print(f"missing clarifications.json (excluded): {missing}")
print(f"turn-count mismatches clarifications vs log Q: {len(mismatches)} {mismatches[:20]}")
print(f"n chats {len(chats)}, n replies {len(rows)}")
for i in rng.choice(len(rows), 3, replace=False):
    r = rows[i]
    print("=" * 60)
    print(r["arm"], r["case_id"], "turn", r["turn"])
    print("REPLY:", r["answer"])
    print("reply numbers:", r["numbers"])
    print("case hidden numbers:", fmt(hidden[r["case_id"]]))
    print("hidden given (excl. question):", r["hidden_given"])

with open(os.path.join(OUT, "replies.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

def content_stats(keys):
    rs = [r for k in keys for r in chats[k]["rows"]]
    words = np.array([r["words"] for r in rs])
    hg = np.array([r["n_hidden_given"] for r in rs])
    shares = [chats[k]["distinct"] / chats[k]["n_hidden"] for k in keys if chats[k]["n_hidden"] > 0]
    return dict(n_chats=len(keys), n_replies=len(rs),
                replies_per_chat=len(rs) / len(keys),
                mean_words=words.mean() if len(rs) else float("nan"),
                median_words=float(np.median(words)) if len(rs) else float("nan"),
                hidden_given_per_reply=hg.mean() if len(rs) else float("nan"),
                distinct_hidden_per_chat=np.mean([chats[k]["distinct"] for k in keys]),
                share_hidden_given=np.mean(shares) if shares else float("nan"),
                n_chats_with_hidden=len(shares),
                hidden_per_100_words=100 * hg.sum() / words.sum() if words.sum() else float("nan"))

def write_csv(path, recs):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        for r in recs:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

arm_recs = []
for arm in ARMS:
    keys = [k for k in chats if k[0] == arm]
    arm_recs.append(dict(arm=arm, pole=pole_of(arm), **content_stats(keys)))
write_csv(os.path.join(OUT, "arm_content.csv"), arm_recs)
POLES = TRAITS + ["neutral"]
pole_recs = []
for p in POLES:
    keys = [k for k in chats if pole_of(k[0]) == p]
    pole_recs.append(dict(pole=p, **content_stats(keys)))
write_csv(os.path.join(OUT, "pole_content.csv"), pole_recs)

# matched first replies
def first(k):
    rs = chats[k]["rows"]
    t1 = [r for r in rs if r["turn"] == 1]
    return t1[0] if t1 else None

def make_pair(a, b, cid):
    ra, rb = first((a, cid)), first((b, cid))
    if ra is None or rb is None:
        return None
    if norm_ws(ra["question"]) != norm_ws(rb["question"]):
        return None
    return dict(arm=a, neutral_arm=b, pole=pole_of(a), case_id=cid, Q1=ra["question"],
                A1_trait=ra["answer"], A1_neutral=rb["answer"],
                words_trait=ra["words"], words_neutral=rb["words"],
                n_hidden_given_trait=ra["n_hidden_given"], n_hidden_given_neutral=rb["n_hidden_given"],
                hidden_given_trait=ra["hidden_given"], hidden_given_neutral=rb["hidden_given"])

all_cases = sorted(cases)
pairs = []
for a in TRAIT_ARMS:
    for b in NEUTRAL_ARMS:
        for cid in all_cases:
            if (a, cid) in chats and (b, cid) in chats:
                p = make_pair(a, b, cid)
                if p: pairs.append(p)
null_pairs = []
for a, b in itertools.combinations(sorted(NEUTRAL_ARMS), 2):
    for cid in all_cases:
        if (a, cid) in chats and (b, cid) in chats:
            p = make_pair(a, b, cid)
            if p:
                p["pole"] = "neutral_vs_neutral"
                null_pairs.append(p)
write_csv(os.path.join(OUT, "pairs.csv"), pairs)
write_csv(os.path.join(OUT, "pairs_null.csv"), null_pairs)

def boot_ci(ps, B=2000, seed=0):
    by = defaultdict(list)
    for p in ps:
        by[p["case_id"]].append(p["n_hidden_given_trait"] - p["n_hidden_given_neutral"])
    cs = list(by)
    sums = np.array([sum(by[c]) for c in cs], float)
    ns = np.array([len(by[c]) for c in cs], float)
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(cs), size=(B, len(cs)))
    means = sums[idx].sum(1) / ns[idx].sum(1)
    return np.percentile(means, [2.5, 97.5])

summ = []
groups = [(p, [x for x in pairs if x["pole"] == p]) for p in TRAITS] + [("neutral_vs_neutral", null_pairs)]
for name, ps in groups:
    if not ps:
        summ.append(dict(pole=name, n_pairs=0)); continue
    d = np.array([x["n_hidden_given_trait"] - x["n_hidden_given_neutral"] for x in ps])
    lo, hi = boot_ci(ps)
    summ.append(dict(pole=name, n_pairs=len(ps), n_cases=len({x["case_id"] for x in ps}),
                     mean_words_trait=np.mean([x["words_trait"] for x in ps]),
                     mean_words_neutral=np.mean([x["words_neutral"] for x in ps]),
                     mean_hg_trait=np.mean([x["n_hidden_given_trait"] for x in ps]),
                     mean_hg_neutral=np.mean([x["n_hidden_given_neutral"] for x in ps]),
                     mean_diff=d.mean(), ci_lo=lo, ci_hi=hi,
                     share_gt=(d > 0).mean(), share_eq=(d == 0).mean(), share_lt=(d < 0).mean()))
write_csv(os.path.join(OUT, "pairs_summary.csv"), summ)

# examples
rng = np.random.default_rng(0)
md = ["# Matched first-reply examples (random, seed 0)\n"]
for p in ["high_E", "high_N", "high_O", "high_A"]:
    ps = [x for x in pairs if x["pole"] == p]
    gt = [x for x in ps if x["n_hidden_given_trait"] > x["n_hidden_given_neutral"]]
    eq = [x for x in ps if x["n_hidden_given_trait"] == x["n_hidden_given_neutral"]]
    md.append(f"## {p}\n\nqualifying: trait>neutral {len(gt)}, equal {len(eq)}\n")
    sel = [("trait > neutral", gt[i]) for i in rng.choice(len(gt), min(2, len(gt)), replace=False)] if gt else []
    sel += [("equal", eq[i]) for i in rng.choice(len(eq), min(1, len(eq)), replace=False)] if eq else []
    for lab, x in sel:
        md.append(f"### {lab} — {x['case_id']}: {x['arm']} vs {x['neutral_arm']}\n")
        md.append(f"case hidden numbers: {fmt(hidden[x['case_id']])}\n")
        md.append(f"**Q1:** {x['Q1']}\n")
        md.append(f"**A1 {x['arm']}** ({x['words_trait']} words; hidden given {x['hidden_given_trait']}):\n\n> " + x["A1_trait"].replace("\n", "\n> ") + "\n")
        md.append(f"**A1 {x['neutral_arm']}** ({x['words_neutral']} words; hidden given {x['hidden_given_neutral']}):\n\n> " + x["A1_neutral"].replace("\n", "\n> ") + "\n")
open(os.path.join(OUT, "examples.md"), "w").write("\n".join(md))

def show(recs, cols):
    print("\t".join(cols))
    for r in recs:
        print("\t".join(f"{r.get(c, ''):.2f}" if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols))
print()
show(pole_recs, ["pole", "n_chats", "replies_per_chat", "mean_words", "median_words", "hidden_given_per_reply", "distinct_hidden_per_chat", "share_hidden_given", "hidden_per_100_words"])
print()
show(summ, ["pole", "n_pairs", "n_cases", "mean_words_trait", "mean_words_neutral", "mean_hg_trait", "mean_hg_neutral", "mean_diff", "ci_lo", "ci_hi", "share_gt", "share_eq", "share_lt"])
