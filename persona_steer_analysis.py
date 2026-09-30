import os
import csv
import json
import random
import argparse
from collections import defaultdict

DEFAULT_ARM = "approve_all"
BASE_ARM = "bf_neutral"
N_BOOT = 2000
BOOT_SEED = 0
VARIANTS = ("fast", "rigorous")


def load_rows(run_dir):
    with open(os.path.join(run_dir, "steer_results.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def defaults(rows):
    """(case, variant, entity) -> approve_all label; only where every approve_all chat agrees on fast/rigorous."""
    seen = defaultdict(set)
    for r in rows:
        if r["arm"] == DEFAULT_ARM:
            for e, lab in r["choice_labels"].items():
                seen[(r["case_id"], r["variant"], e)].add(lab)
    return {k: next(iter(v)) for k, v in seen.items() if len(v) == 1 and next(iter(v)) in VARIANTS}


def default_override(rows, arm, dflt):
    hits = []
    for r in rows:
        if r["arm"] != arm:
            continue
        for e, lab in r["choice_labels"].items():
            d = dflt.get((r["case_id"], r["variant"], e))
            if d is not None and d != r["variant"]:
                hits.append(lab == r["variant"])
    return (sum(hits) / len(hits) if hits else None), len(hits)


def cell_means(rows, arm):
    """(case, variant) -> (choice_match, facts_recovered), averaged over reps."""
    acc = defaultdict(lambda: ([], []))
    for r in rows:
        if r["arm"] == arm:
            acc[(r["case_id"], r["variant"])][0].append(r["choice_match"])
            acc[(r["case_id"], r["variant"])][1].append(r["facts_recovered"])
    return {k: (mean(c), mean(f)) for k, (c, f) in acc.items()}


def paired_deltas(rows, arm, base=BASE_ARM):
    """[(case_id, d_choice, d_facts)] over case-variants both arms have, with both metrics defined."""
    a, b = cell_means(rows, arm), cell_means(rows, base)
    out = []
    for k in sorted(set(a) & set(b)):
        if None in a[k] or None in b[k]:
            continue
        out.append((k[0], a[k][0] - b[k][0], a[k][1] - b[k][1]))
    return out


def summarise(pairs):
    dc = mean([p[1] for p in pairs])
    df = mean([p[2] for p in pairs])
    return dc, df, (dc - df) if dc is not None and df is not None else None


def percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def bootstrap(pairs, n_boot=N_BOOT, seed=BOOT_SEED):
    """Case-level bootstrap 95% CIs for (d_choice, d_facts, did): resample case ids with replacement."""
    by_case = defaultdict(list)
    for p in pairs:
        by_case[p[0]].append(p)
    cases = sorted(by_case)
    if not cases:
        return [(None, None)] * 3
    rng = random.Random(seed)
    draws = [[], [], []]
    for _ in range(n_boot):
        sample = [p for c in (rng.choice(cases) for _ in cases) for p in by_case[c]]
        for i, v in enumerate(summarise(sample)):
            draws[i].append(v)
    return [(percentile(sorted(d), 0.025), percentile(sorted(d), 0.975)) for d in draws]


def arm_table(rows):
    dflt = defaults(rows)
    arms = sorted({r["arm"] for r in rows})
    table = []
    for arm in arms:
        ar = [r for r in rows if r["arm"] == arm]
        do, n_do = default_override(rows, arm, dflt)
        row = {"arm": arm, "n_chats": len(ar),
               "choice_match": mean([r["choice_match"] for r in ar]),
               "facts_recovered": mean([r["facts_recovered"] for r in ar]),
               "default_override": do, "n_override_points": n_do,
               "words_per_reply": mean([r["user_words_per_reply"] for r in ar]),
               "turns": mean([r["turns"] for r in ar]),
               "cap_hits": sum(bool(r["cap_hit"]) for r in ar)}
        if arm != BASE_ARM:
            pairs = paired_deltas(rows, arm)
            dc, df, did = summarise(pairs)
            (dc_lo, dc_hi), (df_lo, df_hi), (did_lo, did_hi) = bootstrap(pairs)
            row.update({"n_pairs": len(pairs), "n_pair_cases": len({p[0] for p in pairs}),
                        "d_choice": dc, "d_choice_lo": dc_lo, "d_choice_hi": dc_hi,
                        "d_facts": df, "d_facts_lo": df_lo, "d_facts_hi": df_hi,
                        "did": did, "did_lo": did_lo, "did_hi": did_hi})
        table.append(row)
    return table


COLUMNS = ["arm", "n_chats", "choice_match", "facts_recovered", "default_override", "n_override_points",
           "words_per_reply", "turns", "cap_hits", "n_pairs", "n_pair_cases", "d_choice", "d_choice_lo",
           "d_choice_hi", "d_facts", "d_facts_lo", "d_facts_hi", "did", "did_lo", "did_hi"]


def fmt(v, nd=3):
    if v is None:
        return "–"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def ci(row, key):
    if row.get(key) is None:
        return "–"
    return f"{fmt(row[key])} [{fmt(row[key + '_lo'])}, {fmt(row[key + '_hi'])}]"


def write_outputs(run_dir, table):
    with open(os.path.join(run_dir, "steer_arms.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for row in table:
            w.writerow({k: row.get(k) for k in COLUMNS})
    lines = ["# Steering pilot: per-arm summary", "",
             f"Run dir: `{run_dir}`. Descriptive only. default_override: share of choice points where the "
             f"`{DEFAULT_ARM}` label on the same case-variant differs from the variant and the arm's label matches "
             f"the variant. Deltas vs `{BASE_ARM}`, paired by case-variant (reps averaged); "
             f"DiD = Δchoice − Δfacts; 95% CI from a case-level bootstrap ({N_BOOT} draws, seed {BOOT_SEED}).", "",
             "| arm | n chats | choice match | facts recovered | default override (n pts) | words/reply | turns | "
             "cap hits |",
             "|---|---|---|---|---|---|---|---|"]
    for r in table:
        lines.append(f"| {r['arm']} | {r['n_chats']} | {fmt(r['choice_match'])} | {fmt(r['facts_recovered'])} | "
                     f"{fmt(r['default_override'])} ({r['n_override_points']}) | {fmt(r['words_per_reply'], 1)} | "
                     f"{fmt(r['turns'], 2)} | {r['cap_hits']} |")
    lines += ["", f"## vs {BASE_ARM}", "",
              "| arm | pairs (cases) | Δchoice | Δfacts | DiD |", "|---|---|---|---|---|"]
    for r in table:
        if "n_pairs" in r:
            lines.append(f"| {r['arm']} | {r['n_pairs']} ({r['n_pair_cases']}) | {ci(r, 'd_choice')} | "
                         f"{ci(r, 'd_facts')} | {ci(r, 'did')} |")
    with open(os.path.join(run_dir, "steer_analysis.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Per-arm summary of steer_results.jsonl")
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args(argv)
    table = arm_table(load_rows(args.run_dir))
    write_outputs(args.run_dir, table)
    print(f"wrote {os.path.join(args.run_dir, 'steer_analysis.md')} and steer_arms.csv ({len(table)} arms)")
    return table


if __name__ == "__main__":
    main()
