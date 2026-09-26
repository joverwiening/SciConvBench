import os
import csv
import json
import argparse
from collections import defaultdict

import numpy as np

B = 10000
SEED = 0
NATIVE_METRICS = ["FR", "CGRR", "SRR"]
ITEM_METRICS = ["asked", "delivered", "partial", "recovered", "grounded", "silent"]
REPLICATION_METRICS = ["FR", "CGRR", "SRR", "IC", "MC", "CR", "CP"]
REFERENCE_ARMS = ("paper", "neutral", "oracle", "mute")


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def case_key(r):
    return (r["domain"], r["mode"], r["case_id"])


def metric(r, m):
    if m in ITEM_METRICS:
        return (r.get("item_rates") or {}).get(m)
    return (r.get("native") or {}).get(m)


def per_case_mean(rows, arm, m):
    acc = defaultdict(list)
    for r in rows:
        if r["arm"] == arm:
            v = metric(r, m)
            if v is not None:
                acc[case_key(r)].append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def bootstrap(n, stat, b=B, seed=SEED):
    """Percentile CI of stat(idx) over case resamples."""
    if n == 0:
        return None, None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        v = stat(rng.integers(0, n, n))
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return None, None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_effect(rows, arm, m, ref="neutral"):
    a, n = per_case_mean(rows, arm, m), per_case_mean(rows, ref, m)
    cases = sorted(set(a) & set(n))
    d = np.array([a[c] - n[c] for c in cases])
    if not len(d):
        return {"arm": arm, "metric": m, "n_cases": 0, "effect": None, "ci_lo": None, "ci_hi": None}
    lo, hi = bootstrap(len(d), lambda idx: d[idx].mean())
    return {"arm": arm, "metric": m, "n_cases": len(d), "effect": float(d.mean()), "ci_lo": lo, "ci_hi": hi}


def screened_cases(rows, m="recovered"):
    o, u = per_case_mean(rows, "oracle", m), per_case_mean(rows, "mute", m)
    return {c for c in set(o) & set(u) if o[c] > u[c]}, o, u


def gap_normalised(rows, arm, m="recovered"):
    screen, o, u = screened_cases(rows, m)
    a, n = per_case_mean(rows, arm, m), per_case_mean(rows, "neutral", m)
    cases = sorted(screen & set(a) & set(n))
    num = np.array([a[c] - n[c] for c in cases])
    den = np.array([o[c] - u[c] for c in cases])
    if not len(cases):
        return {"arm": arm, "metric": m, "n_cases": 0, "effect": None, "ci_lo": None, "ci_hi": None}

    def stat(idx):
        s = den[idx].sum()
        return num[idx].sum() / s if s > 0 else None

    lo, hi = bootstrap(len(cases), stat)
    return {"arm": arm, "metric": m, "n_cases": len(cases), "effect": float(num.sum() / den.sum()),
            "ci_lo": lo, "ci_hi": hi}


def item_counts(rows, arm):
    """Per case: [delivered, recovered&delivered, not delivered, recovered&not delivered], pooled over reps."""
    acc = defaultdict(lambda: np.zeros(4))
    for r in rows:
        if r["arm"] != arm:
            continue
        for it in r.get("items") or []:
            if it.get("delivered") is None or it.get("recovered") is None:
                continue
            c = acc[case_key(r)]
            if it["delivered"]:
                c[0] += 1
                c[1] += bool(it["recovered"])
            else:
                c[2] += 1
                c[3] += bool(it["recovered"])
    return acc


def decomposition(rows, arm):
    a, n = item_counts(rows, arm), item_counts(rows, "neutral")
    da, dn = per_case_mean(rows, arm, "delivered"), per_case_mean(rows, "neutral", "delivered")
    cases = sorted(set(a) & set(n) & set(da) & set(dn))
    if not cases:
        return []
    A = np.array([a[c] for c in cases])
    N = np.array([n[c] for c in cases])
    ddel = np.array([da[c] - dn[c] for c in cases])

    def cond(M, idx, num, den):
        s = M[idx, den].sum()
        return M[idx, num].sum() / s if s > 0 else np.nan

    stats = {
        "delta_delivered_rate": lambda idx: ddel[idx].mean(),
        "delta_P_recovered_given_delivered": lambda idx: cond(A, idx, 1, 0) - cond(N, idx, 1, 0),
        "delta_P_recovered_given_not_delivered": lambda idx: cond(A, idx, 3, 2) - cond(N, idx, 3, 2),
    }
    full = np.arange(len(cases))
    out = []
    for name, f in stats.items():
        est = f(full)
        lo, hi = bootstrap(len(cases), f)
        out.append({"arm": arm, "quantity": name, "n_cases": len(cases),
                    "effect": None if not np.isfinite(est) else float(est), "ci_lo": lo, "ci_hi": hi})
    return out


def manipulation(rows, traits):
    present, answers = defaultdict(lambda: defaultdict(float)), defaultdict(float)
    by_turn = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for r in rows:
        for t in r.get("trait_by_turn") or []:
            answers[r["arm"]] += 1
            for tid in traits:
                present[r["arm"]][tid] += t["present"].get(tid, 0)
            if r["arm"] in traits:
                cell = by_turn[r["arm"]][t["turn"]]
                cell[0] += t["present"].get(r["arm"], 0)
                cell[1] += 1
    table = []
    for arm in sorted(answers):
        rates = {tid: present[arm][tid] / answers[arm] for tid in traits}
        on = rates.get(arm) if arm in traits else None
        off = max((v for tid, v in rates.items() if tid != arm), default=None)
        table.append(dict({"arm": arm, "n_answers": int(answers[arm]), "on_target": on, "max_off_target": off},
                          **rates))
    turn_rows = [{"arm": arm, "turn": t, "n_answers": int(c[1]), "on_target": c[0] / c[1]}
                 for arm in sorted(by_turn) for t, c in sorted(by_turn[arm].items())]
    return table, turn_rows


def arm_means(rows):
    out = []
    for arm in sorted({r["arm"] for r in rows}):
        sub = [r for r in rows if r["arm"] == arm]
        rec = {"arm": arm, "n_rows": len(sub), "n_cases": len({case_key(r) for r in sub})}
        for m in NATIVE_METRICS + ITEM_METRICS:
            vals = [metric(r, m) for r in sub if metric(r, m) is not None]
            rec[m] = float(np.mean(vals)) if vals else None
        out.append(rec)
    return out


def replication(rows):
    sub = [r for r in rows if r["arm"] == "paper" and r.get("native")]
    rec = {"n_rows": len(sub), "n_cases": len({case_key(r) for r in sub})}
    for m in REPLICATION_METRICS:
        vals = [r["native"][m] for r in sub if r["native"].get(m) is not None]
        rec[m] = float(np.mean(vals)) if vals else None
    return rec


def attribution_sanity(rows):
    def share(arm, fn):
        num = den = 0
        for r in rows:
            if r["arm"] != arm:
                continue
            for it in r.get("items") or []:
                if it.get("delivered") is None:
                    continue
                den += 1
                num += fn(it)
        return (num / den if den else None), den
    o, on = share("oracle", lambda it: bool(it["delivered"]))
    m, mn = share("mute", lambda it: bool(it["delivered"] or it["partial"]))
    return {"oracle_full_delivery_rate": o, "oracle_items": on,
            "mute_any_delivery_rate": m, "mute_items": mn}


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def md_table(recs, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in recs:
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def write_csv(path, recs):
    if not recs:
        open(path, "w").close()
        return
    cols = list(recs[0].keys())
    for r in recs[1:]:
        cols += [c for c in r if c not in cols]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(recs)


def analyse(rows):
    arms = sorted({r["arm"] for r in rows})
    trait_arms = [a for a in arms if a not in REFERENCE_ARMS]
    traits = sorted({t for r in rows for t in ((r.get("trait_present_share") or {}).keys())})
    screen, o, u = screened_cases(rows)
    res = {
        "arm_means": arm_means(rows),
        "paired": [paired_effect(rows, a, m) for a in trait_arms for m in NATIVE_METRICS + ITEM_METRICS],
        "screen": [{"domain": c[0], "mode": c[1], "case_id": c[2], "oracle_recovered": o[c],
                    "mute_recovered": u[c], "passes": c in screen} for c in sorted(set(o) & set(u))],
        "gap_normalised": [gap_normalised(rows, a, m) for a in trait_arms for m in ("recovered",)],
        "decomposition": [d for a in trait_arms for d in decomposition(rows, a)],
        "replication": replication(rows),
        "attribution_sanity": attribution_sanity(rows),
    }
    res["manipulation"], res["manipulation_by_turn"] = manipulation(rows, traits)
    res["traits"] = traits
    return res


def render_markdown(res):
    effect_cols = ["arm", "metric", "n_cases", "effect", "ci_lo", "ci_hi"]
    n_pass = sum(s["passes"] for s in res["screen"])
    parts = [
        "# Persona study analysis",
        f"Bootstrap: {B} case resamples, seed {SEED}, 95% percentile CI.",
        "## Per-arm means", md_table(res["arm_means"], ["arm", "n_rows", "n_cases"] + NATIVE_METRICS + ITEM_METRICS),
        "## Paired effect vs neutral (per case mean over reps, trait − neutral)",
        md_table(res["paired"], effect_cols),
        f"## Case screen\n{n_pass}/{len(res['screen'])} cases with oracle item-recovery > mute item-recovery.",
        "## Gap-normalised effect on screened cases: Σ(trait − neutral) / Σ(oracle − mute)",
        md_table(res["gap_normalised"], effect_cols),
        "## Decomposition vs neutral",
        md_table(res["decomposition"], ["arm", "quantity", "n_cases", "effect", "ci_lo", "ci_hi"]),
        "## Manipulation check (share of user answers flagged per trait)",
        md_table(res["manipulation"], ["arm", "n_answers", "on_target", "max_off_target"] + res["traits"]),
        "### On-target rate by turn index",
        md_table(res["manipulation_by_turn"], ["arm", "turn", "n_answers", "on_target"]),
        "## Replication (paper arm, native metrics)",
        md_table([res["replication"]], ["n_rows", "n_cases"] + REPLICATION_METRICS),
        "## Attribution sanity",
        md_table([res["attribution_sanity"]], list(res["attribution_sanity"].keys())),
    ]
    return "\n\n".join(parts) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Paired persona-trait effects from results.jsonl")
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True, help="Output directory for analysis.md and csv files")
    args = parser.parse_args(argv)
    rows = load_rows(args.results)
    res = analyse(rows)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "analysis.md"), "w") as f:
        f.write(render_markdown(res))
    for name in ("arm_means", "paired", "screen", "gap_normalised", "decomposition",
                 "manipulation", "manipulation_by_turn"):
        write_csv(os.path.join(args.out, f"{name}.csv"), res[name])
    write_csv(os.path.join(args.out, "replication.csv"), [res["replication"]])
    write_csv(os.path.join(args.out, "attribution_sanity.csv"), [res["attribution_sanity"]])
    print(f"wrote {args.out}/analysis.md")
    return res


if __name__ == "__main__":
    main()
