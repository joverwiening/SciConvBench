import os
import csv
import json
import argparse
from collections import defaultdict

import numpy as np

import persona_users as pu

B = 10000
B_PRIMARY = 100000
B_CHUNK = 10000
SEED = 0
ALPHA = 0.05
TOP_K = 3
PRIMARY = "recovered_judge"
NATIVE_METRICS = ["FR", "CGRR", "SRR"]
ITEM_METRICS = ["asked", "delivered", "partial", "unasked_delivered", "unasked_partial", "recovered", "grounded",
                "silent"]
PROCESS_METRICS = ["cap_hit", "turns", "questions"]
SECONDARY_METRICS = NATIVE_METRICS + [m for m in ITEM_METRICS if m != PRIMARY]
REPLICATION_METRICS = ["FR", "CGRR", "SRR", "IC", "MC", "CR", "CP"]
REFERENCE_ARMS = ("paper", "neutral", "oracle", "mute")
PERSONAS = pu.load_personas()
PLACEBO_ARMS = tuple(pu.placebo_ids(PERSONAS))
HOLM_FAMILY = tuple(sorted(pid for pid, p in PERSONAS.items() if p["kind"] in ("style", "policy")))
ENFORCED_ARMS = tuple(sorted(pid for pid, p in PERSONAS.items() if p.get("enforced") == "code"))
MAX_ATTEMPTS = 3
EFFECT_COLS = ["arm", "metric", "n_cases", "effect", "ci_lo", "ci_hi", "p", "p_holm", "abs_lo", "beats_placebo"]
# PERSONA_PREREG.md, Analysis 2-3: a trait counts only if Holm, placebo and manipulation check all pass
MANIP_MIN_ON_TARGET = 0.60
MANIP_MIN_ABOVE_NEUTRAL = 0.30
PRIMARY_COLS = EFFECT_COLS + ["manip_pass", "counts"]


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def case_key(r):
    return (r["domain"], r["mode"], r["case_id"])


def b_for(m):
    return B_PRIMARY if m == PRIMARY else B


def metric(r, m):
    if m in ITEM_METRICS or m == PRIMARY:
        return (r.get("item_rates") or {}).get(m)
    if m in PROCESS_METRICS:
        return r.get(m)
    return (r.get("native") or {}).get(m)


def per_case_mean(rows, arm, m):
    acc = defaultdict(list)
    for r in rows:
        if r["arm"] == arm:
            v = metric(r, m)
            if v is not None:
                acc[case_key(r)].append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def resample_idx(n, b=B, seed=SEED):
    return np.random.default_rng(seed).integers(0, n, (b, n))


def bootstrap(n, stat, b=B, seed=SEED, idx=None):
    """Percentile CI over case resamples. stat maps a (b, n) index matrix to b values."""
    if n == 0:
        return None, None, None
    if idx is not None:
        vals = np.asarray(stat(idx), dtype=float)
    else:
        # chunks keep the (b, n) index matrix out of memory at b = B_PRIMARY; b <= B_CHUNK is one draw as before
        rng = np.random.default_rng(seed)
        vals = np.concatenate([np.asarray(stat(rng.integers(0, n, (min(B_CHUNK, b - i), n))), dtype=float)
                               for i in range(0, b, B_CHUNK)])
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return None, None, None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), vals


def boot_p(vals):
    if vals is None or not len(vals):
        return None
    return float(min(1.0, 2 * min(np.mean(vals <= 0), np.mean(vals >= 0))))


def abs_lower(lo, hi):
    if lo is None or hi is None:
        return None
    return 0.0 if lo <= 0 <= hi else float(min(abs(lo), abs(hi)))


def holm(ps):
    """Holm step-down adjusted p-values; None entries stay None and do not count toward m."""
    order = sorted((p, i) for i, p in enumerate(ps) if p is not None)
    m, out, running = len(order), [None] * len(ps), 0.0
    for rank, (p, i) in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p))
        out[i] = running
    return out


def paired_diffs(rows, arm, m, ref="neutral"):
    a, n = per_case_mean(rows, arm, m), per_case_mean(rows, ref, m)
    cases = sorted(set(a) & set(n))
    return cases, np.array([a[c] - n[c] for c in cases])


def mean_effect(d, rec, b=B):
    if not len(d):
        return dict(rec, n_cases=0, effect=None, ci_lo=None, ci_hi=None, p=None)
    lo, hi, vals = bootstrap(len(d), lambda idx: d[idx].mean(axis=1), b=b)
    return dict(rec, n_cases=len(d), effect=float(d.mean()), ci_lo=lo, ci_hi=hi, p=boot_p(vals))


def paired_effect(rows, arm, m, ref="neutral"):
    _, d = paired_diffs(rows, arm, m, ref)
    return mean_effect(d, {"arm": arm, "metric": m}, b=b_for(m))


def placebo_effects(rows, placebos, metrics):
    return [paired_effect(rows, a, m) for a in placebos for m in metrics]


def trait_effects(rows, traits, metrics, placebo, check_m=False):
    """Primary Holm runs over the style/policy family only; secondary Holm (descriptive) over all trait arms."""
    out = []
    for m in metrics:
        recs = [paired_effect(rows, a, m) for a in traits]
        fam = [r for r in recs if r["arm"] in HOLM_FAMILY] if m == PRIMARY else recs
        adj = dict(zip((r["arm"] for r in fam), holm([r["p"] for r in fam])))
        if m == PRIMARY and check_m:
            n_p = sum(r["p"] is not None for r in fam)
            want = len([a for a in HOLM_FAMILY if a in traits])
            assert n_p == want, f"Holm m = {n_p} but {want} family arms are present: an arm has no paired cases"
        for r in recs:
            r["p_holm"] = adj.get(r["arm"])
            r["abs_lo"] = abs_lower(r["ci_lo"], r["ci_hi"])
            pl = [abs(p["effect"]) for p in placebo if p["metric"] == m and p["effect"] is not None]
            r["beats_placebo"] = (r["abs_lo"] > max(pl)) if (pl and r["abs_lo"] is not None) else None
        out += recs
    return out


def selection(rows, traits, m=PRIMARY, k=TOP_K):
    diffs = {}
    for a in traits:
        cases, d = paired_diffs(rows, a, m)
        if cases:
            diffs[a] = dict(zip(cases, d))
    if not diffs:
        return [], 0
    cases = sorted(set.intersection(*(set(v) for v in diffs.values())))
    names = sorted(diffs)
    if not cases:
        return [{"arm": a, "n_cases": 0, "effect": None, "rank": None, "p_top": None} for a in names], 0
    D = np.array([[diffs[a][c] for c in cases] for a in names])  # traits × cases
    idx = resample_idx(len(cases))  # one matrix shared by every trait
    est = np.stack([D[t][idx].mean(axis=1) for t in range(len(names))], axis=1)  # B × traits
    order = np.argsort(-np.abs(est), axis=1, kind="stable")
    top = np.zeros_like(est, dtype=bool)
    np.put_along_axis(top, order[:, :k], True, axis=1)
    point = D.mean(axis=1)
    rank = {names[t]: r + 1 for r, t in enumerate(np.argsort(-np.abs(point), kind="stable"))}
    return [{"arm": a, "n_cases": len(cases), "effect": float(point[t]), "rank": rank[a],
             "p_top": float(top[:, t].mean())} for t, a in enumerate(names)], len(cases)


def oracle_mute_gap(rows, m=PRIMARY):
    o, u = per_case_mean(rows, "oracle", m), per_case_mean(rows, "mute", m)
    return {c: o[c] - u[c] for c in set(o) & set(u)}, o, u


def screened_cases(rows, m=PRIMARY):
    gap, o, u = oracle_mute_gap(rows, m)
    return {c for c, g in gap.items() if g > 0}, o, u


def gap_normalised(rows, arm, m=PRIMARY):
    screen, o, u = screened_cases(rows, m)
    a, n = per_case_mean(rows, arm, m), per_case_mean(rows, "neutral", m)
    cases = sorted(screen & set(a) & set(n))
    num = np.array([a[c] - n[c] for c in cases])
    den = np.array([o[c] - u[c] for c in cases])
    if not len(cases):
        return {"arm": arm, "metric": m, "n_cases": 0, "effect": None, "ci_lo": None, "ci_hi": None}

    def stat(idx):
        s = den[idx].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(s > 0, num[idx].sum(axis=1) / s, np.nan)

    lo, hi, _ = bootstrap(len(cases), stat)
    return {"arm": arm, "metric": m, "n_cases": len(cases), "effect": float(num.sum() / den.sum()),
            "ci_lo": lo, "ci_hi": hi}


def gap_terciles(rows, traits, m=PRIMARY):
    """Δ per trait within oracle−mute gap terciles; cut points are the 1/3 and 2/3 quantiles over all cases."""
    gap, _, _ = oracle_mute_gap(rows, m)
    if not gap:
        return [], None
    q1, q2 = (float(q) for q in np.quantile(list(gap.values()), [1 / 3, 2 / 3]))
    tercile = {c: ("low" if g <= q1 else "mid" if g <= q2 else "high") for c, g in gap.items()}
    out = []
    for a in traits:
        cases, d = paired_diffs(rows, a, m)
        for t in ("low", "mid", "high"):
            sel = np.array([d[i] for i, c in enumerate(cases) if tercile.get(c) == t])
            gaps = [gap[c] for c in cases if tercile.get(c) == t]
            rec = mean_effect(sel, {"arm": a, "metric": m, "tercile": t,
                                    "gap_min": min(gaps) if gaps else None,
                                    "gap_max": max(gaps) if gaps else None})
            out.append(rec)
    return out, (q1, q2)


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
        s = M[idx, den].sum(axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(s > 0, M[idx, num].sum(axis=-1) / np.where(s > 0, s, 1), np.nan)

    stats = {
        "delta_delivered_rate": lambda idx: ddel[idx].mean(axis=-1),
        "delta_P_recovered_given_delivered": lambda idx: cond(A, idx, 1, 0) - cond(N, idx, 1, 0),
        "delta_P_recovered_given_not_delivered": lambda idx: cond(A, idx, 3, 2) - cond(N, idx, 3, 2),
    }
    full = np.arange(len(cases))
    out = []
    for name, f in stats.items():
        est = float(f(full))
        lo, hi, _ = bootstrap(len(cases), f)
        out.append({"arm": arm, "quantity": name, "n_cases": len(cases),
                    "effect": est if np.isfinite(est) else None, "ci_lo": lo, "ci_hi": hi})
    return out


def item_recovery(rows, arm):
    """(case, item index) -> recovered, mean over reps."""
    acc = defaultdict(list)
    for r in rows:
        if r["arm"] != arm:
            continue
        for it in r.get("items") or []:
            if it.get("recovered") is not None:
                acc[(case_key(r), it["index"])].append(float(bool(it["recovered"])))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def by_difficulty(rows, traits):
    """Δ recovery per case restricted to items the mute arm recovers (easy, mean ≥ 0.5) or not (hard)."""
    mute = item_recovery(rows, "mute")
    if not mute:
        return []
    neu = item_recovery(rows, "neutral")
    out = []
    for a in traits:
        tr = item_recovery(rows, a)
        for stratum in ("easy", "hard"):
            per_case = defaultdict(list)
            for k, mv in mute.items():
                if (mv >= 0.5) == (stratum == "easy") and k in tr and k in neu:
                    per_case[k[0]].append((tr[k], neu[k]))
            d = np.array([np.mean([x[0] for x in v]) - np.mean([x[1] for x in v])
                          for _, v in sorted(per_case.items())])
            out.append(mean_effect(d, {"arm": a, "metric": "recovered", "stratum": stratum,
                                       "n_items": sum(len(v) for v in per_case.values())}))
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
    rows_by_arm = {r["arm"]: r for r in table}
    for arm in sorted({r["arm"] for r in rows} & set(ENFORCED_ARMS)):
        rec = rows_by_arm.get(arm)
        if rec is None:
            rec = {"arm": arm, "n_answers": None, "on_target": None, "max_off_target": None}
            table.append(rec)
        rec.update(enforced_check(rows, arm))
    return table, turn_rows


def bigfive_check(rows):
    """Blind Big Five judge (persona_eval --trait_check bigfive): per arm, parsed dialogues only."""
    acc = defaultdict(list)
    for r in rows:
        bf = r.get("bf_trait")
        if bf and not bf.get("parse_failure"):
            acc[r["arm"]].append(bf)
    out = []
    for arm in sorted(acc):
        v = acc[arm]
        hits = [b["hit"] for b in v if b.get("hit") is not None]
        dist = defaultdict(int)
        for b in v:
            dist[b["choice"]] += 1
        out.append({"arm": arm, "n": len(v), "hit_rate": (sum(hits) / len(hits)) if hits else None,
                    "ordinary_share": sum(b["choice"] == "ordinary" for b in v) / len(v),
                    "choices": "; ".join(f"{k} {n}" for k, n in sorted(dist.items(), key=lambda x: (-x[1], x[0])))})
    return out


def enforced_check(rows, arm):
    """Code-enforced persona: passes iff mean deferred turns per case ≥ 1 over cases with ≥ 1 question."""
    acc = defaultdict(list)
    for r in rows:
        if r["arm"] == arm and (r.get("questions") or 0) >= 1 and r.get("deferred") is not None:
            acc[case_key(r)].append(float(r["deferred"]))
    mean = float(np.mean([np.mean(v) for v in acc.values()])) if acc else None
    return {"n_cases_with_q": len(acc), "mean_deferred": mean,
            "manip_pass": None if mean is None else bool(mean >= 1),
            "note": "enforced in code, not in the trait judge; passes iff mean deferred ≥ 1 per case"}


def arm_means(rows):
    out = []
    for arm in sorted({r["arm"] for r in rows}):
        sub = [r for r in rows if r["arm"] == arm]
        rec = {"arm": arm, "n_rows": len(sub), "n_cases": len({case_key(r) for r in sub})}
        for m in [PRIMARY] + NATIVE_METRICS + ITEM_METRICS + PROCESS_METRICS:
            vals = [float(metric(r, m)) for r in sub if metric(r, m) is not None]
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


def job_key(r):
    return (r["arm"], r["rep"], r["domain"], r["mode"], r["case_id"])


def exclude_rows(rows):
    """PERSONA_PREREG.md, Sample: drop cases over the prompt cap and rows with an item whose card failed."""
    over = [r for r in rows if (r.get("native") or {}).get("over_length_cap") is True]
    rows = [r for r in rows if (r.get("native") or {}).get("over_length_cap") is not True]
    bad = defaultdict(int)
    kept = []
    for r in rows:
        if any(it.get("card_ok") is False for it in r.get("items") or []):
            bad[r["arm"]] += 1
        else:
            kept.append(r)
    return kept, {"over_length_cap": len(over), "card_not_ok": dict(sorted(bad.items()))}


def load_runs(run_dir):
    runs = []
    for root, _, files in os.walk(run_dir):
        if "persona_run.json" in files:
            with open(os.path.join(root, "persona_run.json")) as f:
                runs.append(json.load(f))
    return runs


def failure_table(runs, rows):
    have = {job_key(r) for r in rows}
    acc = defaultdict(lambda: {"jobs": 0, "done": 0, "error": 0, "attempts_capped": 0, "no_eval_row": 0})
    for run in runs:
        a = acc[run["arm"]]
        a["jobs"] += 1
        a["done"] += run.get("status") == "done"
        a["error"] += run.get("status") != "done"
        a["attempts_capped"] += run.get("status") != "done" and run.get("attempts", 1) >= MAX_ATTEMPTS
        a["no_eval_row"] += job_key(run) not in have
    return [dict({"arm": arm}, **acc[arm]) for arm in sorted(acc)]


def fill_deferred(rows, runs):
    deferred = {job_key(run): run.get("deferred") for run in runs}
    return [r if r.get("deferred") is not None else dict(r, deferred=deferred.get(job_key(r))) for r in rows]


def rep_counts(rows):
    n = defaultdict(set)
    for r in rows:
        n[(r["arm"], case_key(r))].add(r["rep"])
    arm_max = defaultdict(int)
    for (arm, _), reps in n.items():
        arm_max[arm] = max(arm_max[arm], len(reps))
    per_case = [{"arm": arm, "domain": c[0], "mode": c[1], "case_id": c[2], "n_reps": len(reps),
                 "short": len(reps) < arm_max[arm]} for (arm, c), reps in sorted(n.items())]
    summary = []
    for arm in sorted(arm_max):
        recs = [p for p in per_case if p["arm"] == arm]
        short = [p["case_id"] for p in recs if p["short"]]
        summary.append({"arm": arm, "n_cases": len(recs), "max_reps": arm_max[arm],
                        "min_reps": min(p["n_reps"] for p in recs), "n_short": len(short), "short_cases": short})
    return per_case, summary


def complete_cases(rows):
    """Cases present in every arm but paper (a separate replication run) with that arm's full number of reps."""
    per_case, summary = rep_counts([r for r in rows if r["arm"] != "paper"])
    if not summary:
        return set()
    full = defaultdict(set)
    for p in per_case:
        if not p["short"]:
            full[p["arm"]].add((p["domain"], p["mode"], p["case_id"]))
    return set.intersection(*(full[s["arm"]] for s in summary))


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def md_table(recs, cols, labels=None):
    labels = labels or {}
    lines = ["| " + " | ".join(labels.get(c, c) for c in cols) + " |", "|" + "---|" * len(cols)]
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


def analyse(rows, runs=None):
    raw = rows
    rows, excluded = exclude_rows(rows)
    if runs is not None:
        rows = fill_deferred(rows, runs)
    arms = sorted({r["arm"] for r in rows})
    placebos = [a for a in arms if a in PLACEBO_ARMS]
    trait_arms = [a for a in arms if a not in REFERENCE_ARMS and a not in PLACEBO_ARMS]
    traits = sorted(({t for r in rows for t in ((r.get("trait_present_share") or {}).keys())}
                     | {t for r in rows for turn in (r.get("trait_by_turn") or []) for t in turn["present"]})
                    - set(ENFORCED_ARMS))
    has = set(arms)
    missing = {"neutral": "neutral" not in has, "placebo": not placebos,
               "oracle_mute": not {"oracle", "mute"} <= has, "mute": "mute" not in has}
    screen, o, u = screened_cases(rows)
    placebo = placebo_effects(rows, placebos, [PRIMARY] + SECONDARY_METRICS) if not missing["neutral"] else []
    complete = complete_cases(rows)
    srows = [r for r in rows if case_key(r) in complete]
    splacebo = placebo_effects(srows, placebos, [PRIMARY]) if not missing["neutral"] else []
    reps, reps_summary = rep_counts(rows)
    res = {
        "arms": arms, "placebo_arms": placebos, "trait_arms": trait_arms, "missing": missing,
        "excluded": excluded,
        "holm_family": list(HOLM_FAMILY), "holm_missing": [a for a in HOLM_FAMILY if a not in has],
        "holm_m": len([a for a in HOLM_FAMILY if a in has]),
        "reps": reps, "reps_summary": reps_summary,
        "failures": failure_table(runs, raw) if runs is not None else None,
        "arm_means": arm_means(rows),
        "paired": trait_effects(rows, trait_arms, [PRIMARY] + SECONDARY_METRICS, placebo, check_m=True)
        if not missing["neutral"] else [],
        "sensitivity": trait_effects(srows, trait_arms, [PRIMARY], splacebo) if not missing["neutral"] else [],
        "sensitivity_n": len(complete),
        "placebo": placebo,
        "screen": [{"domain": c[0], "mode": c[1], "case_id": c[2], "oracle_recovered": o[c],
                    "mute_recovered": u[c], "passes": c in screen} for c in sorted(set(o) & set(u))],
        "gap_normalised": [gap_normalised(rows, a) for a in trait_arms]
        if not (missing["neutral"] or missing["oracle_mute"]) else [],
        "decomposition": [d for a in trait_arms for d in decomposition(rows, a)] if not missing["neutral"] else [],
        "difficulty": by_difficulty(rows, trait_arms) if not missing["neutral"] else [],
        "replication": replication(rows),
        "attribution_sanity": attribution_sanity(rows),
    }
    res["selection"], res["selection_n"] = selection(rows, trait_arms) if not missing["neutral"] else ([], 0)
    res["gap_tercile"], res["gap_cuts"] = gap_terciles(rows, trait_arms) \
        if not (missing["neutral"] or missing["oracle_mute"]) else ([], None)
    res["manipulation"], res["manipulation_by_turn"] = manipulation(rows, traits)
    res["traits"] = traits
    res["bigfive_check"] = bigfive_check(rows)
    manip = {r["arm"]: r for r in res["manipulation"]}
    for r in res["paired"] + res["sensitivity"]:
        if r["metric"] != PRIMARY:
            continue
        if r["arm"] in ENFORCED_ARMS:
            r["manip_pass"] = (manip.get(r["arm"]) or {}).get("manip_pass")
        else:
            on = (manip.get(r["arm"]) or {}).get("on_target")
            base = (manip.get("neutral") or {}).get(r["arm"])
            r["manip_pass"] = (on >= MANIP_MIN_ON_TARGET and on - base >= MANIP_MIN_ABOVE_NEUTRAL) \
                if (on is not None and base is not None) else None
        r["counts"] = bool(r["p_holm"] is not None and r["p_holm"] < ALPHA and r["beats_placebo"]
                           and r["manip_pass"])
    passing = sorted((r for r in res["paired"] if r["metric"] == PRIMARY and r["counts"]),
                     key=lambda r: -abs(r["effect"]))
    res["selected"] = [r["arm"] for r in passing[:TOP_K]]
    return res


def render_markdown(res):
    effect_cols = ["arm", "metric", "n_cases", "effect", "ci_lo", "ci_hi"]
    n_pass = sum(s["passes"] for s in res["screen"])
    miss = res["missing"]
    primary = [r for r in res["paired"] if r["metric"] == PRIMARY]
    secondary = [r for r in res["paired"] if r["metric"] != PRIMARY]
    no_neutral = "No `neutral` arm in the results: no paired comparison."
    no_om = "No `oracle` and/or `mute` arm in the results: block skipped."
    per_arm_cols = ["arm", "n_rows", "n_cases"]
    ex = res["excluded"]
    holm_warn = (f"**WARNING: Holm family incomplete. Missing from the data: {', '.join(res['holm_missing'])}. "
                 f"Primary Holm runs with m = {res['holm_m']} of {len(res['holm_family'])} prereg traits.**"
                 if res["holm_missing"] else
                 f"Holm family complete: m = {res['holm_m']} ({', '.join(res['holm_family'])}).")
    fail_cols = ["arm", "jobs", "done", "error", "attempts_capped", "no_eval_row"]
    manip_cols = ["arm", "n_answers", "on_target", "max_off_target"] + res["traits"]
    if any("note" in r for r in res["manipulation"]):
        manip_cols += ["n_cases_with_q", "mean_deferred", "manip_pass", "note"]
    parts = [
        "# Persona study analysis",
        f"**Primary outcome: item-recovery rate (`item_rates.{PRIMARY}`)**, the FR judge's own a/b score per "
        "case, averaged over reps per case. Everything else is secondary or descriptive.",
        holm_warn,
        f"Bootstrap: {B_PRIMARY} case resamples for the primary metric, {B} for everything else, seed {SEED}, "
        "95% percentile CI. Two-sided bootstrap p = 2·min(P(Δ*≤0), P(Δ*≥0)), capped at 1. Primary Holm over the "
        "style/policy personas (placebos excluded); Holm on secondary metrics is descriptive. "
        "|Δ| lower bound (abs_lo) = 0 if the CI spans 0, else min(|lo|, |hi|). beats_placebo = abs_lo > the "
        "largest placebo |Δ| point estimate for that metric.",
        f"Arms: {', '.join(res['arms'])}. Traits: {', '.join(res['trait_arms']) or '—'}. "
        f"Placebos: {', '.join(res['placebo_arms']) or '—'}.",
        "## Exclusions",
        f"Rows over the prompt length cap: {ex['over_length_cap']}. Rows with an item whose card is not ok, "
        f"per arm: {', '.join(f'{a} {n}' for a, n in ex['card_not_ok'].items()) or 'none'}.",
        "## Failure accounting",
        md_table(res["failures"], fail_cols) if res["failures"] is not None
        else "No --run_dir given: job failures not counted.",
        "### Reps per case per arm (short = fewer reps than the arm's max)",
        md_table(res["reps_summary"], ["arm", "n_cases", "max_reps", "min_reps", "n_short", "short_cases"]),
        "## Per-arm means (primary, item and process metrics)",
        md_table(res["arm_means"], per_arm_cols + [PRIMARY] + ITEM_METRICS + PROCESS_METRICS),
        f"## Primary: Paired effect vs neutral on {PRIMARY} (per case mean over reps, trait − neutral)",
        (f"Available cases per arm. counts = Holm p < {ALPHA} AND beats_placebo AND manip_pass (on-target ≥ "
         f"{MANIP_MIN_ON_TARGET:.0%} of the arm's answers and ≥ {MANIP_MIN_ABOVE_NEUTRAL:.0%} above the neutral "
         "arm's rate for that trait; code-enforced personas: see the manipulation check). "
         "Only traits with counts = True are read as trait effects.\n\n" + md_table(primary, PRIMARY_COLS))
        if not miss["neutral"] else no_neutral,
        "### Sensitivity: complete cases only",
        (f"Same test on the {res['sensitivity_n']} cases present in every arm except paper with that arm's full "
         "number of reps.\n\n" + md_table(res["sensitivity"], PRIMARY_COLS))
        if not miss["neutral"] else no_neutral,
        "## Placebo arms: Δ(placebo − neutral)",
        (md_table([r for r in res["placebo"] if r["metric"] == PRIMARY], effect_cols + ["p"])
         if res["placebo"] else ("No placebo arm in the results: beats_placebo is undefined."
                                 if miss["placebo"] else no_neutral)),
        "## Selection",
        f"Selected (traits with counts = True, top {TOP_K} by |Δ|): {', '.join(res['selected']) or 'none'}.",
        f"### Bootstrap P(trait in top {TOP_K} by |Δ| on {PRIMARY}), descriptive",
        (f"One resample index matrix ({B} × {res['selection_n']} cases with neutral and every trait arm) "
         "shared by all traits.\n\n" + md_table(res["selection"], ["arm", "n_cases", "effect", "rank", "p_top"]))
        if res["selection"] else (no_neutral if miss["neutral"] else "No trait arm in the results."),
        "## Secondary",
        "Native strict FR / CGRR / SRR (IC-gated) and the other item metrics, paired Δ vs neutral.",
        "### Per-arm means (native)",
        md_table(res["arm_means"], per_arm_cols + NATIVE_METRICS),
        "### Paired effect vs neutral",
        md_table(secondary, EFFECT_COLS, {"p_holm": "p_holm (descriptive)"}) if not miss["neutral"] else no_neutral,
        "### Placebo Δ, secondary metrics",
        md_table([r for r in res["placebo"] if r["metric"] != PRIMARY], effect_cols + ["p"])
        if res["placebo"] else "No placebo comparison.",
        "## Oracle − mute gap (covariate, descriptive)",
        f"### Case screen\n{n_pass}/{len(res['screen'])} cases with oracle item-recovery > mute item-recovery."
        if not miss["oracle_mute"] else no_om,
        "### Gap-normalised effect on screened cases: Σ(trait − neutral) / Σ(oracle − mute)",
        md_table(res["gap_normalised"], effect_cols) if not miss["oracle_mute"] else no_om,
        "### Δ by gap tercile (all cases with oracle and mute)",
        (f"Cut points (1/3, 2/3 quantiles of the per-case gap): {fmt(res['gap_cuts'][0])}, "
         f"{fmt(res['gap_cuts'][1])}.\n\n"
         + md_table(res["gap_tercile"], ["arm", "tercile", "gap_min", "gap_max", "n_cases", "effect", "ci_lo",
                                         "ci_hi"]))
        if res["gap_cuts"] is not None else (no_om if miss["oracle_mute"] else no_neutral),
        "## Decomposition vs neutral — descriptive (conditions on a post-treatment variable)",
        md_table(res["decomposition"], ["arm", "quantity", "n_cases", "effect", "ci_lo", "ci_hi"])
        if not miss["neutral"] else no_neutral,
        f"### Δ {PRIMARY} by item difficulty (easy = recovered in the mute arm, mean over mute reps ≥ 0.5)",
        (md_table(res["difficulty"], ["arm", "stratum", "n_cases", "n_items", "effect", "ci_lo", "ci_hi"])
         if not miss["mute"] else "No `mute` arm in the results: difficulty strata undefined.")
        if not miss["neutral"] else no_neutral,
        "## Manipulation check (share of user answers flagged per trait)",
        md_table(res["manipulation"], manip_cols),
        "### On-target rate by turn index",
        md_table(res["manipulation_by_turn"], ["arm", "turn", "n_answers", "on_target"]),
        "## Replication (paper arm, native metrics)",
        md_table([res["replication"]], ["n_rows", "n_cases"] + REPLICATION_METRICS),
        "## Attribution sanity",
        md_table([res["attribution_sanity"]], list(res["attribution_sanity"].keys())),
    ]
    if res.get("bigfive_check"):
        parts += ["## Big Five manipulation check (blind judge, one call per dialogue)",
                  "hit_rate: share of trait-arm dialogues whose judged pole is the arm's target.\n\n"
                  + md_table(res["bigfive_check"], ["arm", "n", "hit_rate", "ordinary_share", "choices"])]
    return "\n\n".join(parts) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Paired persona-trait effects from results.jsonl")
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True, help="Output directory for analysis.md and csv files")
    parser.add_argument("--run_dir", default=None, help="Runner output tree, for failure accounting")
    args = parser.parse_args(argv)
    rows = load_rows(args.results)
    res = analyse(rows, load_runs(args.run_dir) if args.run_dir else None)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "analysis.md"), "w") as f:
        f.write(render_markdown(res))
    for name in ("arm_means", "paired", "sensitivity", "placebo", "selection", "screen", "gap_normalised",
                 "gap_tercile", "decomposition", "difficulty", "manipulation", "manipulation_by_turn", "reps",
                 "reps_summary"):
        write_csv(os.path.join(args.out, f"{name}.csv"), res[name])
    if res["failures"] is not None:
        write_csv(os.path.join(args.out, "failures.csv"), res["failures"])
    if res["bigfive_check"]:
        write_csv(os.path.join(args.out, "bigfive_check.csv"), res["bigfive_check"])
    write_csv(os.path.join(args.out, "replication.csv"), [res["replication"]])
    write_csv(os.path.join(args.out, "attribution_sanity.csv"), [res["attribution_sanity"]])
    if res["holm_missing"]:
        print(f"WARNING: Holm family incomplete, missing {res['holm_missing']}")
    print(f"wrote {args.out}/analysis.md")
    return res


if __name__ == "__main__":
    main()
