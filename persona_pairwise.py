"""Pairwise manipulation check for the Big Five persona arms: each trait-arm dialogue against the reference
(neutral) arm's dialogue on the same case; the judge picks which scientist shows the target pole more.
Run after persona_eval.py on a finished run dir."""
import os
import sys
import csv
import glob
import json
import math
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import Config
from utils import LLMService
import persona_users as pu
import persona_eval as pev

PAIRWISE_SYSTEM = """
        You read two chats, [CHAT_A] and [CHAT_B]. In each, an AI assistant is preparing a simulation setup for a
        scientist and messages them with questions; the scientist replies. The two chats are with different
        scientists. Some words in the scientists' lines may be replaced by "[…]".

        **YOUR TASK:**
        Decide which scientist, A or B, is more "__DESCRIPTION__", judged only from how the scientist writes
        (not from the assistant's messages). "choice" is "A" or "B". "confidence" is how sure you are: 0 (a guess),
        1 (slightly), 2 (clearly), 3 (certain). "evidence" is the passage of the chosen scientist's lines that shows
        it most clearly, copied VERBATIM.

        **OUTPUT FORMAT:**
        You must output your final evaluation as a JSON object enclosed EXACTLY within <FINAL_JSON> tags.
        <FINAL_JSON>
        {
            "choice": "A",
            "confidence": 0,
            "evidence": "verbatim passage of the chosen scientist's lines"
        }
        </FINAL_JSON>
        """

PAIRWISE_USER = """
        <CHAT_A>
        {chat_a}
        </CHAT_A>

        <CHAT_B>
        {chat_b}
        </CHAT_B>
        """

PAIRWISE_TEMPLATES_SHA = pu.sha256_text(PAIRWISE_SYSTEM + PAIRWISE_USER +
                                        json.dumps(pev.BIGFIVE_OPTIONS, sort_keys=True))
GATE_HIT_RATE, GATE_CI_LO = 0.75, 0.5
CSV_COLS = ["target", "arm", "n", "hits", "hit_rate", "ci_lo", "ci_hi", "pass", "n_parse_fail", "n_no_partner"]


def _log_binom_pmf(k, n, p):
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def _binom_cdf(k, n, p):
    return min(1.0, sum(math.exp(_log_binom_pmf(i, n, p)) for i in range(0, k + 1)))


def _bisect(f, lo=0.0, hi=1.0, iters=200):
    # f increasing in p, root of f(p) = 0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(k, n, alpha=0.05):
    if n <= 0:
        return None, None
    try:
        from scipy.stats import beta
        lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
        hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
        return lo, hi
    except ImportError:
        pass
    # lo: P(X >= k | p) = alpha/2; hi: P(X <= k | p) = alpha/2
    lo = 0.0 if k == 0 else _bisect(lambda p: (1.0 - _binom_cdf(k - 1, n, p)) - alpha / 2)
    hi = 1.0 if k == n else _bisect(lambda p: alpha / 2 - _binom_cdf(k, n, p))
    return lo, hi


def load_folders(run_dir):
    """Done case dirs as {folder: (arm, rep, domain, agent_label, case_id)}."""
    out = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "*", "rep*", "*", "*", "*", "persona_run.json"))):
        run = pev.read_json(p) or {}
        if run.get("status") != "done":
            continue
        folder = os.path.dirname(p)
        out[folder] = (run["arm"], int(run["rep"]), run["domain"], os.path.basename(os.path.dirname(folder)),
                       run["case_id"])
    return out


def find_pairs(run_dir, bigfive):
    """[(trait folder, arm, target, partner folder or None)] for every done trait-arm case dir."""
    personas = bigfive["personas"]
    refs = [pid for pid, p in personas.items() if p["kind"] == "neutral"]
    if len(refs) != 1:
        raise ValueError(f"expected exactly one neutral Big Five persona, got {refs}")
    ref = refs[0]
    folders = load_folders(run_dir)
    ref_by_case = {}
    for folder, (arm, rep, domain, agent_label, case_id) in folders.items():
        if arm == ref:
            ref_by_case.setdefault((domain, agent_label, case_id), {})[rep] = folder
    pairs = []
    for folder, (arm, rep, domain, agent_label, case_id) in sorted(folders.items()):
        p = personas.get(arm)
        if not p or p["kind"] != "trait":
            continue
        reps = ref_by_case.get((domain, agent_label, case_id)) or {}
        partner = reps.get(rep, reps[min(reps)] if reps else None)
        pairs.append((folder, arm, p["target"], partner))
    return pairs


def trait_is_a(rel):
    return int(pu.sha256_text(rel), 16) % 2 == 0


def pairwise_judge(llm, trait_turns, ref_turns, target, trait_is):
    chats = {"A": trait_turns, "B": ref_turns} if trait_is == "A" else {"A": ref_turns, "B": trait_turns}
    system = PAIRWISE_SYSTEM.replace("__DESCRIPTION__", pev.BIGFIVE_OPTIONS[target])
    user = PAIRWISE_USER.format(chat_a=pev.bigfive_chat(chats["A"]), chat_b=pev.bigfive_chat(chats["B"]))
    parsed, attempts = pev.invoke_parsed(llm, user, system)
    out = {"choice": None, "hit": None, "confidence": None, "evidence": None, "evidence_ok": None,
           "attempts": attempts}
    choice = parsed.get("choice") if parsed is not None else None
    choice = choice.strip().upper() if isinstance(choice, str) else None
    if choice not in ("A", "B"):
        return out
    try:
        conf = int(parsed.get("confidence"))
        conf = conf if 0 <= conf <= 3 else None
    except (TypeError, ValueError):
        conf = None
    ev = parsed.get("evidence") if isinstance(parsed.get("evidence"), str) else ""
    chosen = chats[choice]
    raw = "\n".join(t["answer"] for t in chosen)
    masked = "\n".join(pev.mask_constructs(t["answer"]) for t in chosen)
    out.update(choice=choice, hit=choice == trait_is, confidence=conf, evidence=ev,
               evidence_ok=pu.is_verbatim(ev, raw) or pu.is_verbatim(ev, masked))
    return out


def judge_pair(pair, ctx):
    folder, arm, target, partner = pair
    path = os.path.join(folder, "persona_pairwise.json")
    if not ctx["args"].force:
        rec = pev.read_json(path)
        if rec is not None:
            return rec, "cached"
    if ctx["spend"].exhausted():
        return None, "skipped_budget"
    run_dir = ctx["args"].run_dir
    rel = os.path.relpath(os.path.abspath(folder), os.path.abspath(run_dir))
    trait_is = "A" if trait_is_a(rel) else "B"
    llm = LLMService(ctx["judge_cfg"])
    res = pairwise_judge(llm, pev.dialogue_turns(folder)[0], pev.dialogue_turns(partner)[0], target, trait_is)
    usage = llm.get_statistics()
    rec = {"target": target, "arm": arm,
           "partner": os.path.relpath(os.path.abspath(partner), os.path.abspath(run_dir)),
           "trait_is": trait_is, **{k: res[k] for k in ("choice", "hit", "confidence", "evidence", "evidence_ok",
                                                       "attempts")},
           "usage": usage, "judge": ctx["judge_cfg"].as_dict(), "prompt_sha256": PAIRWISE_TEMPLATES_SHA}
    pev.write_json(path, rec)
    ctx["spend"].add(pu.stats_cost(usage, ctx["prices"].get(ctx["judge_cfg"].model_version)))
    return rec, "done"


def gate_row(target, arm, recs, n_no_partner):
    parsed = [r for r in recs if r.get("hit") is not None]
    n, hits = len(parsed), sum(bool(r["hit"]) for r in parsed)
    rate = hits / n if n else None
    lo, hi = clopper_pearson(hits, n)
    return {"target": target, "arm": arm, "n": n, "hits": hits, "hit_rate": rate, "ci_lo": lo, "ci_hi": hi,
            "pass": bool(rate is not None and rate >= GATE_HIT_RATE and lo > GATE_CI_LO),
            "n_parse_fail": len(recs) - n, "n_no_partner": n_no_partner}


def summarize(pairs, recs_by_folder):
    groups = {}
    for folder, arm, target, partner in pairs:
        g = groups.setdefault((target, arm), {"recs": [], "no_partner": 0})
        if partner is None:
            g["no_partner"] += 1
        elif recs_by_folder.get(folder) is not None:
            g["recs"].append(recs_by_folder[folder])
    rows = []
    for target in sorted({t for t, _ in groups}):
        arms = sorted(a for t, a in groups if t == target)
        for arm in arms:
            g = groups[(target, arm)]
            rows.append(gate_row(target, arm, g["recs"], g["no_partner"]))
        rows.append(gate_row(target, "ALL", [r for a in arms for r in groups[(target, a)]["recs"]],
                             sum(groups[(target, a)]["no_partner"] for a in arms)))
    return rows


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_summary(run_dir, rows):
    with open(os.path.join(run_dir, "pairwise_gate.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r[k]) for k in CSV_COLS})
    lines = [f"# Pairwise Big Five manipulation check",
             "",
             f"pass = hit_rate >= {GATE_HIT_RATE} and Clopper-Pearson 95% ci_lo > {GATE_CI_LO}",
             "",
             "| " + " | ".join(CSV_COLS) + " |",
             "|" + "|".join("---" for _ in CSV_COLS) + "|"]
    lines += ["| " + " | ".join(fmt(r[k]) for k in CSV_COLS) + " |" for r in rows]
    with open(os.path.join(run_dir, "pairwise_gate.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pairwise Big Five manipulation check on a persona run dir")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--judge_model", required=True, choices=list(Config.LLM_PROFILES))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--budget_usd", type=float, required=True,
                        help="Spend cap for this invocation's judge calls; no new pair starts once reached")
    parser.add_argument("--force", action="store_true", help="Rejudge pairs that already have persona_pairwise.json")
    parser.add_argument("--dry_run", action="store_true", help="Count pairs, make no calls")
    args = parser.parse_args(argv)

    manifest = pev.read_json(os.path.join(args.run_dir, "manifest.json"))
    if manifest is None:
        sys.exit(f"No manifest.json in {args.run_dir}")
    pairs = find_pairs(args.run_dir, pu.load_bigfive())
    with_partner = [p for p in pairs if p[3] is not None]
    n_no_partner = len(pairs) - len(with_partner)
    n_cached = 0 if args.force else sum(os.path.exists(os.path.join(p[0], "persona_pairwise.json"))
                                        for p in with_partner)
    print(f"{len(pairs)} trait case dirs: {len(with_partner)} pairs ({n_cached} cached, "
          f"{len(with_partner) - n_cached} to judge), {n_no_partner} without a reference partner")
    if args.dry_run:
        return {"pairs": len(with_partner), "cached": n_cached, "to_judge": len(with_partner) - n_cached,
                "no_partner": n_no_partner}

    judge_cfg = pu.role_config(args.judge_model)
    em = pev.read_json(os.path.join(args.run_dir, "eval_manifest.json")) or {}
    prices = dict(manifest.get("prices") or {}, **{k: v for k, v in (em.get("prices") or {}).items() if v})
    if not pu.has_price(prices.get(judge_cfg.model_version)):
        prices.update(pu.fetch_prices([judge_cfg.model_version]))
    pu.require_prices(prices, [judge_cfg.model_version])
    ctx = {"args": args, "judge_cfg": judge_cfg, "prices": prices, "spend": pu.SpendTracker(args.budget_usd)}

    recs, n_err, n_budget, n_done = {}, 0, 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(judge_pair, p, ctx): p for p in with_partner}
        for fut in as_completed(futures):
            try:
                rec, status = fut.result()
            except Exception as e:
                n_err += 1
                print(f"pairwise failed for {futures[fut][0]}: {type(e).__name__}: {e}")
                continue
            n_budget += status == "skipped_budget"
            n_done += status == "done"
            if rec is not None:
                recs[futures[fut][0]] = rec
    rows = summarize(pairs, recs)
    write_summary(args.run_dir, rows)
    print(f"pairwise_gate.csv: {len(rows)} rows, {n_done} judged, {n_err} errors")
    if n_budget:
        print(f"{n_budget} pairs not judged: budget reached")
    print(ctx["spend"].summary())
    out = {"pairs": len(with_partner), "judged": n_done, "errors": n_err, "no_partner": n_no_partner,
           "rows": rows}
    if n_budget:
        out["skipped_budget"] = n_budget
    return out


if __name__ == "__main__":
    main()
