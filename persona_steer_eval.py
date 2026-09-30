import os
import sys
import glob
import json
import time
import random
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import Config
from utils import LLMService
from llm_judge_disambiguation import LLMJudgeDisambiguation
import persona_users as pu
import persona_eval as pev

PARSE_TRIES = 3

CHOICE_SYSTEM = """
        You check which setup a final simulation specification uses. You are given the
        [FINAL_SPECIFICATION] and, for several settings, two options A and B.
        For each setting answer:
        - "A" if the specification uses option A,
        - "B" if it uses option B,
        - "neither" if it states this setting but uses a clearly different setup from both,
        - "not_stated" if the specification does not state this setting.
        Judge by meaning, not wording. For numeric settings (cell counts, time steps, end times, tolerances and
        the like), pick the option whose value is nearer to the specification's value by ratio (for example
        with options 1000 and 100000 cells, 20000 cells is nearer to 100000); answer "neither" only if it
        sits at about the geometric middle or uses a different kind of setup.

        **OUTPUT FORMAT:**
        You must output your final answer as a JSON object enclosed EXACTLY within <FINAL_JSON> tags, with one
        entry per setting, keyed by the setting's name.
        <FINAL_JSON>
        {"setting_name": "A"}
        </FINAL_JSON>
        """

CHOICE_USER = """
        <FINAL_SPECIFICATION>
        {spec}
        </FINAL_SPECIFICATION>

        <SETTINGS>
        {settings}
        </SETTINGS>
        """

CHOICE_PROMPT_SHA = pu.sha256_text(CHOICE_SYSTEM + CHOICE_USER)


def label_order(arm, case_id, variant, entities):
    """Per entity, {label: variant} with A/B randomised by a seed from sha256(arm|case|variant)."""
    seed = int(hashlib.sha256(f"{arm}|{case_id}|{variant}".encode()).hexdigest(), 16)
    rng = random.Random(seed)
    out = {}
    for e in entities:
        out[e] = {"A": "fast", "B": "rigorous"} if rng.random() < 0.5 else {"A": "rigorous", "B": "fast"}
    return out


def choice_prompt(spec, points, order):
    lines = []
    for p in points:
        lines.append(f"{p['entity']}:")
        for lab in ("A", "B"):
            lines.append(f"  {lab}. {p[order[p['entity']][lab]]}")
    return CHOICE_USER.format(spec=spec, settings="\n".join(lines))


def map_labels(answer, order):
    out = {}
    for e, o in order.items():
        a = str((answer or {}).get(e, "")).strip()
        norm = a.lower().replace(" ", "_")
        if a.upper() in o:
            out[e] = o[a.upper()]
        elif norm in ("neither", "not_stated"):
            out[e] = norm
        else:
            out[e] = "judge_error"
    return out


def choice_judge(llm, spec, steer, arm):
    points = steer["choice_points"]
    order = label_order(arm, steer["case_id"], steer["variant"], [p["entity"] for p in points])
    prompt = choice_prompt(spec, points, order)
    answer = None
    for _ in range(PARSE_TRIES):
        answer = pu.parse_final_json(llm.invoke(prompt, CHOICE_SYSTEM))
        if answer is not None:
            break
    return {"order": order, "answer": answer, "labels": map_labels(answer, order),
            "prompt_sha256": CHOICE_PROMPT_SHA}


def fact_judge(folder, fact_entities, cfg):
    fr = pev.make_judge(LLMJudgeDisambiguation, cfg)
    gt_text = fr.read_file(os.path.join(folder, "complete_prompt.txt"))
    gen_text = fr.read_file(os.path.join(folder, "generated_prompt.txt"))
    result = fr.evaluate(gt_text, gen_text, fact_entities)
    parsed = pev.parse_score(result.get("score"))
    recovered, total = parsed if parsed else (0, len(fact_entities))
    result["strict_score"] = "1/1" if recovered == total and total > 0 else "0/1"
    result["partial_score"] = f"{recovered}/{total}"
    result["fact_entities"] = fact_entities
    return result, fr


def score_folder(folder, ctx):
    steer = pev.read_json(os.path.join(folder, "steer_case.json"))
    run = pev.read_json(os.path.join(folder, "persona_run.json"))
    spec = pev.read_text(os.path.join(folder, "generated_prompt.txt")).strip()
    cfg = ctx["judge_cfg"]
    llm = LLMService(cfg)
    choice = choice_judge(llm, spec, steer, run["arm"])
    choice["usage"] = llm.get_statistics()
    pev.write_json(os.path.join(folder, "steer_choice.json"), choice)
    facts, fr = fact_judge(folder, steer.get("fact_entities", []), cfg)
    facts["usage"] = fr.llm.get_statistics()
    pev.write_json(os.path.join(folder, "steer_facts.json"), facts)
    price = ctx["prices"][cfg.model_version]
    ctx["spend"].add(pu.stats_cost(choice["usage"], price) + pu.stats_cost(facts["usage"], price))


def result_row(folder):
    run = pev.read_json(os.path.join(folder, "persona_run.json")) or {}
    steer = pev.read_json(os.path.join(folder, "steer_case.json")) or {}
    choice = pev.read_json(os.path.join(folder, "steer_choice.json"))
    facts = pev.read_json(os.path.join(folder, "steer_facts.json"))
    if not choice or not facts:
        return None
    clar = pev.read_json(os.path.join(folder, "clarifications.json")) or []
    variant = steer.get("variant", run.get("variant"))
    labels = choice["labels"]
    parsed = pev.parse_score(facts.get("partial_score"))
    words = [len(str(c.get("answer") or "").split()) for c in clar]
    rep = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(folder))))
    return {"arm": run.get("arm"), "rep": rep, "case_id": steer.get("case_id", run.get("case_id")),
            "variant": variant, "turns": run.get("turns"), "cap_hit": run.get("cap_hit"),
            "choice_labels": labels,
            "choice_match": (sum(l == variant for l in labels.values()) / len(labels)) if labels else None,
            "facts_recovered": (parsed[0] / parsed[1]) if parsed and parsed[1] else None,
            "user_words_per_reply": (sum(words) / len(words)) if words else None,
            "n_user_replies": len(clar)}


def find_folders(run_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(run_dir, "*", "rep*", "foam", "*_Disambiguation", "*", "persona_run.json"))):
        if (pev.read_json(p) or {}).get("status") == "done":
            out.append(os.path.dirname(p))
    return out


def is_scored(folder):
    return all(os.path.exists(os.path.join(folder, f)) for f in ("steer_choice.json", "steer_facts.json"))


def rebuild_results(run_dir, folders):
    rows = [r for r in (result_row(f) for f in folders) if r]
    rows.sort(key=lambda r: (r["arm"] or "", r["rep"], r["case_id"] or "", r["variant"] or ""))
    path = os.path.join(run_dir, "steer_results.jsonl")
    with open(path + ".tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(path + ".tmp", path)
    return rows


def score_or_skip(folder, ctx):
    if ctx["spend"].exhausted():
        return "skipped_budget"
    score_folder(folder, ctx)
    return "done"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score steering-pilot chats: choice match and fact recovery")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--judge_model", required=True, choices=list(Config.LLM_PROFILES))
    parser.add_argument("--budget_usd", type=float, required=True,
                        help="Spend cap for this invocation; no new folder starts once reached")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Rescore folders that already have scores")
    args = parser.parse_args(argv)

    folders = find_folders(args.run_dir)
    todo = [f for f in folders if args.force or not is_scored(f)]
    print(f"{len(folders)} done folders, {len(todo)} to score")

    cfg = pu.role_config(args.judge_model)
    prices = pu.fetch_prices([cfg.model_version])
    pu.require_prices(prices, [cfg.model_version])
    ctx = {"judge_cfg": cfg, "prices": prices, "spend": pu.SpendTracker(args.budget_usd)}
    man_path = os.path.join(args.run_dir, "steer_eval_manifest.json")
    man = pev.read_json(man_path) or {"invocations": []}
    man["invocations"].append({"argv": sys.argv if argv is None else ["persona_steer_eval.py"] + list(argv),
                               "git": pu.git_state(), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "judge": cfg.as_dict(), "choice_prompt_sha256": CHOICE_PROMPT_SHA,
                               "budget_usd": args.budget_usd, "prices": prices})
    pev.write_json(man_path, man)

    n_err = n_budget = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(score_or_skip, f, ctx): f for f in todo}
        for fut in as_completed(futures):
            try:
                n_budget += fut.result() == "skipped_budget"
            except Exception as e:
                n_err += 1
                print(f"scoring failed for {futures[fut]}: {type(e).__name__}: {e}")
    rows = rebuild_results(args.run_dir, folders)
    man["invocations"][-1]["spent_usd"] = ctx["spend"].spent
    pev.write_json(man_path, man)
    print(f"steer_results.jsonl: {len(rows)} rows, {n_err} errors")
    if n_budget:
        print(f"{n_budget} folders not scored: budget reached")
    print(ctx["spend"].summary())
    return {"rows": len(rows), "errors": n_err, "skipped_budget": n_budget}


if __name__ == "__main__":
    main()
