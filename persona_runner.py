import os
import sys
import json
import time
import random
import inspect
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import Config
from main import ConvAgent
from utils import LLMService
from llm_user_interface import LLMUserInterface
import test_runner
import persona_users as pu

JOB_SEED = 0
MAX_ATTEMPTS = 3  # PERSONA_PREREG.md, Sample: errored jobs are rerun up to twice


def parse_cases(spec: str):
    if spec.startswith("@"):
        with open(spec[1:]) as f:
            spec = f.read()
    return [c.strip() for c in spec.replace("\n", ",").split(",") if c.strip()]


def case_dir(out, arm, rep, domain, agent_profile, mode, case_id):
    return os.path.join(out, arm, f"rep{rep}", domain,
                        f"{agent_profile}_{pu.MODE_LABEL[mode]}", case_id)


def user_prompt_template_sha() -> str:
    src = "".join(inspect.getsource(getattr(LLMUserInterface, name))
                  for name in ("get_full_context", "build_system_prompt", "answer_clarification_question"))
    src += "".join(inspect.getsource(obj) for obj in (pu.render_persona, pu.turn_note_for_arm, pu.BusyUser))
    src += inspect.getsource(pu.BigFiveUser) + pu.load_bigfive()["shared_text"]
    return pu.sha256_text(src)


def matraix_user_sha() -> str:
    return pu.sha256_text(inspect.getsource(pu.MatrAIxUser) + pu.MX_IDENTITY + pu.MX_REMINDER)


def behaviour_user_sha() -> str:
    return pu.sha256_text(inspect.getsource(pu.behaviour_persona) + inspect.getsource(PrimedAgent)
                          + open(pu.BV_PATH).read())


class PrimedAgent(ConvAgent):
    """ConvAgent whose first reply is a saved one (shared by every arm of a case and rep), so arms only diverge
    from the simulated user's first answer; later turns call the model as usual."""

    def __init__(self, config, prompt_mode=None, first_response=None):
        super().__init__(config, prompt_mode)
        self._first = first_response

    def chat(self, user_input: str) -> str:
        if self._first is None:
            return super().chat(user_input)
        response, self._first = self._first, None
        self.history.append(f"User: {user_input}")
        self.history.append(f"Analyst: {response}")
        return response


def shared_first_path(root, case_id, rep):
    return os.path.join(root, f"{case_id}_rep{rep}.json")


def make_shared_first(case_id, rep, ctx):
    """One real first agent reply for (case, rep), written once; its cost goes to this invocation's spend."""
    args = ctx["args"]
    path = shared_first_path(args.shared_first, case_id, rep)
    if os.path.exists(path):
        return "exists"
    incomplete_req = ctx["cases"][case_id].get(test_runner.TASK_CONFIG[args.mode]["keys"]["input_prompt"], "")
    response, stats = None, {}
    for _ in range(2):
        # chat() is the exact first call run_dialogue makes (history "User: <request>")
        agent = ConvAgent(ctx["agent_cfg"], "baseline" if args.baseline else None)
        try:
            response = agent.chat(incomplete_req)
        except Exception as e:
            print(f"first reply {case_id} rep{rep}: {type(e).__name__}: {e}")
            response = None
        stats = agent.llm.get_statistics()
        ctx["spend"].add(pu.stats_cost(stats, ctx["prices"].get(ctx["agent_cfg"].model_version)))
        if response and response.strip() and not getattr(agent.llm, "failed_calls", 0):
            break
        response = None
    if response is None:
        return "failed"  # its jobs error on the missing file and are rerun next invocation
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"case_id": case_id, "rep": rep, "response": response, "usage": stats,
                   "agent": ctx["agent_cfg"].as_dict(), "agent_system_prompt_sha256": pu.sha256_text(agent.system_prompt),
                   "created": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    os.replace(tmp, path)
    return "made"


def load_cards(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def persona_hashes(personas, bigfive, matraix):
    out = {pid: pu.sha256_file(p["path"]) for pid, p in personas.items()}
    out["bigfive/shared"] = pu.sha256_file(bigfive["shared_path"])
    out.update({pid: pu.sha256_file(p["path"]) for pid, p in bigfive["personas"].items()})
    out.update({pid: pu.sha256_file(p["path"]) for pid, p in matraix["personas"].items()})
    return out


def build_manifest(args, agent_cfg, user_cfg, personas, cards_path, bigfive, prices, matraix):
    prompt_mode = "baseline" if args.baseline else None
    agent_prompt = ConvAgent(agent_cfg, prompt_mode).system_prompt
    return {
        "git": pu.git_state(),
        "argv": sys.argv if args._argv is None else ["persona_runner.py"] + list(args._argv),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "roles": {"agent": agent_cfg.as_dict(), "user": user_cfg.as_dict() if user_cfg else None},
        "baseline": bool(args.baseline),
        "max_turns": args.max_turns,
        "final_fix": True,
        "sha256": {
            "agent_system_prompt": pu.sha256_text(agent_prompt),
            "user_prompt_template": user_prompt_template_sha(),
            "personas": persona_hashes(personas, bigfive, matraix),
            "matraix_user": matraix_user_sha(),
            "behaviour_user": behaviour_user_sha(),
        },
        "shared_first": os.path.abspath(args.shared_first) if args.shared_first else None,
        "cards": {f"{args.domain}/{args.mode}": cards_entry(cards_path)},
        "prices": prices,
    }


def cards_entry(cards_path):
    if not cards_path or not os.path.exists(cards_path):
        return None
    return {"path": os.path.abspath(cards_path), "sha256": pu.sha256_file(cards_path)}


def check_or_write_manifest(args, agent_cfg, user_cfg, personas, cards_path, bigfive, prices, matraix):
    path = os.path.join(args.out, "manifest.json")
    key = f"{args.domain}/{args.mode}"
    new = build_manifest(args, agent_cfg, user_cfg, personas, cards_path, bigfive, prices, matraix)
    if not os.path.exists(path):
        os.makedirs(args.out, exist_ok=True)
        with open(path, "w") as f:
            json.dump(new, f, indent=2)
        return new
    with open(path) as f:
        old = json.load(f)
    diffs = []
    old_user, new_user = old["roles"].get("user"), new["roles"].get("user")
    if old["roles"]["agent"] != new["roles"]["agent"]:
        diffs.append("agent role")
    # a run without an LLM user (only oracle/mute arms) is compatible with any user model
    if old_user and new_user and old_user != new_user:
        diffs.append("user role")
    for k in ("baseline", "max_turns", "final_fix"):
        if old.get(k) != new.get(k):
            diffs.append(k)
    for k in ("agent_system_prompt", "user_prompt_template", "personas", "matraix_user"):
        if old["sha256"].get(k) != new["sha256"].get(k):
            diffs.append(k)
    # keys added after mx1: compared only when the existing manifest has them
    if "behaviour_user" in old["sha256"] and old["sha256"]["behaviour_user"] != new["sha256"]["behaviour_user"]:
        diffs.append("behaviour_user")
    if "shared_first" in old and old["shared_first"] != new["shared_first"]:
        diffs.append("shared_first")
    old_cards, new_cards = old.get("cards", {}).get(key), new["cards"][key]
    if old_cards and new_cards and old_cards["sha256"] != new_cards["sha256"]:
        diffs.append(f"cards {key}")
    if diffs and not args.force:
        sys.exit(f"{path} differs from this invocation in: {', '.join(diffs)}. Use a new --out or --force.")
    changed = False
    if new_cards and not old_cards:
        old.setdefault("cards", {})[key] = new_cards
        changed = True
    if new_user and not old_user:
        old["roles"]["user"] = new_user
        old["prices"] = new["prices"]
        changed = True
    if changed:
        with open(path, "w") as f:
            json.dump(old, f, indent=2)
    return old


def run_job(job, ctx):
    arm, rep, case_id = job
    args = ctx["args"]
    case = ctx["cases"][case_id]
    keys = test_runner.TASK_CONFIG[args.mode]["keys"]
    incomplete_req = case.get(keys["input_prompt"], "")
    complete_req = case.get(keys["ground_truth"], "")
    folder = case_dir(args.out, arm, rep, args.domain, ctx["agent_cfg"].profile, args.mode, case_id)
    os.makedirs(folder, exist_ok=True)
    run_path = os.path.join(folder, "persona_run.json")
    prev = None
    if os.path.exists(run_path):
        with open(run_path) as f:
            prev = json.load(f)
    prev_attempts = prev.get("attempts", 1) if prev else 0
    spend = ctx["spend"]
    if spend.exhausted():
        # not an attempt: an earlier record (e.g. an error) is left as is so its attempt count stands
        rec = {"arm": arm, "rep": rep, "case_id": case_id, "domain": args.domain, "mode": args.mode,
               "status": "skipped_budget", "attempts": prev_attempts, "error": None,
               "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        if prev is None:
            with open(run_path, "w") as f:
                json.dump(rec, f, indent=2)
        return rec
    log_file = os.path.join(folder, "conversation_log.txt")
    result_file = os.path.join(folder, "generated_prompt.txt")
    # a rerun after an error must not append to or reuse the failed attempt's files
    open(log_file, "w").close()
    if os.path.exists(result_file):
        os.remove(result_file)
    with open(os.path.join(folder, "complete_prompt.txt"), "w") as f:
        f.write(complete_req)

    bf_persona = ctx["bigfive"]["personas"].get(arm)
    mx_persona = ctx["matraix"]["personas"].get(arm)
    if arm in pu.DETERMINISTIC_ARMS:
        persona_text = None
    elif arm in ctx["behaviour"]["cards"]:
        persona_text = pu.bigfive_persona_text(pu.behaviour_persona(arm, ctx["bigfive"], ctx["behaviour"]))
    elif bf_persona is not None:
        persona_text = pu.bigfive_persona_text(bf_persona)
    elif mx_persona is not None:
        persona_text = mx_persona["block"]
    else:
        persona_text = pu.persona_text_for_arm(arm, ctx["personas"])
    user_cfg = None if arm in pu.DETERMINISTIC_ARMS else ctx["user_cfg"]
    trait = pu.bigfive_target(arm, ctx["bigfive"]) if bf_persona is not None else \
        (arm if arm in ctx["traits"] else None)
    record = {
        "arm": arm, "trait": trait,
        "persona_text_sha256": None if persona_text is None else pu.sha256_text(persona_text),
        "rep": rep, "case_id": case_id, "domain": args.domain, "mode": args.mode,
        "baseline": bool(args.baseline),
        "models": {"agent": ctx["agent_cfg"].as_dict(), "user": user_cfg.as_dict() if user_cfg else None},
        "turns": None, "cap_hit": None, "completed_by_agent": None,
        "final_attempts": None, "final_valid": None, "truth_retries": None,
        "deferred": None, "attempts": prev_attempts + 1,
        "status": "error", "error": None,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"), "finished": None,
        "usage": {"agent": None, "user": None},
    }
    agent, user_llm, user = None, None, None
    try:
        prompt_mode = "baseline" if args.baseline else None
        if args.shared_first:
            with open(shared_first_path(args.shared_first, case_id, rep)) as f:
                first = json.load(f)["response"]
            record["shared_first_sha256"] = pu.sha256_text(first)
            agent = PrimedAgent(ctx["agent_cfg"], prompt_mode, first_response=first)
        else:
            agent = ConvAgent(ctx["agent_cfg"], prompt_mode)
        user_llm = LLMService(user_cfg) if user_cfg else None
        cards = None
        if ctx["cards"] is not None:
            cards = ctx["cards"]["cases"].get(case_id, {}).get("cards")
        user = pu.make_user(arm, user_llm, incomplete_req, complete_req, log_file,
                            cards=cards, personas=ctx["personas"], bigfive=ctx["bigfive"],
                            matraix=ctx["matraix"], behaviour=ctx["behaviour"])
        result = test_runner.run_dialogue(agent, user, incomplete_req, args.max_turns, final_fix=True)
        record.update(turns=result.turns, cap_hit=result.cap_hit,
                      completed_by_agent=result.completed_by_agent, error=result.error,
                      final_attempts=result.final_attempts, final_valid=result.final_valid)
        if result.final_spec is not None:
            with open(result_file, "w") as f:
                f.write(result.final_spec)
        # LLMUserInterface swallows its own API failure and replies with a canned line
        user_failed = user_llm is not None and user_llm.failed_calls > 0
        if user_failed and record["error"] is None:
            record["error"] = "simulated user LLM call failed"
        record["status"] = "done" if (result.final_spec is not None and record["error"] is None) else "error"
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"

    agent_stats = agent.llm.get_statistics() if agent else {}
    user_stats = user_llm.get_statistics() if user_llm else {}
    record["usage"] = {"agent": agent_stats, "user": user_stats or None}
    prices = ctx["prices"]
    spend.add(pu.stats_cost(agent_stats, prices.get(ctx["agent_cfg"].model_version))
              + (pu.stats_cost(user_stats, prices.get(user_cfg.model_version)) if user_cfg else 0.0))
    test_runner.save_statistics(os.path.join(folder, "statistics.json"), agent_stats,
                                record["turns"] or 0, args.max_turns,
                                os.path.exists(result_file), args.mode)
    with open(os.path.join(folder, "user_statistics.json"), "w") as f:
        json.dump({"user_model": user_cfg.model_version if user_cfg else None,
                   "user_arm": arm, "user_llm_usage": user_stats}, f, indent=4)
    clar = user.clarifications if user is not None else []
    record["deferred"] = sum(bool(c.get("deferred")) for c in clar)
    record["truth_retries"] = getattr(user, "truth_retries", None)
    with open(os.path.join(folder, "clarifications.json"), "w") as f:
        json.dump([{"turn": k, "question": c["question"], "answer": c["answer"],
                    "deferred": bool(c.get("deferred"))} for k, c in enumerate(clar, 1)], f, indent=2)
    record["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(run_path, "w") as f:
        json.dump(record, f, indent=2)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run persona-trait arms on SciConvBench")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--mode", required=True, choices=list(pu.MODE_LABEL))
    parser.add_argument("--cases", required=True, help="Comma list or @file")
    parser.add_argument("--arms", required=True, help="Comma list: paper, neutral, <trait>, oracle, mute")
    parser.add_argument("--agent_model", required=True, choices=list(Config.LLM_PROFILES))
    parser.add_argument("--user_model", default=None, choices=list(Config.LLM_PROFILES))
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cards", default=None, help="Fact-card file (default persona/fact_cards/...)")
    parser.add_argument("--agent_temperature", type=float, default=None)
    parser.add_argument("--user_temperature", type=float, default=None)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--force", action="store_true", help="Continue despite a manifest mismatch")
    parser.add_argument("--shared_first", default=None,
                        help="Dir of saved first agent replies per case and rep (made if missing); every arm starts from it")
    parser.add_argument("--budget_usd", type=float, required=True,
                        help="Spend cap for this invocation; no new job starts once it is reached")
    args = parser.parse_args(argv)
    args._argv = argv

    personas = pu.load_personas()
    bigfive = pu.load_bigfive()
    matraix = pu.load_matraix()
    behaviour = pu.load_behaviour()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    valid = pu.valid_arms(personas, bigfive, matraix, behaviour)
    bad = [a for a in arms if a not in valid]
    if bad:
        sys.exit(f"Unknown arms {bad}; valid: {valid}")
    needs_user = any(a not in pu.DETERMINISTIC_ARMS for a in arms)
    if needs_user and not args.user_model:
        sys.exit("--user_model is required for LLM-user arms")

    cases = pu.load_cases(args.domain, args.mode)
    case_ids = parse_cases(args.cases)
    missing = [c for c in case_ids if c not in cases]
    if missing:
        sys.exit(f"Unknown case ids for {args.domain}/{args.mode}: {missing}")

    cards_path = args.cards or pu.default_cards_path(args.domain, args.mode)
    cards = load_cards(cards_path)
    if "oracle" in arms:
        if cards is None:
            sys.exit(f"oracle arm needs a fact-card file; not found: {cards_path}")
        no_cards = [c for c in case_ids if not cards["cases"].get(c, {}).get("cards")]
        if no_cards:
            sys.exit(f"No fact cards for cases {no_cards} in {cards_path}")

    agent_cfg = pu.role_config(args.agent_model, args.agent_temperature)
    user_cfg = pu.role_config(args.user_model, args.user_temperature) if needs_user else None
    models = sorted({c.model_version for c in (agent_cfg, user_cfg) if c is not None})
    prices = pu.fetch_prices(models)
    pu.require_prices(prices, models)
    check_or_write_manifest(args, agent_cfg, user_cfg, personas, cards_path if cards else None, bigfive, prices,
                            matraix)

    jobs = [(arm, rep, cid) for arm in arms for rep in range(1, args.reps + 1) for cid in case_ids]
    random.Random(JOB_SEED).shuffle(jobs)
    todo, skipped, capped = [], 0, []
    for job in jobs:
        run_path = os.path.join(case_dir(args.out, job[0], job[1], args.domain, agent_cfg.profile,
                                         args.mode, job[2]), "persona_run.json")
        if os.path.exists(run_path):
            with open(run_path) as f:
                old = json.load(f)
            if old.get("status") == "done":
                skipped += 1
                continue
            if old.get("attempts", 1) >= MAX_ATTEMPTS:
                capped.append(job)
                continue
        todo.append(job)
    print(f"{len(jobs)} jobs: {skipped} done, {len(capped)} capped, {len(todo)} to run")

    spend = pu.SpendTracker(args.budget_usd)
    ctx = {"args": args, "cases": cases, "cards": cards, "personas": personas, "bigfive": bigfive,
           "matraix": matraix, "behaviour": behaviour, "traits": set(pu.trait_ids(personas)), "agent_cfg": agent_cfg,
           "user_cfg": user_cfg, "prices": prices, "spend": spend}
    if args.shared_first:
        os.makedirs(args.shared_first, exist_ok=True)
        firsts = sorted({(cid, rep) for _, rep, cid in todo})
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            made = list(ex.map(lambda cr: make_shared_first(cr[0], cr[1], ctx), firsts))
        print(f"shared first replies: {made.count('made')} made, {made.count('exists')} reused, {made.count('failed')} failed")
    n_done = n_err = n_budget = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_job, job, ctx): job for job in todo}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                rec = fut.result()
                ok = rec["status"] == "done"
            except Exception as e:
                rec, ok = {"error": str(e)}, False
            if rec.get("status") == "skipped_budget":
                n_budget += 1
                continue
            n_done += ok
            n_err += not ok
            print(f"[{n_done + n_err}/{len(todo)}] {job}: {'done' if ok else 'error ' + str(rec.get('error'))}")
    print(f"finished: {n_done} done, {n_err} errors, {skipped} skipped")
    if n_budget:
        print(f"{n_budget} jobs not started: budget reached (status skipped_budget; rerun to continue)")
    if capped:
        print(f"{len(capped)} jobs not rerun (error after {MAX_ATTEMPTS} attempts):")
        for job in sorted(capped):
            print(f"  {job}")
    print(spend.summary())
    out = {"jobs": len(jobs), "skipped": skipped, "done": n_done, "errors": n_err}
    if n_budget:
        out["skipped_budget"] = n_budget
    return out


if __name__ == "__main__":
    main()
