import os
import re
import sys
import csv
import glob
import json
import time
import random
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from config import Config
from utils import LLMService
import main as agent_main
import persona_users as pu

DEFAULT_OUT = "/n/holylabs/kempner_sompolinsky_lab/Everyone/jorin/benchmarks/runs/tone1"
NEUTRAL_ARMS = ("bf_neutral", "bf_neutral_b", "bf_neutral_c", "bf_neutral_d", "bf_neutral_e", "bf_neutral_f")
CAL_ARM = "bf_low_A"
DOMAIN, MODE = "foam", "disambiguation"
STYLES = ("para", "lowA")
PARSE_TRIES = 3
PAIR_TYPES = {"orig_vs_lowA": ("orig", "lowA"), "orig_vs_para": ("orig", "para"),
              "lowA_vs_real_lowA": ("lowA", "real_lowA")}
SOFTEN = re.compile(r"sorry|apolog|understand|thank|appreciate|I know|just to confirm", re.I)
TECH = re.compile(r"[A-Za-z][A-Za-z0-9_\-\.]*[A-Za-z0-9]")
T_BINS = (("1-2", 1, 2), ("3-5", 3, 5), ("6+", 6, 10 ** 9))


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path, data):
    with open(path + ".tmp", "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(path + ".tmp", path)


def read_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def chat_dir(run, arm, model, case_id):
    return os.path.join(run, arm, "rep1", DOMAIN, f"{model}_{pu.MODE_LABEL[MODE]}", case_id)


def load_chat(folder):
    """(persona_run, clarifications) of a done chat with >= 1 clarification, else None."""
    run = read_json(os.path.join(folder, "persona_run.json"))
    clar = read_json(os.path.join(folder, "clarifications.json"))
    if not run or run.get("status") != "done" or not clar:
        return None
    return run, clar


def slot_labels(n, completed_by_agent):
    out = []
    for t in range(1, n + 1):
        move = "ask" if t < n else ("stop" if completed_by_agent else "cap")
        out.append({"t": t, "recorded_move": move, "excluded": move == "cap"})
    return out


def build_context(incomplete_req, questions, answers, t):
    """The string ConvAgent.chat passes to the LLM after the scientist's t-th reply."""
    lines = [f"User: {incomplete_req}"]
    for q, a in zip(questions[:t], answers[:t]):
        lines += [f"Analyst: {q}", f"User: {a}"]
    return "\n".join(lines)


def tech_tokens(text):
    return {w for w in TECH.findall(text or "")
            if "_" in w or any(c.isdigit() for c in w) or any(c.isupper() for c in w[1:])}


def fact_check(orig, new):
    """(ok, missing, added): numbers and technical tokens must be the same sets."""
    a = set(pu.NUM.findall(orig or "")) | tech_tokens(orig)
    b = set(pu.NUM.findall(new or "")) | tech_tokens(new)
    missing, added = sorted(a - b), sorted(b - a)
    return not missing and not added, missing, added


def open_manifest(out, sub, argv, extra):
    path = os.path.join(out, "tone_manifest.json")
    man = read_json(path) or {"invocations": []}
    man["invocations"].append({"subcommand": sub, "argv": argv, "git": pu.git_state(),
                               "created": time.strftime("%Y-%m-%d %H:%M:%S"), **extra})
    write_json(path, man)
    return path, man


def close_manifest(path, man, spend):
    man["invocations"][-1]["spent_usd"] = spend.spent
    write_json(path, man)


def prices_for(cfg):
    prices = pu.fetch_prices([cfg.model_version])
    pu.require_prices(prices, [cfg.model_version])
    return prices


def run_pool(fn, jobs, workers, label):
    n_err = n_budget = n_done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fn, j): j for j in jobs}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                n_budget += r == "skipped_budget"
                n_done += r == "done"
            except Exception as e:
                n_err += 1
                print(f"{label} failed for {futures[fut]}: {type(e).__name__}: {e}")
    print(f"{label}: {n_done} done, {n_err} errors" + (f", {n_budget} not started: budget reached" if n_budget else ""))
    return {"done": n_done, "errors": n_err, "skipped_budget": n_budget}


# ---------------------------------------------------------------- select

def cmd_select(args):
    cases = pu.load_cases(DOMAIN, MODE)
    present = sorted({os.path.basename(p) for arm in NEUTRAL_ARMS
                      for p in glob.glob(chat_dir(args.src_run, arm, args.model, "case_*"))})
    picks = {}
    for case_id in present:
        cands = [arm for arm in NEUTRAL_ARMS if load_chat(chat_dir(args.src_run, arm, args.model, case_id))]
        if cands and case_id in cases:
            picks[case_id] = random.Random(f"{args.seed}|{case_id}").choice(cands)
    if args.cases:
        chosen = [c.strip() for c in args.cases.split(",") if c.strip()]
        bad = [c for c in chosen if c not in picks]
        if bad:
            sys.exit(f"no eligible neutral chat for {bad}")
    else:
        eligible = sorted(picks)
        chosen = sorted(random.Random(args.seed).sample(eligible, min(args.n_cases, len(eligible))))
    chats = []
    for case_id in chosen:
        arm = picks[case_id]
        folder = chat_dir(args.src_run, arm, args.model, case_id)
        run, clar = load_chat(folder)
        cal = chat_dir(args.src_run, CAL_ARM, args.model, case_id)
        cal_clar = read_json(os.path.join(cal, "clarifications.json"))
        chats.append({"case_id": case_id, "arm": arm, "folder": folder,
                      "incomplete_req": cases[case_id]["incomplete_user_req"],
                      "clarifications": [[c["question"], c["answer"]] for c in clar],
                      "completed_by_agent": bool(run.get("completed_by_agent")), "cap_hit": bool(run.get("cap_hit")),
                      "slots": slot_labels(len(clar), bool(run.get("completed_by_agent"))),
                      "bf_low_A_folder": cal if cal_clar else None})
    manifest_path = os.path.join(args.src_run, "manifest.json")
    sel = {"src_run": args.src_run, "manifest_path": manifest_path,
           "agent_system_prompt_sha256": ((read_json(manifest_path) or {}).get("sha256") or {}).get("agent_system_prompt"),
           "model": args.model, "seed": args.seed, "n_cases": args.n_cases, "cases_arg": args.cases,
           "n_eligible_cases": len(picks), "created": time.strftime("%Y-%m-%d %H:%M:%S"), "chats": chats}
    os.makedirs(args.out, exist_ok=True)
    write_json(os.path.join(args.out, "selection.json"), sel)
    n_slots = sum(len(c["slots"]) for c in chats)
    n_excl = sum(s["excluded"] for c in chats for s in c["slots"])
    print(f"selection.json: {len(chats)} chats of {len(picks)} eligible cases, {n_slots} slots ({n_excl} excluded cap)")
    return sel


def load_selection(out):
    sel = read_json(os.path.join(out, "selection.json"))
    if sel is None:
        sys.exit(f"no selection.json in {out}; run select first")
    return sel


# ---------------------------------------------------------------- rewrite

REWRITE_SYSTEM = """
        You rewrite the replies a scientist wrote in a chat with an AI assistant. You change only how the replies
        are worded, never what they say.
        """

REWRITE_RULES = """Rules:
- Rewrite ONLY the wording of each reply.
- Keep every value, unit, name and setting, every file, solver and model name, every permission to choose (for example "use whatever is standard"), and every uncertainty.
- Add nothing: no new facts or values, no instructions or requests (never tell the assistant to stop asking, to hurry up, or to read the request), no claim that something was already said or given, no refusals, no questions back that ask for anything (a rhetorical remark that asks for nothing is fine).
- Drop nothing.
- Keep numbers and technical names character-identical (for example 9.81 stays 9.81, buoyantFoam stays buoyantFoam, p_rgh stays p_rgh)."""

PARA_STYLE = ("Reword each reply as the same scientist would say it on another day: different word choice and "
              "sentence order, but the same register as the original: plain conversational English, contractions "
              "where the original has them, the same friendliness, no more formal, no longer or shorter than about "
              "±20%.")

# smoke2: "blunt, curt" alone gave trimmed copies of the neutral replies (the judge still preferred the real
# bf_low_A chat 5/6), so the attitude has to be asked for explicitly, as remarks that carry no task information
LOWA_STYLE_TAIL = ("Rewrite each reply in this person's voice. Every reply must clearly sound cold, blunt, critical, "
                   "impatient and distrustful: someone reading one reply alone should notice the attitude at once. "
                   "A reply that only drops a few words is wrong. Put the attitude in remarks that carry no "
                   "information about the task: irritation (\"Honestly.\", \"Fine.\"), criticism of the question or "
                   "the assistant (\"Is that really so hard to work out?\", \"I'd expect you to know that.\"), open "
                   "distrust (\"I doubt you'll get this right, but here it is.\"), sarcasm, cold short sentences. "
                   "The information must stay exactly the same; the rules below still hold, in particular: never "
                   "claim you already said or gave something, never tell the assistant how or whether to ask, never "
                   "refuse. Do not pad with anything that is not attitude.")

# tone3: lowA puts much of its attitude in remarks about the question or the assistant ("I'd expect you to know
# that"), which can read as "stop asking". lowAc keeps the hostility and forbids every remark of that kind.
LOWAC_STYLE = ("Rewrite each reply in the voice of a cold, irritable, unfriendly person. Every reply must clearly "
               "sound annoyed and unfriendly: someone reading one reply alone should notice the attitude at once. A "
               "reply that only drops a few words is wrong. Put the attitude only in how the reply sounds: irritated "
               "interjections (\"Ugh.\", \"Fine, whatever.\", \"*sigh*\"), a flat cold tone, no greeting, thanks "
               "or softening, grudging or sarcastic phrasing (\"Here, take it.\", \"Great. Just great.\", "
               "\"Wonderful.\"), clipped contemptuous sentences. It should sound as hostile as a rude person, not merely "
               "tired. Vary the wording from reply to reply; do not open or close every reply the same way. Never comment on the question or on the assistant: no remark that the question is "
               "unnecessary, basic, obvious or easy to work out, that the assistant should know or should have known "
               "something, that the information was in the request or already given, about the assistant's "
               "competence, about how many questions it asks, about wanting the conversation to end, or about the scientist's time "
               "or patience. So never write things like \"obviously\", \"of course\", \"again\", \"that's your job\", "
               "\"get on with it\", \"anything else?\", \"is that all?\", \"are we done?\", \"happy now?\" or "
               "\"more questions\". The "
               "information must stay exactly the same; the rules below still hold. Do not pad with anything that is "
               "not attitude.")

# lowAc remarks the prompt alone did not stop in smoke2 (10-03); a hit absent from the original reply fails the
# rewrite (one retry, then the original goes through)
LOWAC_FORBIDDEN = [r"\bobviously\b", r"\bof course\b", r"\bclearly\b", r"\balready\b", r"\bagain\b",
                   r"\bare we\b", r"\bwe'?re (done|finished)\b", r"\banything else\b", r"\bis that (all|it|everything)\b",
                   r"\bget on with\b", r"\bproceed\b", r"\bhurry\b", r"\byour job\b", r"\byou'?re here for\b",
                   r"\bwhat you'?re for\b", r"\byou should\b", r"\bshould (have|'ve)\b", r"\byou'?d (know|think)\b",
                   r"\bexpect you\b", r"\bhow many\b", r"\banother (question|detail|one|thing)\b",
                   r"\bmore (questions|parameters|details)\b", r"\b(keep|stop) asking\b", r"\bsatisfied\b",
                   r"\bhappy now\b", r"\bi (said|told you|gave you|mentioned)\b", r"\bwork it out\b",
                   r"\bfigure it out\b", r"\bdone yet\b", r"\b(sufficient|enough)\?", r"\bjust do it\b",
                   r"\bdragging\b"]


def lowac_remarks(original, new):
    return [p for p in LOWAC_FORBIDDEN if re.search(p, new or "", re.I) and not re.search(p, original or "", re.I)]


REWRITE_USER = """
The scientist's original request to the assistant:
<request>
{request}
</request>

The conversation that followed, numbered. Q is the assistant's message, A is the scientist's reply:
<conversation>
{conversation}
</conversation>

Style:
{style}

{rules}

Output only a JSON object {{"replies": [...]}} with exactly {n} strings: one rewritten reply per original reply A, in the same order.
"""

EQUIV_SYSTEM = """
        You compare a scientist's original replies with rewritten versions of them. For each numbered pair, decide
        whether the rewritten reply carries the same information as the original.
        Same information means: the same values, settings and names; the same permissions to choose or deferrals;
        the same uncertainty; nothing added (no fact, instruction, request, claim of having already said
        something, refusal or question that the original does not have); nothing dropped.
        Differences in tone, politeness, rudeness, length and wording do NOT count, and neither do remarks that
        only express attitude and carry no task information (irritation, criticism of the question, distrust,
        sarcasm, rhetorical questions that ask for nothing).

        Output only a JSON object with one entry per pair:
        {"pairs": [{"i": 1, "same_info": true, "diff": ""}]}
        "diff" says what information differs; leave it empty when same_info is true.
        """

EQUIV_USER = """
<pairs>
{pairs}
</pairs>
"""


def lowa_style(bigfive=None):
    p = (bigfive or pu.load_bigfive())["personas"][CAL_ARM]
    ex = "\n".join(f'Message: "{e["q"]}"\nReply: "{e["a"]}"' for e in p.get("examples") or [])
    return ("The person, described as if speaking to them:\n" + p["sheet"] + "\n" + p["appraisal"] +
            "\n\nExamples of how this person writes (style only):\n" + ex + "\n\n" + LOWA_STYLE_TAIL)


def style_text(style, bigfive=None):
    if style == "lowAc":
        return LOWAC_STYLE
    return PARA_STYLE if style == "para" else lowa_style(bigfive)


def rewrite_prompt(chat, style_instr):
    conv = []
    for i, (q, a) in enumerate(chat["clarifications"], 1):
        conv += [f"[{i}] Q: {q}", f"[{i}] A: {a}"]
    return REWRITE_USER.format(request=chat["incomplete_req"], conversation="\n".join(conv), style=style_instr,
                               rules=REWRITE_RULES, n=len(chat["clarifications"]))


def parse_replies(raw, n):
    parsed = pu.parse_final_json(raw or "")
    reps = (parsed or {}).get("replies")
    if isinstance(reps, list) and len(reps) == n and all(isinstance(r, str) for r in reps):
        return reps
    return None


def equiv_judge(llm, originals, rewrites):
    pairs = "\n\n".join(f"[{i}] ORIGINAL: {o}\n[{i}] REWRITTEN: {r}"
                        for i, (o, r) in enumerate(zip(originals, rewrites), 1))
    prompt = EQUIV_USER.format(pairs=pairs)
    for _ in range(PARSE_TRIES):
        parsed = pu.parse_final_json(llm.invoke(prompt, EQUIV_SYSTEM))
        if parsed is not None and isinstance(parsed.get("pairs"), list):
            return {int(p["i"]): p for p in parsed["pairs"] if isinstance(p, dict) and str(p.get("i", "")).isdigit()}
    return {}


def check_replies(llm, originals, replies):
    if replies is None:
        return [{"i": i, "code_ok": False, "missing": [], "added": [], "judge_same": False,
                 "judge_diff": "rewrite output unparseable or wrong number of replies"}
                for i in range(1, len(originals) + 1)]
    verdicts = equiv_judge(llm, originals, replies)
    checks = []
    for i, (o, r) in enumerate(zip(originals, replies), 1):
        ok, missing, added = fact_check(o, r)
        v = verdicts.get(i)
        checks.append({"i": i, "code_ok": ok, "missing": missing, "added": added,
                       "judge_same": bool(v and v.get("same_info") is True),
                       "judge_diff": "no verdict from judge" if v is None else str(v.get("diff") or "")})
    return checks


def failures(checks, replies):
    out = []
    for c in checks:
        if c["code_ok"] and c["judge_same"]:
            continue
        mine = f' (your version: "{replies[c["i"] - 1]}")' if replies else ""
        why = []
        if c["missing"]:
            why.append(f"numbers or names missing or changed: {', '.join(c['missing'])}")
        if c["added"]:
            why.append(f"numbers or names added: {', '.join(c['added'])}")
        if not c["judge_same"]:
            why.append(f"information differs: {c['judge_diff']}")
        out.append(f"Reply {c['i']}{mine}: " + "; ".join(why))
    return out


def rewrite_one(chat, style, ctx):
    path = os.path.join(ctx["out"], "rewrites", f"{chat['case_id']}__{style}.json")
    if os.path.exists(path):
        return "exists"
    if ctx["spend"].exhausted():
        return "skipped_budget"
    llm = LLMService(ctx["cfg"])
    originals = [a for _, a in chat["clarifications"]]
    prompt = rewrite_prompt(chat, ctx["style_text"][style])
    raws = [llm.invoke(prompt, REWRITE_SYSTEM)]
    replies = parse_replies(raws[0], len(originals))
    first_checks = checks = check_replies(llm, originals, replies)
    fails = failures(checks, replies)
    attempts = 1
    if fails:
        attempts = 2
        retry = prompt + "\nYour previous rewrite had these problems. Fix them and follow every rule:\n" + \
            "\n".join(f"- {f}" for f in fails) + "\n"
        raws.append(llm.invoke(retry, REWRITE_SYSTEM))
        replies = parse_replies(raws[1], len(originals))
        checks = check_replies(llm, originals, replies)
        fails = failures(checks, replies)
    usage = llm.get_statistics()
    cost = pu.stats_cost(usage, ctx["prices"][ctx["cfg"].model_version])
    ctx["spend"].add(cost)
    write_json(path, {"case_id": chat["case_id"], "style": style, "status": "failed" if fails else "ok",
                      "attempts": attempts, "replies": replies, "checks": checks, "first_checks": first_checks,
                      "reasons": fails, "raw": raws, "usage": usage, "cost_usd": cost})
    return "done"


def cmd_rewrite(args, argv):
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    bad = [s for s in styles if s not in STYLES]
    if bad:
        sys.exit(f"unknown style(s) {bad}; valid: {STYLES}")
    sel = load_selection(args.out)
    cfg = pu.role_config(args.model, 0.0)
    prices = prices_for(cfg)
    spend = pu.SpendTracker(args.budget_usd)
    texts = {s: style_text(s) for s in styles}
    os.makedirs(os.path.join(args.out, "rewrites"), exist_ok=True)
    mpath, man = open_manifest(args.out, "rewrite", argv, {
        "model": cfg.as_dict(), "budget_usd": args.budget_usd, "prices": prices,
        "prompt_sha256": pu.sha256_text(REWRITE_SYSTEM + REWRITE_USER + REWRITE_RULES + EQUIV_SYSTEM + EQUIV_USER),
        "style_sha256": {s: pu.sha256_text(t) for s, t in texts.items()}})
    ctx = {"out": args.out, "cfg": cfg, "prices": prices, "spend": spend, "style_text": texts}
    jobs = [(c, s) for c in sel["chats"] for s in styles
            if not os.path.exists(os.path.join(args.out, "rewrites", f"{c['case_id']}__{s}.json"))]
    print(f"{len(jobs)} (chat, style) rewrites to do")
    res = run_pool(lambda j: rewrite_one(j[0], j[1], ctx), jobs, args.workers, "rewrite")
    close_manifest(mpath, man, spend)
    print(spend.summary())
    return res


def load_rewrite(out, case_id, style):
    return read_json(os.path.join(out, "rewrites", f"{case_id}__{style}.json"))


def ok_replies(out, chat, style):
    if style == "orig":
        return [a for _, a in chat["clarifications"]]
    rw = load_rewrite(out, chat["case_id"], style)
    return rw["replies"] if rw and rw.get("status") == "ok" else None


# ---------------------------------------------------------------- replay

def row_key(row):
    return tuple(row["key"])


class JsonlSink:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.done = {row_key(r) for r in read_jsonl(path)}

    def write(self, row):
        with self.lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.done.add(row_key(row))


def orig_subset(sel, n, seed):
    slots = sorted((c["case_id"], s["t"]) for c in sel["chats"] for s in c["slots"] if not s["excluded"])
    return set(random.Random(seed).sample(slots, min(n, len(slots))))


def replay_one(job, ctx):
    chat, slot, arm, answers = job
    if ctx["spend"].exhausted():
        return "skipped_budget"
    agent = agent_main.ConvAgent(ctx["cfg"], None)
    questions = [q for q, _ in chat["clarifications"]]
    context = build_context(chat["incomplete_req"], questions, answers, slot["t"])
    response = agent.llm.invoke(context, agent.system_prompt)
    u = pu.usage_cost(agent.llm.get_statistics(), ctx["prices"][ctx["cfg"].model_version])
    cost = u["cost_usd"] or 0.0
    ctx["spend"].add(cost)
    ctx["sink"].write({"key": [chat["case_id"], slot["t"], arm], "case_id": chat["case_id"], "t": slot["t"],
                       "arm": arm, "src_arm": chat["arm"], "recorded_move": slot["recorded_move"],
                       "move": "stop" if "[COMPLETE]" in response else "ask", "response": response,
                       "response_words": len(response.split()), "input_tokens": u["input_tokens"],
                       "output_tokens": u["output_tokens"], "token_source": u["token_source"], "cost_usd": cost})
    return "done"


def cmd_replay(args, argv):
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arms if a not in STYLES]
    if bad:
        sys.exit(f"unknown arm(s) {bad}; valid: {STYLES} (orig runs via --orig_slots)")
    sel = load_selection(args.out)
    cfg = pu.role_config(args.model)
    probe = agent_main.ConvAgent(cfg, None)
    want = ((read_json(sel["manifest_path"]) or {}).get("sha256") or {}).get("agent_system_prompt")
    have = pu.sha256_text(probe.system_prompt)
    if have != want:
        sys.exit(f"agent system prompt sha256 {have} != source manifest {want}; aborting")
    prices = prices_for(cfg)
    spend = pu.SpendTracker(args.budget_usd)
    sink = JsonlSink(os.path.join(args.out, "replay.jsonl"))
    orig = orig_subset(sel, args.orig_slots, args.seed)
    jobs = []
    for chat in sel["chats"]:
        for arm in arms + ["orig"]:
            answers = ok_replies(args.out, chat, arm)
            if answers is None:
                continue
            for s in chat["slots"]:
                if s["excluded"] or (arm == "orig" and (chat["case_id"], s["t"]) not in orig):
                    continue
                if (chat["case_id"], s["t"], arm) not in sink.done:
                    jobs.append((chat, s, arm, answers))
    mpath, man = open_manifest(args.out, "replay", argv, {
        "agent": cfg.as_dict(), "agent_system_prompt_sha256": have, "budget_usd": args.budget_usd,
        "prices": prices, "orig_slots": sorted(orig)})
    print(f"{len(jobs)} replay calls to do ({len(sink.done)} already in replay.jsonl)")
    ctx = {"cfg": cfg, "prices": prices, "spend": spend, "sink": sink}
    res = run_pool(lambda j: replay_one(j, ctx), jobs, args.workers, "replay")
    close_manifest(mpath, man, spend)
    print(spend.summary())
    return res


# ---------------------------------------------------------------- judge (first order)

JUDGE_SYSTEM = """
        You read two chats, A and B, between an AI assistant and a scientist. Each chat shows the assistant's
        messages and the scientist's replies. Judge only the scientist.
        Which scientist comes across as more disagreeable: colder, blunter, more critical, more distrustful?
        You must pick one.

        Output only a JSON object:
        {"more_disagreeable": "A", "why": "..."}
        """

JUDGE_USER = """
<CHAT_A>
{a}
</CHAT_A>

<CHAT_B>
{b}
</CHAT_B>

Which scientist comes across as more disagreeable: colder, blunter, more critical, more distrustful?
"""


def transcript(qa):
    return "\n\n".join(f"Assistant: {q}\nScientist: {a}" for q, a in qa)


def side_qa(out, chat, style):
    if style == "real_lowA":
        clar = read_json(os.path.join(chat["bf_low_A_folder"], "clarifications.json")) if chat.get("bf_low_A_folder") else None
        return [(c["question"], c["answer"]) for c in clar] if clar else None
    answers = ok_replies(out, chat, style)
    return None if answers is None else list(zip([q for q, _ in chat["clarifications"]], answers))


def judge_one(job, ctx):
    chat, pair, qa = job
    if ctx["spend"].exhausted():
        return "skipped_budget"
    x, y = PAIR_TYPES[pair]
    swap = random.Random(f"{ctx['seed']}|{chat['case_id']}|{pair}").random() < 0.5
    sides = {"A": y, "B": x} if swap else {"A": x, "B": y}
    llm = LLMService(ctx["cfg"])
    prompt = JUDGE_USER.format(a=transcript(qa[sides["A"]]), b=transcript(qa[sides["B"]]))
    answer = None
    for _ in range(PARSE_TRIES):
        answer = pu.parse_final_json(llm.invoke(prompt, JUDGE_SYSTEM))
        if answer is not None:
            break
    label = str((answer or {}).get("more_disagreeable", "")).strip().upper()
    usage = llm.get_statistics()
    u = pu.usage_cost(usage, ctx["prices"][ctx["cfg"].model_version])
    cost = u["cost_usd"] or 0.0
    ctx["spend"].add(cost)
    ctx["sink"].write({"key": [chat["case_id"], pair], "case_id": chat["case_id"], "pair_type": pair,
                       "sides": sides, "answer": answer, "winner": sides.get(label),
                       "usage": usage, "cost_usd": cost})
    return "done"


def cmd_judge(args, argv):
    sel = load_selection(args.out)
    cfg = pu.role_config(args.model, 0.0)
    prices = prices_for(cfg)
    spend = pu.SpendTracker(args.budget_usd)
    sink = JsonlSink(os.path.join(args.out, "judge.jsonl"))
    jobs = []
    for chat in sel["chats"]:
        for pair, (x, y) in PAIR_TYPES.items():
            if (chat["case_id"], pair) in sink.done:
                continue
            qa = {s: side_qa(args.out, chat, s) for s in (x, y)}
            if all(qa.values()):
                jobs.append((chat, pair, qa))
    mpath, man = open_manifest(args.out, "judge", argv, {
        "judge": cfg.as_dict(), "budget_usd": args.budget_usd, "prices": prices,
        "prompt_sha256": pu.sha256_text(JUDGE_SYSTEM + JUDGE_USER)})
    print(f"{len(jobs)} judge calls to do ({len(sink.done)} already in judge.jsonl)")
    ctx = {"cfg": cfg, "prices": prices, "spend": spend, "sink": sink, "seed": args.seed}
    res = run_pool(lambda j: judge_one(j, ctx), jobs, args.workers, "judge")
    close_manifest(mpath, man, spend)
    print(spend.summary())
    return res


# ---------------------------------------------------------------- analyze

def wilson(k, n, z=1.96):
    if n == 0:
        return None, None, None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, mid - half, mid + half


def case_bootstrap(per_case, stat, n_boot=2000, seed=0):
    """per_case: (n_cases, k) array of per-case sums; stat maps the (k,) column sums to a value."""
    arr = np.asarray(per_case, dtype=float)
    if not len(arr):
        return None, None
    idx = np.random.default_rng(seed).integers(0, len(arr), (n_boot, len(arr)))
    vals = np.array([stat(arr[i].sum(axis=0)) for i in idx], dtype=float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return None, None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def fmt(x, d=3):
    return "NA" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def fmt_ci(p, lo, hi):
    return f"{fmt(p)} [{fmt(lo)}, {fmt(hi)}]"


def ratio(a, b):
    return a / b if b else float("nan")


def stop_stats(paired):
    """paired: list of (case_id, t, recorded_move, para_move, lowA_move)."""
    n = len(paired)
    sp = sum(p[3] == "stop" for p in paired)
    sl = sum(p[4] == "stop" for p in paired)
    sr = sum(p[2] == "stop" for p in paired)
    return {"n": n, "stop_para": sp, "stop_lowA": sl, "stop_rec": sr,
            "lowA_only": sum(p[4] == "stop" and p[3] == "ask" for p in paired),
            "para_only": sum(p[3] == "stop" and p[4] == "ask" for p in paired)}


def cmd_analyze(args):
    sel = load_selection(args.out)
    chats = {c["case_id"]: c for c in sel["chats"]}
    rewrites = {(rw["case_id"], rw["style"]): rw for rw in
                (read_json(p) for p in sorted(glob.glob(os.path.join(args.out, "rewrites", "*.json")))) if rw}
    replay = read_jsonl(os.path.join(args.out, "replay.jsonl"))
    judge = read_jsonl(os.path.join(args.out, "judge.jsonl"))
    L = ["# tone1 analysis", "", f"selection: {len(chats)} chats, "
         f"{sum(len(c['slots']) for c in chats.values())} slots, "
         f"{sum(s['excluded'] for c in chats.values() for s in c['slots'])} excluded (cap)", ""]

    L += ["## rewrites", "", "| style | ok | failed | code pass 1st | judge pass 1st | both 1st | code pass final | judge pass final | both final |",
          "|---|---|---|---|---|---|---|---|---|"]
    for st in STYLES:
        rws = [rw for (c, s), rw in rewrites.items() if s == st]
        row = [st, str(sum(r["status"] == "ok" for r in rws)), str(sum(r["status"] == "failed" for r in rws))]
        for key in ("first_checks", "checks"):
            ch = [c for r in rws for c in (r.get(key) or [])]
            n = len(ch)
            row += [f"{sum(c['code_ok'] for c in ch)}/{n}", f"{sum(c['judge_same'] for c in ch)}/{n}",
                    f"{sum(c['code_ok'] and c['judge_same'] for c in ch)}/{n}"]
        L.append("| " + " | ".join(row) + " |")
    both_ok = [cid for cid in chats if all((rewrites.get((cid, s)) or {}).get("status") == "ok" for s in STYLES)]
    L += ["", f"mean words per reply over {len(both_ok)} chats with ok rewrites in both styles:", ""]
    for st in ("orig",) + STYLES:
        w = [len(a.split()) for cid in both_ok for a in ok_replies(args.out, chats[cid], st)]
        L.append(f"- {st}: {fmt(float(np.mean(w)), 1) if w else 'NA'} (n replies = {len(w)})")

    L += ["", "## first order (pairwise disagreeableness judge)", ""]
    for pair, want, label in (("orig_vs_lowA", "lowA", "lowA judged more disagreeable than orig"),
                              ("orig_vs_para", "para", "para judged more disagreeable than orig"),
                              ("lowA_vs_real_lowA", "lowA", "lowA rewrite judged more disagreeable than real bf_low_A")):
        rows = [r for r in judge if r["pair_type"] == pair and r.get("winner")]
        k = sum(r["winner"] == want for r in rows)
        p, lo, hi = wilson(k, len(rows))
        n_bad = sum(r["pair_type"] == pair and not r.get("winner") for r in judge)
        L.append(f"- {label}: {fmt_ci(p, lo, hi)}, n = {len(rows)}" + (f" ({n_bad} unparsed)" if n_bad else ""))

    by_slot = {}
    for r in replay:
        by_slot.setdefault((r["case_id"], r["t"]), {})[r["arm"]] = r
    orig_rows = [r for r in replay if r["arm"] == "orig"]
    L += ["", "## reproduction (orig replay vs recorded move)", ""]
    agree = sum(r["move"] == r["recorded_move"] for r in orig_rows)
    L.append(f"- move == recorded_move: {agree}/{len(orig_rows)}"
             + (f" = {fmt(agree / len(orig_rows))}" if orig_rows else ""))
    L += ["", "| recorded \\ replay | ask | stop |", "|---|---|---|"]
    for rec in ("ask", "stop"):
        L.append(f"| {rec} | " + " | ".join(str(sum(r["recorded_move"] == rec and r["move"] == m for r in orig_rows))
                                          for m in ("ask", "stop")) + " |")

    paired = sorted((cid, t, arms["para"]["recorded_move"], arms["para"]["move"], arms["lowA"]["move"])
                    for (cid, t), arms in by_slot.items() if "para" in arms and "lowA" in arms)
    L += ["", "## main: stop rate per slot, lowA vs para (paired slots)", ""]

    def block(pp, with_ci):
        s = stop_stats(pp)
        n = s["n"]
        if not n:
            return ["- no paired slots"]
        out = [f"- n slots = {n}, n cases = {len({p[0] for p in pp})}",
               f"- stop rate: para {fmt(s['stop_para'] / n)} ({s['stop_para']}), lowA {fmt(s['stop_lowA'] / n)} "
               f"({s['stop_lowA']}), recorded {fmt(s['stop_rec'] / n)} ({s['stop_rec']})"]
        rt, df = ratio(s["stop_lowA"], s["stop_para"]), (s["stop_lowA"] - s["stop_para"]) / n
        if with_ci:
            cases = sorted({p[0] for p in pp})
            per_case = np.array([[sum(p[3] == "stop" for p in pp if p[0] == c),
                                  sum(p[4] == "stop" for p in pp if p[0] == c),
                                  sum(1 for p in pp if p[0] == c)] for c in cases])
            rlo, rhi = case_bootstrap(per_case, lambda v: ratio(v[1], v[0]))
            dlo, dhi = case_bootstrap(per_case, lambda v: (v[1] - v[0]) / v[2])
            out += [f"- ratio lowA/para stop counts: {fmt_ci(rt, rlo, rhi)}",
                    f"- difference of stop rates lowA - para: {fmt_ci(df, dlo, dhi)}"]
        else:
            out += [f"- ratio lowA/para stop counts: {fmt(rt)}", f"- difference of stop rates lowA - para: {fmt(df)}"]
        out.append(f"- discordant: lowA stop & para ask = {s['lowA_only']}, para stop & lowA ask = {s['para_only']}")
        return out

    L += ["case-cluster bootstrap 95% CI, 2000 draws, seed 0", ""] + block(paired, True)
    for name, lo, hi in T_BINS:
        L += ["", f"### t in {name}", ""] + block([p for p in paired if lo <= p[1] <= hi], False)

    L += ["", "## when both para and lowA ask", ""]
    both_ask = [arms for (cid, t), arms in by_slot.items()
                if "para" in arms and "lowA" in arms and arms["para"]["move"] == "ask" and arms["lowA"]["move"] == "ask"]
    L.append(f"- n slots = {len(both_ask)}")
    for st in STYLES:
        if both_ask:
            w = np.mean([a[st]["response_words"] for a in both_ask])
            k = sum(bool(SOFTEN.search(a[st]["response"])) for a in both_ask)
            L.append(f"- {st}: mean response words {fmt(float(w), 1)}, softening share {fmt(k / len(both_ask))} "
                     f"({k}/{len(both_ask)})")

    spend = {"rewrite": sum(rw.get("cost_usd") or 0.0 for rw in rewrites.values()),
             "replay": sum(r.get("cost_usd") or 0.0 for r in replay),
             "judge": sum(r.get("cost_usd") or 0.0 for r in judge)}
    L += ["", "## spend (sum of cost_usd in outputs)", ""]
    L += [f"- {k}: ${v:.4f}" for k, v in spend.items()] + [f"- total: ${sum(spend.values()):.4f}", ""]
    with open(os.path.join(args.out, "analysis.md"), "w") as f:
        f.write("\n".join(L))

    with open(os.path.join(args.out, "slots.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "src_arm", "t", "recorded_move", "excluded", "move_orig", "move_para", "move_lowA"])
        for cid in sorted(chats):
            c = chats[cid]
            for s in c["slots"]:
                arms = by_slot.get((cid, s["t"]), {})
                w.writerow([cid, c["arm"], s["t"], s["recorded_move"], s["excluded"]] +
                           [arms[a]["move"] if a in arms else "" for a in ("orig", "para", "lowA")])
    print("\n".join(L))
    return {"spend": spend, "paired": len(paired)}


# ---------------------------------------------------------------- CLI

def main(argv=None):
    argv_list = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(description="Tone-only replay of full1 neutral foam chats")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p, api):
        p.add_argument("--out", default=DEFAULT_OUT)
        p.add_argument("--model", default="bedrock_glm5", choices=list(Config.LLM_PROFILES))
        p.add_argument("--workers", type=int, default=8)
        if api:
            p.add_argument("--budget_usd", type=float, required=True,
                           help="Spend cap for this invocation; no new call starts once reached")

    p = sub.add_parser("select")
    common(p, False)
    p.add_argument("--src_run", required=True)
    p.add_argument("--n_cases", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cases", default=None, help="comma-separated case ids; overrides --n_cases")
    p = sub.add_parser("rewrite")
    common(p, True)
    p.add_argument("--styles", default="para,lowA")
    p = sub.add_parser("replay")
    common(p, True)
    p.add_argument("--arms", default="para,lowA")
    p.add_argument("--orig_slots", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("judge")
    common(p, True)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("analyze")
    common(p, False)
    args = parser.parse_args(argv_list)
    full_argv = ["persona_tone_replay.py"] + argv_list
    if args.cmd == "select":
        return cmd_select(args)
    if args.cmd == "rewrite":
        return cmd_rewrite(args, full_argv)
    if args.cmd == "replay":
        return cmd_replay(args, full_argv)
    if args.cmd == "judge":
        return cmd_judge(args, full_argv)
    return cmd_analyze(args)


if __name__ == "__main__":
    main()
