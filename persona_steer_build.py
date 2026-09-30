import os
import re
import sys
import json
import time
import argparse
from typing import Dict, List, Any, Optional, Tuple

from config import Config
from utils import LLMService
import persona_users as pu

DOMAIN, MODE = "foam", "disambiguation"
CHOICE = {"mesh", "time_controls", "solver", "turbulence_model", "solver_settings"}
VARIANTS = ("fast", "rigorous")
PARSE_TRIES = 2
MIN_POINTS = 2
DEFAULT_OUT = os.path.join(pu.REPO_DIR, "persona", "steer_cards", "foam_disamb.json")

BANNED_WORDS = ("foam", "epsilon", "omega", "sst", "spalart", "les", "ras", "laminar", "piso", "simple",
                "pimple", "courant", "tolerance", "cells")

BUILDER_SYSTEM = """
        You prepare material for a study in which a simulated scientist steers an assistant towards a
        simulation setup. You are given an OpenFOAM simulation request in two versions: the
        [INCOMPLETE_REQUEST] the assistant sees, and the [COMPLETE_REQUEST] it was derived from. Some
        settings of the complete request are CHOICE POINTS: several setups would be valid there. Each is listed
        in [CHOICE_POINTS] with the passage(s) of the complete request that specify it.

        TASK 1: OPTIONS. For each choice point, write two concrete options that are both physically valid
        for THIS case (same geometry, physics and solver family as the complete request):
        - "fast": the cheaper, quicker option (coarser, looser, simpler, shorter).
        - "rigorous": the more accurate option (finer, tighter, higher fidelity, longer).
        Each option is one or two sentences written in the same style as the original passages, with
        concrete settings and values, so that an OpenFOAM engineer could set it up from that text alone.
        The two options must differ clearly. If no valid pair exists for a choice point (for example the
        case only works with one setting), skip it and give the reason.

        TASK 2: GOAL CARDS. Write two goal cards, 2-3 sentences each, in the first person, as the scientist
        who asked for this simulation. The scientist knows their physical problem but does not know
        OpenFOAM settings.
        - "fast": they want a quick check that the setup runs and shows the rough trend, it will run on a
          laptop, and they need it soon.
        - "rigorous": the numbers go into a grant report or a paper, a reviewer doubted the accuracy of
          earlier results, and runtime is no issue.
        The goal cards must not name any setting, number, solver, model or algorithm, and must not contain
        any digit. They say what the scientist needs, not how to set it up.

        **OUTPUT FORMAT:**
        You must output your final answer as a JSON object enclosed EXACTLY within <FINAL_JSON> tags, with
        one entry per choice point, keyed by its name.
        <FINAL_JSON>
        {
            "options": {
                "choice_point_name": {"fast": "One or two sentences.", "rigorous": "One or two sentences."},
                "other_choice_point": {"skip": "Reason no valid pair exists."}
            },
            "goals": {"fast": "Two or three sentences.", "rigorous": "Two or three sentences."}
        }
        </FINAL_JSON>
        """

BUILDER_USER = """
        <INCOMPLETE_REQUEST>
        {incomplete}
        </INCOMPLETE_REQUEST>

        <COMPLETE_REQUEST>
        {complete}
        </COMPLETE_REQUEST>

        <CHOICE_POINTS>
        {choice_block}
        </CHOICE_POINTS>
        """

REGEN_NOTE = """
        Your previous goal cards were:
        {goals}

        They failed these checks:
        {errors}

        Rewrite only the two goal cards so that they pass. Output
        <FINAL_JSON>
        {{"goals": {{"fast": "...", "rigorous": "..."}}}}
        </FINAL_JSON>
        """

SOLVE_SYSTEM = """
        A scientist asked for a simulation. You see their request, a note in which they describe what they
        need the results for, and for several settings of the simulation two options, A and B.
        For each setting, pick the option that fits what the scientist needs.

        **OUTPUT FORMAT:**
        You must output your final answer as a JSON object enclosed EXACTLY within <FINAL_JSON> tags, with
        one entry per setting, keyed by the setting's name, each "A" or "B".
        <FINAL_JSON>
        {"setting_name": "A"}
        </FINAL_JSON>
        """

SOLVE_USER = """
        <REQUEST>
        {incomplete}
        </REQUEST>

        <SCIENTIST_NOTE>
        {goal}
        </SCIENTIST_NOTE>

        <SETTINGS>
        {settings}
        </SETTINGS>
        """

PROMPT_SHA = {"builder": pu.sha256_text(BUILDER_SYSTEM + BUILDER_USER + REGEN_NOTE),
              "solvability": pu.sha256_text(SOLVE_SYSTEM + SOLVE_USER)}


# ---- deterministic parts ----

def parse_cases(spec: str) -> List[str]:
    if spec.startswith("@"):
        with open(spec[1:]) as f:
            spec = f.read()
    return [c.strip() for c in spec.replace("\n", ",").split(",") if c.strip()]


def split_cards(cards: List[dict]) -> Tuple[List[dict], List[dict]]:
    return [c for c in cards if c["item"] in CHOICE], [c for c in cards if c["item"] not in CHOICE]


def eligibility(rec: Optional[dict]) -> Optional[str]:
    """None if the fact-card record is usable, else the drop reason."""
    if not rec or rec.get("status") != "ok":
        return f"dropped: fact cards status {rec.get('status') if rec else 'missing'}"
    choice, facts = split_cards(rec["cards"])
    if len(choice) < MIN_POINTS:
        return f"dropped: {len(choice)} choice cards"
    if not facts:
        return "dropped: no fact cards"
    return None


def collapse(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def make_notes(complete: str, choice_cards: List[dict], fact_cards: List[dict]) -> Tuple[str, List[str]]:
    notes = complete
    for c in choice_cards:
        for s in c["spans"]:
            notes = notes.replace(s, "")
    notes = collapse(notes)
    errors = []
    for c in fact_cards:
        for s in c["spans"]:
            if not pu.is_verbatim(s, notes):
                errors.append(f"fact span of {c['item']} lost: {s!r}")
    for c in choice_cards:
        for s in c["spans"]:
            if pu.is_verbatim(s, notes):
                errors.append(f"choice span of {c['item']} still present: {s!r}")
    return notes, errors


def split_camel(word: str) -> List[str]:
    return re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", word)


def is_technical(tok: str) -> bool:
    return bool(re.search(r"\d", tok) or re.search(r"[a-z][A-Z]", tok) or re.search(r"[A-Z][a-z]+[A-Z]", tok))


def option_tokens(options: Dict[str, dict]) -> List[str]:
    toks = set()
    for opt in options.values():
        for v in VARIANTS:
            for tok in re.findall(r"[A-Za-z0-9_]+", opt.get(v, "")):
                if is_technical(tok):
                    toks.add(tok)
    return sorted(toks)


def lint_goals(goals: Any, options: Dict[str, dict]) -> List[str]:
    if not isinstance(goals, dict):
        return ["goals missing"]
    errors = []
    tech = option_tokens(options)
    for v in VARIANTS:
        text = goals.get(v)
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{v} card missing")
            continue
        if re.search(r"\d", text):
            errors.append(f"{v} card contains a digit")
        words = {p.lower() for w in re.findall(r"[A-Za-z0-9_]+", text) for p in [w] + split_camel(w)}
        low = text.lower()
        for b in BANNED_WORDS:
            if b in words or (b == "foam" and b in low):
                errors.append(f"{v} card contains banned word {b!r}")
        for tok in tech:
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(tok.lower())}(?![A-Za-z0-9_])", low):
                errors.append(f"{v} card contains the option token {tok!r}")
    return errors


def clean_options(raw: Any, entities: List[str]) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """Kept {entity: {fast, rigorous}} and skipped {entity: reason}."""
    raw = raw if isinstance(raw, dict) else {}
    kept, skipped = {}, {}
    for e in entities:
        o = raw.get(e)
        if not isinstance(o, dict):
            skipped[e] = "missing in builder output"
        elif o.get("skip") and not (o.get("fast") and o.get("rigorous")):
            skipped[e] = str(o["skip"])
        elif all(isinstance(o.get(v), str) and o[v].strip() for v in VARIANTS):
            if pu.norm_ws(o["fast"]) == pu.norm_ws(o["rigorous"]):
                skipped[e] = "fast and rigorous options identical"
            else:
                kept[e] = {v: o[v].strip() for v in VARIANTS}
        else:
            skipped[e] = "fast or rigorous option missing"
    return kept, skipped


def solve_orders() -> List[Dict[str, str]]:
    """Label -> variant, for the two presentation orders."""
    return [{"A": "fast", "B": "rigorous"}, {"A": "rigorous", "B": "fast"}]


def keep_points(picks: Dict[str, List[Dict[str, Optional[str]]]], entities: List[str]) -> List[str]:
    """picks[variant] = one {entity: picked variant or None} per order; kept iff the target is picked every time."""
    return [e for e in entities if all(p.get(e) == v for v in VARIANTS for p in picks[v])]


# ---- LLM parts ----

def invoke_parsed(llm, user_prompt: str, system_prompt: str) -> Tuple[Optional[dict], str]:
    text = ""
    for _ in range(PARSE_TRIES):
        text = llm.invoke(user_prompt, system_prompt)
        parsed = pu.parse_final_json(text)
        if parsed is not None:
            return parsed, text
    return None, text


def builder_prompt(case: dict, choice_cards: List[dict]) -> str:
    block = "\n".join(f"- {c['item']}: " + " | ".join(c["spans"]) for c in choice_cards)
    return BUILDER_USER.format(incomplete=case["incomplete_user_req"], complete=case["complete_user_req"],
                               choice_block=block)


def solve_prompt(incomplete: str, goal: str, options: Dict[str, dict], order: Dict[str, str]) -> str:
    lines = []
    for e, opt in options.items():
        lines.append(f"{e}:")
        for lab in ("A", "B"):
            lines.append(f"  {lab}. {opt[order[lab]]}")
    return SOLVE_USER.format(incomplete=incomplete, goal=goal, settings="\n".join(lines))


def solvability(llm, incomplete: str, goals: Dict[str, str], options: Dict[str, dict]) -> dict:
    picks, raw = {}, {}
    for v in VARIANTS:
        picks[v], raw[v] = [], []
        for order in solve_orders():
            parsed, _ = invoke_parsed(llm, solve_prompt(incomplete, goals[v], options, order), SOLVE_SYSTEM)
            parsed = parsed or {}
            raw[v].append({"order": order, "answer": parsed})
            picks[v].append({e: order.get(str(parsed.get(e, "")).strip().upper()) for e in options})
    kept = keep_points(picks, list(options))
    return {"picks": picks, "raw": raw, "kept": kept}


def build_case(llm, case: dict, cards_rec: Optional[dict]) -> dict:
    rec = {"status": None, "choice_points": [], "fact_entities": [], "notes_without_choices": "",
           "goals": {}, "checks": {}}
    reason = eligibility(cards_rec)
    if reason:
        rec["status"] = reason
        return rec
    choice_cards, fact_cards = split_cards(cards_rec["cards"])
    rec["fact_entities"] = [c["item"] for c in fact_cards]
    notes, note_errors = make_notes(case["complete_user_req"], choice_cards, fact_cards)
    rec["notes_without_choices"] = notes
    rec["checks"]["choice_items"] = [c["item"] for c in choice_cards]
    rec["checks"]["notes_errors"] = note_errors
    if note_errors:
        rec["status"] = "dropped: notes"
        return rec

    user_prompt = builder_prompt(case, choice_cards)
    parsed, raw = invoke_parsed(llm, user_prompt, BUILDER_SYSTEM)
    rec["checks"]["builder_raw"] = raw
    if parsed is None:
        rec["status"] = "dropped: builder parse"
        return rec
    options, skipped = clean_options(parsed.get("options"), [c["item"] for c in choice_cards])
    rec["checks"]["skipped_options"] = skipped
    goals = parsed.get("goals")
    lint = lint_goals(goals, options)
    rec["checks"]["lint"] = [lint]
    if lint:
        regen = REGEN_NOTE.format(goals=json.dumps(goals, indent=2), errors="\n".join(f"- {e}" for e in lint))
        parsed2, raw2 = invoke_parsed(llm, user_prompt + regen, BUILDER_SYSTEM)
        rec["checks"]["regen_raw"] = raw2
        goals = (parsed2 or {}).get("goals")
        lint = lint_goals(goals, options)
        rec["checks"]["lint"].append(lint)
    if isinstance(goals, dict):
        rec["goals"] = {v: goals.get(v) for v in VARIANTS}
    if lint:
        rec["status"] = "dropped: lint"
        return rec
    if len(options) < MIN_POINTS:
        rec["status"] = "dropped: options"
        return rec

    sol = solvability(llm, case["incomplete_user_req"], rec["goals"], options)
    rec["checks"]["solvability"] = sol
    spans = {c["item"]: c["spans"] for c in choice_cards}
    rec["choice_points"] = [{"entity": e, "original_spans": spans[e], **options[e]} for e in sol["kept"]]
    rec["status"] = "ok" if len(sol["kept"]) >= MIN_POINTS else "dropped: solvability"
    return rec


def write_json_atomic(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build steering cards (choice options + goal cards) for foam disambiguation")
    parser.add_argument("--cases", required=True, help="Comma-separated case ids or @file, walked in order")
    parser.add_argument("--n", type=int, required=True, help="Stop once this many cases are ok")
    parser.add_argument("--model", required=True, choices=list(Config.LLM_PROFILES))
    parser.add_argument("--budget_usd", type=float, required=True,
                        help="Spend cap for this invocation; no new case is queried once it is reached")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--cards", default=pu.default_cards_path(DOMAIN, MODE))
    args = parser.parse_args(argv)

    cases = pu.load_cases(DOMAIN, MODE)
    wanted = parse_cases(args.cases)
    unknown = [c for c in wanted if c not in cases]
    if unknown:
        sys.exit(f"Unknown case ids: {unknown}")
    with open(args.cards) as f:
        cards = json.load(f)["cases"]
    items = sorted({c["item"] for cid in wanted for c in (cards.get(cid) or {}).get("cards", [])})
    print("card items in requested cases:", ", ".join(f"{i}{'*' if i in CHOICE else ''}" for i in items),
          "(* = choice)")

    role = pu.role_config(args.model)
    meta = {"model": role.as_dict(), "created": time.strftime("%Y-%m-%d %H:%M:%S"), "git": pu.git_state(),
            "prompt_sha256": PROMPT_SHA, "cards_file": os.path.relpath(os.path.abspath(args.cards), pu.REPO_DIR),
            "cards_sha256": pu.sha256_file(args.cards), "choice_items": sorted(CHOICE),
            "spend": {"invocations": []}}
    data = {"meta": meta, "cases": {}}
    if os.path.exists(args.out):
        with open(args.out) as f:
            data = json.load(f)
        old = data["meta"]
        for k in ("prompt_sha256", "cards_sha256", "model"):
            if old.get(k) != meta[k]:
                sys.exit(f"{args.out}: {k} differs from the current one; refusing to mix cards")

    prices = pu.fetch_prices([role.model_version])
    pu.require_prices(prices, [role.model_version])
    spend = pu.SpendTracker(args.budget_usd)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_ok, n_budget = 0, False
    for cid in wanted:
        if n_ok >= args.n:
            break
        if cid in data["cases"]:
            n_ok += data["cases"][cid]["status"] == "ok"
            continue
        if spend.exhausted():
            n_budget = True
            break
        llm = LLMService(role)
        try:
            rec = build_case(llm, cases[cid], cards.get(cid))
        except Exception as e:
            print(f"{cid}: LLM call failed: {type(e).__name__}: {e}")
            spend.add(pu.stats_cost(llm.get_statistics(), prices[role.model_version]))
            continue
        usage = llm.get_statistics()
        rec["checks"]["usage"] = usage
        spend.add(pu.stats_cost(usage, prices[role.model_version]))
        data["cases"][cid] = rec
        n_ok += rec["status"] == "ok"
        write_json_atomic(args.out, data)
        print(f"{cid}: {rec['status']}")
    data["meta"]["spend"]["invocations"].append({"created": meta["created"], "budget_usd": args.budget_usd,
                                                 "spent_usd": spend.spent})
    write_json_atomic(args.out, data)
    if n_budget:
        print("budget reached before --n ok cases (rerun to continue)")
    elif n_ok < args.n:
        print(f"only {n_ok} ok cases in the requested list")
    print(spend.summary())
    counts = {}
    for rec in data["cases"].values():
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
    print(f"{args.out}: {counts}")
    return data


if __name__ == "__main__":
    main()
