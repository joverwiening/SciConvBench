import os
import re
import sys
import glob
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import Config
from utils import LLMService
from llm_judge_disambiguation import LLMJudgeDisambiguation
from llm_judge_disambiguation_chat import LLMJudgeDisambiguationChat
from llm_judge_inconsistency import LLMJudgeInconsistency
from llm_judge_inconsistency_chat import LLMJudgeInconsistencyChat
from llm_judge_intent import LLMJudgeIntent
import make_paper_artifacts as mpa
import persona_users as pu

ATTRIBUTION_SYSTEM = """
        You label one turn of a clarification dialogue between an Agent and a User.
        The User holds the information on the [FACT_CARDS] (each has a card_id, the item it concerns and its text).
        You get the Agent's [QUESTION] and the User's [ANSWER].

        **YOUR TASK:**
        1. "asked": the card_ids whose information the QUESTION asks for, directly or as part of a broader question.
        2. "conveyed": one entry per card whose information the ANSWER gives to the Agent.
           - "level": "full" if the answer gives all of the card's information, "partial" if it gives only part of it.
             Confirming a value that the question proposes counts as conveying it.
           - "evidence": the shortest passage of the ANSWER that carries the information, copied VERBATIM from the ANSWER.
           Cards whose information is not in the answer are left out.

        **OUTPUT FORMAT:**
        You must output your final evaluation as a JSON object enclosed EXACTLY within <FINAL_JSON> tags.
        <FINAL_JSON>
        {
            "asked": ["card_id"],
            "conveyed": [{"card_id": "card_id", "level": "full", "evidence": "verbatim passage of the answer"}]
        }
        </FINAL_JSON>
        """

TRAIT_SYSTEM = """
        You check which behavioural traits a User's reply shows in a clarification dialogue.
        The [TRAITS] are written as instructions that a User may have been given.
        You get the Agent's [QUESTION] and the User's [ANSWER].

        **YOUR TASK:**
        For EACH trait decide whether the ANSWER shows it: "present" is 1 or 0.
        If present is 1, "evidence" is a passage of the ANSWER, copied VERBATIM, that shows the trait. Otherwise "evidence" is "".

        **OUTPUT FORMAT:**
        You must output your final evaluation as a JSON object enclosed EXACTLY within <FINAL_JSON> tags,
        with one entry for every trait id.
        <FINAL_JSON>
        {
            "traits": {"trait_id": {"present": 0, "evidence": ""}}
        }
        </FINAL_JSON>
        """

NATIVE_FILES = {
    "disambiguation": ["llm_judge_disambiguation_score.json", "llm_judge_disambiguation_chat_score.json",
                       "llm_judge_intent_score.json"],
    "inconsistency": ["llm_judge_inconsistency_score.json", "llm_judge_inconsistency_chat_score.json",
                      "llm_judge_intent_score.json"],
}


class RecordingLLM:
    """LLMService wrapper that keeps the raw responses so parse failures can be counted."""

    def __init__(self, service):
        self.service = service
        self.responses = []

    def invoke(self, user_prompt, system_prompt=None, max_retries=5):
        text = self.service.invoke(user_prompt, system_prompt, max_retries)
        self.responses.append(text)
        return text

    def get_statistics(self, name="LLM Service"):
        return self.service.get_statistics()

    def parse_failures(self):
        return sum(1 for r in self.responses if pu.parse_final_json(r) is None)


def make_judge(cls, cfg):
    judge = cls()
    judge.config = cfg
    judge.llm = RecordingLLM(LLMService(cfg))
    return judge


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def sum_usage(stats_list):
    out = {}
    for s in stats_list:
        for k, v in (s or {}).items():
            if isinstance(v, (int, float)):
                out[k] = out.get(k, 0) + v
    return out


# ---- native judges: per-case bodies of each judge's main() ----

def native_disambiguation(folder, row, cfg):
    fr = make_judge(LLMJudgeDisambiguation, cfg)
    chat = make_judge(LLMJudgeDisambiguationChat, cfg)
    intent = make_judge(LLMJudgeIntent, cfg)
    case_id = row["id"]
    gen_path = os.path.join(folder, "generated_prompt.txt")
    gt_path = os.path.join(folder, "complete_prompt.txt")
    chat_path = os.path.join(folder, "conversation_log.txt")
    missing_entities = row.get("missing_entities", [])

    gt_text = fr.read_file(gt_path)
    gen_text = fr.read_file(gen_path)
    result = fr.evaluate(gt_text, gen_text, missing_entities)
    score_str = result.get('score', '0/0')
    try:
        recovered_val = int(score_str.split('/')[0])
        total_val = int(score_str.split('/')[1])
    except (ValueError, IndexError):
        recovered_val = 0
        total_val = len(missing_entities)
    is_perfect = (recovered_val == total_val and total_val > 0)
    result["strict_score"] = "1/1" if is_perfect else "0/1"
    result["partial_score"] = f"{recovered_val}/{total_val}"
    write_json(os.path.join(folder, "llm_judge_disambiguation_score.json"), result)

    gen_text = chat.read_file(gen_path)
    raw_chat = chat.read_file(chat_path)
    parsed_chat = chat.parse_conversation_log(raw_chat)
    result = chat.evaluate(parsed_chat, missing_entities, gen_text)
    total_questions_asked = len(parsed_chat)
    total_missing_count = len(missing_entities)
    result["case_id"] = case_id
    result["total_questions_asked_in_chat"] = total_questions_asked
    result["total_missing_entities"] = total_missing_count
    relevant_qs = result.get("relevant_questions", {}).get("count", 0)
    asked_about = result.get("questions_about_missing_entities", {}).get("count", 0)
    raw_is_consistent = result.get("memory_consistency", {}).get("is_consistent", 0)
    try:
        m_consistency = float(raw_is_consistent)
    except (ValueError, TypeError):
        m_consistency = 0.0
    resolved_in_final = 0
    t_data = read_json(os.path.join(folder, "llm_judge_disambiguation_score.json")) or {}
    try:
        resolved_in_final = int(t_data.get("partial_score", "0/0").split('/')[0])
    except (ValueError, AttributeError):
        pass
    assumed_count = max(0, resolved_in_final - asked_about)
    c_recall = (asked_about / total_missing_count) if total_missing_count > 0 else 0.0
    c_precision = (relevant_qs / total_questions_asked) if total_questions_asked > 0 else 0.0
    a_rate = 1.0 - (assumed_count / total_missing_count) if total_missing_count > 0 else 0.0
    result["metrics"] = {
        "clarification_precision": round(max(0.0, min(1.0, c_precision)), 4),
        "clarification_recall": round(max(0.0, min(1.0, c_recall)), 4),
        "assumption_rate": round(max(0.0, min(1.0, a_rate)), 4),
        "memory_consistency": m_consistency
    }
    write_json(os.path.join(folder, "llm_judge_disambiguation_chat_score.json"), result)

    run_intent(intent, folder, case_id)
    return [fr, chat, intent]


def native_inconsistency(folder, row, cfg):
    fr = make_judge(LLMJudgeInconsistency, cfg)
    chat = make_judge(LLMJudgeInconsistencyChat, cfg)
    intent = make_judge(LLMJudgeIntent, cfg)
    case_id = row["id"]
    gen_path = os.path.join(folder, "generated_prompt.txt")
    chat_path = os.path.join(folder, "conversation_log.txt")
    inconsistencies_list = row.get("inconsistencies", [])

    gen_text = fr.read_file(gen_path)
    result = fr.evaluate(gen_text, inconsistencies_list, row.get("inconsistent_prompt", ""),
                         row.get("correct_prompt", ""))
    raw_breakdown = result.get("breakdown", [])
    clean_breakdown = []
    for i, inconsistency_text in enumerate(inconsistencies_list):
        found_match = None
        for item in raw_breakdown:
            if item.get("inconsistency_index", -1) == (i + 1):
                found_match = item
                break
        if found_match:
            clean_breakdown.append({
                "description": inconsistency_text,
                "resolved": found_match.get("resolved", False),
                "reason": found_match.get("reasoning", "No reasoning provided.")
            })
        else:
            clean_breakdown.append({
                "description": inconsistency_text,
                "resolved": False,
                "reason": "LLM failed to return a verdict for this specific item."
            })
    result["breakdown"] = clean_breakdown
    result["case_id"] = case_id
    case_resolved = sum(1 for item in clean_breakdown if item.get("resolved") is True)
    result["strict_score"] = result.get("score", "0/1")
    result["partial_score"] = f"{case_resolved}/{len(clean_breakdown)}"
    write_json(os.path.join(folder, "llm_judge_inconsistency_score.json"), result)

    gen_text = chat.read_file(gen_path)
    raw_chat = chat.read_file(chat_path)
    parsed_chat = chat.parse_conversation_log(raw_chat)
    result = chat.evaluate(parsed_chat, inconsistencies_list, gen_text)
    total_questions_asked = len(parsed_chat)
    total_inconsistencies = len(inconsistencies_list)
    result["case_id"] = case_id
    result["total_questions_asked_in_chat"] = total_questions_asked
    result["total_planted_inconsistencies"] = total_inconsistencies
    relevant_qs = result.get("relevant_questions", {}).get("count", 0)
    asked_about = result.get("questions_about_inconsistencies", {}).get("count", 0)
    warnings = result.get("explicit_warnings", {}).get("count", 0)
    raw_is_consistent = result.get("memory_consistency", {}).get("is_consistent", 0)
    try:
        m_consistency = float(raw_is_consistent)
    except (ValueError, TypeError):
        m_consistency = 0.0
    resolved_in_final = 0
    t_data = read_json(os.path.join(folder, "llm_judge_inconsistency_score.json")) or {}
    try:
        resolved_in_final = int(t_data.get("partial_score", "0/0").split('/')[0])
    except (ValueError, AttributeError):
        pass
    assumed_count = max(0, resolved_in_final - asked_about)
    c_recall = (asked_about / total_inconsistencies) if total_inconsistencies > 0 else 0.0
    c_precision = (relevant_qs / total_questions_asked) if total_questions_asked > 0 else 0.0
    e_detection = (warnings / total_inconsistencies) if total_inconsistencies > 0 else 0.0
    a_rate = 1.0 - (assumed_count / total_inconsistencies) if total_inconsistencies > 0 else 0.0
    result["metrics"] = {
        "clarification_precision": round(max(0.0, min(1.0, c_precision)), 4),
        "clarification_recall": round(max(0.0, min(1.0, c_recall)), 4),
        "assumption_rate": round(max(0.0, min(1.0, a_rate)), 4),
        "error_detection_rate": round(max(0.0, min(1.0, e_detection)), 4),
        "memory_consistency": m_consistency
    }
    write_json(os.path.join(folder, "llm_judge_inconsistency_chat_score.json"), result)

    run_intent(intent, folder, case_id)
    return [fr, chat, intent]


def run_intent(intent, folder, case_id):
    gen_text = intent.read_file(os.path.join(folder, "generated_prompt.txt"))
    ground_truth = intent.read_file(os.path.join(folder, "complete_prompt.txt"))
    result = intent.evaluate(gen_text, ground_truth)
    result["case_id"] = case_id
    write_json(os.path.join(folder, "llm_judge_intent_score.json"), result)


def native_score(folder, mode, row, domain, agent_label):
    prompt_len = len(row.get(mpa.PROMPT_FIELD[pu.MODE_LABEL[mode]], ""))
    scorer = mpa._score_disambiguation if mode == "disambiguation" else mpa._score_inconsistency
    sc = scorer(Path(folder), prompt_len, domain, agent_label, row["id"], row)
    if sc is None:
        return None
    s = dict(sc.__dict__)
    s["FR_raw"], s["grounded_raw"] = s["FR"], s["grounded"]
    # IC gate as in make_paper_artifacts.build_case_df
    if s["IC"] == 0:
        s["FR"], s["grounded"] = 0, 0
    s["CGRR"] = int(s["FR"] == 1 and s["grounded"] == 1)
    s["SRR"] = int(s["FR"] == 1 and s["grounded"] == 0)
    s["over_length_cap"] = prompt_len > mpa.MAX_PROMPT_CHARS
    return s


# ---- attribution and trait check ----

def dialogue_turns(folder):
    raw = LLMJudgeDisambiguationChat.read_file(None, os.path.join(folder, "conversation_log.txt"))
    parsed = LLMJudgeDisambiguationChat.parse_conversation_log(None, raw)
    turns = []
    for k, t in enumerate(parsed, 1):
        turns.append((t.get(f"Question_{k}", ""), t.get(f"Answer_{k}", "")))
    return turns


def attribute_turn(llm, cards, question, answer):
    card_view = [{"card_id": c["card_id"], "item": c["item"], "text": c["text"]} for c in cards]
    user_prompt = f"""
        <FACT_CARDS>
        {json.dumps(card_view, indent=2)}
        </FACT_CARDS>

        <QUESTION>
        {question}
        </QUESTION>

        <ANSWER>
        {answer}
        </ANSWER>
        """
    parsed = pu.parse_final_json(llm.invoke(user_prompt, ATTRIBUTION_SYSTEM))
    ids = {c["card_id"] for c in cards}
    out = {"asked": [], "conveyed": {}, "parse_failure": parsed is None,
           "dropped_evidence": 0, "unknown_ids": 0, "invalid_level": 0}
    if parsed is None:
        return out
    for cid in parsed.get("asked") or []:
        if cid in ids:
            if cid not in out["asked"]:
                out["asked"].append(cid)
        else:
            out["unknown_ids"] += 1
    rank = {"partial": 1, "full": 2}
    for entry in parsed.get("conveyed") or []:
        if not isinstance(entry, dict):
            continue
        cid, level, ev = entry.get("card_id"), entry.get("level"), entry.get("evidence") or ""
        if cid not in ids:
            out["unknown_ids"] += 1
            continue
        if level not in rank:
            out["invalid_level"] += 1
            continue
        if not pu.is_verbatim(ev, answer):
            out["dropped_evidence"] += 1
            continue
        prev = out["conveyed"].get(cid)
        if prev is None or rank[level] > rank[prev["level"]]:
            out["conveyed"][cid] = {"level": level, "evidence": ev}
    return out


def trait_turn(llm, trait_defs, question, answer):
    defs = "\n".join(f"- {tid}: {text}" for tid, text in trait_defs.items())
    user_prompt = f"""
        <TRAITS>
        {defs}
        </TRAITS>

        <QUESTION>
        {question}
        </QUESTION>

        <ANSWER>
        {answer}
        </ANSWER>
        """
    parsed = pu.parse_final_json(llm.invoke(user_prompt, TRAIT_SYSTEM))
    out = {"present": {}, "evidence": {}, "parse_failure": parsed is None,
           "dropped_evidence": 0, "missing_traits": 0}
    traits = (parsed or {}).get("traits") or {}
    for tid in trait_defs:
        entry = traits.get(tid)
        if not isinstance(entry, dict):
            out["missing_traits"] += 1
            out["present"][tid] = 0
            continue
        try:
            present = int(entry.get("present", 0)) == 1
        except (TypeError, ValueError):
            present = False
        ev = entry.get("evidence") or ""
        if present and not pu.is_verbatim(ev, answer):
            out["dropped_evidence"] += 1
            present = False
        out["present"][tid] = int(present)
        if present:
            out["evidence"][tid] = ev
    return out


def attribution_flags(card_id, turns):
    levels = [t["conveyed"].get(card_id, {}).get("level") for t in turns]
    delivered = "full" in levels
    # leak: conveyed in full in a turn that does not ask for it, before any turn has asked for it
    asked_before, leak = False, False
    for t in turns:
        asked_now = card_id in t["asked"]
        if t["conveyed"].get(card_id, {}).get("level") == "full" and not asked_now and not asked_before:
            leak = True
        asked_before = asked_before or asked_now
    return {"asked": any(card_id in t["asked"] for t in turns), "delivered": delivered,
            "partial": (not delivered) and "partial" in levels, "unasked_delivered": leak}


# ---- per case ----

def norm_entity(s):
    return re.sub(r"^[\W_]+|[\W_]+$", "", pu.norm_ws(str(s)).lower())


def eval_case(folder, ctx):
    args = ctx["args"]
    run = read_json(os.path.join(folder, "persona_run.json"))
    domain, mode, case_id, arm = run["domain"], run["mode"], run["case_id"], run["arm"]
    row = ctx["datasets"][(domain, mode)][case_id]
    agent_label = os.path.basename(os.path.dirname(folder))
    items = row.get(pu.ITEMS_KEY[mode], []) or []
    prev = read_json(os.path.join(folder, "persona_eval.json")) or {}
    judge_cfg, aux_cfg = ctx["judge_cfg"], ctx["aux_cfg"]
    parse_failures = {"native": 0, "attribution": 0, "traits": 0}

    judge_usage = None
    native_ok = all(os.path.exists(os.path.join(folder, f)) for f in NATIVE_FILES[mode])
    if not args.skip_native:
        if native_ok and prev.get("native_judge") == judge_cfg.as_dict():
            judge_usage = prev.get("usage_raw", {}).get("judge")
            parse_failures["native"] = prev.get("parse_failures", {}).get("native", 0)
        else:
            fn = native_disambiguation if mode == "disambiguation" else native_inconsistency
            judges = fn(folder, row, judge_cfg)
            judge_usage = sum_usage(j.llm.get_statistics() for j in judges)
            parse_failures["native"] = sum(j.llm.parse_failures() for j in judges)
            native_ok = True
    native_judge = judge_cfg.as_dict() if not args.skip_native else prev.get("native_judge")
    native = native_score(folder, mode, row, domain, agent_label) if native_ok else None

    fr_file = read_json(os.path.join(folder, NATIVE_FILES[mode][0])) if native_ok else None
    recovered, unmatched = [None] * len(items), []
    if fr_file is not None:
        if mode == "disambiguation":
            rec_set = {norm_entity(e) for e in fr_file.get("recovered_entities", []) or []}
            item_norms = [norm_entity(e) for e in items]
            recovered = [n in rec_set for n in item_norms]
            unmatched = [e for e in fr_file.get("recovered_entities", []) or []
                         if norm_entity(e) not in set(item_norms)]
        else:
            bd = fr_file.get("breakdown", []) or []
            recovered = [bool(bd[i].get("resolved") is True) if i < len(bd) else False
                         for i in range(len(items))]

    turns = dialogue_turns(folder)
    cards_rec = ctx["cards"].get((domain, mode))
    cards = (cards_rec["data"]["cases"].get(case_id, {}).get("cards") or []) if cards_rec else []
    card_by_index = {c["index"]: c for c in cards if c.get("text")}
    usable_cards = list(card_by_index.values())

    attribution, traits, aux_usage = None, None, []
    if not args.skip_aux:
        attr_path = os.path.join(folder, "persona_attribution.json")
        attribution = read_json(attr_path)
        key = {"aux_model": aux_cfg.as_dict(), "cards_sha256": cards_rec["sha256"] if cards_rec else None,
               "prompt_sha256": pu.sha256_text(ATTRIBUTION_SYSTEM)}
        if not attribution or {k: attribution.get(k) for k in key} != key:
            llm = LLMService(aux_cfg)
            per_turn = [attribute_turn(llm, usable_cards, q, a) for q, a in turns] if usable_cards else []
            attribution = dict(key, turns=per_turn, usage=llm.get_statistics())
            write_json(attr_path, attribution)
        aux_usage.append(attribution.get("usage"))

        if arm not in pu.DETERMINISTIC_ARMS:
            tr_path = os.path.join(folder, "persona_traits.json")
            traits = read_json(tr_path)
            key = {"aux_model": aux_cfg.as_dict(), "personas_sha256": ctx["personas_sha"],
                   "prompt_sha256": pu.sha256_text(TRAIT_SYSTEM)}
            if not traits or {k: traits.get(k) for k in key} != key:
                llm = LLMService(aux_cfg)
                per_turn = [trait_turn(llm, ctx["trait_defs"], q, a) for q, a in turns]
                traits = dict(key, turns=per_turn, usage=llm.get_statistics())
                write_json(tr_path, traits)
            aux_usage.append(traits.get("usage"))

    item_rows = []
    for i, item in enumerate(items):
        card = card_by_index.get(i)
        r = {"index": i, "item": item, "card_id": card["card_id"] if card else None,
             "has_card": card is not None, "asked": None, "delivered": None, "partial": None,
             "unasked_delivered": None, "recovered": recovered[i], "grounded": None, "silent": None}
        if attribution is not None and card is not None:
            r.update(attribution_flags(card["card_id"], attribution["turns"]))
        if r["recovered"] is not None and r["delivered"] is not None:
            r["grounded"] = r["recovered"] and r["delivered"]
            r["silent"] = r["recovered"] and not r["delivered"]
        item_rows.append(r)

    def rate(key):
        vals = [r[key] for r in item_rows if r[key] is not None]
        return (sum(vals) / len(vals)) if vals else None

    trait_share, trait_by_turn = None, None
    if traits is not None:
        trait_by_turn = [{"turn": k, "present": t["present"]} for k, t in enumerate(traits["turns"], 1)]
        trait_share = {tid: (sum(t["present"].get(tid, 0) for t in traits["turns"]) / len(traits["turns"]))
                       if traits["turns"] else None for tid in ctx["trait_defs"]}
    if attribution is not None:
        parse_failures["attribution"] = sum(t["parse_failure"] for t in attribution["turns"])
    if traits is not None:
        parse_failures["traits"] = sum(t["parse_failure"] for t in traits["turns"])

    usage_raw = {"agent": (run.get("usage") or {}).get("agent"), "user": (run.get("usage") or {}).get("user"),
                 "judge": judge_usage, "aux": sum_usage(aux_usage) if aux_usage else None}
    role_models = {"agent": (run["models"].get("agent") or {}).get("model_version"),
                   "user": (run["models"].get("user") or {}).get("model_version"),
                   "judge": (native_judge or {}).get("model_version"),
                   "aux": aux_cfg.model_version if not args.skip_aux else None}
    prices = ctx["prices"]
    usage = {role: (pu.usage_cost(u, (prices or {}).get(role_models[role])) if u else None)
             for role, u in usage_raw.items()}

    out = {
        "arm": arm, "trait": run.get("trait"), "rep": run["rep"], "case_id": case_id,
        "domain": domain, "mode": mode, "agent_label": agent_label, "n_items": len(items),
        "native": native, "items": item_rows,
        "item_rates": {k: rate(k) for k in ("asked", "delivered", "partial", "unasked_delivered", "recovered", "grounded",
                                           "silent")},
        "unmatched_recovered_entities": unmatched,
        "turns": run.get("turns"), "questions": len(turns), "cap_hit": run.get("cap_hit"),
        "completed_by_agent": run.get("completed_by_agent"),
        "trait_present_share": trait_share, "trait_by_turn": trait_by_turn,
        "usage": usage, "usage_raw": usage_raw,
        "native_judge": native_judge,
        "judge_model": role_models["judge"], "aux_model": role_models["aux"],
        "parse_failures": parse_failures,
        "dropped": {
            "attribution_evidence": sum(t["dropped_evidence"] for t in attribution["turns"]) if attribution else 0,
            "attribution_unknown_ids": sum(t["unknown_ids"] for t in attribution["turns"]) if attribution else 0,
            "attribution_invalid_level": sum(t["invalid_level"] for t in attribution["turns"]) if attribution else 0,
            "trait_evidence": sum(t["dropped_evidence"] for t in traits["turns"]) if traits else 0,
            "trait_missing": sum(t["missing_traits"] for t in traits["turns"]) if traits else 0,
        },
    }
    write_json(os.path.join(folder, "persona_eval.json"), out)
    return out


def rebuild_results(run_dir):
    rows = [read_json(p) for p in glob.glob(os.path.join(run_dir, "*", "rep*", "*", "*", "*", "persona_eval.json"))]
    rows = sorted((r for r in rows if r), key=lambda r: (r["arm"], r["rep"], r["domain"], r["mode"], r["case_id"]))
    path = os.path.join(run_dir, "results.jsonl")
    with open(path + ".tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(path + ".tmp", path)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a persona_runner output dir")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--judge_model", required=True, choices=list(Config.LLM_PROFILES))
    parser.add_argument("--aux_model", required=True, choices=list(Config.LLM_PROFILES))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip_native", action="store_true")
    parser.add_argument("--skip_aux", action="store_true")
    args = parser.parse_args(argv)

    manifest = read_json(os.path.join(args.run_dir, "manifest.json"))
    if manifest is None:
        sys.exit(f"No manifest.json in {args.run_dir}")
    folders = []
    for p in sorted(glob.glob(os.path.join(args.run_dir, "*", "rep*", "*", "*", "*", "persona_run.json"))):
        if (read_json(p) or {}).get("status") == "done":
            folders.append(os.path.dirname(p))
    print(f"{len(folders)} done case dirs")

    personas = pu.load_personas()
    trait_defs = {tid: personas[tid]["text"] for tid in pu.trait_ids(personas)}
    ctx = {"args": args, "judge_cfg": pu.role_config(args.judge_model), "aux_cfg": pu.role_config(args.aux_model),
           "trait_defs": trait_defs,
           "personas_sha": pu.sha256_text(json.dumps(trait_defs, sort_keys=True)),
           "datasets": {}, "cards": {}}
    for folder in folders:
        run = read_json(os.path.join(folder, "persona_run.json"))
        key = (run["domain"], run["mode"])
        if key in ctx["datasets"]:
            continue
        ctx["datasets"][key] = pu.load_cases(*key)
        entry = (manifest.get("cards") or {}).get(f"{key[0]}/{key[1]}")
        if entry:
            if pu.sha256_file(entry["path"]) != entry["sha256"]:
                sys.exit(f"Cards file {entry['path']} changed since the run (sha256 mismatch)")
            with open(entry["path"]) as f:
                ctx["cards"][key] = {"data": json.load(f), "sha256": entry["sha256"]}
        elif not args.skip_aux:
            sys.exit(f"No cards recorded in the manifest for {key}; attribution needs them (or --skip_aux)")

    eval_manifest_path = os.path.join(args.run_dir, "eval_manifest.json")
    em = read_json(eval_manifest_path) or {"prices": {}, "invocations": []}
    new_models = [m for m in {ctx["judge_cfg"].model_version, ctx["aux_cfg"].model_version}
                  if m not in em["prices"]]
    if new_models:
        fetched = pu.fetch_openrouter_prices(new_models) or {m: None for m in new_models}
        em["prices"].update(fetched)
    em["invocations"].append({"argv": sys.argv if argv is None else ["persona_eval.py"] + list(argv),
                              "git": pu.git_state(), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                              "judge": ctx["judge_cfg"].as_dict(), "aux": ctx["aux_cfg"].as_dict(),
                              "attribution_prompt_sha256": pu.sha256_text(ATTRIBUTION_SYSTEM),
                              "trait_prompt_sha256": pu.sha256_text(TRAIT_SYSTEM)})
    write_json(eval_manifest_path, em)
    ctx["prices"] = dict(manifest.get("prices") or {}, **{k: v for k, v in em["prices"].items() if v})

    n_err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(eval_case, f, ctx): f for f in folders}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                n_err += 1
                print(f"eval failed for {futures[fut]}: {type(e).__name__}: {e}")
    rows = rebuild_results(args.run_dir)
    print(f"results.jsonl: {len(rows)} rows, {n_err} eval errors")
    return {"rows": len(rows), "errors": n_err}


if __name__ == "__main__":
    main()
