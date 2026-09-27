import os
import re
import sys
import glob
import json
import time
import random
import string
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

ATTRIBUTION_USER = """
        <FACT_CARDS>
        {cards}
        </FACT_CARDS>

        <QUESTION>
        {question}
        </QUESTION>

        <ANSWER>
        {answer}
        </ANSWER>
        """

ASK_SYSTEM = """
        You label one question of a clarification dialogue between an Agent and a User.
        The User holds the information on the [FACT_CARDS] (each has a card_id, the item it concerns and its text).
        You get only the Agent's [QUESTION].

        **YOUR TASK:**
        "asked": the card_ids that the QUESTION explicitly asks for.
        A card is asked only if the question explicitly asks for that card's value or for the entity the card specifies.
        A generic question such as "anything else?" or "any other details?" asks for no card.
        Cards the question does not explicitly ask for are left out.

        **OUTPUT FORMAT:**
        You must output your final evaluation as a JSON object enclosed EXACTLY within <FINAL_JSON> tags.
        <FINAL_JSON>
        {
            "asked": ["card_id"]
        }
        </FINAL_JSON>
        """

ASK_USER = """
        <FACT_CARDS>
        {cards}
        </FACT_CARDS>

        <QUESTION>
        {question}
        </QUESTION>
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

TRAIT_USER = """
        <TRAITS>
        {traits}
        </TRAITS>

        <QUESTION>
        {question}
        </QUESTION>

        <ANSWER>
        {answer}
        </ANSWER>
        """

# pole id -> plain description; letters are assigned per case after a deterministic shuffle
BIGFIVE_OPTIONS = {
    "high_openness": "curious, imaginative, drawn to new ideas",
    "low_openness": "conventional, incurious, prefers the familiar",
    "high_conscientiousness": "organised, careful, thorough, dependable",
    "low_conscientiousness": "disorganised, careless, distractible, hurried",
    "high_extraversion": "outgoing, talkative, energetic, assertive",
    "low_extraversion": "reserved, quiet, withdrawn, low-key",
    "high_agreeableness": "warm, polite, trusting, cooperative",
    "low_agreeableness": "cold, blunt, critical, distrustful of others",
    "high_neuroticism": "anxious, tense, insecure, easily upset",
    "low_neuroticism": "calm, relaxed, emotionally steady, secure",
    "ordinary": "ordinary: no trait stands out",
}

BIGFIVE_SYSTEM = """
        You read a chat between an AI assistant and a scientist. The assistant is preparing a simulation setup for
        the scientist and messages them with questions; the scientist replies.
        Some words in the scientist's lines may be replaced by "[…]".

        **YOUR TASK:**
        Judge the personality of THE SCIENTIST (not the assistant) from how they write.
        From the [OPTIONS], pick the letter of the description that best fits the scientist ("choice") and the
        letter of the next best one ("second"). "intensity" is how strongly the choice shows: 0 (not at all),
        1 (slightly), 2 (clearly), 3 (strongly). "evidence" is the passage of the scientist's lines that shows it
        most clearly, copied VERBATIM.

        **OUTPUT FORMAT:**
        You must output your final evaluation as a JSON object enclosed EXACTLY within <FINAL_JSON> tags.
        <FINAL_JSON>
        {
            "choice": "A",
            "second": "B",
            "intensity": 0,
            "evidence": "verbatim passage of the scientist's lines"
        }
        </FINAL_JSON>
        """

BIGFIVE_USER = """
        <OPTIONS>
        {options}
        </OPTIONS>

        <CHAT>
        {chat}
        </CHAT>
        """

BIGFIVE_MASK = re.compile(r"\w*(?:neurotic|agreeabl|disagreeabl|conscientious|extravert|extrovert|introvert|openness|"
                          r"personality)\w*|\bbig[\s-]*five\b", re.I)

ATTRIBUTION_TEMPLATES_SHA = pu.sha256_text(ATTRIBUTION_SYSTEM + ATTRIBUTION_USER + ASK_SYSTEM + ASK_USER)
TRAIT_TEMPLATES_SHA = pu.sha256_text(TRAIT_SYSTEM + TRAIT_USER)
BIGFIVE_TEMPLATES_SHA = pu.sha256_text(BIGFIVE_SYSTEM + BIGFIVE_USER + json.dumps(BIGFIVE_OPTIONS, sort_keys=True))
PARSE_TRIES = 3

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
    except (ValueError, IndexError, AttributeError):  # AttributeError: judge JSON had "score": null or a number
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
    """Turns as [{"question", "answer", "deferred"}] plus the raw text they came from (for cache keys)."""
    path = os.path.join(folder, "clarifications.json")
    if os.path.exists(path):
        with open(path) as f:
            raw = f.read()
        turns = [{"question": t.get("question") or "", "answer": t.get("answer") or "",
                  "deferred": bool(t.get("deferred"))} for t in sorted(json.loads(raw), key=lambda t: t["turn"])]
        return turns, raw
    raw = LLMJudgeDisambiguationChat.read_file(None, os.path.join(folder, "conversation_log.txt"))
    parsed = LLMJudgeDisambiguationChat.parse_conversation_log(None, raw)
    turns = [{"question": t.get(f"Question_{k}", ""), "answer": t.get(f"Answer_{k}", ""), "deferred": False}
             for k, t in enumerate(parsed, 1)]
    return turns, raw


def invoke_parsed(llm, user_prompt, system_prompt):
    for attempt in range(1, PARSE_TRIES + 1):
        parsed = pu.parse_final_json(llm.invoke(user_prompt, system_prompt))
        if parsed is not None:
            return parsed, attempt
    return None, PARSE_TRIES


def cards_json(cards):
    view = [{"card_id": c["card_id"], "item": c["item"], "text": c["text"]} for c in cards]
    return json.dumps(view, indent=2, ensure_ascii=False)


def attribute_turn(llm, cards, question, answer):
    user_prompt = ATTRIBUTION_USER.format(cards=cards_json(cards), question=question, answer=answer)
    parsed, attempts = invoke_parsed(llm, user_prompt, ATTRIBUTION_SYSTEM)
    ids = {c["card_id"] for c in cards}
    out = {"conveyed": None, "parse_failure": parsed is None, "attempts": attempts,
           "dropped_evidence": 0, "unknown_ids": 0, "invalid_level": 0}
    if parsed is None:
        return out
    out["conveyed"] = {}
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


def ask_turn(llm, cards, question):
    parsed, attempts = invoke_parsed(llm, ASK_USER.format(cards=cards_json(cards), question=question), ASK_SYSTEM)
    ids = {c["card_id"] for c in cards}
    out = {"asked": None, "parse_failure": parsed is None, "attempts": attempts, "unknown_ids": 0}
    if parsed is None:
        return out
    out["asked"] = []
    for cid in parsed.get("asked") or []:
        if cid in ids:
            if cid not in out["asked"]:
                out["asked"].append(cid)
        else:
            out["unknown_ids"] += 1
    return out


def label_turn(llm, cards, turn):
    ask = ask_turn(llm, cards, turn["question"])
    if turn["deferred"]:
        conv = {"conveyed": {}, "parse_failure": False, "attempts": 0,
                "dropped_evidence": 0, "unknown_ids": 0, "invalid_level": 0}
    else:
        conv = attribute_turn(llm, cards, turn["question"], turn["answer"])
    return dict(conv, asked=ask["asked"], ask_parse_failure=ask["parse_failure"], ask_attempts=ask["attempts"],
                ask_unknown_ids=ask["unknown_ids"], deferred=turn["deferred"])


def trait_turn(llm, trait_defs, question, answer):
    defs = "\n".join(f"- {tid}: {text}" for tid, text in trait_defs.items())
    parsed, attempts = invoke_parsed(llm, TRAIT_USER.format(traits=defs, question=question, answer=answer),
                                     TRAIT_SYSTEM)
    out = {"present": None, "evidence": {}, "parse_failure": parsed is None, "attempts": attempts,
           "dropped_evidence": 0, "missing_traits": 0}
    if parsed is None:
        return out
    out["present"] = {}
    traits = parsed.get("traits") or {}
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


def mask_constructs(text):
    return BIGFIVE_MASK.sub("[…]", text)


def bigfive_order(folder, run_dir):
    rel = os.path.relpath(os.path.abspath(folder), os.path.abspath(run_dir))
    order = list(BIGFIVE_OPTIONS)
    random.Random(int(pu.sha256_text(rel), 16)).shuffle(order)
    return order


def bigfive_chat(turns):
    lines = []
    for t in turns:
        lines.append(f"Assistant: {t['question']}")
        lines.append(f"Scientist: {mask_constructs(t['answer'])}")
    return "\n".join(lines)


def bigfive_judge(llm, turns, order):
    letters = string.ascii_uppercase[:len(order)]
    options = "\n".join(f"{l}. {BIGFIVE_OPTIONS[pid]}" for l, pid in zip(letters, order))
    parsed, attempts = invoke_parsed(llm, BIGFIVE_USER.format(options=options, chat=bigfive_chat(turns)),
                                     BIGFIVE_SYSTEM)
    by_letter = dict(zip(letters, order))

    def pole(v):
        return by_letter.get(v.strip().upper()) if isinstance(v, str) else None

    out = {"order": order, "choice": None, "second": None, "intensity": None, "evidence": None,
           "evidence_ok": None, "parse_failure": True, "attempts": attempts, "raw": parsed}
    if parsed is None or pole(parsed.get("choice")) is None:
        return out
    try:
        intensity = int(parsed.get("intensity"))
        intensity = intensity if 0 <= intensity <= 3 else None
    except (TypeError, ValueError):
        intensity = None
    ev = parsed.get("evidence") if isinstance(parsed.get("evidence"), str) else ""
    scientist = "\n".join(t["answer"] for t in turns)
    out.update(choice=pole(parsed.get("choice")), second=pole(parsed.get("second")), intensity=intensity,
               evidence=ev, evidence_ok=pu.is_verbatim(ev, scientist), parse_failure=False)
    return out


# three-valued logic: None = unknown because a turn's judge output could not be parsed
def _or(vals):
    vals = list(vals)
    return True if True in vals else (None if None in vals else False)


def _and(*vals):
    return False if False in vals else (None if None in vals else True)


def _not(v):
    return None if v is None else not v


def attribution_flags(card_id, turns):
    def has_level(t, level):
        return None if t["conveyed"] is None else t["conveyed"].get(card_id, {}).get("level") == level

    def asked(t):
        return None if t["asked"] is None else card_id in t["asked"]

    delivered = _or(has_level(t, "full") for t in turns)
    # leak: conveyed in a turn that does not ask for it, before any turn has asked for it
    leak, leak_partial, asked_before = False, False, False
    for t in turns:
        unasked = _and(_not(asked(t)), _not(asked_before))
        leak = _or([leak, _and(has_level(t, "full"), unasked)])
        leak_partial = _or([leak_partial, _and(has_level(t, "partial"), unasked)])
        asked_before = _or([asked_before, asked(t)])
    return {"asked": _or(asked(t) for t in turns), "delivered": delivered,
            "partial": _and(_not(delivered), _or(has_level(t, "partial") for t in turns)),
            "unasked_delivered": leak, "unasked_partial": leak_partial}


# ---- per case ----

def norm_entity(s):
    return re.sub(r"^[\W_]+|[\W_]+$", "", pu.norm_ws(str(s)).lower())


def match_entities(items, entities):
    """Per item, the index of the judge entity it matches or None: exact normalised match first, then
    containment either way; each entity matches at most one item; longest items first."""
    item_n = [norm_entity(e) for e in items]
    ent_n = [norm_entity(e) for e in entities]
    order = sorted(range(len(items)), key=lambda i: -len(item_n[i]))
    match, used = [None] * len(items), set()
    for exact in (True, False):
        for i in order:
            if match[i] is not None or not item_n[i]:
                continue
            for j, e in enumerate(ent_n):
                if j in used or not e:
                    continue
                if e == item_n[i] if exact else (e in item_n[i] or item_n[i] in e):
                    match[i] = j
                    used.add(j)
                    break
    return match


def parse_score(s):
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", s) if isinstance(s, str) else None
    return (int(m.group(1)), int(m.group(2))) if m else None


def read_text(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def eval_case(folder, ctx):
    args = ctx["args"]
    run = read_json(os.path.join(folder, "persona_run.json"))
    domain, mode, case_id, arm = run["domain"], run["mode"], run["case_id"], run["arm"]
    row = ctx["datasets"][(domain, mode)][case_id]
    agent_label = os.path.basename(os.path.dirname(folder))
    items = row.get(pu.ITEMS_KEY[mode], []) or []
    prev = read_json(os.path.join(folder, "persona_eval.json")) or {}
    judge_cfg, aux_cfg = ctx["judge_cfg"], ctx["aux_cfg"]
    trait_check = getattr(args, "trait_check", "per_turn")
    parse_failures = {"native": 0, "attribution": 0, "ask": 0, "traits": 0}
    if trait_check == "bigfive":
        parse_failures["bigfive"] = 0
    fresh = []  # (model_version, stats) of calls made by this invocation, for the spend cap

    cards_rec = ctx["cards"].get((domain, mode))
    case_cards = cards_rec["data"]["cases"].get(case_id, {}) if cards_rec else {}
    cards = case_cards.get("cards") or []
    if cards_rec:
        src_sha = (cards_rec["data"].get("meta") or {}).get("source_sha256")
        if src_sha != cards_rec["dataset_sha256"]:
            raise ValueError(f"{case_id}: cards file source_sha256 {src_sha} != current dataset sha256 "
                             f"{cards_rec['dataset_sha256']}")
        for c in cards:
            i = c.get("index")
            if not (isinstance(i, int) and 0 <= i < len(items)) or \
                    pu.norm_ws(str(c.get("item"))) != pu.norm_ws(str(items[i])):
                raise ValueError(f"{case_id}: card {c.get('card_id')} item {c.get('item')!r} does not match "
                                 f"dataset item {i}")
    card_by_index = {c["index"]: c for c in cards if c.get("text")}
    usable_cards = list(card_by_index.values())

    turns, turns_text = dialogue_turns(folder)
    text_key = {"turns_sha256": pu.sha256_text(turns_text),
                "generated_prompt_sha256": pu.sha256_text(read_text(os.path.join(folder, "generated_prompt.txt")))}

    judge_usage = None
    native_ok = all(os.path.exists(os.path.join(folder, f)) for f in NATIVE_FILES[mode])
    native_key = dict(text_key, judge=judge_cfg.as_dict())
    if not args.skip_native:
        judge_used = judge_cfg.as_dict()
        if native_ok and prev.get("native_key") == native_key:
            judge_usage = prev.get("usage_raw", {}).get("judge")
            parse_failures["native"] = prev.get("parse_failures", {}).get("native", 0)
        else:
            fn = native_disambiguation if mode == "disambiguation" else native_inconsistency
            judges = fn(folder, row, judge_cfg)
            judge_usage = sum_usage(j.llm.get_statistics() for j in judges)
            fresh.append((judge_cfg.model_version, judge_usage))
            parse_failures["native"] = sum(j.llm.parse_failures() for j in judges)
            native_ok = True
    else:
        judge_used, native_key = prev.get("native_judge"), prev.get("native_key")
        parse_failures["native"] = prev.get("parse_failures", {}).get("native", 0)
    native_failed = parse_failures["native"] > 0
    if native_failed:
        # not recorded, so the next eval reruns the native judges instead of reusing the failure
        native_key = None
    native = native_score(folder, mode, row, domain, agent_label) if native_ok and not native_failed else None

    fr_file = read_json(os.path.join(folder, NATIVE_FILES[mode][0])) if native_ok else None
    recovered, unmatched, judge_a, recovered_judge = [None] * len(items), [], None, None
    if fr_file is not None:
        if mode == "disambiguation":
            ents = fr_file.get("recovered_entities", []) or []
            match = match_entities(items, ents)
            recovered = [m is not None for m in match]
            unmatched = [e for j, e in enumerate(ents) if j not in set(match)]
            sc = parse_score(fr_file.get("score"))
            if sc is not None and sc[1] > 0:
                judge_a, recovered_judge = sc[0], sc[0] / sc[1]
        else:
            bd = fr_file.get("breakdown", []) or []
            recovered = [bool(bd[i].get("resolved") is True) if i < len(bd) else False
                         for i in range(len(items))]
            if items:
                judge_a = sum(recovered)
                recovered_judge = judge_a / len(items)
    if native_failed:
        recovered_judge = None
    item_match_mismatch = (sum(recovered) != judge_a) if fr_file is not None and judge_a is not None else None

    attribution, traits, bigfive, aux_usage = None, None, None, []
    if not args.skip_aux:
        attr_path = os.path.join(folder, "persona_attribution.json")
        attribution = read_json(attr_path)
        key = dict(text_key, aux_model=aux_cfg.as_dict(),
                   cards_sha256=cards_rec["sha256"] if cards_rec else None, prompt_sha256=ATTRIBUTION_TEMPLATES_SHA)
        if not attribution or {k: attribution.get(k) for k in key} != key:
            llm = LLMService(aux_cfg)
            per_turn = [label_turn(llm, usable_cards, t) for t in turns] if usable_cards else []
            attribution = dict(key, turns=per_turn, usage=llm.get_statistics())
            write_json(attr_path, attribution)
            fresh.append((aux_cfg.model_version, attribution["usage"]))
        aux_usage.append(attribution.get("usage"))

        if arm not in pu.DETERMINISTIC_ARMS and trait_check == "per_turn":
            tr_path = os.path.join(folder, "persona_traits.json")
            traits = read_json(tr_path)
            key = dict(text_key, aux_model=aux_cfg.as_dict(), personas_sha256=ctx["personas_sha"],
                       prompt_sha256=TRAIT_TEMPLATES_SHA)
            if not traits or {k: traits.get(k) for k in key} != key:
                llm = LLMService(aux_cfg)
                per_turn = [trait_turn(llm, ctx["trait_defs"], t["question"], t["answer"]) for t in turns]
                traits = dict(key, turns=per_turn, usage=llm.get_statistics())
                write_json(tr_path, traits)
                fresh.append((aux_cfg.model_version, traits["usage"]))
            aux_usage.append(traits.get("usage"))

        if arm not in pu.DETERMINISTIC_ARMS and trait_check == "bigfive":
            bf_path = os.path.join(folder, "persona_bigfive_trait.json")
            bigfive = read_json(bf_path)
            key = dict(text_key, aux_model=aux_cfg.as_dict(), prompt_sha256=BIGFIVE_TEMPLATES_SHA)
            if not bigfive or {k: bigfive.get(k) for k in key} != key:
                llm = LLMService(aux_cfg)
                res = bigfive_judge(llm, turns, bigfive_order(folder, args.run_dir))
                bigfive = dict(key, **res, usage=llm.get_statistics())
                write_json(bf_path, bigfive)
                fresh.append((aux_cfg.model_version, bigfive["usage"]))
            aux_usage.append(bigfive.get("usage"))

    card_case_status = case_cards.get("status") if cards_rec else None
    item_rows = []
    for i, item in enumerate(items):
        card = card_by_index.get(i)
        r = {"index": i, "item": item, "card_id": card["card_id"] if card else None,
             "has_card": card is not None,
             "card_ok": card is not None and card_case_status == "ok" and card.get("status") == "ok",
             "asked": None, "delivered": None, "partial": None, "unasked_delivered": None, "unasked_partial": None,
             "recovered": recovered[i], "grounded": None, "silent": None}
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
        valid = [t for t in traits["turns"] if t["present"] is not None]
        trait_share = {tid: (sum(t["present"].get(tid, 0) for t in valid) / len(valid)) if valid else None
                       for tid in ctx["trait_defs"]}
    if attribution is not None:
        parse_failures["attribution"] = sum(t["parse_failure"] for t in attribution["turns"])
        parse_failures["ask"] = sum(t["ask_parse_failure"] for t in attribution["turns"])
    if traits is not None:
        parse_failures["traits"] = sum(t["parse_failure"] for t in traits["turns"])
    bf_trait = None
    if bigfive is not None:
        parse_failures["bigfive"] = int(bigfive["parse_failure"])
        target = pu.bigfive_target(arm, ctx.get("bigfive"))
        bf_trait = {"choice": bigfive["choice"], "second": bigfive["second"], "intensity": bigfive["intensity"],
                    "evidence_ok": bigfive["evidence_ok"], "target": target,
                    "hit": None if target is None or bigfive["choice"] is None else bigfive["choice"] == target,
                    "parse_failure": bigfive["parse_failure"]}

    usage_raw = {"agent": (run.get("usage") or {}).get("agent"), "user": (run.get("usage") or {}).get("user"),
                 "judge": judge_usage, "aux": sum_usage(aux_usage) if aux_usage else None}
    role_models = {"agent": (run["models"].get("agent") or {}).get("model_version"),
                   "user": (run["models"].get("user") or {}).get("model_version"),
                   "judge": (judge_used or {}).get("model_version"),
                   "aux": aux_cfg.model_version if not args.skip_aux else None}
    prices = ctx["prices"]
    usage = {role: (pu.usage_cost(u, (prices or {}).get(role_models[role])) if u else None)
             for role, u in usage_raw.items()}

    item_rates = {k: rate(k) for k in ("asked", "delivered", "partial", "unasked_delivered", "unasked_partial",
                                       "recovered", "grounded", "silent")}
    item_rates["recovered_judge"] = recovered_judge
    out = {
        "arm": arm, "trait": run.get("trait"), "rep": run["rep"], "case_id": case_id,
        "domain": domain, "mode": mode, "agent_label": agent_label, "n_items": len(items),
        "native": native, "items": item_rows, "item_rates": item_rates,
        "unmatched_recovered_entities": unmatched, "item_match_mismatch": item_match_mismatch,
        "card_case_status": card_case_status,
        "turns": run.get("turns"), "questions": len(turns), "cap_hit": run.get("cap_hit"),
        "completed_by_agent": run.get("completed_by_agent"),
        "deferred": run.get("deferred"), "attempts": run.get("attempts"),
        "trait_present_share": trait_share, "trait_by_turn": trait_by_turn,
        **({"bf_trait": bf_trait} if trait_check == "bigfive" else {}),
        "usage": usage, "usage_raw": usage_raw,
        "native_judge": None if native_failed else judge_used, "native_key": native_key,
        "judge_model": role_models["judge"], "aux_model": role_models["aux"],
        "parse_failures": parse_failures,
        "dropped": {
            "attribution_evidence": sum(t["dropped_evidence"] for t in attribution["turns"]) if attribution else 0,
            "attribution_unknown_ids": sum(t["unknown_ids"] + t["ask_unknown_ids"] for t in attribution["turns"])
            if attribution else 0,
            "attribution_invalid_level": sum(t["invalid_level"] for t in attribution["turns"]) if attribution else 0,
            "trait_evidence": sum(t["dropped_evidence"] for t in traits["turns"]) if traits else 0,
            "trait_missing": sum(t["missing_traits"] for t in traits["turns"]) if traits else 0,
        },
    }
    write_json(os.path.join(folder, "persona_eval.json"), out)
    if ctx.get("spend") is not None:
        ctx["spend"].add(sum(pu.stats_cost(u, (prices or {}).get(m)) for m, u in fresh))
    return out


def eval_or_skip(folder, ctx):
    if ctx["spend"].exhausted():
        return "skipped_budget"
    eval_case(folder, ctx)
    return "done"


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
    parser.add_argument("--trait_check", choices=["per_turn", "bigfive", "none"], default="per_turn",
                        help="per_turn: per-answer multi-label trait judge; bigfive: one blind Big Five call per "
                             "dialogue; none: no trait check")
    parser.add_argument("--budget_usd", type=float, required=True,
                        help="Spend cap for this invocation's judge/aux calls; no new case starts once reached")
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
    # code-enforced traits (e.g. busy deferrals) are not judged from the text
    trait_defs = {tid: personas[tid]["text"] for tid in pu.trait_ids(personas)
                  if personas[tid].get("enforced") != "code"}
    ctx = {"args": args, "judge_cfg": pu.role_config(args.judge_model), "aux_cfg": pu.role_config(args.aux_model),
           "trait_defs": trait_defs,
           "personas_sha": pu.sha256_text(json.dumps(trait_defs, sort_keys=True)),
           "datasets": {}, "cards": {}, "bigfive": pu.load_bigfive(), "spend": pu.SpendTracker(args.budget_usd)}
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
                ctx["cards"][key] = {"data": json.load(f), "sha256": entry["sha256"],
                                     "dataset_sha256": pu.sha256_file(pu.data_path(*key))}
        elif not args.skip_aux:
            print(f"WARNING: no cards recorded for {key}; per-item attribution is skipped, trait check still runs")

    eval_manifest_path = os.path.join(args.run_dir, "eval_manifest.json")
    em = read_json(eval_manifest_path) or {"prices": {}, "invocations": []}
    role_models = sorted({ctx["judge_cfg"].model_version, ctx["aux_cfg"].model_version})
    new_models = [m for m in role_models if not pu.has_price(em["prices"].get(m))]
    if new_models:
        em["prices"].update(pu.fetch_prices(new_models))
    pu.require_prices(em["prices"], role_models)
    em["invocations"].append({"argv": sys.argv if argv is None else ["persona_eval.py"] + list(argv),
                              "git": pu.git_state(), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                              "judge": ctx["judge_cfg"].as_dict(), "aux": ctx["aux_cfg"].as_dict(),
                              "attribution_prompt_sha256": ATTRIBUTION_TEMPLATES_SHA,
                              "trait_prompt_sha256": TRAIT_TEMPLATES_SHA,
                              "trait_check": args.trait_check,
                              "bigfive_prompt_sha256": BIGFIVE_TEMPLATES_SHA,
                              "budget_usd": args.budget_usd})
    write_json(eval_manifest_path, em)
    ctx["prices"] = dict(manifest.get("prices") or {}, **{k: v for k, v in em["prices"].items() if v})

    n_err = n_budget = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(eval_or_skip, f, ctx): f for f in folders}
        for fut in as_completed(futures):
            try:
                n_budget += fut.result() == "skipped_budget"
            except Exception as e:
                n_err += 1
                print(f"eval failed for {futures[fut]}: {type(e).__name__}: {e}")
    rows = rebuild_results(args.run_dir)
    print(f"results.jsonl: {len(rows)} rows, {n_err} eval errors")
    if n_budget:
        print(f"{n_budget} case dirs not evaluated: budget reached")
    print(ctx["spend"].summary())
    out = {"rows": len(rows), "errors": n_err}
    if n_budget:
        out["skipped_budget"] = n_budget
    return out


if __name__ == "__main__":
    main()
