import os
import re
import sys
import json
import time
import hashlib
import threading
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

import requests

from config import Config
from llm_user_interface import LLMUserInterface

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
PERSONA_DIR = os.path.join(REPO_DIR, "persona", "personas")
BIGFIVE_DIR = os.path.join(REPO_DIR, "persona", "bigfive")

PRECEDENCE = ("The persona below describes how you talk and behave in this conversation. "
              "Where it conflicts with the style instructions above, follow the persona, with two limits: "
              "every value you give must come from the COMPLETE requirement, and you do not give information "
              "that was not asked for unless the persona explicitly tells you to.")
MUTE_REPLY = "Make a reasonable assumption based on best practices."
DETERMINISTIC_ARMS = ("oracle", "mute")
MODE_LABEL = {"disambiguation": "Disambiguation", "inconsistency": "Inconsistency"}
ITEMS_KEY = {"disambiguation": "missing_entities", "inconsistency": "inconsistencies"}


def data_path(domain: str, mode: str) -> str:
    return os.path.join(REPO_DIR, domain, f"{mode}_{domain}.json")


def default_cards_path(domain: str, mode: str) -> str:
    return os.path.join(REPO_DIR, "persona", "fact_cards", domain, f"{mode}_{domain}.cards.json")


def load_cases(domain: str, mode: str) -> Dict[str, dict]:
    with open(data_path(domain, mode)) as f:
        return {c["id"]: c for c in json.load(f) if c.get("id")}


def load_personas(persona_dir: str = PERSONA_DIR) -> Dict[str, dict]:
    out = {}
    for name in sorted(os.listdir(persona_dir)):
        if name.endswith(".json"):
            with open(os.path.join(persona_dir, name)) as f:
                p = json.load(f)
            p["path"] = os.path.join(persona_dir, name)
            out[p["id"]] = p
    return out


def trait_ids(personas: Dict[str, dict]) -> List[str]:
    return [pid for pid, p in personas.items() if p["kind"] not in ("neutral", "placebo")]


def placebo_ids(personas: Dict[str, dict]) -> List[str]:
    return [pid for pid, p in personas.items() if p["kind"] == "placebo"]


EXAMPLES_HEADER = ("Examples of how you write (style only; [brackets] stand for values that must come from the "
                   "COMPLETE requirement, never invented):")


def persona_block(p: dict) -> str:
    if not p.get("examples"):
        return p["text"]
    return p["text"] + "\n\n" + "\n".join([EXAMPLES_HEADER] + [f'"{e}"' for e in p["examples"]])


def render_persona(trait_id: Optional[str], personas: Optional[Dict[str, dict]] = None) -> str:
    personas = personas or load_personas()
    parts = [PRECEDENCE, personas["neutral"]["text"]]
    if trait_id is not None and trait_id != "neutral":
        parts.append(persona_block(personas[trait_id]))
    return "\n\n".join(parts)


def persona_text_for_arm(arm: str, personas: Optional[Dict[str, dict]] = None) -> str:
    if arm == "paper" or arm in DETERMINISTIC_ARMS:
        return ""
    if arm == "neutral":
        return render_persona(None, personas)
    return render_persona(arm, personas)


def turn_note_for_arm(arm: str, personas: Optional[Dict[str, dict]] = None) -> str:
    if arm == "paper" or arm in DETERMINISTIC_ARMS:
        return ""
    personas = personas or load_personas()
    notes = [personas["neutral"]["reminder"]]
    if arm != "neutral":
        notes.append(personas[arm]["reminder"])
    return f"(Stay in character. {' '.join(notes)})"


def load_bigfive(bigfive_dir: str = BIGFIVE_DIR) -> dict:
    shared_path = os.path.join(bigfive_dir, "shared.json")
    with open(shared_path) as f:
        shared_text = f.read()
    personas_dir = os.path.join(bigfive_dir, "personas")
    personas = {}
    for name in sorted(os.listdir(personas_dir)):
        if name.endswith(".json"):
            with open(os.path.join(personas_dir, name)) as f:
                p = json.load(f)
            p["path"] = os.path.join(personas_dir, name)
            personas[p["id"]] = p
    return {"shared": json.loads(shared_text), "shared_text": shared_text, "shared_path": shared_path,
            "personas": personas}


def bigfive_persona_text(p: dict) -> str:
    return "\n\n".join([p["sheet"], p["appraisal"], p["reminder"]] + [e["a"] for e in p.get("examples") or []])


def bigfive_target(arm: str, bigfive: Optional[dict] = None) -> Optional[str]:
    """Target pole of a Big Five trait arm; None for bf neutral/placebo and for every other arm."""
    p = (bigfive or load_bigfive())["personas"].get(arm)
    return p["target"] if p and p["kind"] == "trait" else None


def valid_arms(personas: Dict[str, dict], bigfive: Optional[dict] = None) -> List[str]:
    old = ["paper", "neutral"] + placebo_ids(personas) + trait_ids(personas) + list(DETERMINISTIC_ARMS)
    bf = list((bigfive or load_bigfive())["personas"])
    clash = sorted(set(old) & set(bf))
    if clash:
        raise ValueError(f"Big Five persona ids collide with existing arms: {clash}")
    return old + bf


@dataclass
class ScriptedUser(LLMUserInterface):
    def reply(self, question: str) -> str:
        raise NotImplementedError

    def answer_clarification_question(self, question: str) -> str:
        self._log_qa(question, "Thinking...")
        answer = self.reply(question)
        self.clarifications.append({
            "question": question,
            "answer": answer,
            "timestamp": time.time(),
            "deferred": False
        })
        self._update_last_log_entry(answer)
        return answer


@dataclass
class OracleUser(ScriptedUser):
    cards: List[dict] = field(default_factory=list)

    def reply(self, question: str) -> str:
        return "\n".join(c["text"] for c in sorted(self.cards, key=lambda c: c["index"]) if c.get("text"))


@dataclass
class MuteUser(ScriptedUser):
    def reply(self, question: str) -> str:
        return MUTE_REPLY


@dataclass
class BusyUser(LLMUserInterface):
    """Defers questions 1, 3, 5, ... with a fixed reply and no LLM call; the rest go to the LLM."""
    defer_reply: str = ""
    n_questions: int = 0

    def answer_clarification_question(self, question: str) -> str:
        self.n_questions += 1
        if self.n_questions % 2 == 0:
            return super().answer_clarification_question(question)
        self._log_qa(question, "Thinking...")
        self.clarifications.append({
            "question": question,
            "answer": self.defer_reply,
            "timestamp": time.time(),
            "deferred": True
        })
        self._update_last_log_entry(self.defer_reply)
        return self.defer_reply


USER_FAILURE_REPLY = "I am unable to answer that at this time."
_THINK_BLOCK = re.compile(r"<\s*(think(?:ing)?|thought)\b[^<>\n]*>?.*?(?:<\s*/\s*\1\s*>?|$)", re.S | re.I)
_THINK_CLOSE = re.compile(r"<\s*/\s*(?:think(?:ing)?|thought)\s*>?", re.I)
_SPEAKER_LABEL = re.compile(r"^\s*(?:\*\*)?(?:Dr\.?\s+)?(?:Sam\s+Keller|Sam|Keller)(?:\*\*)?\s*:(?:\*\*)?\s*", re.I)
# same patterns as scripts/false_claims.py
CLAIM = re.compile(r"\b(already (gave|told|sent|said|provided|specified|stated|listed|mentioned)|i (gave|told|sent|provided|specified|stated|listed) you|(as|like) i (said|told you|mentioned|wrote)|it'?s (right )?(there )?in (my|the) (request|message)|read (it|my request|what i sent|the request))", re.I)
NUM = re.compile(r"\d+(?:\.\d+)?")
TRUTH_RETRY_NOTE = ("\n\n(Your draft reply said you had already given the assistant something that is not in your "
                    "request or your earlier replies. Write your reply again without claiming that.)")


@dataclass
class BigFiveUser(LLMUserInterface):
    """Big Five simulated scientist: own system prompt (COMPLETE requirement as <notes>, INCOMPLETE requirement as
    its own opening message <your_request>), history as real chat turns, optional persona example turns before
    the real history, one regeneration when a reply falsely claims an earlier disclosure."""
    persona: dict = field(default_factory=dict)
    shared: dict = field(default_factory=dict)
    truth_retries: int = 0

    def __post_init__(self):
        super().__post_init__()
        folder = os.path.dirname(self.log_file)
        self.calls_file = os.path.join(folder, "user_calls.jsonl")
        # one user instance = one dialogue; a rerun must not append to a failed attempt's calls
        open(self.calls_file, "w").close()
        with open(os.path.join(folder, "user_system_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(self.build_system_prompt())

    def build_system_prompt(self) -> str:
        sh, p = self.shared, self.persona
        # the user sees its own opening request, as in a real chat window; without it the sim accuses the agent
        # of ignoring facts that were never sent (stage-1 smoke, bf_low_A)
        return "\n\n".join([sh["identity"], p["sheet"], sh["situation"], p["appraisal"], sh["notes_intro"],
                             f"<notes>\n{self.complete_requirement}\n</notes>", sh["request_intro"],
                             f"<your_request>\n{self.incomplete_requirement}\n</your_request>", sh["rules"]]
                            + ([sh["examples_intro"]] if p.get("examples") else []))

    def build_messages(self, question: str) -> List[dict]:
        messages = []
        for e in self.persona.get("examples") or []:
            messages.append({"role": "user", "content": f"Message from the assistant:\n{e['q']}"})
            messages.append({"role": "assistant", "content": e["a"]})
        for c in self.clarifications:
            messages.append({"role": "user", "content": f"Message from the assistant:\n{c['question']}"})
            messages.append({"role": "assistant", "content": c["answer"]})
        # shared suffix: the rule in the system prompt alone did not stop false "I already told you" claims
        reminder = " ".join(x for x in (self.persona["reminder"], self.shared.get("reminder_suffix", "")) if x)
        messages.append({"role": "user", "content": f"Message from the assistant:\n{question}\n\n{reminder}"})
        return messages

    @staticmethod
    def clean_reply(raw: str) -> str:
        text = (raw or "").strip()
        text = _THINK_BLOCK.sub("", text)
        # a closing tag with no opening one: the reasoning is everything before it, unless nothing follows it
        closes = list(_THINK_CLOSE.finditer(text))
        if closes:
            after = text[closes[-1].end():]
            text = after if after.strip() else text[:closes[-1].start()]
        text = _SPEAKER_LABEL.sub("", text.strip(), count=1).strip()
        return text or (raw or "").strip()

    def false_claim(self, reply: str) -> bool:
        """Claims an earlier disclosure and carries a number from the notes that the sim has not sent yet."""
        if not CLAIM.search(reply):
            return False
        sent = set(NUM.findall(self.incomplete_requirement or ""))
        for c in self.clarifications:
            sent |= set(NUM.findall(c["answer"] or ""))
        novel = (set(NUM.findall(reply)) & set(NUM.findall(self.complete_requirement or ""))) - sent
        return bool(novel)

    def answer_clarification_question(self, question: str) -> str:
        self._log_qa(question, "Thinking...")
        system_prompt = self.build_system_prompt()
        messages = self.build_messages(question)
        svc = self.llm_service
        before = [getattr(svc, k, 0) for k in ("api_input_tokens", "api_output_tokens", "api_reasoning_chars")]
        try:
            raw = svc.invoke_messages(messages, system_prompt)
            reply = self.clean_reply(raw)
            extra = {"truth_retry": False}
            if self.false_claim(reply):
                extra = {"truth_retry": True, "draft": reply, "draft_raw": raw}
                self.truth_retries += 1
                messages = [dict(m) for m in messages]
                messages[-1]["content"] += TRUTH_RETRY_NOTE
                raw = svc.invoke_messages(messages, system_prompt)
                reply = self.clean_reply(raw)
            after = [getattr(svc, k, 0) for k in ("api_input_tokens", "api_output_tokens", "api_reasoning_chars")]
            with open(self.calls_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"turn": len(self.clarifications) + 1, "messages": messages, "raw": raw,
                                    "reply": reply, **extra,
                                    "usage": {"input_tokens": after[0] - before[0],
                                              "output_tokens": after[1] - before[1]},
                                    "reasoning_chars": after[2] - before[2]}, ensure_ascii=False) + "\n")
            self.clarifications.append({
                "question": question,
                "answer": reply,
                "timestamp": time.time(),
                "deferred": False
            })
            self._update_last_log_entry(reply)
            return reply
        except Exception as e:
            print(f"Error generating answer: {e}")
            return USER_FAILURE_REPLY


def make_user(arm: str, llm_service, incomplete_requirement: str, complete_requirement: str,
              log_file: str, cards: Optional[List[dict]] = None,
              personas: Optional[Dict[str, dict]] = None, bigfive: Optional[dict] = None) -> LLMUserInterface:
    common = dict(incomplete_requirement=incomplete_requirement,
                  complete_requirement=complete_requirement, log_file=log_file)
    if arm == "oracle":
        if not cards:
            raise ValueError("oracle arm needs fact cards for this case")
        return OracleUser(llm_service=None, cards=cards, **common)
    if arm == "mute":
        return MuteUser(llm_service=None, **common)
    bigfive = bigfive or load_bigfive()
    if arm in bigfive["personas"]:
        return BigFiveUser(llm_service=llm_service, persona=bigfive["personas"][arm], shared=bigfive["shared"],
                           **common)
    personas = personas or load_personas()
    common.update(llm_service=llm_service, persona_text=persona_text_for_arm(arm, personas),
                  turn_note=turn_note_for_arm(arm, personas))
    if personas.get(arm, {}).get("defer") == "odd":
        return BusyUser(defer_reply=personas[arm]["defer_reply"], **common)
    return LLMUserInterface(**common)


@dataclass
class RoleConfig:
    profile: str
    model_provider: str
    model_version: str
    temperature: float
    max_tokens: Optional[int] = None

    def as_dict(self) -> dict:
        d = {"profile": self.profile, "provider": self.model_provider,
             "model_version": self.model_version, "temperature": self.temperature}
        if self.max_tokens is not None:
            d["max_tokens"] = self.max_tokens
        return d


def role_config(profile: str, temperature: Optional[float] = None) -> RoleConfig:
    prof = Config.LLM_PROFILES[profile]
    return RoleConfig(profile, prof["provider"], prof["model_version"],
                      prof["temperature"] if temperature is None else temperature, prof.get("max_tokens"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_ws(text: str) -> str:
    return " ".join((text or "").split())


def is_verbatim(quote: str, text: str) -> bool:
    q = norm_ws(quote)
    return bool(q) and q in norm_ws(text)


def parse_final_json(response_text: str) -> Optional[dict]:
    """Same extraction as the repo judges: last <FINAL_JSON> block, else outermost braces."""
    try:
        matches = re.findall(r'<FINAL_JSON>(.*?)</FINAL_JSON>', response_text, re.DOTALL)
        if matches:
            clean_json_str = matches[-1].strip()
        else:
            cleaned_text = re.sub(r'```json\s*', '', response_text)
            cleaned_text = re.sub(r'```\s*', '', cleaned_text)
            start_idx = cleaned_text.find('{')
            end_idx = cleaned_text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                clean_json_str = cleaned_text[start_idx:end_idx+1]
            else:
                return None
        try:
            parsed = json.loads(clean_json_str)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
        clean_json_str = re.sub(r'[\xa0 ​]', ' ', clean_json_str)
        clean_json_str = re.sub(r'```json\s*', '', clean_json_str)
        clean_json_str = re.sub(r'```\s*', '', clean_json_str)
        clean_json_str = re.sub(r'(?<!\\)\\(?!["\\/])', r'\\\\', clean_json_str)
        parsed = json.loads(clean_json_str, strict=False)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def fetch_openrouter_prices(model_ids: List[str]) -> Optional[Dict[str, Optional[dict]]]:
    try:
        r = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
        r.raise_for_status()
        table = {m["id"]: m.get("pricing") for m in r.json().get("data", [])}
    except Exception as e:
        print(f"Price fetch failed: {e}")
        return None
    return {mid: table.get(mid) for mid in model_ids}


# USD per token; rates from AWS Cost Explorer, 2026-09-27
BEDROCK_PRICES = {"zai.glm-5": {"prompt": 1.2e-6, "completion": 3.2e-6}}


def fetch_prices(model_ids: List[str]) -> Dict[str, Optional[dict]]:
    """Static Bedrock entry when present; OpenRouter only for the other ids (no network call if none)."""
    out = {mid: BEDROCK_PRICES[mid] for mid in model_ids if mid in BEDROCK_PRICES}
    rest = [mid for mid in model_ids if mid not in BEDROCK_PRICES]
    if rest:
        fetched = fetch_openrouter_prices(rest) or {}
        out.update({mid: fetched.get(mid) for mid in rest})
    return out


def has_price(pricing: Optional[dict]) -> bool:
    if not isinstance(pricing, dict):
        return False
    try:
        float(pricing["prompt"])
        float(pricing["completion"])
    except (KeyError, TypeError, ValueError):
        return False
    return True


def require_prices(prices: Optional[dict], model_ids: List[str]) -> None:
    missing = [m for m in model_ids if not has_price((prices or {}).get(m))]
    if missing:
        sys.exit(f"No price for model(s) {missing}; refusing to start (the spend cap needs a price per model)")


class SpendTracker:
    """Thread-safe spend accumulator for --budget_usd."""

    def __init__(self, budget_usd: float):
        self.budget = float(budget_usd)
        self.spent = 0.0
        self._lock = threading.Lock()

    def add(self, usd: float) -> None:
        with self._lock:
            self.spent += usd or 0.0

    def exhausted(self) -> bool:
        with self._lock:
            return self.spent >= self.budget

    def summary(self) -> str:
        return f"spent ${self.spent:.4f} of ${self.budget:.4f} budget"


def stats_cost(stats: Optional[dict], pricing: Optional[dict]) -> float:
    """Cost of one LLMService's statistics; estimate tokens count when the provider reported none."""
    if not stats:
        return 0.0
    return usage_cost(stats, pricing)["cost_usd"] or 0.0


def usage_cost(usage: Optional[dict], pricing: Optional[dict]) -> dict:
    """Tokens from the provider when it reported them, else the tiktoken estimates."""
    usage = usage or {}
    if usage.get("api_usage_calls"):
        source = "api"
        tin, tout = usage.get("api_input_tokens", 0), usage.get("api_output_tokens", 0)
    else:
        source = "estimate"
        tin, tout = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    cost = None
    if pricing and usage:
        try:
            cost = tin * float(pricing.get("prompt", 0)) + tout * float(pricing.get("completion", 0))
        except (TypeError, ValueError):
            cost = None
    return {"token_source": source, "input_tokens": tin, "output_tokens": tout,
            "reasoning_tokens": usage.get("api_reasoning_tokens", 0),
            "reasoning_chars": usage.get("api_reasoning_chars", 0),
            "calls": usage.get("total_calls", 0), "cost_usd": cost}


def git_state() -> dict:
    try:
        commit = subprocess.run(["git", "-C", REPO_DIR, "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        # -uno: untracked enumeration takes ~30 s on this NFS tree
        dirty = subprocess.run(["git", "-C", REPO_DIR, "status", "--porcelain", "-uno"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": commit, "dirty": bool(dirty)}
    except Exception as e:
        return {"commit": None, "dirty": None, "error": str(e)}
