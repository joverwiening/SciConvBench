import os
import re
import json
import time
import hashlib
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

import requests

from config import Config
from llm_user_interface import LLMUserInterface

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
PERSONA_DIR = os.path.join(REPO_DIR, "persona", "personas")

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


def valid_arms(personas: Dict[str, dict]) -> List[str]:
    return ["paper", "neutral"] + placebo_ids(personas) + trait_ids(personas) + list(DETERMINISTIC_ARMS)


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


def make_user(arm: str, llm_service, incomplete_requirement: str, complete_requirement: str,
              log_file: str, cards: Optional[List[dict]] = None,
              personas: Optional[Dict[str, dict]] = None) -> LLMUserInterface:
    common = dict(incomplete_requirement=incomplete_requirement,
                  complete_requirement=complete_requirement, log_file=log_file)
    if arm == "oracle":
        if not cards:
            raise ValueError("oracle arm needs fact cards for this case")
        return OracleUser(llm_service=None, cards=cards, **common)
    if arm == "mute":
        return MuteUser(llm_service=None, **common)
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

    def as_dict(self) -> dict:
        return {"profile": self.profile, "provider": self.model_provider,
                "model_version": self.model_version, "temperature": self.temperature}


def role_config(profile: str, temperature: Optional[float] = None) -> RoleConfig:
    prof = Config.LLM_PROFILES[profile]
    return RoleConfig(profile, prof["provider"], prof["model_version"],
                      prof["temperature"] if temperature is None else temperature)


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
