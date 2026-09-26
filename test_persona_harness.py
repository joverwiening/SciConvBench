import os
import re
import sys
import json
import time
import shutil
import tempfile
import threading
import subprocess
import unittest
import importlib.util

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

import main as sc_main
import test_runner as tr
import llm_judge_disambiguation
import llm_judge_disambiguation_chat
import llm_judge_inconsistency
import llm_judge_inconsistency_chat
import llm_judge_intent
import persona_users as pu
import persona_fact_cards as pfc
import persona_runner as prun
import persona_eval as pev
import persona_analysis as pan
from config import Config
from llm_user_interface import LLMUserInterface

UPSTREAM_COMMIT = "8d1528ed8"
DOMAIN, MODE = "pde", "disambiguation"
CASES = pu.load_cases(DOMAIN, MODE)
CASE_IDS = list(CASES)[:2]
GT_TO_ENTITIES = {c["complete_user_req"].strip(): c["missing_entities"] for c in CASES.values()}

BUILT = []
CALLS = []
_lock = threading.Lock()


def between(tag, text):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def jload(path):
    with open(path) as f:
        return json.load(f)


def readf(path):
    with open(path) as f:
        return f.read()


def fj(obj):
    return f"reasoning...\n<FINAL_JSON>\n{json.dumps(obj)}\n</FINAL_JSON>"


class FakeLLM:
    complete_after = 3

    def __init__(self, config):
        self.model_version = config.model_version
        self.calls = 0
        self.failed_calls = 0
        with _lock:
            BUILT.append(self)

    def invoke(self, user_prompt, system_prompt=None, max_retries=5):
        self.calls += 1
        sp = system_prompt or ""
        with _lock:
            CALLS.append((self.model_version, sp, user_prompt))
        if "Requirement Analyst" in sp:
            if "Time is up" in sp:
                return "FINAL SPEC (forced)"
            if FakeLLM.complete_after and self.calls >= FakeLLM.complete_after:
                return "[COMPLETE]\nUse the boundary value 42."
            return f"What is detail number {self.calls}?"
        if "You are a User who knows the COMPLETE requirement" in sp:
            return "The boundary value is 42." + (" (persona)" if "PERSONA:" in sp else "")
        if "has successfully recovered information that was known to be missing" in sp:
            ents = GT_TO_ENTITIES.get(between("GROUND_TRUTH", user_prompt), ["?"])
            return fj({"score": f"1/{len(ents)}", "recovered_entities": [ents[0], "Nonexistent entity"],
                       "still_missing_entities": ents[1:]})
        if "[MISSING_ENTITIES]" in sp:
            return fj({"questions_about_missing_entities": {"reasoning": "r", "count": 1},
                       "relevant_questions": {"reasoning": "r", "count": 1},
                       "memory_consistency": {"reasoning": "r", "is_consistent": 1}})
        if "faithful to the overall intent" in sp:
            return fj({"score": "1/1", "status": "FAITHFUL", "reasoning": "r"})
        if "You build fact cards" in sp:
            complete = between("COMPLETE_TEXT", user_prompt)
            n_items = len(re.findall(r"^\s*\[\d+\]", between("ITEMS", user_prompt), re.M))
            span = " ".join(complete.split()[:6])
            spans = [span] if "previous answer failed" in user_prompt else [span, "a span that is not in the text"]
            return fj({"items": [{"index": i, "spans": spans, "statement": "s"} for i in range(n_items)]})
        if "You label one turn of a clarification dialogue" in sp:
            cards = json.loads(between("FACT_CARDS", user_prompt))
            answer = between("ANSWER", user_prompt)
            conveyed = [{"card_id": c["card_id"], "level": "full", "evidence": c["text"]}
                        for c in cards if pu.norm_ws(c["text"]) in pu.norm_ws(answer)]
            conveyed.append({"card_id": cards[0]["card_id"], "level": "partial", "evidence": "NOT IN THE ANSWER"})
            conveyed.append({"card_id": "unknown#9", "level": "full", "evidence": "x"})
            return fj({"asked": [c["card_id"] for c in cards], "conveyed": conveyed})
        if "You check which behavioural traits" in sp:
            ids = re.findall(r"^\s*- (\w+):", between("TRAITS", user_prompt), re.M)
            answer = between("ANSWER", user_prompt)
            traits = {t: {"present": 0, "evidence": ""} for t in ids}
            traits["terse"] = {"present": 1, "evidence": " ".join(answer.split()[:3])}
            traits["verbose"] = {"present": 1, "evidence": "NOT IN THE ANSWER"}
            return fj({"traits": traits})
        raise AssertionError(f"FakeLLM: unrecognised system prompt: {sp[:120]!r}")

    def get_statistics(self, name="x"):
        return {"total_calls": self.calls, "failed_calls": self.failed_calls, "total_retries": 0,
                "total_tokens": 30 * self.calls, "prompt_tokens": 20 * self.calls,
                "completion_tokens": 10 * self.calls, "api_usage_calls": self.calls,
                "api_input_tokens": 100 * self.calls, "api_output_tokens": 50 * self.calls,
                "api_reasoning_tokens": 0}


PATCHED = [sc_main, tr, llm_judge_disambiguation, llm_judge_disambiguation_chat, llm_judge_inconsistency,
           llm_judge_inconsistency_chat, llm_judge_intent, prun, pev, pfc]
_saved = {}


def setUpModule():
    for mod in PATCHED:
        _saved[mod] = mod.LLMService
        mod.LLMService = FakeLLM
    _saved["sleep"] = time.sleep
    time.sleep = lambda s: None
    _saved["prices"] = pu.fetch_openrouter_prices
    pu.fetch_openrouter_prices = lambda ids: {i: {"prompt": "0.000001", "completion": "0.000002"} for i in ids}


def tearDownModule():
    for mod in PATCHED:
        mod.LLMService = _saved[mod]
    time.sleep = _saved["sleep"]
    pu.fetch_openrouter_prices = _saved["prices"]


class Recorder:
    def __init__(self):
        self.system_prompts = []

    def invoke(self, user_prompt, system_prompt=None):
        self.system_prompts.append(system_prompt)
        return f"answer {len(self.system_prompts)}"


class TestUserPrompt(unittest.TestCase):
    def test_empty_persona_matches_upstream(self):
        try:
            src = subprocess.run(["git", "-C", REPO, "show", f"{UPSTREAM_COMMIT}:llm_user_interface.py"],
                                 capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("git or upstream commit unavailable")
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "upstream_llm_user_interface.py")
        with open(path, "w") as f:
            f.write(src)
        spec = importlib.util.spec_from_file_location("upstream_llm_user_interface", path)
        up = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(up)
        case = CASES[CASE_IDS[0]]
        prompts = []
        for cls in (up.LLMUserInterface, LLMUserInterface):
            rec = Recorder()
            u = cls(llm_service=rec, incomplete_requirement=case["incomplete_user_req"],
                    complete_requirement=case["complete_user_req"], log_file=os.path.join(tmp, f"{cls.__module__}.log"))
            u.answer_clarification_question("What is the domain?")
            u.answer_clarification_question("What is the boundary condition?")
            prompts.append(rec.system_prompts)
        self.assertEqual(len(prompts[0]), 2)
        self.assertEqual(prompts[0], prompts[1])
        shutil.rmtree(tmp)

    def test_persona_block_appended(self):
        tmp = tempfile.mkdtemp()
        case = CASES[CASE_IDS[0]]
        base = LLMUserInterface(None, case["incomplete_user_req"], case["complete_user_req"],
                                log_file=os.path.join(tmp, "a.log"))
        text = pu.render_persona("terse")
        withp = LLMUserInterface(None, case["incomplete_user_req"], case["complete_user_req"],
                                 log_file=os.path.join(tmp, "b.log"), persona_text=text)
        self.assertTrue(withp.build_system_prompt().startswith(base.build_system_prompt()))
        self.assertIn(text, withp.build_system_prompt())
        self.assertTrue(text.startswith(pu.PRECEDENCE))
        self.assertEqual(pu.persona_text_for_arm("paper"), "")
        shutil.rmtree(tmp)


class TestApiUsage(unittest.TestCase):
    def test_usage_metadata_recorded(self):
        from langchain_core.messages import AIMessage
        from utils import LLMService
        svc = LLMService(pu.role_config("bedrock_4_6_sonnet_sdk"))  # claude_sdk: no client built
        svc._record_api_usage(AIMessage(content="x", usage_metadata={
            "input_tokens": 3, "output_tokens": 5, "total_tokens": 8, "output_token_details": {"reasoning": 2}}))
        svc._record_api_usage(AIMessage(content="y"))
        s = svc.get_statistics()
        self.assertEqual((s["api_usage_calls"], s["api_input_tokens"], s["api_output_tokens"],
                          s["api_reasoning_tokens"]), (1, 3, 5, 2))
        self.assertEqual(s["prompt_tokens"], 0)


class TestRunSingleCase(unittest.TestCase):
    def test_upstream_entry_point_unchanged(self):
        saved = (Config.active_conf, Config.model_provider, Config.model_version, Config.temperature)
        FakeLLM.complete_after = None
        BUILT.clear()
        try:
            Config.active_conf = Config.LLM_PROFILES["or_gpt_5_2"]
            Config.model_provider = Config.active_conf["provider"]
            Config.model_version = Config.active_conf["model_version"]
            Config.temperature = 0.0
            case = CASES[CASE_IDS[0]]
            out = tempfile.mkdtemp()
            tr.run_single_case(case, tr.TASK_CONFIG["disambiguation"], "disambiguation", out,
                               user_profile="or_sonnet_4_6", max_turns=4)
            agent, user = BUILT[0], BUILT[1]
            self.assertEqual(agent.model_version, "openai/gpt-5.2")
            self.assertEqual(user.model_version, "anthropic/claude-sonnet-4.6")
            self.assertEqual(agent.calls, 4 + 1)
            self.assertEqual(user.calls, 4)
            d = os.path.join(out, case["id"])
            self.assertEqual(readf(os.path.join(d, "generated_prompt.txt")), "FINAL SPEC (forced)")
            stats = jload(os.path.join(d, "statistics.json"))
            self.assertEqual((stats["max_turns_allowed"], stats["turns_needed"]), (4, 4))
            self.assertEqual(stats["result"], "SUCCESS")
            ustats = jload(os.path.join(d, "user_statistics.json"))
            self.assertEqual(ustats["user_model"], "anthropic/claude-sonnet-4.6")
            shutil.rmtree(out)
        finally:
            FakeLLM.complete_after = 3
            Config.active_conf, Config.model_provider, Config.model_version, Config.temperature = saved


ARMS = ["paper", "neutral", "terse", "oracle", "mute"]
RUN_FILES = ["conversation_log.txt", "generated_prompt.txt", "complete_prompt.txt", "statistics.json",
             "user_statistics.json", "persona_run.json"]


def synthetic_cards(path):
    data = {"meta": {"model": None}, "cases": {}}
    for cid in CASE_IDS:
        cards = []
        for i, item in enumerate(CASES[cid]["missing_entities"]):
            cards.append({"card_id": f"{DOMAIN}/{MODE}/{cid}#{i}", "case_id": cid, "index": i, "item": item,
                          "spans": [], "statement": "", "text": f"FACT {cid} number {i}",
                          "checks": {}, "status": "ok"})
        data["cases"][cid] = {"status": "ok", "cards": cards}
    with open(path, "w") as f:
        json.dump(data, f)


class TestRunnerAndEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.cards = os.path.join(cls.tmp, "cards.json")
        synthetic_cards(cls.cards)
        cls.out = os.path.join(cls.tmp, "run")
        cls.argv = ["--domain", DOMAIN, "--mode", MODE, "--cases", ",".join(CASE_IDS), "--arms", ",".join(ARMS),
                    "--agent_model", "or_gpt_5_2", "--user_model", "or_sonnet_4_6", "--reps", "2",
                    "--workers", "4", "--out", cls.out, "--cards", cls.cards]
        cls.first = prun.main(cls.argv)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def folder(self, arm, rep, cid):
        return prun.case_dir(self.out, arm, rep, DOMAIN, "or_gpt_5_2", MODE, cid)

    def test_a_tree_and_manifest(self):
        self.assertEqual(self.first, {"jobs": 20, "skipped": 0, "done": 20, "errors": 0})
        for arm in ARMS:
            for rep in (1, 2):
                for cid in CASE_IDS:
                    d = self.folder(arm, rep, cid)
                    for f in RUN_FILES:
                        self.assertTrue(os.path.exists(os.path.join(d, f)), (d, f))
                    run = jload(os.path.join(d, "persona_run.json"))
                    self.assertEqual(run["status"], "done")
                    self.assertEqual((run["turns"], run["completed_by_agent"], run["cap_hit"]), (3, True, False))
                    if arm in pu.DETERMINISTIC_ARMS:
                        self.assertIsNone(run["models"]["user"])
                    else:
                        self.assertEqual(run["persona_text_sha256"],
                                         pu.sha256_text(pu.persona_text_for_arm(arm)))
        log = readf(os.path.join(self.folder("oracle", 1, CASE_IDS[0]), "conversation_log.txt"))
        self.assertIn(f"FACT {CASE_IDS[0]} number 0", log)
        log = readf(os.path.join(self.folder("terse", 1, CASE_IDS[0]), "conversation_log.txt"))
        self.assertIn("(persona)", log)
        log = readf(os.path.join(self.folder("paper", 1, CASE_IDS[0]), "conversation_log.txt"))
        self.assertNotIn("(persona)", log)
        m = jload(os.path.join(self.out, "manifest.json"))
        self.assertEqual(m["roles"]["agent"]["model_version"], "openai/gpt-5.2")
        self.assertEqual(m["roles"]["user"]["model_version"], "anthropic/claude-sonnet-4.6")
        self.assertIn("agent_system_prompt", m["sha256"])
        self.assertEqual(set(m["sha256"]["personas"]), set(pu.load_personas()))
        self.assertEqual(m["cards"][f"{DOMAIN}/{MODE}"]["sha256"], pu.sha256_file(self.cards))

    def test_b_resume_skips_done(self):
        n_built = len(BUILT)
        again = prun.main(self.argv)
        self.assertEqual(again, {"jobs": 20, "skipped": 20, "done": 0, "errors": 0})
        self.assertEqual(len(BUILT), n_built + 1)  # only the manifest-check ConvAgent
        path = os.path.join(self.folder("terse", 2, CASE_IDS[1]), "persona_run.json")
        run = jload(path)
        run["status"] = "error"
        with open(path, "w") as f:
            json.dump(run, f)
        self.assertEqual(prun.main(self.argv), {"jobs": 20, "skipped": 19, "done": 1, "errors": 0})
        self.assertEqual(jload(path)["status"], "done")
        self.assertEqual(readf(os.path.join(self.folder("terse", 2, CASE_IDS[1]), "conversation_log.txt"))
                         .count("=== "), 2)

    def test_c_manifest_mismatch_refused(self):
        argv = list(self.argv)
        argv[argv.index("--agent_model") + 1] = "or_sonnet_4_6"
        with self.assertRaises(SystemExit):
            prun.main(argv)
        argv = list(self.argv)
        argv[argv.index("--user_model") + 1] = "or_gpt_5_2"
        with self.assertRaises(SystemExit):
            prun.main(argv)

    def test_d_eval(self):
        argv = ["--run_dir", self.out, "--judge_model", "or_gemini_2_5_pro", "--aux_model", "or_gemini_2_5_flash",
                "--workers", "4"]
        res = pev.main(argv)
        self.assertEqual(res, {"rows": 20, "errors": 0})
        d = self.folder("neutral", 1, CASE_IDS[0])
        for f in pev.NATIVE_FILES[MODE] + ["persona_eval.json", "persona_attribution.json", "persona_traits.json"]:
            self.assertTrue(os.path.exists(os.path.join(d, f)), f)
        fr = jload(os.path.join(d, "llm_judge_disambiguation_score.json"))
        self.assertIn("strict_score", fr)
        chat = jload(os.path.join(d, "llm_judge_disambiguation_chat_score.json"))
        self.assertEqual(chat["total_questions_asked_in_chat"], 2)
        self.assertIn("metrics", chat)
        self.assertFalse(os.path.exists(os.path.join(self.folder("oracle", 1, CASE_IDS[0]), "persona_traits.json")))

        rows = [json.loads(l) for l in readf(os.path.join(self.out, "results.jsonl")).splitlines()]
        self.assertEqual(len(rows), 20)
        for r in rows:
            self.assertEqual(r["questions"], 2)
            self.assertIsNotNone(r["native"])
            self.assertEqual(r["native"]["IC"], 1)
            self.assertEqual(r["items"][0]["recovered"], True)
            self.assertIn("Nonexistent entity", r["unmatched_recovered_entities"])
            self.assertGreater(r["dropped"]["attribution_evidence"], 0)
            self.assertGreater(r["dropped"]["attribution_unknown_ids"], 0)
            self.assertEqual(r["parse_failures"], {"native": 0, "attribution": 0, "traits": 0})
            self.assertEqual(r["usage"]["agent"]["token_source"], "api")
            self.assertIsNotNone(r["usage"]["judge"]["cost_usd"])
            if r["arm"] == "oracle":
                self.assertTrue(all(it["delivered"] for it in r["items"]))
                self.assertEqual(r["item_rates"]["delivered"], 1.0)
            else:
                self.assertTrue(not any(it["delivered"] for it in r["items"]))
            if r["arm"] in pu.DETERMINISTIC_ARMS:
                self.assertIsNone(r["trait_present_share"])
            else:
                self.assertEqual(r["trait_present_share"]["terse"], 1.0)
                self.assertEqual(r["trait_present_share"]["verbose"], 0.0)
                self.assertGreater(r["dropped"]["trait_evidence"], 0)

        n_calls = len(CALLS)
        self.assertEqual(pev.main(argv), {"rows": 20, "errors": 0})
        self.assertEqual(len(CALLS), n_calls)  # all judge/aux results reused

        res = pan.main(["--results", os.path.join(self.out, "results.jsonl"), "--out", os.path.join(self.tmp, "an")])
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "an", "analysis.md")))
        self.assertEqual(res["attribution_sanity"]["oracle_full_delivery_rate"], 1.0)
        self.assertEqual(res["attribution_sanity"]["mute_any_delivery_rate"], 0.0)


class TestFactCards(unittest.TestCase):
    def test_coverage(self):
        cov = pfc.coverage("a b c", "a b X Y c", ["X"])
        self.assertEqual(cov["changed_tokens"], 2)
        self.assertEqual(cov["uncovered_tokens"], ["Y"])
        self.assertAlmostEqual(cov["uncovered_share"], 0.5)

    def test_non_verbatim_span_rejected(self):
        case = CASES[CASE_IDS[0]]
        words = case["complete_user_req"].split()
        good = "  ".join(words[3:9])  # whitespace differs from the source: still verbatim after normalisation
        rec = pfc.check_case(DOMAIN, MODE, case, [{"index": 0, "spans": [good, "made-up span 123"],
                                                   "statement": "s"}], 1.0)
        card = rec["cards"][0]
        self.assertEqual(card["spans"], [good])
        self.assertEqual(card["checks"]["dropped_spans"], ["made-up span 123"])
        self.assertEqual(card["status"], "needs_review")
        self.assertEqual(card["text"], good)
        self.assertEqual(card["card_id"], f"{DOMAIN}/{MODE}/{case['id']}#0")
        self.assertTrue(0.0 <= rec["uncovered_share"] <= 1.0)
        self.assertGreater(rec["changed_tokens"], 0)
        rec = pfc.check_case(DOMAIN, MODE, case, [{"index": 0, "spans": ["made-up span 123"]}], 1.0)
        self.assertEqual(rec["cards"][0]["status"], "failed")
        self.assertEqual(rec["status"], "failed")

    def test_build_with_retry_and_resume(self):
        tmp = tempfile.mkdtemp()
        out = os.path.join(tmp, "cards.json")
        argv = ["--domain", DOMAIN, "--mode", MODE, "--model", "or_gpt_5_2", "--cases", CASE_IDS[0],
                "--out", out, "--max_uncovered_share", "1.0"]
        n0 = len(CALLS)
        data = pfc.main(argv)
        rec = data["cases"][CASE_IDS[0]]
        self.assertEqual(rec["attempts"], 2)
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(len(CALLS) - n0, 2)
        self.assertEqual(data["meta"]["source_sha256"], pu.sha256_file(pu.data_path(DOMAIN, MODE)))
        pfc.main(argv)
        self.assertEqual(len(CALLS) - n0, 2)
        data = pfc.main(argv + ["--verify_only"])
        self.assertEqual(data["cases"][CASE_IDS[0]]["status"], "ok")
        shutil.rmtree(tmp)


class TestAnalysis(unittest.TestCase):
    def test_planted_effect(self):
        import random
        rng = random.Random(1)
        p = {"neutral": 0.6, "terse": 0.3, "oracle": 0.9, "mute": 0.1}
        rows = []
        for c in range(40):
            for arm, pr in p.items():
                for rep in (1, 2):
                    items = []
                    for i in range(3):
                        delivered = arm == "oracle" or (arm != "mute" and rng.random() < pr)
                        rec = rng.random() < (0.9 if delivered else 0.1) if arm not in ("oracle", "mute") \
                            else rng.random() < pr
                        items.append({"index": i, "asked": True, "delivered": delivered, "partial": False,
                                      "recovered": rec, "grounded": rec and delivered,
                                      "silent": rec and not delivered})
                    rate = lambda k: sum(it[k] for it in items) / len(items)
                    fr = int(all(it["recovered"] for it in items))
                    rows.append({
                        "arm": arm, "rep": rep, "domain": "pde", "mode": "disambiguation", "case_id": f"c{c}",
                        "native": {"FR": fr, "CGRR": fr, "SRR": 0, "IC": 1, "MC": 1, "CR": 1.0, "CP": 1.0},
                        "items": items,
                        "item_rates": {k: rate(k) for k in pan.ITEM_METRICS},
                        "trait_present_share": None if arm in ("oracle", "mute") else {"terse": float(arm == "terse")},
                        "trait_by_turn": None if arm in ("oracle", "mute") else
                        [{"turn": 1, "present": {"terse": int(arm == "terse")}}],
                    })
        res = pan.analyse(rows)
        eff = [e for e in res["paired"] if e["arm"] == "terse" and e["metric"] == "recovered"][0]
        self.assertLess(eff["effect"], 0)
        self.assertLess(eff["ci_hi"], 0)
        gap = [e for e in res["gap_normalised"] if e["arm"] == "terse"][0]
        self.assertLess(gap["effect"], 0)
        self.assertLess(gap["ci_hi"], 0)
        dec = {d["quantity"]: d for d in res["decomposition"] if d["arm"] == "terse"}
        self.assertLess(dec["delta_delivered_rate"]["ci_hi"], 0)
        man = {m["arm"]: m for m in res["manipulation"]}
        self.assertEqual(man["terse"]["on_target"], 1.0)
        self.assertEqual(man["neutral"]["terse"], 0.0)
        md = pan.render_markdown(res)
        self.assertIn("Paired effect vs neutral", md)


if __name__ == "__main__":
    unittest.main(verbosity=2)
