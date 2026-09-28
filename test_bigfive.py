import os
import re
import sys
import json
import time
import shutil
import tempfile
import unittest
from collections import defaultdict

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

import utils
import persona_users as pu
import persona_runner as prun
import persona_eval as pev
import persona_analysis as pan
import test_persona_harness as th

BF = pu.load_bigfive()
SENTINEL = "INCOMPLETE-SENTINEL-7f3a"
LINT = re.compile(r"\b(ask|shar|detail|valu|number|volunteer|withh|length|know|question|answer|inform|tell)", re.I)


def jl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


class FakeService:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.api_input_tokens = self.api_output_tokens = self.api_reasoning_chars = 0
        self.failed_calls = 0

    def invoke_messages(self, messages, system_prompt=None, max_retries=5):
        self.calls.append((json.loads(json.dumps(messages)), system_prompt))
        self.api_input_tokens += 10
        self.api_output_tokens += 4
        self.api_reasoning_chars += 3
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            self.failed_calls += 1
            raise r
        return r


def make(arm, svc, tmp):
    return pu.make_user(arm, svc, SENTINEL, "COMPLETE spec: value 42.", os.path.join(tmp, "case", "conversation_log.txt"))


class TestBigFiveUser(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_prompt_and_turns(self):
        svc = FakeService(["<think>hmm</think>Sam Keller: first", "second", "third"])
        u = make("bf_low_A", svc, self.tmp)
        self.assertIsInstance(u, pu.BigFiveUser)
        answers = [u.answer_clarification_question(q) for q in ("Q1?", "Q2?", "Q3?")]
        self.assertEqual(answers, ["first", "second", "third"])
        p, sh = BF["personas"]["bf_low_A"], BF["shared"]
        sp = svc.calls[0][1]
        rem = p["reminder"] + " " + sh["reminder_suffix"]
        self.assertEqual(sp, "\n\n".join([sh["identity"], p["sheet"], sh["situation"], p["appraisal"],
                                          sh["notes_intro"], "<notes>\nCOMPLETE spec: value 42.\n</notes>",
                                          sh["request_intro"], f"<your_request>\n{SENTINEL}\n</your_request>",
                                          sh["rules"]]))
        for msgs, s in svc.calls:
            # the incomplete request appears once, as the user's own message, never in the chat turns
            self.assertEqual(s.count(SENTINEL), 1)
            self.assertNotIn(SENTINEL, json.dumps(msgs))
            self.assertEqual([m["role"] for m in msgs], ["user", "assistant"] * (len(msgs) // 2) + ["user"])
            self.assertTrue(msgs[-1]["content"].endswith("\n\n" + rem))
            for m in msgs[:-1]:
                self.assertNotIn(p["reminder"], m["content"])
        last = svc.calls[2][0]
        self.assertEqual(last[0]["content"], "Message from the assistant:\nQ1?")
        self.assertEqual([last[1]["content"], last[3]["content"]], ["first", "second"])
        self.assertEqual(last[4]["content"], "Message from the assistant:\nQ3?\n\n" + rem)
        lines = jl(os.path.join(self.tmp, "case", "user_calls.jsonl"))
        self.assertEqual([l["turn"] for l in lines], [1, 2, 3])
        self.assertEqual(lines[0]["raw"], "<think>hmm</think>Sam Keller: first")
        self.assertEqual(lines[0]["reply"], "first")
        self.assertEqual(lines[0]["usage"], {"input_tokens": 10, "output_tokens": 4})
        self.assertEqual(lines[0]["reasoning_chars"], 3)
        with open(os.path.join(self.tmp, "case", "user_system_prompt.txt")) as f:
            self.assertEqual(f.read(), sp)
        self.assertEqual([c["deferred"] for c in u.clarifications], [False] * 3)
        with open(u.log_file) as f:
            self.assertIn("A: first", f.read())

    def test_shared_sections_identical(self):
        prompts = {}
        for arm in BF["personas"]:
            svc = FakeService(["ok"])
            make(arm, svc, os.path.join(self.tmp, arm)).answer_clarification_question("Q?")
            prompts[arm] = svc.calls[0][1]
        for arm, sp in prompts.items():
            p = BF["personas"][arm]
            rest = sp.replace(p["sheet"], "SHEET").replace(p["appraisal"], "APPRAISAL")
            self.assertEqual(rest, prompts["bf_neutral"].replace(BF["personas"]["bf_neutral"]["sheet"], "SHEET")
                             .replace(BF["personas"]["bf_neutral"]["appraisal"], "APPRAISAL"))

    def test_failure_reply(self):
        svc = FakeService([RuntimeError("down")])
        u = make("bf_neutral", svc, self.tmp)
        self.assertEqual(u.answer_clarification_question("Q?"), "I am unable to answer that at this time.")
        self.assertEqual(u.clarifications, [])

    def test_cleaning(self):
        c = pu.BigFiveUser.clean_reply
        self.assertEqual(c("<thought reasoning here</thought>Fine."), "Fine.")
        self.assertEqual(c("<thought>reasoning\n</thought\nFine."), "Fine.")
        self.assertEqual(c("Fine, 42.\n<think>trailing reasoning"), "Fine, 42.")
        self.assertEqual(c("Answer 42.</thought"), "Answer 42.")
        self.assertEqual(c("Dr. Keller: Use 42."), "Use 42.")
        self.assertEqual(c("Sam Keller: Use 42."), "Use 42.")
        self.assertEqual(c("<think>only reasoning</think>"), "<think>only reasoning</think>")

    def test_lint_and_ids(self):
        self.assertIn("Sam Keller", BF["shared"]["identity"])
        for pid, p in BF["personas"].items():
            text = " ".join([p["sheet"], p["appraisal"], p["reminder"]])
            self.assertIsNone(LINT.search(text), (pid, LINT.search(text)))
        arms = pu.valid_arms(pu.load_personas(), BF)
        self.assertEqual(len(arms), len(set(arms)))
        self.assertTrue(set(BF["personas"]) <= set(arms))


class FakeClient:
    def __init__(self):
        self.kwargs = []

    def converse(self, **kw):
        self.kwargs.append(kw)
        return {"output": {"message": {"content": [
            {"reasoningContent": {"reasoningText": {"text": "abcdef", "signature": "s"}}}, {"text": "hi"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 5}}


class TestBoto3(unittest.TestCase):
    def test_converse(self):
        made = []
        saved = utils.boto3.client

        def factory(*a, **kw):
            made.append(kw)
            return FakeClient()
        utils.boto3.client = factory
        try:
            svc = utils.LLMService(pu.role_config("bedrock_glm5"))
            self.assertEqual(svc.invoke("hello", "sys"), "hi")
            self.assertEqual(svc.invoke_messages([{"role": "user", "content": "a"},
                                                  {"role": "assistant", "content": "b"},
                                                  {"role": "user", "content": "c"}], "sys"), "hi")
            with self.assertRaises(ValueError):
                svc.invoke_messages([{"role": "assistant", "content": "a"}])
            with self.assertRaises(ValueError):
                svc.invoke_messages([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])
        finally:
            utils.boto3.client = saved
        self.assertEqual(len(made), 1)
        self.assertEqual(made[0]["region_name"], os.getenv("AWS_DEFAULT_REGION", "us-west-2").strip())
        client = svc._boto3_client
        self.assertEqual(client.kwargs[0]["inferenceConfig"], {"temperature": 0.0, "maxTokens": 8192})
        self.assertEqual([m["role"] for m in client.kwargs[1]["messages"]], ["user", "assistant", "user"])
        s = svc.get_statistics()
        self.assertEqual((s["api_usage_calls"], s["api_input_tokens"], s["api_output_tokens"],
                          s["api_reasoning_chars"]), (2, 20, 10, 12))
        self.assertEqual(pu.usage_cost(s, pu.BEDROCK_PRICES["zai.glm-5"])["reasoning_chars"], 12)
        self.assertEqual(pu.role_config("bedrock_glm5").as_dict()["max_tokens"], 8192)
        self.assertNotIn("max_tokens", pu.role_config("or_gpt_5_2").as_dict())


class TestPrices(unittest.TestCase):
    def test_static_no_network(self):
        saved = (pu.requests.get, pu.fetch_openrouter_prices)

        def boom(*a, **k):
            raise AssertionError("network call")
        pu.requests.get = boom
        pu.fetch_openrouter_prices = th._saved["prices"]
        try:
            self.assertEqual(pu.fetch_prices(["zai.glm-5"]), {"zai.glm-5": {"prompt": 1.2e-6, "completion": 3.2e-6}})
        finally:
            pu.requests.get, pu.fetch_openrouter_prices = saved
        with self.assertRaises(SystemExit):
            pu.require_prices({"m": None}, ["m"])


class BFFake(th.FakeLLM):
    def __init__(self, config):
        super().__init__(config)
        self.api_input_tokens = self.api_output_tokens = self.api_reasoning_chars = 0

    def invoke_messages(self, messages, system_prompt=None, max_retries=5):
        self.calls += 1
        with th._lock:
            th.CALLS.append((self.model_version, system_prompt, json.dumps(messages)))
        return "Dr. Keller: Fine, it is 42. Stop wasting my time."

    def invoke(self, user_prompt, system_prompt=None, max_retries=5):
        if system_prompt == pev.BIGFIVE_SYSTEM:
            self.calls += 1
            with th._lock:
                th.CALLS.append((self.model_version, system_prompt, user_prompt))
            letter = re.search(r"^\s*([A-K])\. cold, blunt", user_prompt, re.M).group(1)
            return th.fj({"choice": letter, "second": letter, "intensity": 2, "evidence": "Stop wasting my time."})
        return super().invoke(user_prompt, system_prompt, max_retries)


_saved = {}


def setUpModule():
    for mod in th.PATCHED:
        _saved[mod] = mod.LLMService
        mod.LLMService = BFFake
    _saved["sleep"] = time.sleep
    time.sleep = lambda s: None
    th._saved["prices"] = pu.fetch_openrouter_prices
    pu.fetch_openrouter_prices = lambda ids: {i: {"prompt": "0.000001", "completion": "0.000002"} for i in ids}


def tearDownModule():
    for mod in th.PATCHED:
        mod.LLMService = _saved[mod]
    time.sleep = _saved["sleep"]
    pu.fetch_openrouter_prices = th._saved["prices"]


class TestTraitJudge(unittest.TestCase):
    def test_mask_order_parse(self):
        self.assertEqual(pev.mask_constructs("I'm Disagreeable and neurotic; Big Five personality"),
                         "I'm […] and […]; […] […]")
        o1, o2 = pev.bigfive_order("/r/a/x", "/r"), pev.bigfive_order("/r/b/x", "/r")
        self.assertEqual(o1, pev.bigfive_order("/r/a/x", "/r"))
        self.assertEqual(sorted(o1), sorted(pev.BIGFIVE_OPTIONS))
        self.assertNotEqual(o1, o2)
        turns = [{"question": "Q?", "answer": "I am so neurotic about this, ugh.", "deferred": False}]

        class L:
            def __init__(self, reply):
                self.reply, self.prompts = reply, []

            def invoke(self, u, s=None):
                self.prompts.append(u)
                return self.reply
        letter = "ABCDEFGHIJK"[o1.index("high_neuroticism")]
        llm = L(th.fj({"choice": letter, "second": "Z", "intensity": 3, "evidence": "so neurotic about"}))
        r = pev.bigfive_judge(llm, turns, o1)
        self.assertEqual((r["choice"], r["second"], r["intensity"], r["evidence_ok"], r["parse_failure"]),
                         ("high_neuroticism", None, 3, True, False))
        self.assertIn("Scientist: I am so […] about this", llm.prompts[0])
        llm = L(th.fj({"choice": letter, "intensity": 1, "evidence": "so […] about this"}))
        self.assertEqual(pev.bigfive_judge(llm, turns, o1)["evidence_ok"], True)
        llm = L(th.fj({"choice": letter, "intensity": 1, "evidence": "not said"}))
        self.assertEqual(pev.bigfive_judge(llm, turns, o1)["evidence_ok"], False)
        r = pev.bigfive_judge(L("garbage"), turns, o1)
        self.assertTrue(r["parse_failure"])
        self.assertIsNone(r["choice"])


class TestRunnerBigFive(unittest.TestCase):
    ARMS = ["bf_neutral", "bf_low_A", "paper", "mute"]

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cards = os.path.join(self.tmp, "cards.json")
        th.synthetic_cards(self.cards)
        self.out = os.path.join(self.tmp, "run")
        th.FakeLLM.complete_after = 3

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def argv(self, budget, workers="2"):
        return ["--domain", th.DOMAIN, "--mode", th.MODE, "--cases", ",".join(th.CASE_IDS),
                "--arms", ",".join(self.ARMS), "--agent_model", "or_gpt_5_2", "--user_model", "bedrock_glm5",
                "--workers", workers, "--out", self.out, "--cards", self.cards, "--budget_usd", budget]

    def folder(self, arm, cid):
        return prun.case_dir(self.out, arm, 1, th.DOMAIN, "or_gpt_5_2", th.MODE, cid)

    def test_budget_then_resume(self):
        res = prun.main(self.argv("0.0000001", workers="1"))
        self.assertEqual(res, {"jobs": 8, "skipped": 0, "done": 1, "errors": 0, "skipped_budget": 7})
        runs = [th.jload(os.path.join(self.folder(a, c), "persona_run.json")) for a in self.ARMS for c in th.CASE_IDS]
        self.assertEqual(sorted(r["status"] for r in runs), ["done"] + ["skipped_budget"] * 7)
        self.assertTrue(all(r["attempts"] == 0 for r in runs if r["status"] == "skipped_budget"))
        self.assertEqual(prun.main(self.argv("1000")), {"jobs": 8, "skipped": 1, "done": 7, "errors": 0})

    def test_end_to_end_and_eval(self):
        self.assertEqual(prun.main(self.argv("1000")), {"jobs": 8, "skipped": 0, "done": 8, "errors": 0})
        for cid in th.CASE_IDS:
            for arm in ("bf_neutral", "bf_low_A"):
                d = self.folder(arm, cid)
                run = th.jload(os.path.join(d, "persona_run.json"))
                self.assertEqual(run["trait"], "low_agreeableness" if arm == "bf_low_A" else None)
                self.assertEqual(run["models"]["user"]["max_tokens"], 8192)
                self.assertEqual(len(jl(os.path.join(d, "user_calls.jsonl"))), 2)
                clar = th.jload(os.path.join(d, "clarifications.json"))
                self.assertEqual(clar[0]["answer"], "Fine, it is 42. Stop wasting my time.")
                self.assertIn(f"<your_request>\n{th.CASES[cid]['incomplete_user_req']}\n</your_request>",
                              th.readf(os.path.join(d, "user_system_prompt.txt")))
            self.assertIn("The boundary value is 42.",
                          th.readf(os.path.join(self.folder("paper", cid), "conversation_log.txt")))
        m = th.jload(os.path.join(self.out, "manifest.json"))
        self.assertIn("bigfive/shared", m["sha256"]["personas"])
        self.assertEqual(m["prices"]["zai.glm-5"], pu.BEDROCK_PRICES["zai.glm-5"])

        argv = ["--run_dir", self.out, "--judge_model", "or_gemini_2_5_pro", "--aux_model", "or_gemini_2_5_flash",
                "--workers", "2", "--trait_check", "bigfive", "--budget_usd", "1000"]
        self.assertEqual(pev.main(argv), {"rows": 8, "errors": 0})
        rows = {(r["arm"], r["case_id"]): r for r in jl(os.path.join(self.out, "results.jsonl"))}
        for cid in th.CASE_IDS:
            self.assertEqual(rows[("bf_low_A", cid)]["bf_trait"],
                             {"choice": "low_agreeableness", "second": "low_agreeableness", "intensity": 2,
                              "evidence_ok": True, "target": "low_agreeableness", "hit": True,
                              "parse_failure": False})
            self.assertIsNone(rows[("bf_neutral", cid)]["bf_trait"]["hit"])
            self.assertEqual(rows[("paper", cid)]["bf_trait"]["evidence_ok"], False)
            self.assertIsNone(rows[("mute", cid)]["bf_trait"])
            self.assertIsNone(rows[("paper", cid)]["trait_present_share"])
            cache = th.jload(os.path.join(self.folder("bf_low_A", cid), "persona_bigfive_trait.json"))
            self.assertEqual(cache["prompt_sha256"], pev.BIGFIVE_TEMPLATES_SHA)
            self.assertEqual(cache["order"], pev.bigfive_order(self.folder("bf_low_A", cid), self.out))
        n = len(th.CALLS)
        self.assertEqual(pev.main(argv), {"rows": 8, "errors": 0})
        self.assertEqual(len(th.CALLS), n)
        res = pan.main(["--results", os.path.join(self.out, "results.jsonl"), "--out", os.path.join(self.tmp, "an")])
        bf = {r["arm"]: r for r in res["bigfive_check"]}
        self.assertEqual((bf["bf_low_A"]["n"], bf["bf_low_A"]["hit_rate"]), (2, 1.0))
        self.assertIsNone(bf["bf_neutral"]["hit_rate"])


def bf_rows(n_cases=20):
    shift = {"bf_neutral": 0.0, "bf_neutral_b": 0.0, "bf_low_A": -0.3, "bf_high_N": 0.1, "bf_low_C": 0.0,
             "neutral": 0.2, "terse": -0.2, "neutral_b": 0.2, "paper": 0.0, "oracle": 0.4, "mute": -0.3}
    rows = []
    for c in range(n_cases):
        base = 0.4 + 0.01 * c
        for arm, d in shift.items():
            items = [{"index": i, "asked": True, "delivered": i < 2, "partial": False, "unasked_delivered": False,
                      "unasked_partial": False, "recovered": i < 2, "grounded": False, "silent": False}
                     for i in range(4)]
            rates = {k: sum(it[k] for it in items) / len(items) for k in pan.ITEM_METRICS}
            rates[pan.PRIMARY] = base + d
            rows.append({"arm": arm, "rep": 1, "domain": "pde", "mode": "disambiguation", "case_id": f"c{c}",
                         "native": {"FR": 0, "CGRR": 0, "SRR": 0, "IC": 1}, "items": items, "item_rates": rates,
                         "turns": 3, "questions": 2, "cap_hit": False, "deferred": 0,
                         "trait_present_share": None, "trait_by_turn": None})
    return rows


class TestAnalysisBigFive(unittest.TestCase):
    def test_family_bigfive(self):
        rows = bf_rows()
        res = pan.analyse(rows, family="bigfive")
        self.assertEqual((res["family"], res["reference"]), ("bigfive", "bf_neutral"))
        self.assertEqual(res["placebo_arms"], ["bf_neutral_b"])
        self.assertEqual(res["trait_arms"], ["bf_high_N", "bf_low_A", "bf_low_C"])
        family = sorted(pid for pid, p in pu.load_bigfive()["personas"].items() if p["kind"] == "trait")
        run = ["bf_high_N", "bf_low_A", "bf_low_C"]
        self.assertEqual((res["holm_family"], res["holm_m"], res["holm_missing"]),
                         (family, 3, [a for a in family if a not in run]))
        prim = {r["arm"]: r for r in res["paired"] if r["metric"] == pan.PRIMARY}
        self.assertEqual(sorted(prim), res["trait_arms"])
        self.assertAlmostEqual(prim["bf_low_A"]["effect"], -0.3)
        self.assertAlmostEqual(prim["bf_high_N"]["effect"], 0.1)
        self.assertAlmostEqual(prim["bf_low_C"]["effect"], 0.0)
        self.assertEqual({r["arm"] for r in res["placebo"]}, {"bf_neutral_b"})
        self.assertAlmostEqual([r for r in res["placebo"] if r["metric"] == pan.PRIMARY][0]["effect"], 0.0)
        self.assertEqual({r["arm"] for r in res["selection"]}, set(res["trait_arms"]))
        self.assertEqual({r["arm"] for r in res["decomposition"]}, set(res["trait_arms"]))
        self.assertEqual({r["arm"] for r in res["difficulty"]}, set(res["trait_arms"]))
        gap = {r["arm"]: r for r in res["gap_normalised"]}
        self.assertAlmostEqual(gap["bf_low_A"]["effect"], -0.3 / 0.7)
        md = pan.render_markdown(res)
        self.assertIn(f"Paired effect vs bf_neutral on {pan.PRIMARY}", md)
        self.assertIn("Δ(placebo − bf_neutral)", md)
        self.assertIn("Big Five trait personas", md)
        self.assertNotIn("terse", res["trait_arms"])
        old = pan.analyse(rows)
        self.assertEqual(old["reference"], "neutral")
        self.assertAlmostEqual({r["arm"]: r for r in old["paired"] if r["metric"] == pan.PRIMARY}["terse"]["effect"],
                               -0.4)
        res = pan.analyse([r for r in rows if r["arm"] != "bf_neutral"], family="bigfive")
        self.assertTrue(res["missing"]["bf_neutral"])
        self.assertEqual(res["paired"], [])
        self.assertIn("No `bf_neutral` arm in the results", pan.render_markdown(res))

    def test_bigfive_manipulation_gate(self):
        rows = [dict(r) for r in bf_rows()]
        pole = {"bf_low_A": "low_agreeableness", "bf_high_N": "high_neuroticism", "bf_low_C": "low_conscientiousness"}
        seen = defaultdict(int)
        for r in rows:
            if not r["arm"].startswith("bf_"):
                continue
            k = seen[r["arm"]]
            seen[r["arm"]] += 1
            tgt = pole.get(r["arm"])
            if r["arm"] == "bf_low_A":      # target picked always; neutral never → pass
                choice = tgt
            elif r["arm"] == "bf_high_N":   # target picked in every other dialogue → below 60 %
                choice = tgt if k % 2 == 0 else "ordinary"
            elif r["arm"] == "bf_low_C":    # always picked, but neutral also reads as low C → not 30 pts above
                choice = tgt
            else:
                choice = "low_conscientiousness" if r["arm"] == "bf_neutral" else "ordinary"
            r["bf_trait"] = {"choice": choice, "target": tgt, "hit": (choice == tgt) if tgt else None,
                             "parse_failure": False}
        res = pan.analyse(rows, family="bigfive")
        prim = {r["arm"]: r for r in res["paired"] if r["metric"] == pan.PRIMARY}
        self.assertEqual({a: prim[a]["manip_pass"] for a in pole},
                         {"bf_low_A": True, "bf_high_N": False, "bf_low_C": False})

    def test_failure_table_skipped_budget(self):
        runs = [{"arm": "bf_low_A", "rep": 1, "domain": "pde", "mode": "disambiguation", "case_id": f"c{i}",
                 "status": st, "attempts": att} for i, (st, att) in
                enumerate([("done", 1), ("skipped_budget", 0), ("skipped_budget", 3), ("error", 3)])]
        runs.append(dict(runs[0], arm="bf_neutral"))
        rows = [{"arm": r["arm"], "rep": 1, "domain": "pde", "mode": "disambiguation", "case_id": r["case_id"]}
                for r in runs if r["status"] == "done"]
        fail = {f["arm"]: f for f in pan.failure_table(runs, rows)}
        self.assertEqual(fail["bf_low_A"], {"arm": "bf_low_A", "jobs": 4, "done": 1, "error": 1, "attempts_capped": 1,
                                            "no_eval_row": 3, "skipped_budget": 2})
        self.assertEqual(fail["bf_neutral"], {"arm": "bf_neutral", "jobs": 1, "done": 1, "error": 0,
                                              "attempts_capped": 0, "no_eval_row": 0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
