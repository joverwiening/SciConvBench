import os
import re
import ast
import sys
import json
import time
import types
import inspect
import shutil
import tempfile
import subprocess
import unittest

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

import persona_users as pu
import persona_runner as prun
import persona_tone_replay as ptr
import test_runner as tr

BF = pu.load_bigfive()
FULL1_COMMIT = "4c4b51d03"
INC, COMPLETE = "INCOMPLETE request", "COMPLETE: gravity 9.81 m/s2 in buoyantFoam, p_rgh 1e5."
ORIG = ["Use 9.81 for gravity in buoyantFoam.", "Set p_rgh to 1e5, thanks."]
PRICE = {"prompt": 1e-6, "completion": 2e-6}


def jl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def readf(path):
    with open(path) as f:
        return f.read()


def between(tag, text):
    return re.search(rf"<{tag}>\n(.*?)\n</{tag}>", text, re.S).group(1)


class FakeUserLLM:
    """Routes by content: invoke_messages = inner sim; invoke with REWRITE_SYSTEM = restyle; EQUIV_SYSTEM = judge.
    rewrite(orig, attempt) -> reply text or None (unparseable); same_info(orig, attempt) -> judge verdict."""

    def __init__(self, config=None, originals=ORIG, rewrite=None, same_info=None, fail_inner=False):
        self.originals, self.fail_inner = list(originals), fail_inner
        self.rewrite = rewrite or (lambda o, k: "Fine. " + o)
        self.same_info = same_info or (lambda o, k: True)
        self.inner_calls, self.rewrite_prompts, self.judge_prompts = [], [], []
        self.attempt = {}
        self.total_calls = self.failed_calls = self.api_usage_calls = 0
        self.api_input_tokens = self.api_output_tokens = self.api_reasoning_chars = 0

    def _count(self):
        self.total_calls += 1
        self.api_usage_calls += 1
        self.api_input_tokens += 10
        self.api_output_tokens += 4

    def invoke_messages(self, messages, system_prompt=None, max_retries=5):
        self._count()
        self.inner_calls.append((json.loads(json.dumps(messages)), system_prompt))
        if self.fail_inner:
            self.failed_calls += 1
            raise RuntimeError("api down")
        return self.originals.pop(0)

    def invoke(self, user_prompt, system_prompt=None, max_retries=5):
        self._count()
        if system_prompt == ptr.REWRITE_SYSTEM:
            self.rewrite_prompts.append(user_prompt)
            orig = between("reply", user_prompt)
            k = self.attempt[orig] = self.attempt.get(orig, 0) + 1
            new = self.rewrite(orig, k)
            return "not json" if new is None else json.dumps({"reply": new})
        if system_prompt == ptr.EQUIV_SYSTEM:
            self.judge_prompts.append(user_prompt)
            orig = re.search(r"\[1\] ORIGINAL: (.*)\n\[1\] REWRITTEN:", user_prompt).group(1)
            same = self.same_info(orig, self.attempt[orig])
            return json.dumps({"pairs": [{"i": 1, "same_info": same, "diff": "" if same else "dropped a value"}]})
        raise AssertionError(f"unrouted call: {system_prompt!r}")

    def get_statistics(self):
        return {"total_calls": self.total_calls, "failed_calls": self.failed_calls, "api_usage_calls":
                self.api_usage_calls, "api_input_tokens": self.api_input_tokens,
                "api_output_tokens": self.api_output_tokens, "api_reasoning_tokens": 0,
                "api_reasoning_chars": self.api_reasoning_chars, "prompt_tokens": 0, "completion_tokens": 0}


class FakeAgentLLM:
    def get_statistics(self):
        return {"total_calls": 3, "failed_calls": 0, "api_usage_calls": 3, "api_input_tokens": 100,
                "api_output_tokens": 50, "api_reasoning_tokens": 0, "api_reasoning_chars": 0,
                "prompt_tokens": 0, "completion_tokens": 0}


class FakeAgent:
    built = []

    def __init__(self, cfg=None, mode=None):
        self.inputs, self.llm = [], FakeAgentLLM()
        FakeAgent.built.append(self)

    def chat(self, x):
        self.inputs.append(x)
        n = len(self.inputs)
        return f"Question {n}?" if n <= len(ORIG) else "[COMPLETE]\nFinal spec: 9.81 and 1e5."

    def get_final_attempt(self):
        return None


def make(arm, svc, tmp):
    return pu.make_user(arm, svc, INC, COMPLETE, os.path.join(tmp, "case", "conversation_log.txt"), bigfive=BF)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._sleep = time.sleep
        time.sleep = lambda s: None

    def tearDown(self):
        time.sleep = self._sleep
        shutil.rmtree(self.tmp)


class TestRunJob(Base):
    def run_arm(self, arm):
        svcs = []

        def service(cfg):
            svcs.append(FakeUserLLM(cfg))
            return svcs[-1]

        saved = prun.LLMService, prun.ConvAgent
        prun.LLMService, prun.ConvAgent = service, FakeAgent
        FakeAgent.built = []
        keys = tr.TASK_CONFIG["disambiguation"]["keys"]
        ctx = {"args": types.SimpleNamespace(out=self.tmp, domain="foam", mode="disambiguation", max_turns=10,
                                             baseline=False),
               "cases": {"c1": {keys["input_prompt"]: INC, keys["ground_truth"]: COMPLETE}},
               "cards": None, "personas": pu.load_personas(), "bigfive": BF, "traits": set(),
               "agent_cfg": pu.RoleConfig("fake_agent", "x", "agent-model", 0.0),
               "user_cfg": pu.RoleConfig("fake_user", "x", "user-model", 0.7),
               "prices": {"agent-model": PRICE, "user-model": PRICE}, "spend": pu.SpendTracker(100)}
        try:
            rec = prun.run_job((arm, 1, "c1"), ctx)
        finally:
            prun.LLMService, prun.ConvAgent = saved
        folder = prun.case_dir(self.tmp, arm, 1, "foam", "fake_agent", "disambiguation", "c1")
        return rec, ctx, svcs[0], FakeAgent.built[0], folder

    def test_restyled_reaches_agent_originals_stay_inner(self):
        rec, ctx, svc, agent, folder = self.run_arm("rs_lowA")
        restyled = ["Fine. " + o for o in ORIG]
        self.assertEqual((rec["status"], rec["error"], rec["turns"], rec["truth_retries"], rec["deferred"]),
                         ("done", None, 3, 0, 0))
        self.assertIsNone(rec["trait"])
        self.assertEqual(rec["persona_text_sha256"], pu.sha256_text(pu.restyle_persona_text("rs_lowA", BF)))
        self.assertNotEqual(rec["persona_text_sha256"],
                            pu.sha256_text(pu.bigfive_persona_text(BF["personas"]["bf_neutral"])))
        self.assertEqual(agent.inputs, [INC] + restyled)
        clar = jl(os.path.join(folder, "restyle_log.jsonl"))
        self.assertEqual([(r["turn"], r["original"], r["restyled"], r["attempts"], r["fallback"], r["code_ok"],
                           r["judge_same"]) for r in clar],
                         [(k, o, n, 1, False, True, True) for k, (o, n) in enumerate(zip(ORIG, restyled), 1)])
        self.assertEqual(clar[0]["usage"], {"input_tokens": 20, "output_tokens": 8})
        saved = json.load(open(os.path.join(folder, "clarifications.json")))
        self.assertEqual([c["answer"] for c in saved], restyled)
        self.assertEqual([c["question"] for c in saved], ["Question 1?", "Question 2?"])
        log, orig_log = (readf(os.path.join(folder, f)) for f in
                         ("conversation_log.txt", "conversation_log_original.txt"))
        for o, n in zip(ORIG, restyled):
            self.assertIn(f"A: {n}\n", log)
            self.assertIn(f"A: {o}\n", orig_log)
        self.assertNotIn("Fine.", orig_log)
        self.assertNotIn("Thinking...", log + orig_log)
        # inner sim memory: its own original reply, not the restyled one
        msgs = svc.inner_calls[1][0]
        self.assertIn({"role": "assistant", "content": ORIG[0]}, msgs)
        self.assertNotIn("Fine.", json.dumps(msgs))
        # restyle prompt: lowA style, agent-side history for voice consistency
        self.assertIn(ptr.LOWA_STYLE_TAIL, svc.rewrite_prompts[0])
        self.assertIn(ptr.REWRITE_RULES, svc.rewrite_prompts[0])
        self.assertIn("(none yet)", svc.rewrite_prompts[0])
        self.assertIn(f"[1] Q: Question 1?\n[1] A: {restyled[0]}", svc.rewrite_prompts[1])
        self.assertIn(f"<question>\nQuestion 2?\n</question>", svc.rewrite_prompts[1])
        self.assertIn(INC, svc.rewrite_prompts[1])
        # cost: 2 inner + 2 restyle + 2 judge calls, all on the user service
        us = rec["usage"]["user"]
        self.assertEqual((us["total_calls"], us["api_input_tokens"], us["api_output_tokens"]), (6, 60, 24))
        agent_cost = pu.stats_cost(FakeAgentLLM().get_statistics(), PRICE)
        self.assertAlmostEqual(ctx["spend"].spent, agent_cost + 60 * 1e-6 + 24 * 2e-6)
        ustats = json.load(open(os.path.join(folder, "user_statistics.json")))
        self.assertEqual(ustats["user_llm_usage"]["total_calls"], 6)
        self.assertEqual(ustats["user_arm"], "rs_lowA")

    def test_para_style_prompt(self):
        rec, ctx, svc, agent, folder = self.run_arm("rs_para")
        self.assertEqual(rec["status"], "done")
        self.assertIn(ptr.PARA_STYLE, svc.rewrite_prompts[0])
        self.assertNotIn(ptr.LOWA_STYLE_TAIL, svc.rewrite_prompts[0])


class TestRestyledUser(Base):
    def test_retry_then_success(self):
        svc = FakeUserLLM(rewrite=lambda o, k: "Fine." if k == 1 else "Fine. " + o)
        u = make("rs_para", svc, self.tmp)
        self.assertEqual(u.answer_clarification_question("Q1?"), "Fine. " + ORIG[0])
        self.assertEqual(len(svc.rewrite_prompts), 2)
        self.assertIn(pu.RESTYLE_RETRY, svc.rewrite_prompts[1])
        self.assertIn("numbers or names missing or changed: 9.81, buoyantFoam", svc.rewrite_prompts[1])
        self.assertEqual(len(svc.judge_prompts), 2)
        row = jl(u.restyle_file)[0]
        self.assertEqual((row["attempts"], row["fallback"], row["code_ok"]), (2, False, True))

    def test_retry_then_fallback(self):
        svc = FakeUserLLM(same_info=lambda o, k: False)
        u = make("rs_lowA", svc, self.tmp)
        self.assertEqual(u.answer_clarification_question("Q1?"), ORIG[0])
        self.assertIn("information differs: dropped a value", svc.rewrite_prompts[1])
        row = jl(u.restyle_file)[0]
        self.assertEqual((row["attempts"], row["fallback"], row["code_ok"], row["judge_same"], row["restyled"]),
                         (2, True, True, False, "Fine. " + ORIG[0]))
        self.assertEqual(row["judge_diff"], "dropped a value")
        self.assertEqual(u.clarifications[0]["answer"], ORIG[0])
        self.assertEqual(u.fallbacks, 1)
        self.assertIn(f"A: {ORIG[0]}\n", readf(u.log_file))

    def test_unparseable_twice_fallback(self):
        svc = FakeUserLLM(rewrite=lambda o, k: None)
        u = make("rs_lowA", svc, self.tmp)
        self.assertEqual(u.answer_clarification_question("Q1?"), ORIG[0])
        row = jl(u.restyle_file)[0]
        self.assertEqual((row["attempts"], row["fallback"], row["restyled"]), (2, True, None))
        self.assertEqual(svc.judge_prompts, [])

    def test_canned_failure_passthrough(self):
        svc = FakeUserLLM(fail_inner=True)
        u = make("rs_lowA", svc, self.tmp)
        self.assertEqual(u.answer_clarification_question("Q1?"), pu.USER_FAILURE_REPLY)
        self.assertEqual((svc.rewrite_prompts, u.clarifications, u.inner.clarifications), ([], [], []))
        row = jl(u.restyle_file)[0]
        self.assertEqual((row["inner_failed"], row["attempts"], row["fallback"]), (True, 0, False))
        # same records a plain BigFiveUser leaves after a failed call
        self.assertIn("A: Thinking...", readf(u.log_file))

    def test_rerun_truncates_files(self):
        for _ in range(2):
            u = make("rs_para", FakeUserLLM(), self.tmp)
            u.answer_clarification_question("Q1?")
        self.assertEqual(len(jl(u.restyle_file)), 1)
        self.assertEqual(readf(u.inner.log_file).count("Q: Q1?"), 1)


class TestUnchangedBigFive(Base):
    def test_bigfive_source_matches_full1(self):
        try:
            old = subprocess.run(["git", "-C", REPO, "show", f"{FULL1_COMMIT}:persona_users.py"],
                                 capture_output=True, text=True, check=True, timeout=300).stdout
        except Exception as e:
            self.skipTest(f"git show failed: {e}")
        node = next(n for n in ast.parse(old).body if isinstance(n, ast.ClassDef) and n.name == "BigFiveUser")
        lines = old.splitlines(keepends=True)
        start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
        self.assertEqual("".join(lines[start - 1:node.end_lineno]), inspect.getsource(pu.BigFiveUser))

    def test_inner_prompts_equal_plain_bf_neutral(self):
        svc = FakeUserLLM()
        u = make("rs_lowA", svc, self.tmp)
        self.assertIs(type(u.inner), pu.BigFiveUser)
        for q in ("Q1?", "Q2?"):
            u.answer_clarification_question(q)
        ref_svc = FakeUserLLM()
        ref = make("bf_neutral", ref_svc, os.path.join(self.tmp, "ref"))
        for q in ("Q1?", "Q2?"):
            ref.answer_clarification_question(q)
        self.assertEqual(svc.inner_calls, ref_svc.inner_calls)
        self.assertEqual(readf(os.path.join(self.tmp, "case", "user_system_prompt.txt")),
                         readf(os.path.join(self.tmp, "ref", "case", "user_system_prompt.txt")))
        self.assertEqual(len(jl(os.path.join(self.tmp, "case", "user_calls.jsonl"))), 2)


class TestDispatchAndManifest(Base):
    def test_make_user_and_valid_arms(self):
        for arm, style, text in (("rs_para", "para", ptr.PARA_STYLE), ("rs_lowA", "lowA", ptr.lowa_style(BF))):
            u = make(arm, FakeUserLLM(), os.path.join(self.tmp, arm))
            self.assertIsInstance(u, pu.RestyledUser)
            self.assertEqual((u.style, u.style_text), (style, text))
            self.assertEqual(u.inner.persona, BF["personas"]["bf_neutral"])
            self.assertTrue(u.inner.log_file.endswith("conversation_log_original.txt"))
        self.assertIs(type(make("bf_neutral", FakeUserLLM(), self.tmp)), pu.BigFiveUser)
        valid = pu.valid_arms(pu.load_personas(), BF)
        self.assertTrue({"rs_para", "rs_lowA", "bf_neutral"} <= set(valid))

    def test_persona_hashes(self):
        h = prun.persona_hashes(pu.load_personas(), BF)
        self.assertEqual(h["rs_para"], pu.sha256_text(pu.restyle_persona_text("rs_para", BF)))
        self.assertEqual(len({h["rs_para"], h["rs_lowA"], h["bf_neutral"]}), 3)

    def test_restyle_sha_and_manifest_check(self):
        sha = prun.restyle_sha(BF)
        self.assertEqual(sha, prun.restyle_sha(BF))
        saved = ptr.PARA_STYLE
        ptr.PARA_STYLE = saved + " x"
        try:
            self.assertNotEqual(prun.restyle_sha(BF), sha)
        finally:
            ptr.PARA_STYLE = saved
        man = {"roles": {"agent": {}, "user": {}}, "baseline": False, "max_turns": 10, "final_fix": True,
               "sha256": {"agent_system_prompt": "a", "user_prompt_template": "u", "personas": {}, "restyle": sha},
               "cards": {"foam/disambiguation": None}, "prices": {}}
        args = types.SimpleNamespace(out=self.tmp, domain="foam", mode="disambiguation", force=False)
        saved_build = prun.build_manifest
        try:
            prun.build_manifest = lambda *a: json.loads(json.dumps(man))
            prun.check_or_write_manifest(args, None, None, None, None, None, None)
            prun.check_or_write_manifest(args, None, None, None, None, None, None)
            man["sha256"]["restyle"] = "other"
            with self.assertRaises(SystemExit) as cm:
                prun.check_or_write_manifest(args, None, None, None, None, None, None)
            self.assertIn("restyle", str(cm.exception))
        finally:
            prun.build_manifest = saved_build


if __name__ == "__main__":
    unittest.main()
