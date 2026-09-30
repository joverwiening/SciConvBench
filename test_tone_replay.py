import os
import sys
import csv
import json
import shutil
import tempfile
import threading
import unittest

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

import main as agent_main
import persona_users as pu
import persona_tone_replay as tr

_lock = threading.Lock()
CALLS = []
CASES = ["case_003", "case_004", "case_005"]


def fake_stats(n):
    return {"total_calls": n, "failed_calls": 0, "total_retries": 0, "total_tokens": 110 * n,
            "prompt_tokens": 100 * n, "completion_tokens": 10 * n, "api_usage_calls": n,
            "api_input_tokens": 100 * n, "api_output_tokens": 10 * n, "api_reasoning_tokens": 0,
            "api_reasoning_chars": 0}


class FakeLLM:
    """Routes on the system prompt. rewrite_mode: 'good' keeps facts, 'bad' changes a number,
    'bad_then_good' fails on the first rewrite call of each chat and passes on the retry."""
    rewrite_mode = "good"
    agent_script = None

    def __init__(self, config=None):
        self.calls = 0

    def invoke(self, user_prompt, system_prompt=None, max_retries=5):
        self.calls += 1
        sp = system_prompt or ""
        with _lock:
            CALLS.append((sp, user_prompt))
        if sp is tr.REWRITE_SYSTEM:
            answers = [l.split("A: ", 1)[1] for l in tr_between(user_prompt, "conversation").splitlines()
                       if "] A: " in l]
            bad = self.rewrite_mode == "bad" or (self.rewrite_mode == "bad_then_good"
                                                 and "Your previous rewrite" not in user_prompt)
            lowa = "cold, blunt, critical" in user_prompt
            out = []
            for a in answers:
                r = a.replace("9.81", "9.8") if bad else a
                out.append(r + (" Obviously." if lowa else " Thanks."))
            return json.dumps({"replies": out})
        if sp is tr.EQUIV_SYSTEM:
            n = user_prompt.count("ORIGINAL:")
            return json.dumps({"pairs": [{"i": i, "same_info": True, "diff": ""} for i in range(1, n + 1)]})
        if sp is tr.JUDGE_SYSTEM:
            a = tr_between(user_prompt, "CHAT_A")
            return json.dumps({"more_disagreeable": "A" if "Obviously" in a else "B", "why": "x"})
        if "Requirement Analyst" in sp:
            if self.agent_script is not None:
                return self.agent_script(user_prompt)
            last = user_prompt.rsplit("\n", 1)[-1]
            return "[COMPLETE]\nDo it." if "Obviously" in last else "What mesh size?"
        raise AssertionError(f"unrouted call: {sp[:80]}")

    def get_statistics(self, name=None):
        return fake_stats(self.calls)


def tr_between(text, tag):
    s = text.index(f"<{tag}>") + len(tag) + 2
    return text[s:text.index(f"</{tag}>")].strip()


_saved = {}


def setUpModule():
    _saved["tr"], _saved["main"] = tr.LLMService, agent_main.LLMService
    tr.LLMService = agent_main.LLMService = FakeLLM


def tearDownModule():
    tr.LLMService, agent_main.LLMService = _saved["tr"], _saved["main"]


def make_chat(folder, clar, completed=True, status="done"):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "persona_run.json"), "w") as f:
        json.dump({"status": status, "completed_by_agent": completed, "cap_hit": not completed}, f)
    with open(os.path.join(folder, "clarifications.json"), "w") as f:
        json.dump([{"turn": i, "question": q, "answer": a, "deferred": False}
                   for i, (q, a) in enumerate(clar, 1)], f)


def make_src(root, sha=None):
    if sha is None:
        sha = pu.sha256_text(agent_main.ConvAgent(pu.role_config("bedrock_glm5"), None).system_prompt)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump({"sha256": {"agent_system_prompt": sha}}, f)
    for k, cid in enumerate(CASES):
        clar = [(f"Q{i} for {cid}?", f"Use 9.81 and buoyantFoam, answer {i}.") for i in range(1, 3 + k)]
        make_chat(tr.chat_dir(root, "bf_neutral", "bedrock_glm5", cid), clar, completed=(k != 2))
        make_chat(tr.chat_dir(root, "bf_neutral_b", "bedrock_glm5", cid), [], completed=True)
        make_chat(tr.chat_dir(root, "bf_low_A", "bedrock_glm5", cid), [(f"Q1 {cid}?", "9.81. Obviously.")])


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = os.path.join(self.tmp, "src")
        self.out = os.path.join(self.tmp, "out")
        FakeLLM.rewrite_mode, FakeLLM.agent_script = "good", None
        with _lock:
            CALLS.clear()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def select(self):
        make_src(self.src)
        return tr.main(["select", "--src_run", self.src, "--out", self.out, "--cases", ",".join(CASES)])

    def n_calls(self, marker):
        with _lock:
            return sum(1 for sp, _ in CALLS if marker(sp))


class TestContext(unittest.TestCase):
    def test_replay_context_equals_convagent_context(self):
        seen = []
        qs = ["Which solver?", "Which mesh?", "Which end time?"]

        def script(ctx):
            seen.append(ctx)
            return qs[len(seen) - 1] if len(seen) <= len(qs) else "[COMPLETE]\nDone."

        FakeLLM.agent_script = staticmethod(script)
        try:
            agent = agent_main.ConvAgent(pu.role_config("bedrock_glm5"), None)
            req = "Simulate a heated cavity.\nUse OpenFOAM."
            answers = ["buoyantFoam, please.", "Use 40x40 cells.\nUniform.", "End at 1000 s."]
            out = [agent.chat(req)]
            for a in answers:
                out.append(agent.chat(a))
        finally:
            FakeLLM.agent_script = None
        self.assertEqual(out[:3], qs)
        self.assertEqual(len(seen), 4)
        for t in range(4):
            self.assertEqual(tr.build_context(req, qs, answers, t), seen[t])


class TestSlots(unittest.TestCase):
    def test_labels(self):
        self.assertEqual([s["recorded_move"] for s in tr.slot_labels(3, True)], ["ask", "ask", "stop"])
        cap = tr.slot_labels(3, False)
        self.assertEqual([s["recorded_move"] for s in cap], ["ask", "ask", "cap"])
        self.assertEqual([s["excluded"] for s in cap], [False, False, True])
        self.assertEqual(tr.slot_labels(1, True), [{"t": 1, "recorded_move": "stop", "excluded": False}])


class TestFactCheck(unittest.TestCase):
    O = "Use buoyantFoam with p_rgh solved by GAMG. Gravity is 9.81 m/s2 and the mesh is 40 cells."

    def test_value_changed(self):
        ok, missing, added = tr.fact_check(self.O, self.O.replace("9.81", "9.8"))
        self.assertFalse(ok)
        self.assertIn("9.81", missing)
        self.assertIn("9.8", added)

    def test_value_dropped(self):
        ok, missing, added = tr.fact_check(self.O, "Use buoyantFoam with p_rgh solved by GAMG. Gravity is 9.81 m/s2.")
        self.assertFalse(ok)
        self.assertEqual(missing, ["40"])
        self.assertEqual(added, [])

    def test_identifier_added(self):
        ok, missing, added = tr.fact_check(self.O, self.O + " Use PBiCGStab.")
        self.assertFalse(ok)
        self.assertEqual(added, ["PBiCGStab"])

    def test_reordered_passes(self):
        new = "The mesh is 40 cells, gravity 9.81 m/s2. Solver: buoyantFoam, with GAMG for p_rgh."
        self.assertEqual(tr.fact_check(self.O, new), (True, [], []))

    def test_plain_hyphenated_word_not_technical(self):
        self.assertEqual(tr.tech_tokens("use k-epsilon and kEpsilon or k-omega2"), {"kEpsilon", "k-omega2"})


class TestSelectRewrite(Base):
    def test_select(self):
        sel = self.select()
        self.assertEqual([c["case_id"] for c in sel["chats"]], CASES)
        self.assertTrue(all(c["arm"] == "bf_neutral" for c in sel["chats"]))
        self.assertEqual([s["recorded_move"] for s in sel["chats"][2]["slots"]], ["ask", "ask", "ask", "cap"])
        self.assertTrue(all(c["bf_low_A_folder"] for c in sel["chats"]))
        self.assertEqual(sel["chats"][0]["incomplete_req"], pu.load_cases("foam", "disambiguation")["case_003"]["incomplete_user_req"])

    def test_retry_then_fail(self):
        self.select()
        FakeLLM.rewrite_mode = "bad"
        tr.main(["rewrite", "--out", self.out, "--styles", "para", "--budget_usd", "10", "--workers", "1"])
        rw = tr.load_rewrite(self.out, "case_003", "para")
        self.assertEqual((rw["status"], rw["attempts"]), ("failed", 2))
        self.assertFalse(rw["checks"][0]["code_ok"])
        self.assertIn("9.81", rw["checks"][0]["missing"])
        self.assertTrue(rw["reasons"])
        self.assertEqual(self.n_calls(lambda sp: sp is tr.REWRITE_SYSTEM), 2 * len(CASES))
        self.assertEqual(self.n_calls(lambda sp: sp is tr.EQUIV_SYSTEM), 2 * len(CASES))
        self.assertAlmostEqual(rw["cost_usd"], 4 * (100 * 1.2e-6 + 10 * 3.2e-6))

    def test_retry_then_ok(self):
        self.select()
        FakeLLM.rewrite_mode = "bad_then_good"
        tr.main(["rewrite", "--out", self.out, "--styles", "lowA", "--budget_usd", "10", "--workers", "1"])
        rw = tr.load_rewrite(self.out, "case_004", "lowA")
        self.assertEqual((rw["status"], rw["attempts"]), ("ok", 2))
        self.assertFalse(rw["first_checks"][0]["code_ok"])
        self.assertTrue(all(c["code_ok"] and c["judge_same"] for c in rw["checks"]))
        retry = [u for sp, u in CALLS if sp is tr.REWRITE_SYSTEM and "Your previous rewrite" in u]
        self.assertEqual(len(retry), len(CASES))


class TestReplay(Base):
    def prepare(self):
        self.select()
        tr.main(["rewrite", "--out", self.out, "--budget_usd", "10", "--workers", "2"])

    def test_resume_no_new_calls(self):
        self.prepare()
        agent = lambda sp: "Requirement Analyst" in sp
        tr.main(["replay", "--out", self.out, "--budget_usd", "10", "--orig_slots", "4", "--workers", "3"])
        n1 = self.n_calls(agent)
        rows = tr.read_jsonl(os.path.join(self.out, "replay.jsonl"))
        # non-excluded slots: 2 + 3 + 3 = 8 per rewrite arm, plus 4 orig
        self.assertEqual(n1, 8 * 2 + 4)
        self.assertEqual(len(rows), n1)
        self.assertEqual(len({tuple(r["key"]) for r in rows}), n1)
        self.assertAlmostEqual(rows[0]["cost_usd"], 100 * 1.2e-6 + 10 * 3.2e-6)
        tr.main(["replay", "--out", self.out, "--budget_usd", "10", "--orig_slots", "4", "--workers", "3"])
        self.assertEqual(self.n_calls(agent), n1)
        self.assertEqual(len(tr.read_jsonl(os.path.join(self.out, "replay.jsonl"))), n1)

    def test_replay_uses_rewritten_answers(self):
        self.prepare()
        tr.main(["replay", "--out", self.out, "--budget_usd", "10", "--orig_slots", "0", "--arms", "lowA",
                 "--workers", "1"])
        ctxs = [u for sp, u in CALLS if "Requirement Analyst" in sp]
        sel = tr.load_selection(self.out)
        rw = tr.load_rewrite(self.out, "case_003", "lowA")
        chat = sel["chats"][0]
        want = tr.build_context(chat["incomplete_req"], [q for q, _ in chat["clarifications"]], rw["replies"], 2)
        self.assertIn(want, ctxs)
        self.assertTrue(all(r["move"] == "stop" for r in tr.read_jsonl(os.path.join(self.out, "replay.jsonl"))))

    def test_budget_stop(self):
        self.prepare()
        tr.main(["replay", "--out", self.out, "--budget_usd", "1e-9", "--orig_slots", "0", "--workers", "1"])
        self.assertEqual(len(tr.read_jsonl(os.path.join(self.out, "replay.jsonl"))), 1)

    def test_sha_mismatch_aborts(self):
        self.prepare()
        with open(os.path.join(self.src, "manifest.json"), "w") as f:
            json.dump({"sha256": {"agent_system_prompt": "0" * 64}}, f)
        with self.assertRaises(SystemExit):
            tr.main(["replay", "--out", self.out, "--budget_usd", "10"])
        self.assertEqual(self.n_calls(lambda sp: "Requirement Analyst" in sp), 0)


class TestJudgeAnalyze(Base):
    def test_pipeline_and_analyze(self):
        self.select()
        tr.main(["rewrite", "--out", self.out, "--budget_usd", "10", "--workers", "2"])
        tr.main(["replay", "--out", self.out, "--budget_usd", "10", "--orig_slots", "8", "--workers", "2"])
        tr.main(["judge", "--out", self.out, "--budget_usd", "10", "--workers", "2"])
        judge = tr.read_jsonl(os.path.join(self.out, "judge.jsonl"))
        self.assertEqual(len(judge), 3 * len(CASES))
        for r in judge:
            self.assertEqual(set(r["sides"].values()), set(tr.PAIR_TYPES[r["pair_type"]]))
        self.assertTrue(all(r["winner"] == "lowA" for r in judge if r["pair_type"] == "orig_vs_lowA"))
        n_before = len(CALLS)
        tr.main(["judge", "--out", self.out, "--budget_usd", "10"])
        self.assertEqual(len(CALLS), n_before)

        res = tr.main(["analyze", "--out", self.out])
        self.assertEqual(res["paired"], 8)
        md = open(os.path.join(self.out, "analysis.md")).read()
        self.assertIn("stop rate: para 0.000 (0), lowA 1.000 (8), recorded 0.250 (2)", md)
        self.assertIn("discordant: lowA stop & para ask = 8, para stop & lowA ask = 0", md)
        self.assertIn("lowA judged more disagreeable than orig: 1.000", md)
        self.assertIn("move == recorded_move: 6/8", md)
        with open(os.path.join(self.out, "slots.csv")) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2 + 3 + 4)
        self.assertEqual(sum(r["excluded"] == "True" for r in rows), 1)
        self.assertEqual(sum(r["move_lowA"] == "stop" for r in rows), 8)


if __name__ == "__main__":
    unittest.main()
