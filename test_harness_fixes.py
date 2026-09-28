import os
import sys
import csv
import json
import shutil
import tempfile
import unittest

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

import main as sc_main
import test_runner as tr
import persona_users as pu
import persona_runner as prun
import persona_analysis as pan
import persona_pairwise as ppw
import test_persona_harness as th
import test_bigfive as tb
from test_pole_effects import synth, split

setUpModule, tearDownModule = tb.setUpModule, tb.tearDownModule
LONG_SPEC = "\n".join(f"Use setting number {i} for the simulation run." for i in range(12))


class ScriptLLM:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def invoke(self, user_prompt, system_prompt=None, max_retries=5):
        self.calls.append((user_prompt, system_prompt))
        return self.replies.pop(0)


class ScriptUser:
    def __init__(self, replies):
        self.replies = list(replies)

    def answer_clarification_question(self, q):
        return self.replies.pop(0)


def agent(replies):
    a = sc_main.ConvAgent.__new__(sc_main.ConvAgent)
    a.history, a.system_prompt, a.llm = [], "SYS", ScriptLLM(replies)
    return a


class TestCapFix(unittest.TestCase):
    def test_retry_on_invalid(self):
        a = agent(["Q1?", "Q2?", "[COMPLETE] Is that all?", "[COMPLETE]\n" + LONG_SPEC])
        res = tr.run_dialogue(a, ScriptUser(["R1", "R2"]), "INIT", 2, sleep=0, final_fix=True)
        self.assertEqual((res.cap_hit, res.final_attempts, res.final_valid), (True, 2, True))
        self.assertEqual(res.final_spec, LONG_SPEC)
        self.assertEqual(len(a.llm.calls), 4)
        (ctx1, sp1), (ctx2, sp2) = a.llm.calls[2:]
        self.assertTrue(ctx1.endswith("Analyst: Q2?\nUser: R2"))
        self.assertEqual(ctx1, ctx2)
        self.assertNotIn(tr.FINAL_RETRY_INSTRUCTION, sp1)
        self.assertTrue(sp2.endswith("\n\n" + tr.FINAL_RETRY_INSTRUCTION))
        self.assertIn("Time is up", sp2)

    def test_valid_first_and_invalid_retry(self):
        a = agent(["Q1?", LONG_SPEC])
        res = tr.run_dialogue(a, ScriptUser(["R1"]), "INIT", 1, sleep=0, final_fix=True)
        self.assertEqual((res.final_attempts, res.final_valid, len(a.llm.calls)), (1, True, 2))
        a = agent(["Q1?", "short", LONG_SPEC + "\nWhich solver?"])
        res = tr.run_dialogue(a, ScriptUser(["R1"]), "INIT", 1, sleep=0, final_fix=True)
        self.assertEqual((res.final_attempts, res.final_valid), (2, False))
        self.assertTrue(res.final_spec.endswith("Which solver?"))

    def test_upstream_path_unchanged(self):
        a = agent(["Q1?", "Q2?", "one line?"])
        res = tr.run_dialogue(a, ScriptUser(["R1", "R2"]), "INIT", 2, sleep=0)
        self.assertEqual((res.final_spec, res.final_attempts, res.final_valid), ("one line?", 1, None))
        self.assertNotIn("User: R2", a.llm.calls[-1][0])

    def test_valid_rule(self):
        self.assertFalse(tr.final_spec_valid("x" * 299))
        self.assertTrue(tr.final_spec_valid("x" * 300))
        self.assertFalse(tr.final_spec_valid("x" * 300 + "\nok?\n\n"))


INC = "Run the solver with 5 cells."
COMP = "Run the solver with 5 cells, 42 steps and 7 layers."


class TestTruthRetry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def user(self, replies, name="case"):
        svc = tb.FakeService(replies)
        return svc, pu.make_user("bf_neutral", svc, INC, COMP, os.path.join(self.tmp, name, "conversation_log.txt"))

    def test_fires_on_claim_with_novel_number(self):
        svc, u = self.user(["I already told you: 42 steps.", "Use 42 steps."])
        self.assertEqual(u.answer_clarification_question("Steps?"), "Use 42 steps.")
        self.assertEqual((len(svc.calls), u.truth_retries), (2, 1))
        self.assertTrue(svc.calls[1][0][-1]["content"].endswith(pu.TRUTH_RETRY_NOTE))
        self.assertEqual(svc.calls[1][0][:-1], svc.calls[0][0][:-1])
        self.assertNotIn(pu.TRUTH_RETRY_NOTE, svc.calls[0][0][-1]["content"])
        line = tb.jl(os.path.join(self.tmp, "case", "user_calls.jsonl"))[0]
        self.assertEqual((line["truth_retry"], line["draft"], line["reply"]),
                         (True, "I already told you: 42 steps.", "Use 42 steps."))
        self.assertEqual(line["usage"], {"input_tokens": 20, "output_tokens": 8})
        self.assertEqual([c["answer"] for c in u.clarifications], ["Use 42 steps."])

    def test_no_fire(self):
        cases = [["I already told you, 5 cells."],                 # true claim: number in the request
                 ["Use 7 layers."],                                # novel number, no claim
                 ["I already told you it is 99."],                 # claim, number not in the notes
                 ["42 steps.", "As I said, 42 steps."]]            # number sent in an earlier reply
        for k, replies in enumerate(cases):
            svc, u = self.user(list(replies), f"c{k}")
            for _ in replies:
                u.answer_clarification_question("Q?")
            self.assertEqual((len(svc.calls), u.truth_retries), (len(replies), 0), replies)
            self.assertFalse(any(l["truth_retry"] for l in tb.jl(os.path.join(self.tmp, f"c{k}", "user_calls.jsonl"))))


class TestExamples(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def user(self, svc, persona, name):
        return pu.BigFiveUser(llm_service=svc, persona=persona, shared=dict(tb.BF["shared"], examples_intro="EXINTRO"),
                              incomplete_requirement=INC, complete_requirement=COMP,
                              log_file=os.path.join(self.tmp, name, "conversation_log.txt"))

    def test_examples_in_messages_only(self):
        base = tb.BF["personas"]["bf_neutral"]
        p = dict(base, examples=[{"q": "EXQ1?", "a": "EXA1"}, {"q": "EXQ2?", "a": "EXA2"}])
        svc = tb.FakeService(["first", "second"])
        u = self.user(svc, p, "ex")
        u.answer_clarification_question("Q1?")
        u.answer_clarification_question("Q2?")
        msgs, sp = svc.calls[1]
        self.assertEqual(msgs[:4], [{"role": "user", "content": "Message from the assistant:\nEXQ1?"},
                                    {"role": "assistant", "content": "EXA1"},
                                    {"role": "user", "content": "Message from the assistant:\nEXQ2?"},
                                    {"role": "assistant", "content": "EXA2"}])
        self.assertEqual(msgs[4]["content"], "Message from the assistant:\nQ1?")
        self.assertEqual(len(msgs), 7)
        for m in msgs[:-1]:
            self.assertNotIn(base["reminder"], m["content"])
        self.assertTrue(sp.endswith(tb.BF["shared"]["rules"] + "\n\nEXINTRO"))
        self.assertNotIn("EX", json.dumps(u.clarifications))
        with open(u.log_file) as f:
            self.assertNotIn("EX", f.read())
        self.assertIn("EXA1", pu.bigfive_persona_text(p))
        self.assertNotIn("EXQ1", pu.bigfive_persona_text(p))
        # empty / missing examples: identical to now, no intro
        prompts = []
        for k, pp in enumerate([base, dict(base, examples=[])]):
            s = tb.FakeService(["a"])
            self.user(s, pp, f"n{k}").answer_clarification_question("Q?")
            prompts.append(s.calls[0])
        self.assertEqual(prompts[0], prompts[1])
        self.assertNotIn("EXINTRO", prompts[0][1])
        self.assertEqual(len(prompts[0][0]), 1)


FAKE_BF = {"personas": {"bf_neutral": {"kind": "neutral", "target": None},
                        "bf_neutral_b": {"kind": "placebo", "target": None},
                        "bf_low_A": {"kind": "trait", "target": "low_agreeableness"},
                        "bf_abl": {"kind": "ablation", "target": None}}}


class TestAblationKind(unittest.TestCase):
    def setUp(self):
        self.saved = pu.load_bigfive
        pu.load_bigfive = lambda *a, **k: FAKE_BF
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        pu.load_bigfive = self.saved
        shutil.rmtree(self.tmp)

    def test_excluded_from_families(self):
        fam = pan.family_arms("bigfive")
        self.assertEqual((fam["ref"], fam["placebos"], fam["traits"]), ("bf_neutral", ("bf_neutral_b",), ("bf_low_A",)))
        self.assertIsNone(pu.bigfive_target("bf_abl", FAKE_BF))
        rows = tb.bf_rows()
        rows += [dict(r, arm="bf_abl") for r in rows if r["arm"] == "bf_neutral"]
        res = pan.analyse(rows, family="bigfive")
        self.assertEqual((res["trait_arms"], res["placebo_arms"]), (["bf_low_A"], ["bf_neutral_b"]))
        self.assertIn("bf_abl", res["arms"])
        means = {m["arm"]: m for m in res["arm_means"]}
        self.assertAlmostEqual(means["bf_abl"][pan.PRIMARY], means["bf_neutral"][pan.PRIMARY])
        self.assertEqual({p["target"] for p in res["pole_effects"]}, {"low_agreeableness"})
        self.assertIn("| bf_abl |", pan.render_markdown(res))
        for arm in ("bf_neutral", "bf_low_A", "bf_abl"):
            d = os.path.join(self.tmp, arm, "rep1", "pde", "agent_Disambiguation", "c0")
            os.makedirs(d)
            with open(os.path.join(d, "persona_run.json"), "w") as f:
                json.dump({"status": "done", "arm": arm, "rep": 1, "domain": "pde", "case_id": "c0"}, f)
        pairs = ppw.find_pairs(self.tmp, FAKE_BF)
        self.assertEqual([(p[1], p[2]) for p in pairs], [("bf_low_A", "low_agreeableness")])


class TestGateTruthTable(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def write(self, name, rows):
        with open(os.path.join(self.tmp, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["target", "arm", "pass"])
            w.writeheader()
            for t, a, p in rows:
                w.writerow({"target": t, "arm": a, "pass": p})

    def test_truth_table(self):
        import itertools
        for tall, oall, tw, ow in itertools.product([True, False], repeat=4):
            self.write("pairwise_gate.csv", [("t", "w1", True), ("t", "w2", tw), ("t", "ALL", tall)])
            self.write("pairwise_gate_opposite.csv", [("t", "w1", ow), ("t", "w2", True), ("t", "ALL", oall)])
            g = pan.read_gate(self.tmp)["t"]
            self.assertEqual((g["gate_target"], g["gate_opposite"], g["gate_all_wordings"]), (tall, oall, tw and ow))
            self.assertEqual(g["gate"], tall and oall and tw and ow)
        self.write("pairwise_gate.csv", [("t", "w1", True), ("t", "ALL", True)])
        self.write("pairwise_gate_opposite.csv", [("u", "ALL", True)])
        g = pan.read_gate(self.tmp)
        self.assertEqual((g["t"]["gate_opposite"], g["t"]["gate"]), (None, None))
        os.remove(os.path.join(self.tmp, "pairwise_gate_opposite.csv"))
        self.assertIsNone(pan.read_gate(self.tmp))


class TestPoleP(unittest.TestCase):
    def test_p_is_max_and_assert(self):
        arms = {"n0": "neutral", "n1": "neutral", "a1": "pa", "a2": "pa", "b1": "pb"}
        word = {"n0": 0.0, "n1": 0.02, "a1": 0.03, "a2": -0.03, "b1": 0.0}
        rows = synth(arms, 30, word, pole={"pa": 0.05, "pb": 0.1}, seed=2)
        neutral, targets = split(arms)
        poles = pan.pole_effects(rows, neutral, targets, [pan.PRIMARY], b=500)[0]
        for p in poles:
            self.assertEqual(p["p"], max(p["p_z"], p["boot2_p"]))
        self.assertEqual([p["p_holm"] for p in poles], pan.holm([p["p"] for p in poles]))
        with self.assertRaises(AssertionError):
            pan.pole_effects(rows, neutral, dict(targets, ghost="pghost"), [pan.PRIMARY], b=100)

    def test_asked_delivered(self):
        items = [{"asked": True, "delivered": True}, {"asked": False, "delivered": True},
                 {"asked": True, "delivered": False}, {"asked": None, "delivered": None}]
        self.assertAlmostEqual(pan.metric({"items": items}, "asked_delivered"), 1 / 3)
        i = pan.POLE_METRICS.index("unasked_delivered")
        self.assertEqual(pan.POLE_METRICS[i + 1], "asked_delivered")


class TestRunnerCap(unittest.TestCase):
    ARMS = ["bf_neutral"]

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cards = os.path.join(self.tmp, "cards.json")
        th.synthetic_cards(self.cards)
        self.out = os.path.join(self.tmp, "run")
        th.FakeLLM.complete_after = None

    def tearDown(self):
        th.FakeLLM.complete_after = 3
        shutil.rmtree(self.tmp)

    def test_cap_records_and_manifest(self):
        argv = tb.TestRunnerBigFive.argv(self, "1000") + ["--max_turns", "2"]
        self.assertEqual(prun.main(argv)["done"], 2)
        for cid in th.CASE_IDS:
            d = prun.case_dir(self.out, "bf_neutral", 1, th.DOMAIN, "or_gpt_5_2", th.MODE, cid)
            run = th.jload(os.path.join(d, "persona_run.json"))
            self.assertEqual((run["cap_hit"], run["final_attempts"], run["final_valid"], run["truth_retries"]),
                             (True, 2, False, 0))
            self.assertEqual(run["usage"]["agent"]["total_calls"], 4)
            self.assertEqual(th.readf(os.path.join(d, "generated_prompt.txt")), "FINAL SPEC (forced)")
        mpath = os.path.join(self.out, "manifest.json")
        m = th.jload(mpath)
        self.assertIs(m["final_fix"], True)
        del m["final_fix"]
        with open(mpath, "w") as f:
            json.dump(m, f)
        with self.assertRaises(SystemExit):
            prun.main(argv)


if __name__ == "__main__":
    unittest.main(verbosity=2)
