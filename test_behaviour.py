import os
import sys
import json
import shutil
import tempfile
import unittest

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

import persona_users as pu
import persona_runner as prun
import test_persona_harness as th
from test_bigfive import FakeService, setUpModule, tearDownModule  # noqa: F401 (module fixtures)
from test_matraix import run_user, INC, COMP, FULL1_TEMPLATE_SHA

BF = pu.load_bigfive()
BV = pu.load_behaviour()


class TestBehaviourUser(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_prompt_is_neutral_plus_card(self):
        base = BF["personas"]["bf_neutral"]
        for arm, card in BV["cards"].items():
            tmp = os.path.join(self.tmp, arm)
            user, svc = run_user(arm, tmp)
            self.assertIs(type(user), pu.BigFiveUser)
            neutral_tmp = os.path.join(self.tmp, arm + "_neutral")
            _, nsvc = run_user("bf_neutral", neutral_tmp)
            # system prompt: the neutral one with the card right after the sheet, nothing else changed
            self.assertEqual(svc.calls[0][1], nsvc.calls[0][1].replace(base["sheet"], base["sheet"] + "\n\n" + card["card"], 1))
            last = svc.calls[1][0][-1]["content"]
            self.assertIn("(Stay in character as Sam Keller. " + card["reminder"] + ")", last)
            self.assertEqual(svc.calls[1][0][:-1], nsvc.calls[1][0][:-1])

    def test_valid_arms_and_template_sha(self):
        arms = pu.valid_arms(pu.load_personas(), BF, pu.load_matraix(), BV)
        self.assertTrue({"bv_terse", "bv_volunteer"} <= set(arms))
        self.assertEqual(len(arms), len(set(arms)))
        self.assertEqual(prun.user_prompt_template_sha(), FULL1_TEMPLATE_SHA)


class FakeAgentLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, user_prompt, system_prompt=None, max_retries=5):
        self.calls.append(user_prompt)
        return f"model reply {len(self.calls)}"


class TestPrimedAgent(unittest.TestCase):
    def test_first_reply_is_saved_one(self):
        cfg = pu.role_config("bedrock_glm5")
        a = prun.PrimedAgent(cfg, None, first_response="Saved first question?")
        a.llm = FakeAgentLLM()
        self.assertEqual(a.chat("REQ"), "Saved first question?")
        self.assertEqual(a.llm.calls, [])
        self.assertEqual(a.chat("answer 1"), "model reply 1")
        self.assertEqual(a.llm.calls, ["User: REQ\nAnalyst: Saved first question?\nUser: answer 1"])


class TestRunnerSharedFirst(unittest.TestCase):
    ARMS = ["bf_neutral", "bv_terse", "bv_volunteer"]

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cards = os.path.join(self.tmp, "cards.json")
        th.synthetic_cards(self.cards)
        self.out = os.path.join(self.tmp, "run")
        self.first = os.path.join(self.tmp, "first")
        th.FakeLLM.complete_after = 3

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def argv(self):
        return ["--domain", th.DOMAIN, "--mode", th.MODE, "--cases", ",".join(th.CASE_IDS),
                "--arms", ",".join(self.ARMS), "--agent_model", "or_gpt_5_2", "--user_model", "bedrock_glm5",
                "--workers", "2", "--out", self.out, "--cards", self.cards, "--budget_usd", "1000",
                "--shared_first", self.first]

    def test_arms_share_first_reply(self):
        n = len(th.CASE_IDS) * len(self.ARMS)
        del th.CALLS[:]
        self.assertEqual(prun.main(self.argv()), {"jobs": n, "skipped": 0, "done": n, "errors": 0})
        cases = pu.load_cases(th.DOMAIN, th.MODE)
        key = prun.test_runner.TASK_CONFIG[th.MODE]["keys"]["input_prompt"]
        for cid in th.CASE_IDS:
            saved = th.jload(prun.shared_first_path(self.first, cid, 1))
            # the bare first context reaches the agent model once per case: in the pre-pass, not in any arm
            bare = [c for c in th.CALLS if "Requirement Analyst" in c[1] and c[2] == f"User: {cases[cid][key]}"]
            self.assertEqual(len(bare), 1)
            for arm in self.ARMS:
                d = prun.case_dir(self.out, arm, 1, th.DOMAIN, "or_gpt_5_2", th.MODE, cid)
                clar = th.jload(os.path.join(d, "clarifications.json"))
                self.assertEqual(clar[0]["question"], saved["response"])
                run = th.jload(os.path.join(d, "persona_run.json"))
                self.assertEqual(run["shared_first_sha256"], pu.sha256_text(saved["response"]))
                if arm.startswith("bv_"):
                    p = pu.behaviour_persona(arm, BF, BV)
                    self.assertEqual(run["persona_text_sha256"], pu.sha256_text(pu.bigfive_persona_text(p)))
        m = th.jload(os.path.join(self.out, "manifest.json"))
        self.assertEqual(m["shared_first"], os.path.abspath(self.first))
        self.assertEqual(m["sha256"]["behaviour_user"], prun.behaviour_user_sha())
        # rerun reuses the saved replies and skips done jobs
        self.assertEqual(prun.main(self.argv()), {"jobs": n, "skipped": n, "done": 0, "errors": 0})


if __name__ == "__main__":
    unittest.main()
