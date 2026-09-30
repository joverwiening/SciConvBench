import io
import os
import sys
import json
import time
import shutil
import tempfile
import unittest
import contextlib

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

import main as sc_main
import persona_users as pu
import persona_runner as prun
import test_persona_harness as th

BF = pu.load_bigfive()
GOAL_INTRO = ("What you need from this run, in your own words. Your notes do not fix these settings, you have no "
              "values in mind for them, and you don't know which settings achieve it; judge the assistant's "
              "proposals by whether they fit this:")
OK_IDS = th.CASE_IDS
BAD_ID = list(th.CASES)[2]
ARMS = ["approve_all", "oracle_criteria", "bf_neutral"]
VARIANTS = ("fast", "rigorous")


class SteerFake(th.FakeLLM):
    def __init__(self, config):
        super().__init__(config)
        self.api_input_tokens = self.api_output_tokens = self.api_reasoning_chars = 0

    def invoke_messages(self, messages, system_prompt=None, max_retries=5):
        self.calls += 1
        with th._lock:
            th.CALLS.append((self.model_version, system_prompt, json.dumps(messages)))
        return "Hmm, use the faster mesh."


_saved = {}


def setUpModule():
    for mod in th.PATCHED:
        _saved[mod] = mod.LLMService
        mod.LLMService = SteerFake
    _saved["sleep"] = time.sleep
    time.sleep = lambda s: None
    _saved["prices"] = pu.fetch_openrouter_prices
    pu.fetch_openrouter_prices = lambda ids: {i: {"prompt": "0.000001", "completion": "0.000002"} for i in ids}


def tearDownModule():
    for mod in th.PATCHED:
        mod.LLMService = _saved[mod]
    time.sleep = _saved["sleep"]
    pu.fetch_openrouter_prices = _saved["prices"]


def goal(cid, var):
    return f"GOAL {var} for {cid}: finish tonight" if var == "fast" else f"GOAL {var} for {cid}: publishable"


def steer_cards(path):
    cases = {}
    for cid in OK_IDS:
        cases[cid] = {"status": "ok",
                      "choice_points": [{"entity": "mesh", "original_spans": ["a mesh"], "fast": "coarse mesh",
                                         "rigorous": "fine mesh"},
                                        {"entity": "solver", "original_spans": ["a solver"], "fast": "explicit",
                                         "rigorous": "implicit"}],
                      "fact_entities": ["geometry_2D"],
                      "notes_without_choices": f"NOTES WITHOUT CHOICES {cid}",
                      "goals": {v: goal(cid, v) for v in VARIANTS}}
    cases[BAD_ID] = {"status": "no_choice_points"}
    with open(path, "w") as f:
        json.dump({"meta": {}, "cases": cases}, f)


class TestGoalBlock(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def user(self, name, goal_text):
        return pu.make_user("bf_low_A", None, "INCOMPLETE", "COMPLETE 42.", os.path.join(self.tmp, name, "log.txt"),
                            bigfive=BF, goal=goal_text)

    def test_goal_block_iff_goal(self):
        sh, p = BF["shared"], BF["personas"]["bf_low_A"]
        old = "\n\n".join([sh["identity"], p["sheet"], sh["situation"], p["appraisal"], sh["notes_intro"],
                           "<notes>\nCOMPLETE 42.\n</notes>", sh["request_intro"],
                           "<your_request>\nINCOMPLETE\n</your_request>", sh["rules"]]
                          + ([sh["examples_intro"]] if p.get("examples") else []))
        plain = self.user("a", None)
        self.assertEqual(plain.build_system_prompt(), old)
        self.assertNotIn("<goal>", plain.build_system_prompt())
        withg = self.user("b", "Get it done by Friday.")
        block = f"{GOAL_INTRO}\n<goal>\nGet it done by Friday.\n</goal>"
        self.assertEqual(withg.build_system_prompt(),
                         old.replace("</notes>\n\n", f"</notes>\n\n{block}\n\n", 1))
        self.assertIn(f"</notes>\n\n{block}\n\n{sh['request_intro']}", withg.build_system_prompt())

    def test_approve_all_user(self):
        for arm in pu.STEER_DETERMINISTIC_ARMS:
            u = pu.make_user(arm, None, "I", "C", os.path.join(self.tmp, arm, "log.txt"))
            self.assertIsInstance(u, pu.ApproveAllUser)
            self.assertEqual(u.answer_clarification_question("Draft... ok?"), pu.APPROVE_REPLY)
        self.assertNotIn("approve_all", pu.valid_arms(pu.load_personas(), BF))


class TestSteerRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.steer = os.path.join(cls.tmp, "steer.json")
        steer_cards(cls.steer)
        cls.cards = os.path.join(cls.tmp, "cards.json")
        th.synthetic_cards(cls.cards)
        cls.out = os.path.join(cls.tmp, "run")
        th.FakeLLM.complete_after = 3
        th.CALLS.clear()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cls.res = prun.main(cls.argv())
        cls.stdout = buf.getvalue()
        cls.calls = list(th.CALLS)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    @classmethod
    def argv(cls, arms=ARMS):
        return ["--domain", th.DOMAIN, "--mode", th.MODE, "--cases", ",".join(OK_IDS + [BAD_ID]),
                "--arms", ",".join(arms), "--agent_model", "or_gpt_5_2", "--user_model", "bedrock_glm5",
                "--workers", "2", "--out", cls.out, "--cards", cls.cards, "--budget_usd", "1000",
                "--steer_cards", cls.steer]

    def folder(self, arm, cid, var):
        return prun.case_dir(self.out, arm, 1, th.DOMAIN, "or_gpt_5_2", th.MODE, f"{cid}__{var}")

    def agent_calls(self):
        return [(sp, up) for _, sp, up in self.calls if sp and "Requirement Analyst" in sp and "Time is up" not in sp]

    def test_jobs_and_skip(self):
        self.assertEqual(self.res, {"jobs": len(ARMS) * len(OK_IDS) * 2, "skipped": 0,
                                    "done": len(ARMS) * len(OK_IDS) * 2, "errors": 0})
        self.assertIn(f"Skipping cases without an ok steer card: ['{BAD_ID}']", self.stdout)
        for arm in ARMS:
            base = os.path.dirname(self.folder(arm, BAD_ID, "fast"))
            self.assertEqual(sorted(os.listdir(base)), sorted(f"{c}__{v}" for c in OK_IDS for v in VARIANTS))

    def test_propose_prompt(self):
        propose = sc_main.ConvAgent(pu.role_config("or_gpt_5_2"), "propose").system_prompt
        self.assertIn("**HOW YOU WORK:**", propose)
        self.assertNotEqual(propose, sc_main.ConvAgent(pu.role_config("or_gpt_5_2")).system_prompt)
        sps = {sp for sp, _ in self.agent_calls()}
        self.assertEqual(sps, {propose})
        m = th.jload(os.path.join(self.out, "manifest.json"))
        self.assertEqual(m["sha256"]["agent_system_prompt"], pu.sha256_text(propose))
        self.assertEqual(m["steer_cards"], {"path": os.path.abspath(self.steer), "sha256": pu.sha256_file(self.steer)})

    def test_first_input(self):
        ups = [up for _, up in self.agent_calls()]
        for cid in OK_IDS:
            inc = th.CASES[cid]["incomplete_user_req"]
            for var in VARIANTS:
                g = goal(cid, var)
                self.assertIn(f"User: {inc}\n\nWhat I need from this run: {g}", ups)
                # 3 agent calls per job, only the oracle_criteria job carries the goal
                self.assertEqual(sum(g in up for up in ups), 3)
                self.assertEqual(sum(up == f"User: {inc}" for up in ups), 4)  # approve_all + bf_neutral, 2 variants
                log = th.readf(os.path.join(self.folder("approve_all", cid, var), "conversation_log.txt"))
                self.assertIn(pu.APPROVE_REPLY, log)

    def test_folder_contents(self):
        for arm in ARMS:
            for cid in OK_IDS:
                for var in VARIANTS:
                    d = self.folder(arm, cid, var)
                    self.assertEqual(th.readf(os.path.join(d, "complete_prompt.txt")),
                                     th.CASES[cid]["complete_user_req"])
                    sc = th.jload(os.path.join(d, "steer_case.json"))
                    self.assertEqual(sc, {"case_id": cid, "variant": var, "goal": goal(cid, var),
                                          "choice_points": [
                                              {"entity": "mesh", "fast": "coarse mesh", "rigorous": "fine mesh",
                                               "target": "coarse mesh" if var == "fast" else "fine mesh"},
                                              {"entity": "solver", "fast": "explicit", "rigorous": "implicit",
                                               "target": "explicit" if var == "fast" else "implicit"}],
                                          "fact_entities": ["geometry_2D"]})
                    run = th.jload(os.path.join(d, "persona_run.json"))
                    self.assertEqual((run["case_id"], run["variant"], run["status"]), (cid, var, "done"))
                    if arm in pu.STEER_DETERMINISTIC_ARMS:
                        self.assertIsNone(run["models"]["user"])
                        self.assertFalse(os.path.exists(os.path.join(d, "user_system_prompt.txt")))
                    else:
                        usp = th.readf(os.path.join(d, "user_system_prompt.txt"))
                        self.assertIn(f"<notes>\nNOTES WITHOUT CHOICES {cid}\n</notes>\n\n{GOAL_INTRO}\n"
                                      f"<goal>\n{goal(cid, var)}\n</goal>", usp)
                        self.assertIn(f"<your_request>\n{th.CASES[cid]['incomplete_user_req']}\n</your_request>",
                                      usp)
                        self.assertNotIn(th.CASES[cid]["complete_user_req"], usp)

    def test_unknown_arm_refused(self):
        for arm in ("mute", "paper", "oracle", "neutral"):
            with self.assertRaises(SystemExit) as cm:
                prun.main(self.argv([arm]))
            self.assertIn("Unknown arms", str(cm.exception.code))

    def test_z_resume_and_sha_change(self):
        with contextlib.redirect_stdout(io.StringIO()):
            again = prun.main(self.argv())
        self.assertEqual(again["skipped"], len(ARMS) * len(OK_IDS) * 2)
        shutil.copy(self.steer, self.steer + ".bak")
        try:
            with open(self.steer, "a") as f:
                f.write("\n")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as cm:
                prun.main(self.argv())
            self.assertIn("steer_cards", str(cm.exception.code))
        finally:
            shutil.move(self.steer + ".bak", self.steer)


if __name__ == "__main__":
    unittest.main()
