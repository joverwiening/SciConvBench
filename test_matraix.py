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
import persona_analysis as pan
import persona_matraix_seed as seed
import test_persona_harness as th
from test_bigfive import FakeService, setUpModule, tearDownModule  # noqa: F401 (module fixtures)
from test_pole_effects import synth

FULL1_TEMPLATE_SHA = "f3007eb808cf9dd7b752f18b38e9f94f9c854d022d4b1386fba6a5efb1ea7771"
MX = pu.load_matraix()
BF = pu.load_bigfive()
MX_ARM = "mx_candidate_002"
INC, COMP = "INCOMPLETE request: run it.", "COMPLETE spec: value 42, mesh 3."


def run_user(arm, tmp, replies=("reply 1", "reply 2"), questions=("Q1: what value?", "Q2: which mesh?")):
    svc = FakeService(list(replies))
    user = pu.make_user(arm, svc, INC, COMP, os.path.join(tmp, "case", "conversation_log.txt"))
    for q in questions:
        user.answer_clarification_question(q)
    return user, svc


class TestUsers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_bf_neutral_matches_fixture(self):
        with open(os.path.join(REPO, "fixtures", "bf_neutral_4c4b51d03.json")) as f:
            fx = json.load(f)
        self.assertEqual((fx["arm"], fx["incomplete"], fx["complete"]), ("bf_neutral", INC, COMP))
        _, svc = run_user("bf_neutral", self.tmp, questions=fx["questions"])
        self.assertEqual([{"system_prompt": sp, "messages": m} for m, sp in svc.calls], fx["calls"])
        with open(os.path.join(self.tmp, "case", "user_system_prompt.txt"), encoding="utf-8") as f:
            self.assertEqual(f.read(), fx["user_system_prompt_file"])

    def test_load_matraix(self):
        self.assertEqual(len(MX["personas"]), 30)
        self.assertEqual(len(MX["selected"]), 10)
        self.assertTrue(set(MX["selected"]) <= set(MX["personas"]))
        p = MX["personas"][MX_ARM]
        with open(os.path.join(pu.MX_DIR, "personas", "candidate_002.txt"), encoding="utf-8") as f:
            self.assertEqual(p["block"], f.read())
        self.assertEqual((p["id"], p["kind"], p["examples"], p["reminder"]),
                         (MX_ARM, "trait", [], "(Stay in character as the person described in your instructions.)"))

    def test_mx_system_prompt_and_messages(self):
        user, svc = run_user(MX_ARM, self.tmp)
        self.assertIsInstance(user, pu.MatrAIxUser)
        sh = BF["shared"]
        want = "\n\n".join([pu.MX_IDENTITY, MX["personas"][MX_ARM]["block"], sh["situation"], sh["notes_intro"],
                            f"<notes>\n{COMP}\n</notes>", sh["request_intro"],
                            f"<your_request>\n{INC}\n</your_request>", sh["rules"]])
        self.assertTrue(pu.MX_IDENTITY.startswith("You are the person described below. You sent an AI assistant"))
        self.assertNotIn("Sam Keller", want)
        for _, sp in svc.calls:
            self.assertEqual(sp, want)
        with open(os.path.join(self.tmp, "case", "user_system_prompt.txt"), encoding="utf-8") as f:
            self.assertEqual(f.read(), want)
        reminder = pu.MX_REMINDER + " " + sh["reminder_suffix"]
        self.assertEqual(svc.calls[0][0], [{"role": "user", "content": f"Message from the assistant:\nQ1: what value?"
                                                                       f"\n\n{reminder}"}])
        self.assertEqual(svc.calls[1][0], [
            {"role": "user", "content": "Message from the assistant:\nQ1: what value?"},
            {"role": "assistant", "content": "reply 1"},
            {"role": "user", "content": f"Message from the assistant:\nQ2: which mesh?\n\n{reminder}"}])

    def test_dispatch_and_valid_arms(self):
        u = pu.make_user("bf_neutral", FakeService([]), INC, COMP, os.path.join(self.tmp, "a", "log.txt"))
        self.assertIs(type(u), pu.BigFiveUser)
        u = pu.make_user(MX_ARM, FakeService([]), INC, COMP, os.path.join(self.tmp, "b", "log.txt"), matraix=MX)
        self.assertIs(type(u), pu.MatrAIxUser)
        self.assertEqual(u.shared, BF["shared"])
        arms = pu.valid_arms(pu.load_personas(), BF, MX)
        self.assertTrue(set(MX["personas"]) <= set(arms))
        self.assertEqual(len([a for a in arms if a.startswith("mx_")]), 30)
        self.assertEqual(len(arms), len(set(arms)))

    def test_template_sha_unchanged(self):
        self.assertEqual(prun.user_prompt_template_sha(), FULL1_TEMPLATE_SHA)


class TestRunnerMatrAIx(unittest.TestCase):
    ARMS = ["bf_neutral", MX_ARM]

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cards = os.path.join(self.tmp, "cards.json")
        th.synthetic_cards(self.cards)
        self.out = os.path.join(self.tmp, "run")
        th.FakeLLM.complete_after = 3

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def argv(self):
        return ["--domain", th.DOMAIN, "--mode", th.MODE, "--cases", ",".join(th.CASE_IDS),
                "--arms", ",".join(self.ARMS), "--agent_model", "or_gpt_5_2", "--user_model", "bedrock_glm5",
                "--workers", "2", "--out", self.out, "--cards", self.cards, "--budget_usd", "1000"]

    def test_run_manifest_and_resume(self):
        self.assertEqual(prun.main(self.argv()), {"jobs": 4, "skipped": 0, "done": 4, "errors": 0})
        for cid in th.CASE_IDS:
            run = th.jload(os.path.join(prun.case_dir(self.out, MX_ARM, 1, th.DOMAIN, "or_gpt_5_2", th.MODE, cid),
                                        "persona_run.json"))
            self.assertEqual(run["persona_text_sha256"], pu.sha256_text(MX["personas"][MX_ARM]["block"]))
        m = th.jload(os.path.join(self.out, "manifest.json"))
        self.assertEqual(m["sha256"]["user_prompt_template"], FULL1_TEMPLATE_SHA)
        self.assertEqual(m["sha256"]["matraix_user"], prun.matraix_user_sha())
        for pid, p in MX["personas"].items():
            self.assertEqual(m["sha256"]["personas"][pid], pu.sha256_file(p["path"]))
        self.assertEqual(prun.main(self.argv()), {"jobs": 4, "skipped": 4, "done": 0, "errors": 0})
        m["sha256"]["matraix_user"] = "0" * 64
        with open(os.path.join(self.out, "manifest.json"), "w") as f:
            json.dump(m, f)
        with self.assertRaises(SystemExit) as cm:
            prun.main(self.argv())
        self.assertIn("matraix_user", str(cm.exception))


class TestSeedBaseline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src, self.dst = os.path.join(self.tmp, "src"), os.path.join(self.tmp, "dst")
        self.cases = ["case_001", "case_002", "case_003"]
        hashes = prun.persona_hashes(pu.load_personas(), BF, MX)
        self.manifest = {"sha256": {"personas": hashes, "user_prompt_template": prun.user_prompt_template_sha()}}
        os.makedirs(self.src)
        with open(os.path.join(self.src, "manifest.json"), "w") as f:
            json.dump(self.manifest, f)
        rows = []
        for arm in seed.NEUTRAL_ARMS + ("bf_low_A",):
            for cid in self.cases + ["case_009"]:
                if arm == "bf_neutral_c" and cid == "case_002":
                    continue
                d = prun.case_dir(self.src, arm, 1, "foam", "bedrock_glm5", "disambiguation", cid)
                os.makedirs(d)
                with open(os.path.join(d, "persona_run.json"), "w") as f:
                    json.dump({"arm": arm, "case_id": cid}, f)
                rows.append({"arm": arm, "rep": 1, "domain": "foam", "mode": "disambiguation", "case_id": cid})
        with open(os.path.join(self.src, "results.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        self.casefile = os.path.join(self.tmp, "cases.txt")
        with open(self.casefile, "w") as f:
            f.write("\n".join(self.cases) + "\n")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_seed(self):
        argv = ["seed-baseline", "--src_run", self.src, "--dst_run", self.dst, "--cases", "@" + self.casefile]
        prov = seed.main(argv)
        self.assertEqual(len(prov["copied"]), 17)
        self.assertEqual(prov["missing"], ["bf_neutral_c/case_002"])
        self.assertEqual(prov["src_manifest_sha256"], pu.sha256_file(os.path.join(self.src, "manifest.json")))
        self.assertFalse(os.path.exists(os.path.join(self.dst, "manifest.json")))
        self.assertFalse(os.path.exists(os.path.join(self.dst, "bf_low_A")))
        d = prun.case_dir(self.dst, "bf_neutral_f", 1, "foam", "bedrock_glm5", "disambiguation", "case_003")
        self.assertEqual(th.jload(os.path.join(d, "persona_run.json")), {"arm": "bf_neutral_f", "case_id": "case_003"})
        with open(os.path.join(self.dst, "results.jsonl")) as f:
            rows = [json.loads(l) for l in f]
        self.assertEqual(len(rows), 17)
        self.assertEqual({r["case_id"] for r in rows}, set(self.cases))
        self.assertEqual(th.jload(os.path.join(self.dst, "baseline_provenance.json"))["copied"], prov["copied"])
        self.assertEqual(len(seed.main(argv)["copied"]), 17)
        with open(os.path.join(self.dst, "results.jsonl")) as f:
            self.assertEqual(len(f.readlines()), 17)

    def test_refuses_changed_neutral(self):
        self.manifest["sha256"]["personas"]["bf_neutral_b"] = "0" * 64
        with open(os.path.join(self.src, "manifest.json"), "w") as f:
            json.dump(self.manifest, f)
        with self.assertRaises(SystemExit) as cm:
            seed.seed_baseline(self.src, self.dst, self.cases)
        self.assertIn("bf_neutral_b", str(cm.exception))
        self.assertFalse(os.path.exists(self.dst))


class TestAnalysisMatrAIx(unittest.TestCase):
    def test_family_and_counts(self):
        fam = pan.family_arms("matraix")
        self.assertEqual(fam["ref"], "bf_neutral")
        self.assertEqual(fam["placebos"], tuple(f"bf_neutral_{x}" for x in "bcdef"))
        self.assertEqual(fam["traits"], tuple(sorted(MX["selected"])))
        self.assertEqual(fam["holm"], fam["traits"])
        sel = sorted(MX["selected"])
        arms = {a: "neutral" for a in ("bf_neutral",) + fam["placebos"]}
        arms.update({a: a for a in sel})
        arms["mx_candidate_001"] = "not_selected"
        shift = {a: 0.0 for a in arms}
        shift[sel[0]], shift[sel[1]] = -0.3, 0.3
        rows = synth(arms, 40, {a: 0.0 for a in arms}, pole=shift, noise_sd=0.05, seed=5)
        res = pan.analyse(rows, family="matraix")
        self.assertEqual(res["trait_arms"], sel)
        self.assertEqual(res["holm_m"], 10)
        prim = {p["target"]: p for p in res["pole_effects"] if p["metric"] == pan.PRIMARY}
        self.assertEqual(set(prim), set(sel))
        self.assertTrue(all(p["k_wordings"] == 1 and p["k_neutral"] == 6 and p["wordings"] == t
                            for t, p in prim.items()))
        self.assertTrue(all(p["gate"] is None for p in prim.values()))
        want = {t: bool(p["p_holm"] < pan.ALPHA) for t, p in prim.items()}
        self.assertEqual({t: p["counts"] for t, p in prim.items()}, want)
        self.assertTrue(want[sel[0]] and want[sel[1]])
        self.assertEqual(sum(want.values()), 2)
        pair = {r["arm"]: r for r in res["paired"] if r["metric"] == pan.PRIMARY}
        self.assertEqual({a: r["counts"] for a, r in pair.items()},
                         {a: bool(r["p_holm"] < pan.ALPHA) for a, r in pair.items()})
        md = pan.render_markdown(res)
        self.assertEqual(md.count(pan.MX_NO_GATE), 1)
        self.assertNotIn("pairwise manipulation gate", md)

    def test_main(self):
        tmp = tempfile.mkdtemp()
        try:
            sel = sorted(MX["selected"])
            arms = {a: "n" for a in ("bf_neutral",) + tuple(f"bf_neutral_{x}" for x in "bcdef")}
            arms.update({a: a for a in sel})
            rows = synth(arms, 12, {a: 0.0 for a in arms}, seed=1)
            results = os.path.join(tmp, "results.jsonl")
            with open(results, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
            res = pan.main(["--results", results, "--out", os.path.join(tmp, "an"), "--family", "matraix"])
            self.assertEqual(res["family"], "matraix")
            self.assertTrue(os.path.getsize(os.path.join(tmp, "an", "pole_effects.csv")) > 0)
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
