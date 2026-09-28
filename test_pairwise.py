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
import persona_eval as pev
import persona_pairwise as ppw

JUDGE = "or_gemini_2_5_pro"
MODEL = pu.role_config(JUDGE).model_version
TRAIT_ANSWER = "Stop wasting my time."
CALLS = []


def fj(d):
    return f"<FINAL_JSON>{json.dumps(d)}</FINAL_JSON>"


class FakeJudge:
    def __init__(self, config):
        self.calls = 0

    def invoke(self, user_prompt, system_prompt=None, max_retries=5):
        self.calls += 1
        CALLS.append((system_prompt, user_prompt))
        if "case c2" in user_prompt:
            return "garbage"
        chat_a = user_prompt.split("<CHAT_A>")[1].split("</CHAT_A>")[0]
        trait_side = "A" if TRAIT_ANSWER in chat_a else "B"
        # the neuroticism arm is judged wrong on purpose
        wrong = pev.BIGFIVE_OPTIONS["high_neuroticism"] in system_prompt
        side = ("B" if trait_side == "A" else "A") if wrong else trait_side
        ev = TRAIT_ANSWER if side == trait_side else "not said"
        return fj({"choice": side, "confidence": 2, "evidence": ev})

    def get_statistics(self):
        return {"api_usage_calls": self.calls, "api_input_tokens": 1000 * self.calls,
                "api_output_tokens": 10 * self.calls, "total_calls": self.calls}


def write_case(run_dir, arm, rep, cid, answer, status="done"):
    d = os.path.join(run_dir, arm, f"rep{rep}", "dom", "agent_dis", cid)
    os.makedirs(d)
    with open(os.path.join(d, "persona_run.json"), "w") as f:
        json.dump({"status": status, "arm": arm, "rep": rep, "domain": "dom", "case_id": cid}, f)
    with open(os.path.join(d, "clarifications.json"), "w") as f:
        json.dump([{"turn": 1, "question": "What value?", "answer": answer, "deferred": False}], f)
    return d


class TestPairwise(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.run = os.path.join(self.tmp, "run")
        os.makedirs(self.run)
        with open(os.path.join(self.run, "manifest.json"), "w") as f:
            json.dump({"prices": {MODEL: {"prompt": "0.000001", "completion": "0.000002"}}}, f)
        self.f = {}
        for arm, rep, cid in [("bf_low_A", 1, "c1"), ("bf_low_A", 1, "c2"), ("bf_low_A", 1, "c3"),
                              ("bf_low_A", 2, "c1"), ("bf_high_N", 1, "c1")]:
            self.f[(arm, rep, cid)] = write_case(self.run, arm, rep, cid, TRAIT_ANSWER)
        for rep, cid in [(1, "c1"), (1, "c2"), (3, "c1")]:
            self.f[("bf_neutral", rep, cid)] = write_case(self.run, "bf_neutral", rep, cid, f"Sure, case {cid}.")
        write_case(self.run, "bf_neutral", 1, "c3", "Sure, case c3.", status="skipped_budget")
        write_case(self.run, "bf_neutral_b", 1, "c1", "Placebo.")
        self._saved = ppw.LLMService
        ppw.LLMService = FakeJudge
        CALLS.clear()

    def tearDown(self):
        ppw.LLMService = self._saved
        shutil.rmtree(self.tmp)

    def argv(self, budget="1000", *extra):
        return ["--run_dir", self.run, "--judge_model", JUDGE, "--workers", "1", "--budget_usd", budget, *extra]

    def rec(self, key):
        return pev.read_json(os.path.join(self.f[key], "persona_pairwise.json"))

    def test_pairing(self):
        pairs = {(p[0], p[1], p[2]): p[3] for p in ppw.find_pairs(self.run, pu.load_bigfive())}
        get = lambda k: pairs[(self.f[k], k[0], pu.load_bigfive()["personas"][k[0]]["target"])]
        self.assertEqual(len(pairs), 5)
        self.assertEqual(get(("bf_low_A", 1, "c1")), self.f[("bf_neutral", 1, "c1")])
        # rep2 missing on the reference: lowest reference rep
        self.assertEqual(get(("bf_low_A", 2, "c1")), self.f[("bf_neutral", 1, "c1")])
        # reference c3 exists but is not done
        self.assertIsNone(get(("bf_low_A", 1, "c3")))
        self.assertEqual(ppw.main(self.argv("1000", "--dry_run")),
                         {"pairs": 4, "cached": 0, "to_judge": 4, "no_partner": 1})
        self.assertEqual(CALLS, [])
        self.assertFalse(os.path.exists(os.path.join(self.run, "pairwise_gate.csv")))

    def test_order_deterministic(self):
        rel = os.path.join("bf_low_A", "rep1", "dom", "agent_dis", "c1")
        self.assertEqual(ppw.trait_is_a(rel), int(pu.sha256_text(rel), 16) % 2 == 0)
        self.assertEqual(ppw.trait_is_a(rel), ppw.trait_is_a(rel))
        seen = {ppw.trait_is_a(f"x/{i}") for i in range(20)}
        self.assertEqual(seen, {True, False})

    def test_judge_both_orders(self):
        t = [{"question": "Q?", "answer": TRAIT_ANSWER, "deferred": False}]
        r = [{"question": "Q?", "answer": "Sure, fine.", "deferred": False}]
        for trait_is in ("A", "B"):
            CALLS.clear()
            out = ppw.pairwise_judge(FakeJudge(None), t, r, "low_agreeableness", trait_is)
            self.assertEqual((out["choice"], out["hit"], out["confidence"], out["evidence_ok"], out["attempts"]),
                             (trait_is, True, 2, True, 1))
            sp, up = CALLS[0]
            self.assertIn('more "' + pev.BIGFIVE_OPTIONS["low_agreeableness"] + '"', sp)
            a = up.split("<CHAT_A>")[1].split("</CHAT_A>")[0]
            self.assertEqual(TRAIT_ANSWER in a, trait_is == "A")
        out = ppw.pairwise_judge(FakeJudge(None), t, r, "high_neuroticism", "A")
        self.assertEqual((out["choice"], out["hit"], out["evidence_ok"]), ("B", False, False))

    def test_end_to_end(self):
        res = ppw.main(self.argv())
        self.assertEqual((res["pairs"], res["judged"], res["errors"], res["no_partner"]), (4, 4, 0, 1))
        r = self.rec(("bf_low_A", 1, "c1"))
        rel = os.path.join("bf_low_A", "rep1", "dom", "agent_dis", "c1")
        self.assertEqual(r["trait_is"], "A" if ppw.trait_is_a(rel) else "B")
        self.assertEqual((r["target"], r["partner"], r["choice"], r["hit"], r["confidence"], r["evidence"],
                          r["evidence_ok"], r["attempts"]),
                         ("low_agreeableness", os.path.join("bf_neutral", "rep1", "dom", "agent_dis", "c1"),
                          r["trait_is"], True, 2, TRAIT_ANSWER, True, 1))
        self.assertEqual(r["usage"]["api_usage_calls"], 1)
        fail = self.rec(("bf_low_A", 1, "c2"))
        self.assertEqual((fail["hit"], fail["choice"], fail["attempts"]), (None, None, pev.PARSE_TRIES))
        self.assertFalse(self.rec(("bf_high_N", 1, "c1"))["hit"])
        self.assertIsNone(pev.read_json(os.path.join(self.f[("bf_low_A", 1, "c3")], "persona_pairwise.json")))

        rows = {(x["target"], x["arm"]): x for x in res["rows"]}
        self.assertEqual(set(rows), {("low_agreeableness", "bf_low_A"), ("low_agreeableness", "ALL"),
                                     ("high_neuroticism", "bf_high_N"), ("high_neuroticism", "ALL")})
        la = rows[("low_agreeableness", "bf_low_A")]
        self.assertEqual((la["n"], la["hits"], la["hit_rate"], la["n_parse_fail"], la["n_no_partner"]),
                         (2, 2, 1.0, 1, 1))
        self.assertAlmostEqual(la["ci_lo"], 0.025 ** 0.5, places=6)
        self.assertFalse(la["pass"])
        self.assertEqual(rows[("low_agreeableness", "ALL")], dict(la, arm="ALL"))
        hn = rows[("high_neuroticism", "bf_high_N")]
        self.assertEqual((hn["n"], hn["hits"], hn["hit_rate"], hn["ci_lo"]), (1, 0, 0.0, 0.0))
        with open(os.path.join(self.run, "pairwise_gate.csv")) as f:
            lines = f.read().splitlines()
        self.assertEqual(lines[0], ",".join(ppw.CSV_COLS))
        self.assertEqual(len(lines), 5)
        with open(os.path.join(self.run, "pairwise_gate.md")) as f:
            self.assertIn("| low_agreeableness | ALL | 2 | 2 | 1.000 |", f.read())

        n = len(CALLS)
        res = ppw.main(self.argv())
        self.assertEqual(res["judged"], 0)
        self.assertEqual(len(CALLS), n)
        res = ppw.main(self.argv("1000", "--force"))
        self.assertEqual(res["judged"], 4)

    def test_budget_skip(self):
        # one call costs 1000 * 1e-6 + 10 * 2e-6 = 0.00102 USD
        res = ppw.main(self.argv("0.0001"))
        self.assertEqual((res["judged"], res["skipped_budget"]), (1, 3))
        res = ppw.main(self.argv("1000"))
        self.assertEqual((res["judged"], res.get("skipped_budget")), (3, None))
        self.assertEqual(sum(x["n"] + x["n_parse_fail"] for x in res["rows"] if x["arm"] != "ALL"), 4)


class TestStats(unittest.TestCase):
    def test_clopper_pearson(self):
        lo, hi = ppw.clopper_pearson(8, 10)
        self.assertAlmostEqual(lo, 0.44390, places=4)
        self.assertAlmostEqual(hi, 0.97479, places=4)
        self.assertEqual(ppw.clopper_pearson(0, 0), (None, None))
        lo, hi = ppw.clopper_pearson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 1 - 0.025 ** 0.1, places=6)
        lo, hi = ppw.clopper_pearson(10, 10)
        self.assertAlmostEqual(lo, 0.025 ** 0.1, places=6)
        self.assertEqual(hi, 1.0)

    def test_pooled_and_gate(self):
        hit = lambda h: {"hit": h}
        pairs = [(f"a{i}", "arm1", "t", "p") for i in range(10)] + \
                [(f"b{i}", "arm2", "t", "p") for i in range(10)] + [("c", "arm2", "t", None)]
        recs = {f"a{i}": hit(i < 8) for i in range(10)}
        recs.update({f"b{i}": hit(i < 9 if i < 9 else None) for i in range(10)})
        rows = {r["arm"]: r for r in ppw.summarize(pairs, recs)}
        self.assertEqual([r["arm"] for r in ppw.summarize(pairs, recs)], ["arm1", "arm2", "ALL"])
        self.assertEqual((rows["arm1"]["n"], rows["arm1"]["hits"], rows["arm1"]["pass"]), (10, 8, False))
        self.assertEqual((rows["arm2"]["n"], rows["arm2"]["hits"], rows["arm2"]["n_parse_fail"],
                          rows["arm2"]["n_no_partner"]), (9, 9, 1, 1))
        a = rows["ALL"]
        self.assertEqual((a["n"], a["hits"], a["n_parse_fail"], a["n_no_partner"]), (19, 17, 1, 1))
        self.assertAlmostEqual(a["hit_rate"], 17 / 19)
        self.assertGreater(a["ci_lo"], 0.5)
        self.assertTrue(a["pass"])


if __name__ == "__main__":
    unittest.main()
