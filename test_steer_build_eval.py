import os
import re
import sys
import json
import shutil
import tempfile
import threading
import unittest

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

import llm_judge_disambiguation
import persona_users as pu
import persona_eval as pev
import persona_steer_build as sb
import persona_steer_eval as se
import persona_steer_analysis as sa

_lock = threading.Lock()
CALLS = []
GOOD_GOALS = {"fast": "I just want to see that this runs and shows the rough trend. It has to work on my laptop "
                      "and I need it by tomorrow.",
              "rigorous": "These results go into a paper and a reviewer doubted our earlier accuracy. Take as long "
                          "as it needs; I want numbers I can defend."}


def fj(obj):
    return f"<FINAL_JSON>\n{json.dumps(obj)}\n</FINAL_JSON>"


def between(tag, text):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def settings(text, tag="SETTINGS"):
    """{entity: {label: option text}} from a SETTINGS block."""
    out, cur = {}, None
    for line in between(tag, text).splitlines():
        m = re.match(r"\s*([AB])\. (.*)", line)
        if m and cur:
            out[cur][m.group(1)] = m.group(2)
        elif line.strip().endswith(":"):
            cur = line.strip()[:-1]
            out[cur] = {}
    return out


class FakeLLM:
    """Routes on the system prompt; option texts carry FASTOPT / RIGOPT markers."""
    position_biased = False
    first_goals = {"fast": "Run it on my laptop in 5 minutes.", "rigorous": GOOD_GOALS["rigorous"]}

    def __init__(self, config=None):
        self.calls = 0

    def invoke(self, user_prompt, system_prompt=None, max_retries=5):
        self.calls += 1
        sp = system_prompt or ""
        with _lock:
            CALLS.append(sp[:40])
        if sp is sb.BUILDER_SYSTEM:
            if "Rewrite only the two goal cards" in user_prompt:
                return fj({"goals": GOOD_GOALS})
            ents = re.findall(r"^\s*- (\w+):", between("CHOICE_POINTS", user_prompt), re.M)
            return fj({"options": {e: {"fast": f"FASTOPT {e} coarse.", "rigorous": f"RIGOPT {e} fine."}
                                   for e in ents},
                       "goals": self.first_goals})
        if sp is sb.SOLVE_SYSTEM:
            want = "FASTOPT" if "laptop" in between("SCIENTIST_NOTE", user_prompt) else "RIGOPT"
            ans = {}
            for e, opts in settings(user_prompt).items():
                ans[e] = "A" if self.position_biased else next(l for l, t in opts.items() if want in t)
            return fj(ans)
        if sp is se.CHOICE_SYSTEM:
            spec = between("FINAL_SPECIFICATION", user_prompt)
            ans = {}
            for e, opts in settings(user_prompt).items():
                hit = [l for l, t in opts.items() if t in spec]
                ans[e] = hit[0] if hit else "not_stated"
            return fj(ans)
        if "expert AI Evaluator" in sp:
            n = len(between("EXPECTED_MISSING_ENTITIES", user_prompt).split(","))
            return fj({"score": f"1/{n}", "recovered_entities": [], "still_missing_entities": []})
        raise AssertionError(f"unrouted call: {sp[:80]}")

    def get_statistics(self, name=None):
        return {"total_calls": self.calls, "failed_calls": 0, "total_retries": 0, "total_tokens": 10 * self.calls,
                "prompt_tokens": 8 * self.calls, "completion_tokens": 2 * self.calls, "api_usage_calls": 0}


class BiasedLLM(FakeLLM):
    position_biased = True


PATCHED = [sb, se, pev, llm_judge_disambiguation]
_saved = {}


def setUpModule():
    for mod in PATCHED:
        _saved[mod] = mod.LLMService
        mod.LLMService = FakeLLM
    _saved["prices"] = pu.fetch_prices
    pu.fetch_prices = lambda ids: {i: {"prompt": "0.000001", "completion": "0.000002"} for i in ids}


def tearDownModule():
    for mod in PATCHED:
        mod.LLMService = _saved[mod]
    pu.fetch_prices = _saved["prices"]


def card(item, spans):
    return {"item": item, "spans": spans}


class TestNotes(unittest.TestCase):
    def test_removal_and_collapse(self):
        text = "Run a cavity flow.\nUse 20 cells in x.\nWalls are no-slip.  Use the icoFoam solver.\nEnd at 1 s."
        notes, err = sb.make_notes(text, [card("mesh", ["Use 20 cells in x."]),
                                          card("solver", ["Use the icoFoam solver."])],
                                   [card("boundary_conditions", ["Walls are no-slip."])])
        self.assertEqual(err, [])
        self.assertEqual(notes, "Run a cavity flow.\nWalls are no-slip.\nEnd at 1 s.")

    def test_fact_span_lost_and_choice_remaining(self):
        text = "Use 20 cells in x and walls are no-slip."
        _, err = sb.make_notes(text, [card("mesh", ["20 cells in x"])], [card("bc", [text])])
        self.assertTrue(any("fact span of bc lost" in e for e in err))
        _, err = sb.make_notes("Use 20  cells.", [card("mesh", ["Use 20 cells."])], [])
        self.assertTrue(any("still present" in e for e in err))

    def test_real_case_088_dropped(self):
        cards = pev.read_json(pu.default_cards_path("foam", "disambiguation"))["cases"]
        case = pu.load_cases("foam", "disambiguation")["case_088"]
        rec = sb.build_case(FakeLLM(), case, cards["case_088"])
        self.assertEqual(rec["status"], "dropped: notes")


class TestLint(unittest.TestCase):
    OPTS = {"time_controls": {"fast": "Set maxCo 5 and end at 10 s.", "rigorous": "Set maxCo 0.5 and end at 100 s."}}

    def test_clean(self):
        self.assertEqual(sb.lint_goals(GOOD_GOALS, self.OPTS), [])

    def test_digit(self):
        errs = sb.lint_goals(dict(GOOD_GOALS, fast="Done in 2 days please."), self.OPTS)
        self.assertEqual(errs, ["fast card contains a digit"])

    def test_camel_banned(self):
        errs = sb.lint_goals(dict(GOOD_GOALS, rigorous="Use kEpsilon if that is what it takes."), {})
        self.assertIn("rigorous card contains banned word 'epsilon'", errs)
        errs = sb.lint_goals(dict(GOOD_GOALS, fast="A simple run on OpenFOAM."), {})
        self.assertIn("fast card contains banned word 'simple'", errs)
        self.assertIn("fast card contains banned word 'foam'", errs)

    def test_option_token(self):
        errs = sb.lint_goals(dict(GOOD_GOALS, fast="Keep maxCo loose."), self.OPTS)
        self.assertEqual(errs, ["fast card contains the option token 'maxCo'"])

    def test_banned_is_word_level(self):
        self.assertEqual(sb.lint_goals(dict(GOOD_GOALS, fast="The rules of the lesson are simpler."), {}), [])


class TestSolvability(unittest.TestCase):
    def test_keep_logic(self):
        ok = {"m": "fast", "t": "fast"}
        picks = {"fast": [ok, {"m": "fast", "t": "rigorous"}],
                 "rigorous": [{"m": "rigorous", "t": "rigorous"}, {"m": "rigorous", "t": "rigorous"}]}
        self.assertEqual(sb.keep_points(picks, ["m", "t"]), ["m"])
        picks["rigorous"][1]["m"] = None
        self.assertEqual(sb.keep_points(picks, ["m", "t"]), [])

    def test_orders_are_swapped(self):
        o1, o2 = sb.solve_orders()
        self.assertEqual({o1["A"], o2["A"]}, {"fast", "rigorous"})
        self.assertEqual(o1["A"], o2["B"])

    def test_position_bias_dropped(self):
        opts = {"mesh": {"fast": "FASTOPT m.", "rigorous": "RIGOPT m."},
                "solver": {"fast": "FASTOPT s.", "rigorous": "RIGOPT s."}}
        good = sb.solvability(FakeLLM(), "req", GOOD_GOALS, opts)
        self.assertEqual(good["kept"], ["mesh", "solver"])
        bad = sb.solvability(BiasedLLM(), "req", GOOD_GOALS, opts)
        self.assertEqual(bad["kept"], [])
        self.assertEqual(bad["picks"]["fast"], [{"mesh": "fast", "solver": "fast"},
                                                {"mesh": "rigorous", "solver": "rigorous"}])


class TestBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_main_walks_until_n_ok(self):
        out = os.path.join(self.tmp, "steer.json")
        data = sb.main(["--cases", "case_088,case_001,case_002", "--n", "1", "--model", "bedrock_glm5",
                        "--budget_usd", "1", "--out", out])
        self.assertEqual(list(data["cases"]), ["case_088", "case_001"])
        rec = data["cases"]["case_001"]
        self.assertEqual(rec["status"], "ok")
        self.assertEqual([p["entity"] for p in rec["choice_points"]], ["mesh", "time_controls"])
        self.assertEqual(rec["fact_entities"], ["geometry_2D"])
        self.assertEqual(rec["goals"], GOOD_GOALS)
        self.assertEqual(len(rec["checks"]["lint"]), 2)
        self.assertEqual(rec["checks"]["lint"][1], [])
        self.assertTrue(set(rec["choice_points"][0]) >= {"entity", "original_spans", "fast", "rigorous"})
        self.assertIn("Use 90 cells", rec["choice_points"][0]["original_spans"][0])
        self.assertNotIn("90 cells", rec["notes_without_choices"])
        self.assertGreater(data["meta"]["spend"]["invocations"][0]["spent_usd"], 0)
        self.assertEqual(rec["checks"]["usage"]["total_calls"], 2 + 4)


class TestEval(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_label_mapping_round_trip(self):
        pts = [{"entity": f"e{i}", "fast": f"FASTOPT {i}", "rigorous": f"RIGOPT {i}"} for i in range(6)]
        seen = set()
        for k in range(8):
            for variant in sb.VARIANTS:
                steer = {"case_id": f"case_{k}", "variant": variant, "choice_points": pts}
                spec = " ".join(p[variant] for p in pts[:5])
                res = se.choice_judge(FakeLLM(), spec, steer, "armX")
                self.assertEqual(res["labels"], {**{f"e{i}": variant for i in range(5)}, "e5": "not_stated"})
                seen.update(o["A"] for o in res["order"].values())
        self.assertEqual(seen, {"fast", "rigorous"})
        o1 = se.label_order("a", "c", "fast", ["x", "y"])
        self.assertEqual(o1, se.label_order("a", "c", "fast", ["x", "y"]))
        self.assertEqual(se.map_labels({"x": "b", "y": "Not stated"}, o1),
                         {"x": o1["x"]["B"], "y": "not_stated"})

    def make_folder(self, arm, case_id, variant, spec_variant):
        d = os.path.join(self.tmp, arm, "rep0", "foam", "bedrock_glm5_Disambiguation", f"{case_id}__{variant}")
        os.makedirs(d)
        pts = [{"entity": "mesh", "fast": "FASTOPT mesh.", "rigorous": "RIGOPT mesh."},
               {"entity": "solver", "fast": "FASTOPT solver.", "rigorous": "RIGOPT solver."}]
        for p in pts:
            p["target"] = p[variant]
        pev.write_json(os.path.join(d, "persona_run.json"), {"status": "done", "arm": arm, "case_id": case_id,
                                                             "variant": variant, "turns": 3, "cap_hit": False})
        pev.write_json(os.path.join(d, "steer_case.json"), {"case_id": case_id, "variant": variant, "goal": "g",
                                                            "choice_points": pts,
                                                            "fact_entities": ["geometry", "bcs"]})
        pev.write_json(os.path.join(d, "clarifications.json"),
                       [{"turn": 1, "question": "q", "answer": "one two three"},
                        {"turn": 2, "question": "q", "answer": "four"}])
        with open(os.path.join(d, "generated_prompt.txt"), "w") as f:
            f.write(f"Spec. {pts[0][spec_variant]} Something else for the solver.")
        with open(os.path.join(d, "complete_prompt.txt"), "w") as f:
            f.write("Complete request.")
        return d

    def test_eval_end_to_end(self):
        d = self.make_folder("bf_neutral", "case_001", "fast", "fast")
        self.make_folder("bf_neutral", "case_001", "rigorous", "fast")
        out = se.main(["--run_dir", self.tmp, "--judge_model", "bedrock_glm5", "--budget_usd", "1",
                       "--workers", "2"])
        self.assertEqual(out, {"rows": 2, "errors": 0, "skipped_budget": 0})
        rows = sa.load_rows(self.tmp)
        by_v = {r["variant"]: r for r in rows}
        self.assertEqual(by_v["fast"]["choice_labels"], {"mesh": "fast", "solver": "not_stated"})
        self.assertEqual(by_v["fast"]["choice_match"], 0.5)
        self.assertEqual(by_v["rigorous"]["choice_match"], 0.0)
        self.assertEqual(by_v["fast"]["facts_recovered"], 0.5)
        self.assertEqual(by_v["fast"]["user_words_per_reply"], 2.0)
        self.assertEqual(by_v["fast"]["n_user_replies"], 2)
        self.assertEqual(by_v["fast"]["rep"], "rep0")
        facts = pev.read_json(os.path.join(d, "steer_facts.json"))
        self.assertEqual(facts["partial_score"], "1/2")
        again = se.main(["--run_dir", self.tmp, "--judge_model", "bedrock_glm5", "--budget_usd", "1"])
        self.assertEqual(again["rows"], 2)


def row(arm, case, variant, labels, cm, fr):
    return {"arm": arm, "rep": "rep0", "case_id": case, "variant": variant, "turns": 2, "cap_hit": False,
            "choice_labels": labels, "choice_match": cm, "facts_recovered": fr,
            "user_words_per_reply": 5.0, "n_user_replies": 1}


class TestAnalysis(unittest.TestCase):
    ROWS = [
        row("approve_all", "c1", "fast", {"m": "rigorous", "t": "fast"}, 0.5, 0.5),
        row("approve_all", "c1", "rigorous", {"m": "rigorous", "t": "fast"}, 0.5, 0.5),
        row("approve_all", "c2", "fast", {"m": "fast"}, 1.0, 0.5),
        row("approve_all", "c2", "rigorous", {"m": "neither"}, 0.0, 0.5),
        row("X", "c1", "fast", {"m": "fast", "t": "fast"}, 1.0, 0.5),
        row("X", "c1", "rigorous", {"m": "rigorous", "t": "fast"}, 0.5, 0.5),
        row("X", "c2", "fast", {"m": "fast"}, 1.0, 1.0),
        row("X", "c2", "rigorous", {"m": "rigorous"}, 1.0, 0.0),
        row("bf_neutral", "c1", "fast", {}, 0.5, 0.5),
        row("bf_neutral", "c1", "rigorous", {}, 0.5, 0.25),
        row("bf_neutral", "c2", "fast", {}, 0.0, 1.0),
        row("bf_neutral", "c2", "rigorous", {}, 0.5, 0.5),
    ]

    def test_default_override(self):
        dflt = sa.defaults(self.ROWS)
        self.assertEqual(dflt, {("c1", "fast", "m"): "rigorous", ("c1", "fast", "t"): "fast",
                                ("c1", "rigorous", "m"): "rigorous", ("c1", "rigorous", "t"): "fast",
                                ("c2", "fast", "m"): "fast"})
        self.assertEqual(sa.default_override(self.ROWS, "X", dflt), (0.5, 2))
        self.assertEqual(sa.default_override(self.ROWS, "approve_all", dflt), (0.0, 2))

    def test_did(self):
        pairs = sa.paired_deltas(self.ROWS, "X")
        dc, df, did = sa.summarise(pairs)
        self.assertAlmostEqual(dc, 0.5)
        self.assertAlmostEqual(df, -0.0625)
        self.assertAlmostEqual(did, 0.5625)
        (_, _), (_, _), (lo, hi) = sa.bootstrap(pairs)
        self.assertLessEqual(lo, did)
        self.assertGreaterEqual(hi, did)
        self.assertEqual(sa.bootstrap(pairs), sa.bootstrap(pairs))
        one = [p for p in pairs if p[0] == "c1"]
        d1 = sa.summarise(one)[2]
        self.assertAlmostEqual(d1, 0.125)
        self.assertEqual(sa.bootstrap(one)[2], (d1, d1))

    def test_outputs(self):
        tmp = tempfile.mkdtemp()
        try:
            with open(os.path.join(tmp, "steer_results.jsonl"), "w") as f:
                for r in self.ROWS:
                    f.write(json.dumps(r) + "\n")
            table = {r["arm"]: r for r in sa.main(["--run_dir", tmp])}
            self.assertEqual(table["X"]["n_chats"], 4)
            self.assertAlmostEqual(table["X"]["did"], 0.5625)
            self.assertNotIn("did", table["bf_neutral"])
            self.assertTrue(os.path.exists(os.path.join(tmp, "steer_arms.csv")))
            with open(os.path.join(tmp, "steer_analysis.md")) as f:
                self.assertIn("| X | 4 |", f.read())
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
