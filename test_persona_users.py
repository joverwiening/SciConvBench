import io
import os
import sys
import json
import random
import shutil
import tempfile
import subprocess
import unittest
import importlib.util
from contextlib import redirect_stdout

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

import test_persona_harness as th
import persona_users as pu
import persona_runner as prun
import persona_analysis as pan
from llm_user_interface import LLMUserInterface

UPSTREAM_COMMIT = "8d1528ed8"
PERSONAS = pu.load_personas()


def setUpModule():
    th.setUpModule()


def tearDownModule():
    th.tearDownModule()


def jload(path):
    with open(path) as f:
        return json.load(f)


class Recorder:
    def __init__(self):
        self.calls = []

    def invoke(self, user_prompt, system_prompt=None):
        self.calls.append((user_prompt, system_prompt))
        return f"answer {len(self.calls)}"


class TestParseFinalJson(unittest.TestCase):
    def test_valid_escapes_survive(self):
        raw = r'{"a": "line1\nline2", "b": "3 \u00d7 4", "c": "say \"hi\"", "d": "C:\\temp"}'
        out = pu.parse_final_json(f"x\n<FINAL_JSON>\n{raw}\n</FINAL_JSON>")
        self.assertEqual(out, {"a": "line1\nline2", "b": "3 \u00d7 4", "c": 'say "hi"', "d": "C:\\temp"})
        self.assertEqual(pu.parse_final_json(f"```json\n{raw}\n```")["b"], "3 \u00d7 4")

    def test_invalid_escape_repaired(self):
        out = pu.parse_final_json(r'<FINAL_JSON>{"eq": "\alpha + \beta_2"}</FINAL_JSON>')
        self.assertEqual(out, {"eq": r"\alpha + \beta_2"})
        self.assertIsNone(pu.parse_final_json("<FINAL_JSON>[1, 2]</FINAL_JSON>"))
        self.assertIsNone(pu.parse_final_json("no json here"))


class TestRenderPersona(unittest.TestCase):
    def block(self, pid):
        p = PERSONAS[pid]
        return "\n\n".join([p["text"], "\n".join([pu.EXAMPLES_HEADER] + [f'"{e}"' for e in p["examples"]])])

    def test_neutral(self):
        self.assertEqual(pu.render_persona(None), pu.PRECEDENCE + "\n\n" + PERSONAS["neutral"]["text"])
        self.assertEqual(pu.persona_text_for_arm("neutral"), pu.render_persona(None))

    def test_placebo_and_trait_appended(self):
        base = pu.PRECEDENCE + "\n\n" + PERSONAS["neutral"]["text"] + "\n\n"
        for pid in ("neutral_b", "neutral_c", "terse", "busy", "annoyed"):
            text = pu.persona_text_for_arm(pid)
            self.assertEqual(text, base + self.block(pid), pid)
            for e in PERSONAS[pid]["examples"]:
                self.assertIn(f'\n"{e}"', text)
        self.assertIn("Examples of how you write (style only; [brackets] stand for values that must come from "
                      "the COMPLETE requirement, never invented):", pu.render_persona("terse"))

    def test_no_examples_no_block(self):
        ps = {"neutral": {"text": "N"}, "x": {"text": "X", "kind": "style"}}
        self.assertEqual(pu.render_persona("x", ps), pu.PRECEDENCE + "\n\nN\n\nX")


def upstream_module(tmp):
    src = subprocess.run(["git", "-C", REPO, "show", f"{UPSTREAM_COMMIT}:llm_user_interface.py"],
                         capture_output=True, text=True, check=True).stdout
    path = os.path.join(tmp, "upstream_llm_user_interface.py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("upstream_llm_user_interface", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestTurnNote(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.case = th.CASES[th.CASE_IDS[0]]

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def ask(self, user):
        user.answer_clarification_question("What is the domain?")
        user.answer_clarification_question("What is the boundary condition?")

    def test_paper_arm_byte_identical(self):
        try:
            up = upstream_module(self.tmp)
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("git or upstream commit unavailable")
        recs = []
        for k, cls in enumerate((up.LLMUserInterface, LLMUserInterface)):
            rec = Recorder()
            self.ask(cls(llm_service=rec, incomplete_requirement=self.case["incomplete_user_req"],
                         complete_requirement=self.case["complete_user_req"],
                         log_file=os.path.join(self.tmp, f"{k}.log")))
            recs.append(rec.calls)
        rec = Recorder()
        self.ask(pu.make_user("paper", rec, self.case["incomplete_user_req"], self.case["complete_user_req"],
                             os.path.join(self.tmp, "paper.log")))
        recs.append(rec.calls)
        self.assertEqual(len(recs[0]), 2)
        self.assertEqual(recs[0], recs[1])
        self.assertEqual(recs[0], recs[2])
        self.assertEqual(pu.turn_note_for_arm("paper"), "")
        self.assertEqual(pu.turn_note_for_arm("oracle"), "")
        self.assertEqual(pu.turn_note_for_arm("mute"), "")

    def test_reminder_appended(self):
        nr = PERSONAS["neutral"]["reminder"]
        self.assertEqual(pu.turn_note_for_arm("neutral"), f"(Stay in character. {nr})")
        for arm in ("neutral_b", "terse"):
            self.assertEqual(pu.turn_note_for_arm(arm), f"(Stay in character. {nr} {PERSONAS[arm]['reminder']})")
        paper, terse = Recorder(), Recorder()
        for arm, rec in (("paper", paper), ("terse", terse)):
            self.ask(pu.make_user(arm, rec, self.case["incomplete_user_req"], self.case["complete_user_req"],
                                 os.path.join(self.tmp, f"{arm}.log")))
        for (pu_p, _), (tu_p, ts_p) in zip(paper.calls, terse.calls):
            self.assertEqual(tu_p, pu_p + "\n\n" + pu.turn_note_for_arm("terse"))
            self.assertIn(pu.persona_text_for_arm("terse"), ts_p)


class TestBusyUser(unittest.TestCase):
    def test_deferral_schedule(self):
        tmp = tempfile.mkdtemp()
        case = th.CASES[th.CASE_IDS[0]]
        rec = Recorder()
        log = os.path.join(tmp, "busy.log")
        user = pu.make_user("busy", rec, case["incomplete_user_req"], case["complete_user_req"], log)
        self.assertIsInstance(user, pu.BusyUser)
        defer = PERSONAS["busy"]["defer_reply"]
        answers = [user.answer_clarification_question(f"Q{k}?") for k in range(1, 6)]
        self.assertEqual(answers, [defer, "answer 1", defer, "answer 2", defer])
        self.assertEqual(len(rec.calls), 2)
        self.assertEqual([c["deferred"] for c in user.clarifications], [True, False, True, False, True])
        # the LLM sees the deferred exchange in its history
        self.assertIn(f"<question>Q1?</question>\n<answer>{defer}</answer>", rec.calls[0][1])
        self.assertIn(f"<answer>{defer}</answer>", rec.calls[1][1])
        self.assertIn("\n\n" + pu.turn_note_for_arm("busy"), rec.calls[0][0])
        with open(log) as f:
            text = f.read()
        self.assertEqual(text.count("=== "), 5)
        self.assertEqual(text.count(f"A: {defer}"), 3)
        self.assertNotIn("Thinking...", text)
        for arm in ("terse", "mute"):
            u = pu.make_user(arm, Recorder(), case["incomplete_user_req"], case["complete_user_req"],
                             os.path.join(tmp, f"{arm}.log"))
            self.assertNotIsInstance(u, pu.BusyUser)
            u.answer_clarification_question("Q?")
            self.assertIs(u.clarifications[0]["deferred"], False)
        u = pu.make_user("oracle", None, case["incomplete_user_req"], case["complete_user_req"],
                         os.path.join(tmp, "o.log"), cards=[{"index": 0, "text": "FACT"}])
        u.answer_clarification_question("Q?")
        self.assertIs(u.clarifications[0]["deferred"], False)
        shutil.rmtree(tmp)


class TestRunnerRecords(unittest.TestCase):
    def test_clarifications_and_attempts(self):
        tmp = tempfile.mkdtemp()
        cards = os.path.join(tmp, "cards.json")
        th.synthetic_cards(cards)
        out = os.path.join(tmp, "run")
        arms = ["neutral", "busy", "oracle", "mute"]
        cid = th.CASE_IDS[0]
        argv = ["--domain", th.DOMAIN, "--mode", th.MODE, "--cases", cid, "--arms", ",".join(arms),
                "--agent_model", "or_gpt_5_2", "--user_model", "or_sonnet_4_6", "--reps", "1",
                "--workers", "2", "--out", out, "--cards", cards]
        th.FakeLLM.complete_after = 3
        with redirect_stdout(io.StringIO()):
            self.assertEqual(prun.main(argv), {"jobs": 4, "skipped": 0, "done": 4, "errors": 0})

        def folder(arm):
            return prun.case_dir(out, arm, 1, th.DOMAIN, "or_gpt_5_2", th.MODE, cid)
        defer = PERSONAS["busy"]["defer_reply"]
        for arm in arms:
            clar = jload(os.path.join(folder(arm), "clarifications.json"))
            run = jload(os.path.join(folder(arm), "persona_run.json"))
            self.assertEqual([c["turn"] for c in clar], [1, 2])
            self.assertEqual(set(clar[0]), {"turn", "question", "answer", "deferred"})
            want = [True, False] if arm == "busy" else [False, False]
            self.assertEqual([c["deferred"] for c in clar], want, arm)
            self.assertEqual(run["deferred"], sum(want))
            self.assertEqual(run["attempts"], 1)
        self.assertEqual(jload(os.path.join(folder("busy"), "clarifications.json"))[0]["answer"], defer)
        self.assertEqual(jload(os.path.join(folder("mute"), "clarifications.json"))[1]["answer"], pu.MUTE_REPLY)

        def set_run(arm, **kw):
            path = os.path.join(folder(arm), "persona_run.json")
            run = jload(path)
            run.update(kw)
            with open(path, "w") as f:
                json.dump(run, f)
        set_run("neutral", status="error", attempts=3)
        set_run("busy", status="error", attempts=2)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(prun.main(argv), {"jobs": 4, "skipped": 2, "done": 1, "errors": 0})
        self.assertIn(f"1 jobs not rerun (error after {prun.MAX_ATTEMPTS} attempts):", buf.getvalue())
        self.assertIn(str(("neutral", 1, cid)), buf.getvalue())
        self.assertEqual(jload(os.path.join(folder("neutral"), "persona_run.json"))["status"], "error")
        busy = jload(os.path.join(folder("busy"), "persona_run.json"))
        self.assertEqual((busy["status"], busy["attempts"]), ("done", 3))
        shutil.rmtree(tmp)

    def test_template_sha_covers_persona_code(self):
        import inspect
        base = "".join(inspect.getsource(getattr(LLMUserInterface, n))
                       for n in ("get_full_context", "build_system_prompt", "answer_clarification_question"))
        full = base + "".join(inspect.getsource(o) for o in (pu.render_persona, pu.turn_note_for_arm, pu.BusyUser))
        self.assertEqual(prun.user_prompt_template_sha(), pu.sha256_text(full))


N_CASES = 60


def analysis_rows():
    rng = random.Random(5)
    shift = {"neutral": 0.0, "terse": -0.25, "verbose": 0.0, "busy": 0.15}
    rows = []
    for c in range(N_CASES):
        base = rng.uniform(0.3, 0.6)
        probs = dict({a: base + d for a, d in shift.items()}, oracle=0.95, mute=0.1)
        for arm, pr in probs.items():
            for rep in (1, 2):
                items = [{"index": i, "asked": True, "delivered": True, "partial": False, "unasked_delivered": False,
                          "unasked_partial": False, "recovered": i % 2 == 0, "grounded": i % 2 == 0,
                          "silent": False} for i in range(4)]
                rates = {k: sum(it[k] for it in items) / len(items) for k in pan.ITEM_METRICS}
                rates["recovered_judge"] = sum(rng.random() < pr for _ in range(10)) / 10
                llm = arm not in ("oracle", "mute", "busy")
                rows.append({"arm": arm, "rep": rep, "domain": "pde", "mode": "disambiguation",
                             "case_id": f"c{c}", "native": {"FR": 0, "CGRR": 0, "SRR": 0, "IC": 1},
                             "items": items, "item_rates": rates, "turns": 3, "questions": 2, "cap_hit": False,
                             "deferred": 1 if arm == "busy" else 0,
                             "trait_present_share": None,
                             "trait_by_turn": [{"turn": 1, "present": {"terse": int(arm == "terse")}}] if llm
                             else None})
    for r in [r for r in rows if r["arm"] == "neutral"]:
        for arm in ("neutral_b", "neutral_c"):
            rows.append(dict(json.loads(json.dumps(r)), arm=arm))

    def find(arm, case, rep):
        return [r for r in rows if (r["arm"], r["case_id"], r["rep"]) == (arm, case, rep)][0]
    rows.remove(find("terse", "c0", 2))
    rows.remove(find("verbose", "c1", 1))
    rows.remove(find("verbose", "c1", 2))
    find("neutral", "c2", 1)["native"]["over_length_cap"] = True
    find("verbose", "c3", 1)["items"][1]["card_ok"] = False
    return rows


class TestAnalysisPrimary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = analysis_rows()
        cls.res = pan.analyse(cls.rows)
        cls.md = pan.render_markdown(cls.res)

    def prim(self, arm, table="paired"):
        return [e for e in self.res[table] if e["arm"] == arm and e["metric"] == pan.PRIMARY][0]

    def test_primary_field(self):
        self.assertEqual(pan.PRIMARY, "recovered_judge")
        self.assertIn("unasked_partial", pan.ITEM_METRICS)
        self.assertIn("recovered", pan.SECONDARY_METRICS)
        terse = self.prim("terse")
        self.assertLess(terse["effect"], -0.15)
        rec = [e for e in self.res["paired"] if e["arm"] == "terse" and e["metric"] == "recovered"][0]
        self.assertEqual(rec["effect"], 0.0)
        self.assertIn("Primary outcome: item-recovery rate (`item_rates.recovered_judge`)", self.md)
        self.assertIn("p_holm (descriptive)", self.md)

    def test_exclusions_and_reps(self):
        self.assertEqual(self.res["excluded"], {"over_length_cap": 1, "card_not_ok": {"verbose": 1}})
        self.assertIn("Rows over the prompt length cap: 1", self.md)
        summ = {s["arm"]: s for s in self.res["reps_summary"]}
        self.assertEqual((summ["terse"]["n_short"], summ["terse"]["short_cases"]), (1, ["c0"]))
        self.assertEqual((summ["verbose"]["n_cases"], summ["verbose"]["short_cases"]), (N_CASES - 1, ["c3"]))
        self.assertEqual(summ["neutral"]["short_cases"], ["c2"])

    def test_missing_family_warning(self):
        missing = [a for a in pan.HOLM_FAMILY if a not in ("terse", "verbose", "busy")]
        self.assertEqual(self.res["holm_missing"], missing)
        self.assertEqual(self.res["holm_m"], 3)
        self.assertIn("WARNING: Holm family incomplete", self.md)
        self.assertIn("m = 3 of 10", self.md)
        # Holm with m = 3: the smallest p is multiplied by 3
        recs = sorted((self.prim(a) for a in ("terse", "verbose", "busy")), key=lambda r: r["p"])
        self.assertEqual(recs[0]["p_holm"], min(1.0, 3 * recs[0]["p"]))
        rows = [dict(r, case_id="x" + r["case_id"]) if r["arm"] == "terse" else r for r in self.rows]
        with self.assertRaises(AssertionError):
            pan.analyse(rows)

    def test_selection_fewer_than_three(self):
        self.assertTrue(self.prim("terse")["counts"])
        self.assertTrue(self.prim("busy")["counts"])
        self.assertFalse(self.prim("verbose")["counts"])
        self.assertEqual(self.res["selected"], ["terse", "busy"])
        self.assertIn("Selected (traits with counts = True, top 3 by |Δ|): terse, busy.", self.md)
        self.assertTrue(all("p_top" in s for s in self.res["selection"]))

    def test_enforced_manipulation(self):
        man = {m["arm"]: m for m in self.res["manipulation"]}
        self.assertEqual((man["busy"]["mean_deferred"], man["busy"]["manip_pass"]), (1.0, True))
        self.assertNotIn("busy", self.res["traits"])
        self.assertIn("enforced in code", self.md)
        rows = [dict(r, deferred=0) if r["arm"] == "busy" else r for r in self.rows]
        res = pan.analyse(rows)
        self.assertFalse([e for e in res["paired"] if e["arm"] == "busy" and e["metric"] == pan.PRIMARY][0]
                         ["manip_pass"])
        self.assertNotIn("busy", res["selected"])

    def test_sensitivity(self):
        self.assertEqual(self.res["sensitivity_n"], N_CASES - 4)
        self.assertEqual(self.prim("terse")["n_cases"], N_CASES)
        self.assertEqual(self.prim("verbose")["n_cases"], N_CASES - 1)
        for a in ("terse", "verbose", "busy"):
            self.assertEqual(self.prim(a, "sensitivity")["n_cases"], N_CASES - 4)
        self.assertLess(self.prim("terse", "sensitivity")["ci_hi"], 0)
        self.assertIn("Sensitivity: complete cases only", self.md)

    def test_failure_table(self):
        tmp = tempfile.mkdtemp()
        n = 0
        jobs = {(r["arm"], r["rep"], r["case_id"]) for r in self.rows}
        extra = [("terse", 2, "c0", "error", 3), ("verbose", 1, "c1", "done", 1), ("verbose", 2, "c1", "error", 2)]
        for arm, rep, cid, status, att in [(a, r, c, "done", 1) for a, r, c in sorted(jobs)] + extra:
            d = os.path.join(tmp, arm, f"rep{rep}", cid)
            os.makedirs(d)
            with open(os.path.join(d, "persona_run.json"), "w") as f:
                json.dump({"arm": arm, "rep": rep, "domain": "pde", "mode": "disambiguation", "case_id": cid,
                           "status": status, "attempts": att, "deferred": 1 if arm == "busy" else 0}, f)
            n += 1
        res = pan.analyse([{k: v for k, v in r.items() if k != "deferred"} for r in self.rows], pan.load_runs(tmp))
        fail = {f["arm"]: f for f in res["failures"]}
        self.assertEqual(fail["terse"], {"arm": "terse", "jobs": 2 * N_CASES, "done": 2 * N_CASES - 1, "error": 1,
                                         "attempts_capped": 1, "no_eval_row": 1})
        self.assertEqual(fail["verbose"], {"arm": "verbose", "jobs": 2 * N_CASES, "done": 2 * N_CASES - 1,
                                           "error": 1, "attempts_capped": 0, "no_eval_row": 2})
        self.assertEqual(fail["neutral"]["no_eval_row"], 0)
        self.assertEqual(sum(f["jobs"] for f in res["failures"]), n)
        # deferred filled from persona_run.json when the eval row lacks it
        self.assertEqual({m["arm"]: m for m in res["manipulation"]}["busy"]["manip_pass"], True)
        out = os.path.join(tmp, "an")
        rpath = os.path.join(tmp, "results.jsonl")
        with open(rpath, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in self.rows)
        with redirect_stdout(io.StringIO()):
            pan.main(["--results", rpath, "--out", out, "--run_dir", tmp])
        for name in ("analysis.md", "failures.csv", "sensitivity.csv", "reps.csv", "reps_summary.csv"):
            self.assertTrue(os.path.exists(os.path.join(out, name)), name)
        with open(os.path.join(out, "analysis.md")) as f:
            self.assertIn("attempts_capped", f.read())
        shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
