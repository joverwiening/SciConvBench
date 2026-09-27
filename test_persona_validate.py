import os
import re
import sys
import csv
import json
import time
import shutil
import tempfile
import threading
import unittest

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

import main as sc_main
import test_runner as tr
import llm_judge_disambiguation
import llm_judge_disambiguation_chat
import llm_judge_inconsistency
import llm_judge_inconsistency_chat
import llm_judge_intent
import persona_users as pu
import persona_fact_cards as pfc
import persona_runner as prun
import persona_eval as pev
import persona_validate as pv

DOMAIN, MODE = "pde", "disambiguation"
CASES = pu.load_cases(DOMAIN, MODE)
CASE_IDS = [cid for cid, c in CASES.items() if len(c["missing_entities"]) >= 2][:2]
GT_TO_ENTITIES = {c["complete_user_req"].strip(): c["missing_entities"] for c in CASES.values()}
ARMS = ["neutral", "oracle", "mute"]
_lock = threading.Lock()


def between(tag, text):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def fj(obj):
    return f"<FINAL_JSON>\n{json.dumps(obj)}\n</FINAL_JSON>"


class FakeLLM:
    """Same routing as test_persona_harness.FakeLLM, reduced to the calls this pipeline makes."""

    def __init__(self, config):
        self.model_version = config.model_version
        self.calls = 0
        self.failed_calls = 0

    def invoke(self, user_prompt, system_prompt=None, max_retries=5):
        with _lock:
            self.calls += 1
        sp = system_prompt or ""
        if "Requirement Analyst" in sp:
            if "Time is up" in sp:
                return "FINAL SPEC (forced)"
            if self.calls >= 3:
                return "[COMPLETE]\nUse the boundary value 42."
            return f"What is detail number {self.calls}?"
        if "You are a User who knows the COMPLETE requirement" in sp:
            return "The boundary value is 42."
        if "has successfully recovered information that was known to be missing" in sp:
            ents = GT_TO_ENTITIES.get(between("GROUND_TRUTH", user_prompt), ["?"])
            return fj({"score": f"1/{len(ents)}", "recovered_entities": [ents[0]], "still_missing_entities": ents[1:]})
        if "[MISSING_ENTITIES]" in sp:
            return fj({"questions_about_missing_entities": {"reasoning": "r", "count": 1},
                       "relevant_questions": {"reasoning": "r", "count": 1},
                       "memory_consistency": {"reasoning": "r", "is_consistent": 1}})
        if "faithful to the overall intent" in sp:
            return fj({"score": "1/1", "status": "FAITHFUL", "reasoning": "r"})
        if "You label one question of a clarification dialogue" in sp:
            return fj({"asked": [c["card_id"] for c in json.loads(between("FACT_CARDS", user_prompt))]})
        if "You label one turn of a clarification dialogue" in sp:
            cards = json.loads(between("FACT_CARDS", user_prompt))
            answer = between("ANSWER", user_prompt)
            conveyed = [{"card_id": c["card_id"], "level": "full", "evidence": c["text"]}
                        for c in cards if pu.norm_ws(c["text"]) in pu.norm_ws(answer)]
            return fj({"asked": [c["card_id"] for c in cards], "conveyed": conveyed})
        if "You check which behavioural traits" in sp:
            ids = re.findall(r"^\s*- (\w+):", between("TRAITS", user_prompt), re.M)
            return fj({"traits": {t: {"present": 0, "evidence": ""} for t in ids}})
        raise AssertionError(f"FakeLLM: unrecognised system prompt: {sp[:120]!r}")

    def get_statistics(self, name="x"):
        return {"total_calls": self.calls, "failed_calls": 0, "total_retries": 0,
                "total_tokens": 30 * self.calls, "prompt_tokens": 20 * self.calls,
                "completion_tokens": 10 * self.calls, "api_usage_calls": 0,
                "api_input_tokens": 0, "api_output_tokens": 0, "api_reasoning_tokens": 0}


PATCHED = [sc_main, tr, llm_judge_disambiguation, llm_judge_disambiguation_chat, llm_judge_inconsistency,
           llm_judge_inconsistency_chat, llm_judge_intent, prun, pev, pfc]
_saved = {}


def setUpModule():
    for mod in PATCHED:
        _saved[mod] = mod.LLMService
        mod.LLMService = FakeLLM
    _saved["sleep"] = time.sleep
    time.sleep = lambda s: None
    _saved["prices"] = pu.fetch_openrouter_prices
    pu.fetch_openrouter_prices = lambda ids: {i: None for i in ids}


def tearDownModule():
    for mod in PATCHED:
        mod.LLMService = _saved[mod]
    time.sleep = _saved["sleep"]
    pu.fetch_openrouter_prices = _saved["prices"]


def synthetic_cards(path):
    data = {"meta": {"model": None, "source_sha256": pu.sha256_file(pu.data_path(DOMAIN, MODE))}, "cases": {}}
    for cid in CASE_IDS:
        words = CASES[cid]["complete_user_req"].split()
        cards = []
        for i, item in enumerate(CASES[cid]["missing_entities"]):
            cards.append({"card_id": f"{DOMAIN}/{MODE}/{cid}#{i}", "case_id": cid, "index": i, "item": item,
                          "spans": [" ".join(words[:5])], "statement": "s", "text": f"FACT {cid} number {i}",
                          "checks": {}, "status": "ok"})
        data["cases"][cid] = {"status": "ok", "cards": cards, "uncovered_share": 0.25}
    with open(path, "w") as f:
        json.dump(data, f)


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)


class TestValidate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.cards = os.path.join(cls.tmp, "cards.json")
        synthetic_cards(cls.cards)
        cls.run_dir = os.path.join(cls.tmp, "run")
        prun.main(["--domain", DOMAIN, "--mode", MODE, "--cases", ",".join(CASE_IDS), "--arms", ",".join(ARMS),
                   "--agent_model", "or_gpt_5_2", "--user_model", "or_sonnet_4_6", "--reps", "2",
                   "--workers", "4", "--out", cls.run_dir, "--cards", cls.cards])
        res = pev.main(["--run_dir", cls.run_dir, "--judge_model", "or_gemini_2_5_pro",
                        "--aux_model", "or_gemini_2_5_flash", "--workers", "4"])
        assert res == {"rows": 12, "errors": 0}, res
        cls.exp = os.path.join(cls.tmp, "export")
        cls.counts = pv.export(cls.run_dir, 3, 0, cls.exp)
        cls.key = pev.read_json(os.path.join(cls.exp, pv.KEY_FILE))
        cls.attr = read_rows(os.path.join(cls.exp, pv.ATTR_FILE))
        cls.fr = read_rows(os.path.join(cls.exp, pv.FR_FILE))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def filled_copy(self, name, attr_fn=None, fr_fn=None):
        d = os.path.join(self.tmp, name)
        os.makedirs(d)
        shutil.copy(os.path.join(self.exp, pv.KEY_FILE), d)
        for fname, rows, fn in ((pv.ATTR_FILE, self.attr, attr_fn), (pv.FR_FILE, self.fr, fr_fn)):
            out = []
            for r in rows:
                r = dict(r)
                r["label_a"] = str(self.key[r["row_id"]]["model_label"])
                if fn:
                    fn(r)
                out.append(r)
            write_rows(os.path.join(d, fname), out)
        return d

    def test_blind(self):
        self.assertEqual(list(self.attr[0]), pv.ATTR_COLS)
        self.assertEqual(list(self.fr[0]), pv.FR_COLS)
        banned = ARMS + ["rep1", "rep2", self.tmp, self.run_dir, "persona", "or_gpt_5_2_"]
        for rows in (self.attr, self.fr):
            for r in rows:
                self.assertEqual((r["label_a"], r["label_b"]), ("", ""))
                self.assertRegex(r["row_id"], r"^[AF]\d{5}$")
                for col, cell in r.items():
                    for b in banned:
                        self.assertNotIn(b.lower(), cell.lower(), (col, b))

    def test_key_one_to_one(self):
        ids = [r["row_id"] for r in self.attr + self.fr]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), set(self.key))
        for r in self.attr:
            k = self.key[r["row_id"]]
            self.assertEqual(k["task"], "attribution")
            self.assertIn(k["model_label"], pv.LEVELS)
            self.assertTrue(os.path.isdir(os.path.join(self.run_dir, k["case_dir"])))
            self.assertEqual(k["model_label"], "full" if k["arm"] == "oracle" else "none")
        for r in self.fr:
            k = self.key[r["row_id"]]
            self.assertEqual(k["task"], "fr")
            self.assertEqual(k["model_label"], int(k["index"] == 0))
            self.assertEqual(r["entity"], CASES[k["case_id"]]["missing_entities"][k["index"]])

    def test_stratified(self):
        self.assertEqual(self.counts["attribution_turns"], {a: 3 for a in ARMS})
        self.assertEqual(self.counts["fr_items"], {a: 3 for a in ARMS})
        n_cards = {cid: len(CASES[cid]["missing_entities"]) for cid in CASE_IDS}
        turns = {}
        for r in self.attr:
            k = self.key[r["row_id"]]
            turns.setdefault(k["arm"], set()).add((k["case_dir"], k["turn"]))
        for arm, ts in turns.items():
            self.assertEqual(len(ts), 3)
            n_rows = sum(1 for r in self.attr if self.key[r["row_id"]]["arm"] == arm)
            self.assertEqual(n_rows, sum(n_cards[d.split(os.sep)[-1]] for d, _ in ts))
        # 2 cases x 2 reps x 2 questions = 8 turns per arm available
        big = pv.export(self.run_dir, 100, 0, os.path.join(self.tmp, "export_all"))
        self.assertEqual(big["attribution_turns"], {a: 8 for a in ARMS})
        items = 2 * sum(n_cards.values())
        self.assertEqual(big["fr_items"], {a: items for a in ARMS})
        again = pv.export(self.run_dir, 3, 0, os.path.join(self.tmp, "export_again"))
        self.assertEqual(pev.read_json(os.path.join(self.tmp, "export_again", pv.KEY_FILE)), self.key)
        self.assertEqual(again, self.counts)

    def test_score_perfect_and_flipped(self):
        res = pv.score(self.filled_copy("perfect"), B=500)
        for task in ("attribution", "fr"):
            t = res[task]
            self.assertEqual(t["n_labelled"], t["n_rows"])
            for d in t["per_arm"].values():
                self.assertEqual(d["accuracy"], 1.0)
            for m in t["arm_vs_neutral"].values():
                self.assertEqual((m["diff"], m["ci_lo"], m["ci_hi"], m["ci_excludes_0"]), (0.0, 0.0, 0.0, False))
        self.assertEqual(res["attribution"]["overall"]["per_class"]["full"]["precision"], 1.0)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "perfect", "validation_report.md")))

        arm = "neutral"
        attr_ids = sorted(r["row_id"] for r in self.attr if self.key[r["row_id"]]["arm"] == arm)[:2]
        fr_ids = sorted(r["row_id"] for r in self.fr if self.key[r["row_id"]]["arm"] == arm)[:1]
        flip_a = {"full": "none", "none": "full", "partial": "none"}

        def fa(r):
            if r["row_id"] in attr_ids:
                r["label_a"] = flip_a[r["label_a"]]

        def ff(r):
            if r["row_id"] in fr_ids:
                r["label_a"] = "0" if r["label_a"] == "1" else "1"

        res = pv.score(self.filled_copy("flipped", fa, ff), B=500)
        a = res["attribution"]["per_arm"][arm]
        self.assertAlmostEqual(a["accuracy"], (a["n"] - 2) / a["n"])
        self.assertAlmostEqual(a["binary_accuracy"], (a["n"] - 2) / a["n"])
        for m in res["attribution"]["arm_vs_neutral"].values():
            self.assertAlmostEqual(m["diff"], -2 / a["n"])
            self.assertLessEqual(m["ci_lo"], m["diff"])
            self.assertGreaterEqual(m["ci_hi"], m["diff"])
        f = res["fr"]["per_arm"][arm]
        self.assertAlmostEqual(f["accuracy"], (f["n"] - 1) / f["n"])
        for m in res["fr"]["arm_vs_neutral"].values():
            self.assertAlmostEqual(m["diff"], -1 / f["n"])
        for other in ARMS:
            if other != arm:
                self.assertEqual(res["fr"]["per_arm"][other]["accuracy"], 1.0)

    def test_label_b_fallback_and_kappa(self):
        def both(r):
            r["label_b"] = r["label_a"]

        res = pv.score(self.filled_copy("kappa", both, both), B=200)
        for task in ("attribution", "fr"):
            k = res[task]["kappa_a_b"]
            self.assertEqual(k["n"], res[task]["n_rows"])
            self.assertAlmostEqual(k["kappa"], 1.0)

        def only_b(r):
            r["label_b"], r["label_a"] = r["label_a"], ""

        res = pv.score(self.filled_copy("only_b", only_b, only_b), B=200)
        self.assertEqual(res["fr"]["overall"]["accuracy"], 1.0)
        self.assertEqual(res["fr"]["kappa_a_b"]["n"], 0)
        self.assertAlmostEqual(pv.cohen_kappa(["1", "0", "1", "0"], ["1", "0", "0", "1"], pv.FR_LABELS), 0.0)

    def test_invalid_label_rejected(self):
        def bad(r):
            r["label_a"] = "maybe"

        with self.assertRaises(SystemExit):
            pv.score(self.filled_copy("bad", bad, None), B=10)

    def test_export_cards(self):
        out = os.path.join(self.tmp, "audit.md")
        sampled = pv.export_cards(DOMAIN, MODE, 5, 0, self.cards, out)
        self.assertEqual(sorted(sampled), sorted(CASE_IDS))
        with open(out) as f:
            text = f.read()
        sections = re.findall(r"^## (\S+)$", text, re.M)
        self.assertEqual(sorted(sections), sorted(CASE_IDS))
        n_cards = sum(len(CASES[c]["missing_entities"]) for c in CASE_IDS)
        self.assertEqual(text.count("verdict: correct / wrong / incomplete"), n_cards)
        self.assertIn("{+", text)
        self.assertEqual(len(pv.export_cards(DOMAIN, MODE, 1, 0, self.cards, out)), 1)

    def test_word_diff(self):
        self.assertEqual(pv.word_diff("a b c", "a X c d"), "a [-b-] {+X+} c {+d+}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
