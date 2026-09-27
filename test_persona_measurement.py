import os
import re
import sys
import csv
import json
import shutil
import argparse
import tempfile
import unittest

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

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

DOMAIN, MODE, INC = "pde", "disambiguation", "inconsistency"
CASES = pu.load_cases(DOMAIN, MODE)
CID = [cid for cid, c in CASES.items() if len(c["missing_entities"]) >= 2][0]
ITEMS = CASES[CID]["missing_entities"]
INC_CASES = pu.load_cases(DOMAIN, INC)
INC_CID = [cid for cid, c in INC_CASES.items() if len(c["inconsistencies"]) >= 1][0]


def between(tag, text):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def fj(obj):
    return f"<FINAL_JSON>\n{json.dumps(obj)}\n</FINAL_JSON>"


def card_id(i, mode=MODE, cid=CID):
    return f"{DOMAIN}/{mode}/{cid}#{i}"


def conveys_text(cards, answer):
    """Default conveyed judge: a card is conveyed in full when its text is in the answer."""
    return fj({"asked": [c["card_id"] for c in cards],
               "conveyed": [{"card_id": c["card_id"], "level": "full", "evidence": c["text"]}
                            for c in cards if c["text"] in answer]})


class FakeLLM:
    """Routes by system prompt; behaviour per call type is set on the class by each test."""
    calls = []
    fr = staticmethod(lambda ents: {"score": f"1/{len(ents)}", "recovered_entities": [ents[0]]})
    chat_bad = False
    ask = staticmethod(lambda cards, q: fj({"asked": [cards[0]["card_id"]]}))
    conv = staticmethod(lambda cards, q, a: conveys_text(cards, a))
    trait = staticmethod(lambda ids, a: fj({"traits": {t: {"present": 1, "evidence": a.split()[0]} for t in ids}}))
    inc_resolved = staticmethod(lambda n: [i == 0 for i in range(n)])

    def __init__(self, config):
        self.model_version = config.model_version
        self.n = 0

    def invoke(self, user_prompt, system_prompt=None, max_retries=5):
        self.n += 1
        sp = system_prompt or ""
        kind = None
        if "has successfully recovered information that was known to be missing" in sp:
            kind = "fr"
            ents = CASES[CID]["missing_entities"]
            out = fj(FakeLLM.fr(ents))
        elif "[MISSING_ENTITIES]" in sp:
            kind = "chat"
            out = "no json here" if FakeLLM.chat_bad else fj(
                {"questions_about_missing_entities": {"reasoning": "r", "count": 1},
                 "relevant_questions": {"reasoning": "r", "count": 1},
                 "memory_consistency": {"reasoning": "r", "is_consistent": 1}})
        elif "[KNOWN_INCONSISTENCIES]" in sp:
            kind = "fr"
            n = len(re.findall(r"Inconsistency #\d+:", between("KNOWN_INCONSISTENCIES", user_prompt)))
            res = FakeLLM.inc_resolved(n)
            out = fj({"score": "0/1", "status": "FAILED",
                      "breakdown": [{"inconsistency_index": i + 1, "resolved": r, "reasoning": "r"}
                                    for i, r in enumerate(res)]})
        elif "[PLANTED_INCONSISTENCIES]" in sp:
            kind = "chat"
            out = fj({"questions_about_inconsistencies": {"reasoning": "r", "count": 1},
                      "relevant_questions": {"reasoning": "r", "count": 1},
                      "explicit_warnings": {"reasoning": "r", "count": 0},
                      "memory_consistency": {"reasoning": "r", "is_consistent": 1}})
        elif "faithful to the overall intent" in sp:
            kind = "intent"
            out = fj({"score": "1/1", "status": "FAITHFUL", "reasoning": "r"})
        elif sp == pev.ASK_SYSTEM:
            kind = "ask"
            out = FakeLLM.ask(json.loads(between("FACT_CARDS", user_prompt)), between("QUESTION", user_prompt))
        elif sp == pev.ATTRIBUTION_SYSTEM:
            kind = "conv"
            out = FakeLLM.conv(json.loads(between("FACT_CARDS", user_prompt)), between("QUESTION", user_prompt),
                               between("ANSWER", user_prompt))
        elif sp == pev.TRAIT_SYSTEM:
            kind = "trait"
            ids = re.findall(r"^\s*- (\w+):", between("TRAITS", user_prompt), re.M)
            out = FakeLLM.trait(ids, between("ANSWER", user_prompt))
        else:
            raise AssertionError(f"FakeLLM: unrecognised system prompt: {sp[:120]!r}")
        FakeLLM.calls.append((kind, user_prompt))
        return out

    def get_statistics(self, name="x"):
        return {"total_calls": self.n, "failed_calls": 0, "total_retries": 0, "total_tokens": 0,
                "prompt_tokens": 0, "completion_tokens": 0}


PATCHED = [llm_judge_disambiguation, llm_judge_disambiguation_chat, llm_judge_inconsistency,
           llm_judge_inconsistency_chat, llm_judge_intent, pev, pfc]
DEFAULTS = {k: FakeLLM.__dict__[k] for k in ("fr", "chat_bad", "ask", "conv", "trait", "inc_resolved")}
_saved = {}


def setUpModule():
    for mod in PATCHED:
        _saved[mod] = mod.LLMService
        mod.LLMService = FakeLLM


def tearDownModule():
    for mod in PATCHED:
        mod.LLMService = _saved[mod]


def cards_data(mode=MODE, cid=None, texts=None):
    cid = cid or (CID if mode == MODE else INC_CID)
    items = (CASES if mode == MODE else INC_CASES)[cid][pu.ITEMS_KEY[mode]]
    texts = texts or [f"FACT {i} × value" for i in range(len(items))]
    cards = [{"card_id": card_id(i, mode, cid), "case_id": cid, "index": i, "item": it, "spans": [],
              "statement": "", "text": texts[i], "checks": {}, "status": "ok"} for i, it in enumerate(items)]
    return {"meta": {"source_sha256": pu.sha256_file(pu.data_path(DOMAIN, mode))},
            "cases": {cid: {"status": "ok", "cards": cards}}}


class Case:
    """One persona case dir plus an eval ctx, built without the runner."""

    def __init__(self, tmp, turns, mode=MODE, clar=True, arm="terse", cards=None, run_extra=None):
        self.mode = mode
        cases = CASES if mode == MODE else INC_CASES
        self.cid = CID if mode == MODE else INC_CID
        self.run_dir = os.path.join(tmp, "run")
        self.folder = prun.case_dir(self.run_dir, arm, 1, DOMAIN, "or_gpt_5_2", mode, self.cid)
        os.makedirs(self.folder)
        row = cases[self.cid]
        run = {"domain": DOMAIN, "mode": mode, "case_id": self.cid, "arm": arm, "trait": arm, "rep": 1,
               "status": "done", "models": {"agent": {"model_version": "a"}, "user": {"model_version": "u"}},
               "usage": {}, "turns": len(turns) + 1}
        run.update(run_extra or {})
        self.write("persona_run.json", json.dumps(run))
        self.write("generated_prompt.txt", "FINAL SPEC")
        self.write("complete_prompt.txt", row["complete_user_req"] if mode == MODE else row["correct_prompt"])
        self.write("conversation_log.txt", "".join(
            f"=== 2026-01-01 00:00:0{k} ===\nQ: LOG {q}\nA: LOG {a}\n" for k, (q, a, _) in enumerate(turns)))
        if clar:
            self.write("clarifications.json", json.dumps(
                [{"turn": k, "question": q, "answer": a, "deferred": d} for k, (q, a, d) in enumerate(turns, 1)]))
        self.cards = cards or cards_data(mode)
        self.cards_path = os.path.join(tmp, f"cards_{mode}.json")
        with open(self.cards_path, "w") as f:
            json.dump(self.cards, f)
        self.ctx = {"args": argparse.Namespace(skip_native=False, skip_aux=False),
                    "judge_cfg": pu.role_config("or_gemini_2_5_pro"),
                    "aux_cfg": pu.role_config("or_gemini_2_5_flash"),
                    "trait_defs": {"terse": "short", "verbose": "long"}, "personas_sha": "p",
                    "datasets": {(DOMAIN, mode): cases}, "prices": {},
                    "cards": {(DOMAIN, mode): {"data": self.cards, "sha256": pu.sha256_file(self.cards_path),
                                               "dataset_sha256": pu.sha256_file(pu.data_path(DOMAIN, mode))}}}

    def write(self, name, text):
        with open(os.path.join(self.folder, name), "w") as f:
            f.write(text)

    def eval(self):
        return pev.eval_case(self.folder, self.ctx)

    def read(self, name):
        return pev.read_json(os.path.join(self.folder, name))


def kinds():
    return [k for k, _ in FakeLLM.calls]


class Base(unittest.TestCase):
    def setUp(self):
        for k, v in DEFAULTS.items():
            setattr(FakeLLM, k, v)
        FakeLLM.calls = []
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)


TWO_TURNS = [("What is the first value?", "It is FACT 0 × value.", False),
             ("Anything else?", "No.", False)]


class TestRecovery(Base):
    def test_match_entities(self):
        items = ["Boundary condition at x=0", "Viscosity", "Time step"]
        ents = ["viscosity.", "boundary condition", "time step size", "extra"]
        self.assertEqual(pev.match_entities(items, ents), [1, 0, 2])
        # each judge entity is used once; the exact match wins over containment
        self.assertEqual(pev.match_entities(["mesh", "mesh size"], ["mesh size"]), [None, 0])
        # containment: longest item first
        self.assertEqual(pev.match_entities(["grid", "grid spacing"], ["grid spacing value"]), [None, 0])
        self.assertEqual(pev.match_entities(["a"], []), [None])

    def test_recovered_judge_and_mismatch(self):
        FakeLLM.fr = staticmethod(lambda ents: {"score": f"2/{len(ents)}",
                                                "recovered_entities": [" " + ents[0].upper() + ".", "Other thing"]})
        c = Case(self.tmp, TWO_TURNS)
        r = c.eval()
        self.assertAlmostEqual(r["item_rates"]["recovered_judge"], 2 / len(ITEMS))
        self.assertEqual(r["items"][0]["recovered"], True)
        self.assertAlmostEqual(r["item_rates"]["recovered"], 1 / len(ITEMS))
        self.assertTrue(r["item_match_mismatch"])
        self.assertEqual(r["unmatched_recovered_entities"], ["Other thing"])

    def test_recovered_judge_unparseable(self):
        for score in ("abc", f"1/0", None):
            FakeLLM.fr = staticmethod(lambda ents, s=score: {"score": s, "recovered_entities": [ents[0]]})
            c = Case(os.path.join(self.tmp, str(score)), TWO_TURNS)
            r = c.eval()
            self.assertIsNone(r["item_rates"]["recovered_judge"], score)
            self.assertIsNone(r["item_match_mismatch"])
            self.assertEqual(r["items"][0]["recovered"], True)

    def test_recovered_judge_consistent(self):
        c = Case(self.tmp, TWO_TURNS)
        r = c.eval()
        self.assertAlmostEqual(r["item_rates"]["recovered_judge"], 1 / len(ITEMS))
        self.assertFalse(r["item_match_mismatch"])

    def test_inconsistency_recovered_judge(self):
        c = Case(self.tmp, TWO_TURNS, mode=INC)
        r = c.eval()
        n = len(INC_CASES[INC_CID]["inconsistencies"])
        self.assertAlmostEqual(r["item_rates"]["recovered_judge"], 1 / n)
        self.assertEqual([it["recovered"] for it in r["items"]], [i == 0 for i in range(n)])
        self.assertFalse(r["item_match_mismatch"])


class TestParseFailures(Base):
    def test_retry_then_missing(self):
        state = {"n": 0}

        def conv(cards, q, a):
            if "Anything" in q:
                return "garbage"
            state["n"] += 1
            return "garbage" if state["n"] <= 2 else conveys_text(cards, a)

        FakeLLM.conv = staticmethod(conv)
        c = Case(self.tmp, TWO_TURNS)
        r = c.eval()
        self.assertEqual(kinds().count("conv"), 3 + 3)
        att = c.read("persona_attribution.json")
        t1, t2 = att["turns"]
        self.assertEqual((t1["parse_failure"], t1["attempts"]), (False, 3))
        self.assertEqual((t2["parse_failure"], t2["attempts"], t2["conveyed"]), (True, 3, None))
        self.assertEqual(r["parse_failures"]["attribution"], 1)
        # card 0 delivered in the valid turn; card 1 could have been delivered in the unparsed turn
        self.assertEqual(r["items"][0]["delivered"], True)
        self.assertIsNone(r["items"][1]["delivered"])
        self.assertIsNone(r["items"][1]["partial"])
        self.assertIsNone(r["items"][1]["grounded"])
        self.assertEqual(r["item_rates"]["delivered"], 1.0)

    def test_ask_parse_failure(self):
        FakeLLM.ask = staticmethod(lambda cards, q: "garbage" if "Anything" in q else fj({"asked": []}))
        c = Case(self.tmp, TWO_TURNS)
        r = c.eval()
        self.assertEqual(kinds().count("ask"), 1 + 3)
        self.assertEqual(r["parse_failures"]["ask"], 1)
        self.assertIsNone(r["items"][1]["asked"])
        # card 0 was conveyed in turn 1, which provably did not ask for it: a leak regardless of turn 2
        self.assertEqual(r["items"][0]["unasked_delivered"], True)
        # card 1 is never conveyed, so it cannot have leaked even though turn 2's asks are unknown
        self.assertIs(r["items"][1]["unasked_delivered"], False)

    def test_flags_three_valued(self):
        f = pev.attribution_flags
        ok = {"asked": ["A"], "conveyed": {"A": {"level": "full"}}}
        bad = {"asked": None, "conveyed": None}
        self.assertEqual(f("A", [ok, bad])["delivered"], True)
        self.assertEqual(f("A", [ok, bad])["asked"], True)
        self.assertIsNone(f("B", [ok, bad])["delivered"])
        self.assertIsNone(f("B", [ok, bad])["asked"])
        none = {"asked": [], "conveyed": {}}
        self.assertEqual(f("B", [ok, none]), {"asked": False, "delivered": False, "partial": False,
                                              "unasked_delivered": False, "unasked_partial": False})
        # asked unknown in an earlier turn: a later unasked full delivery may or may not be a leak
        later = {"asked": [], "conveyed": {"B": {"level": "full"}}}
        self.assertIsNone(f("B", [{"asked": None, "conveyed": {}}, later])["unasked_delivered"])
        self.assertEqual(f("B", [{"asked": ["B"], "conveyed": None}, later])["unasked_delivered"], False)

    def test_trait_share_excludes_unparsed(self):
        FakeLLM.trait = staticmethod(lambda ids, a: "garbage" if a == "No." else
                                     fj({"traits": {t: {"present": int(t == "terse"), "evidence": a.split()[0]}
                                                    for t in ids}}))
        c = Case(self.tmp, TWO_TURNS)
        r = c.eval()
        self.assertEqual(kinds().count("trait"), 1 + 3)
        self.assertEqual(r["trait_present_share"], {"terse": 1.0, "verbose": 0.0})
        self.assertIsNone(r["trait_by_turn"][1]["present"])
        self.assertEqual(r["parse_failures"]["traits"], 1)

    def test_native_failure_not_cached(self):
        FakeLLM.chat_bad = True
        c = Case(self.tmp, TWO_TURNS)
        r = c.eval()
        self.assertGreater(r["parse_failures"]["native"], 0)
        self.assertIsNone(r["native"])
        self.assertIsNone(r["item_rates"]["recovered_judge"])
        self.assertIsNone(r["native_judge"])
        self.assertIsNone(r["native_key"])
        FakeLLM.chat_bad = False
        n = kinds().count("chat")
        r = c.eval()
        self.assertEqual(kinds().count("chat"), n + 1)
        self.assertEqual(r["parse_failures"]["native"], 0)
        self.assertIsNotNone(r["native"])
        self.assertIsNotNone(r["native_judge"])
        self.assertIsNotNone(r["item_rates"]["recovered_judge"])
        n = len(FakeLLM.calls)
        c.eval()
        self.assertEqual(len(FakeLLM.calls), n)


class TestAsk(Base):
    def test_asked_from_ask_call_only(self):
        FakeLLM.ask = staticmethod(lambda cards, q: fj({"asked": [cards[1]["card_id"]] if "second" in q else []}))
        turns = [("What is the second value?", "It is FACT 0 × value.", False),
                 ("Anything else?", "FACT 1 × value", False)]
        c = Case(self.tmp, turns)
        r = c.eval()
        ask_prompts = [p for k, p in FakeLLM.calls if k == "ask"]
        self.assertEqual(len(ask_prompts), 2)
        for p in ask_prompts:
            self.assertIsNone(between("ANSWER", p))
            self.assertNotIn("It is FACT", p)
        self.assertIn("What is the second value?", ask_prompts[0])
        # the conveyed call claims every card was asked; that output is ignored
        self.assertEqual(r["items"][0]["asked"], False)
        self.assertEqual(r["items"][1]["asked"], True)
        self.assertEqual(r["items"][0]["unasked_delivered"], True)
        self.assertEqual(r["items"][1]["unasked_delivered"], False)  # asked in turn 1, delivered in turn 2

    def test_prompt_definition(self):
        self.assertIn('"anything else?"', pev.ASK_SYSTEM)
        self.assertIn("asks for no card", pev.ASK_SYSTEM)
        self.assertNotIn("<ANSWER>", pev.ASK_USER)


class TestTurns(Base):
    def test_clarifications_and_deferred(self):
        turns = [("What is the first value?", "Sorry, busy. FACT 0 × value", True),
                 ("And the second?", "FACT 1 × value", False)]
        c = Case(self.tmp, turns, run_extra={"deferred": 1, "attempts": 2})
        r = c.eval()
        self.assertEqual(r["questions"], 2)
        self.assertEqual(kinds().count("ask"), 2)
        conv = [p for k, p in FakeLLM.calls if k == "conv"]
        self.assertEqual(len(conv), 1)
        self.assertIn("And the second?", conv[0])
        self.assertNotIn("LOG", conv[0])
        att = c.read("persona_attribution.json")
        self.assertEqual(att["turns"][0]["conveyed"], {})
        self.assertTrue(att["turns"][0]["deferred"])
        self.assertEqual(att["turns"][0]["asked"], [card_id(0)])
        self.assertEqual(r["items"][0]["delivered"], False)
        self.assertEqual(r["items"][1]["delivered"], True)
        self.assertEqual((r["deferred"], r["attempts"]), (1, 2))

    def test_log_fallback(self):
        c = Case(self.tmp, TWO_TURNS, clar=False)
        r = c.eval()
        self.assertEqual(r["questions"], 2)
        self.assertIn("LOG What is the first value?", [p for k, p in FakeLLM.calls if k == "ask"][0])
        self.assertEqual((r["deferred"], r["attempts"]), (None, None))

    def test_cache_keys(self):
        c = Case(self.tmp, TWO_TURNS)
        c.eval()
        n = len(FakeLLM.calls)
        c.eval()
        self.assertEqual(len(FakeLLM.calls), n)
        c.write("generated_prompt.txt", "NEW SPEC")
        c.eval()
        self.assertEqual({k for k, _ in FakeLLM.calls[n:]}, {"fr", "chat", "intent", "ask", "conv", "trait"})
        n = len(FakeLLM.calls)
        c.write("clarifications.json", json.dumps([{"turn": 1, "question": "Q?", "answer": "A.", "deferred": False}]))
        r = c.eval()
        self.assertEqual({k for k, _ in FakeLLM.calls[n:]}, {"fr", "chat", "intent", "ask", "conv", "trait"})
        self.assertEqual(r["questions"], 1)
        att = c.read("persona_attribution.json")
        self.assertEqual(att["prompt_sha256"], pev.ATTRIBUTION_TEMPLATES_SHA)
        self.assertEqual(att["turns_sha256"], pu.sha256_file(os.path.join(c.folder, "clarifications.json")))


class TestLeakPartial(Base):
    def test_unasked_partial_flags(self):
        f = pev.attribution_flags
        turns = [{"asked": [], "conveyed": {"A": {"level": "partial"}}},
                 {"asked": ["A", "B"], "conveyed": {"A": {"level": "full"}, "B": {"level": "partial"}}}]
        a, b = f("A", turns), f("B", turns)
        self.assertEqual((a["unasked_partial"], a["unasked_delivered"], a["delivered"]), (True, False, True))
        self.assertEqual((b["unasked_partial"], b["partial"]), (False, True))

    def test_unasked_partial_rate(self):
        FakeLLM.ask = staticmethod(lambda cards, q: fj({"asked": []}))
        FakeLLM.conv = staticmethod(lambda cards, q, a: fj({"conveyed": [
            {"card_id": cards[0]["card_id"], "level": "partial", "evidence": a}]}))
        c = Case(self.tmp, TWO_TURNS)
        r = c.eval()
        self.assertEqual(r["items"][0]["unasked_partial"], True)
        self.assertAlmostEqual(r["item_rates"]["unasked_partial"], 1 / len(ITEMS))
        self.assertEqual(r["item_rates"]["unasked_delivered"], 0.0)


class TestCardIntegrity(Base):
    def test_item_mismatch(self):
        cards = cards_data()
        cards["cases"][CID]["cards"][1]["item"] = "something else"
        c = Case(self.tmp, TWO_TURNS, cards=cards)
        with self.assertRaisesRegex(ValueError, CID):
            c.eval()

    def test_item_whitespace_ok(self):
        cards = cards_data()
        cards["cases"][CID]["cards"][0]["item"] = "  " + ITEMS[0].replace(" ", "  ") + "\n"
        Case(self.tmp, TWO_TURNS, cards=cards).eval()

    def test_source_sha_mismatch(self):
        cards = cards_data()
        cards["meta"]["source_sha256"] = "0" * 64
        c = Case(self.tmp, TWO_TURNS, cards=cards)
        with self.assertRaisesRegex(ValueError, CID):
            c.eval()
        cards["meta"].pop("source_sha256")
        with self.assertRaisesRegex(ValueError, CID):
            c.eval()

    def test_card_status_recorded(self):
        cards = cards_data()
        cards["cases"][CID]["status"] = "needs_review"
        c = Case(self.tmp, TWO_TURNS, cards=cards)
        r = c.eval()
        self.assertEqual(r["card_case_status"], "needs_review")
        self.assertFalse(any(it["card_ok"] for it in r["items"]))
        self.assertEqual(r["items"][0]["delivered"], True)
        cards["cases"][CID]["status"] = "ok"
        cards["cases"][CID]["cards"][1]["status"] = "needs_review"
        r = c.eval()
        self.assertEqual([it["card_ok"] for it in r["items"]][:2], [True, False])


class TestTraitDefs(Base):
    def test_code_enforced_excluded(self):
        c = Case(self.tmp, TWO_TURNS)
        with open(os.path.join(c.run_dir, "manifest.json"), "w") as f:
            json.dump({"cards": {}}, f)
        captured = {}
        saved = (pev.eval_case, pu.fetch_openrouter_prices)
        pev.eval_case = lambda folder, ctx: captured.update(ctx)
        pu.fetch_openrouter_prices = lambda ids: {i: None for i in ids}
        try:
            pev.main(["--run_dir", c.run_dir, "--judge_model", "or_gemini_2_5_pro",
                      "--aux_model", "or_gemini_2_5_flash", "--workers", "1"])
        finally:
            pev.eval_case, pu.fetch_openrouter_prices = saved
        personas = pu.load_personas()
        code = {p for p, d in personas.items() if d.get("enforced") == "code"}
        self.assertIn("busy", code)
        self.assertTrue(captured["trait_defs"])
        self.assertFalse(code & set(captured["trait_defs"]))
        self.assertEqual(set(captured["trait_defs"]), set(pu.trait_ids(personas)) - code)
        em = pev.read_json(os.path.join(c.run_dir, "eval_manifest.json"))
        self.assertEqual(em["invocations"][0]["attribution_prompt_sha256"], pev.ATTRIBUTION_TEMPLATES_SHA)


class TestFactCardReview(Base):
    def runs(self, cid):
        c = CASES[cid]
        idx = pfc.token_index(c["incomplete_user_req"], c["complete_user_req"])
        runs, prev = [], None
        for j in idx["changed"]:
            if prev is None or j != prev + 1:
                runs.append([])
            runs[-1].append(idx["tokens"][j])
            prev = j
        return [" ".join(r) for r in runs]

    def case_with(self, n_items, n_runs=1):
        for cid, c in CASES.items():
            if len(c["missing_entities"]) == n_items and len(self.runs(cid)) >= n_runs and \
                    all(len(r.split()) >= 4 for r in self.runs(cid)):
                return cid
        self.skipTest(f"no case with {n_items} items")

    def test_clean_span_ok(self):
        cid = self.case_with(1)
        rec = pfc.check_case(DOMAIN, MODE, CASES[cid], [{"index": 0, "spans": [self.runs(cid)[0]],
                                                         "statement": "s"}], 1.0)
        self.assertEqual(rec["cards"][0]["status"], "ok")
        self.assertIsNone(rec["cards"][0]["review_reason"])

    def test_span_too_broad(self):
        cid = self.case_with(1)
        rec = pfc.check_case(DOMAIN, MODE, CASES[cid], [{"index": 0, "spans": [CASES[cid]["complete_user_req"]],
                                                         "statement": "s"}], 1.0)
        card = rec["cards"][0]
        self.assertEqual(card["status"], "needs_review")
        self.assertIn("too broad", card["review_reason"])
        self.assertLess(card["checks"]["span_in_diff_share"], 0.5)
        self.assertEqual(rec["status"], "needs_review")

    def test_span_overlap(self):
        cid = self.case_with(2)
        span = self.runs(cid)[0]
        words = span.split()
        half = " ".join(words[:len(words) // 2 + 1])
        rec = pfc.check_case(DOMAIN, MODE, CASES[cid], [{"index": 0, "spans": [span], "statement": "s"},
                                                        {"index": 1, "spans": [half], "statement": "s"}], 1.0)
        self.assertEqual([c["status"] for c in rec["cards"]], ["needs_review", "needs_review"])
        self.assertIn("overlaps card 1", rec["cards"][0]["review_reason"])
        self.assertIn("overlaps card 0", rec["cards"][1]["review_reason"])
        # overlap of at most half of the shorter span is fine
        one = words[0]
        rest = " ".join(words[1:])
        rec = pfc.check_case(DOMAIN, MODE, CASES[cid], [{"index": 0, "spans": [" ".join(words[:2])], "statement": "s"},
                                                        {"index": 1, "spans": [rest], "statement": "s"}], 1.0)
        self.assertNotIn("overlaps", rec["cards"][0]["review_reason"] or "")
        del one


class TestValidateMeasurement(Base):
    def test_planted_difference(self):
        errors = [("neutral", f"c{i}", 0) for i in range(20)] + [("terse", f"c{i}", 1) for i in range(20)] + \
                 [("verbose", f"c{i}", 0) for i in range(20)]
        m = pv.arm_vs_neutral(errors, B=500)
        self.assertEqual(set(m), {"terse", "verbose"})
        self.assertEqual((m["terse"]["diff"], m["terse"]["ci_lo"], m["terse"]["ci_hi"]), (1.0, 1.0, 1.0))
        self.assertTrue(m["terse"]["ci_excludes_0"])
        self.assertFalse(m["verbose"]["ci_excludes_0"])
        self.assertIsNone(pv.arm_vs_neutral([("terse", "c0", 1)], B=10))

    def test_clustered_by_case(self):
        # all terse errors sit in one case: rows are not independent, so the CI must reach 0
        errors = [("neutral", f"c{i}", 0) for i in range(10)] + [("terse", "c0", 1)] * 100 + \
                 [("terse", f"c{i}", 0) for i in range(1, 10)]
        m = pv.arm_vs_neutral(errors, B=2000)["terse"]
        self.assertAlmostEqual(m["diff"], 100 / 109)
        self.assertEqual(m["ci_lo"], 0.0)
        self.assertFalse(m["ci_excludes_0"])
        self.assertEqual(m["n_cases"], 10)

    def test_asked_column_scored(self):
        self.assertEqual(pv.ATTR_COLS[5:], ["label_a", "asked_a", "label_b", "asked_b"])
        d = os.path.join(self.tmp, "exp")
        os.makedirs(d)
        key, rows = {}, []
        for k in range(8):
            rid = f"A{k:05d}"
            arm = "neutral" if k < 4 else "terse"
            key[rid] = {"task": "attribution", "arm": arm, "domain": DOMAIN, "mode": MODE, "case_id": f"c{k % 4}",
                        "model_label": "full", "model_asked": None if k == 7 else "1"}
            asked = "0" if k in (4, 5) else "1"
            rows.append({"row_id": rid, "question": "q", "answer": "a", "card_item": "i", "card_text": "t",
                         "label_a": "full", "asked_a": asked, "label_b": "", "asked_b": ""})
        pev.write_json(os.path.join(d, pv.KEY_FILE), key)
        with open(os.path.join(d, pv.ATTR_FILE), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=pv.ATTR_COLS)
            w.writeheader()
            w.writerows(rows)
        res = pv.score(d, B=200)
        a = res["attribution_asked"]
        self.assertEqual(a["n_labelled"], 7)  # the unparsed ask label is not scored
        self.assertEqual(a["per_arm"]["neutral"]["accuracy"], 1.0)
        self.assertAlmostEqual(a["per_arm"]["terse"]["accuracy"], 1 / 3)
        self.assertAlmostEqual(a["arm_vs_neutral"]["terse"]["diff"], 2 / 3)
        self.assertEqual(res["attribution"]["per_arm"]["terse"]["accuracy"], 1.0)
        with open(os.path.join(d, "validation_report.md")) as f:
            self.assertIn("## Ask judge", f.read())

    def test_export_has_asked(self):
        FakeLLM.ask = staticmethod(lambda cards, q: "garbage" if "Anything" in q else
                                   fj({"asked": [cards[0]["card_id"]]}))
        c = Case(self.tmp, TWO_TURNS)
        c.eval()
        with open(os.path.join(c.run_dir, "manifest.json"), "w") as f:
            json.dump({"cards": {f"{DOMAIN}/{MODE}": {"path": c.cards_path,
                                                        "sha256": pu.sha256_file(c.cards_path)}}}, f)
        out = os.path.join(self.tmp, "export")
        pv.export(c.run_dir, 10, 0, out)
        key = pev.read_json(os.path.join(out, pv.KEY_FILE))
        attr = {(k["turn"], k["card_index"]): k for k in key.values() if k["task"] == "attribution"}
        self.assertEqual(attr[(1, 0)]["model_asked"], "1")
        self.assertEqual(attr[(1, 1)]["model_asked"], "0")
        self.assertIsNone(attr[(2, 0)]["model_asked"])
        with open(os.path.join(out, pv.ATTR_FILE), newline="") as f:
            self.assertEqual(next(csv.reader(f)), pv.ATTR_COLS)


class TestEscapes(Base):
    def test_parse_final_json_keeps_valid_escapes(self):
        # needs the persona_users.parse_final_json fix (json.loads first, backslash repair only on failure)
        text = '<FINAL_JSON>{"evidence": "line one\\nline two", "v": "3 \\u00d7 4", "q": "say \\"x\\""}</FINAL_JSON>'
        self.assertEqual(pu.parse_final_json(text), {"evidence": "line one\nline two", "v": "3 × 4", "q": 'say "x"'})

    def test_non_ascii_round_trip(self):
        c = Case(self.tmp, TWO_TURNS)
        r = c.eval()
        conv = [p for k, p in FakeLLM.calls if k == "conv"][0]
        self.assertIn("FACT 0 × value", conv)  # ensure_ascii=False in the card view
        self.assertEqual(r["items"][0]["delivered"], True)  # evidence came back as × and still matched
        self.assertEqual(r["dropped"]["attribution_evidence"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
