import os
import sys
import csv
import json
import shutil
import tempfile
import subprocess
import importlib.util
import unittest

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

import persona_analysis as pan
from test_bigfive import bf_rows

NEW_KEYS = {"pole_effects", "pole_wordings", "tau", "pole_gate_found"}


def row(arm, c, y):
    items = [{"index": i, "asked": True, "delivered": i < 2, "partial": False, "unasked_delivered": False,
              "unasked_partial": False, "recovered": i < 2, "grounded": False, "silent": False} for i in range(4)]
    rates = {k: sum(it[k] for it in items) / len(items) for k in pan.ITEM_METRICS}
    rates[pan.PRIMARY] = float(y)
    return {"arm": arm, "rep": 1, "domain": "pde", "mode": "disambiguation", "case_id": f"c{c}",
            "native": {"FR": 0, "CGRR": 0, "SRR": 0, "IC": 1}, "items": items, "item_rates": rates,
            "turns": 3, "questions": 2, "cap_hit": False, "deferred": 0,
            "trait_present_share": None, "trait_by_turn": None}


def synth(arms, n_cases, word, pole=None, mu=0.5, case_sd=0.2, noise_sd=0.1, seed=0):
    """y[a, c] = mu + pole[a] + word[a] + case_c + eps; arms: arm -> family (pole id or 'neutral')."""
    rng = np.random.default_rng(seed)
    case = rng.normal(0, case_sd, n_cases)
    pole = pole or {}
    return [row(a, c, mu + pole.get(f, 0.0) + word[a] + case[c] + rng.normal(0, noise_sd))
            for a, f in arms.items() for c in range(n_cases)]


def design(n_poles, k, k_neutral):
    arms = {f"n{j}": "neutral" for j in range(k_neutral)}
    arms.update({f"p{i}_w{j}": f"pole{i}" for i in range(n_poles) for j in range(k)})
    return arms


def split(arms):
    return [a for a, f in arms.items() if f == "neutral"], {a: f for a, f in arms.items() if f != "neutral"}


def pooled(taus, m=pan.PRIMARY):
    return next(t for t in taus if t["metric"] == m and t["family"] == "POOLED")


class TestTau(unittest.TestCase):
    def run_tau(self, word_sd):
        arms = design(20, 5, 6)
        rng = np.random.default_rng(1)
        word = {a: rng.normal(0, word_sd) if word_sd else 0.0 for a in arms}
        neutral, targets = split(arms)
        _, _, taus = pan.pole_effects(synth(arms, 100, word), neutral, targets, [pan.PRIMARY], b=200)
        fams = [t for t in taus if t["family"] != "POOLED"]
        self.assertEqual(len(fams), 21)
        self.assertEqual({(t["k"], t["n"]) for t in fams}, {(5, 100), (6, 100)})
        return pooled(taus)

    def test_recovers_planted_sd(self):
        t = self.run_tau(0.1)
        self.assertTrue(t["tau_estimable"])
        self.assertAlmostEqual(t["tau"], 0.1, delta=0.025)

    def test_zero_when_none_planted(self):
        t = self.run_tau(0.0)
        self.assertLess(t["tau"], 0.02)

    def test_moments_by_hand(self):
        Y = np.array([[1.0, 2.0, 4.0], [2.0, 2.5, 3.0]])
        s = pan.wording_family_stats(Y)
        r = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + Y.mean()
        self.assertAlmostEqual(s["s2"], np.var([7 / 3, 2.5], ddof=1))
        self.assertAlmostEqual(s["sigma2"], (r ** 2).sum() / 2)
        self.assertAlmostEqual(pan.pooled_tau2([s]), max(0.0, s["s2"] - s["sigma2"] / 3))
        self.assertIsNone(pan.pooled_tau2([{"k": 1, "n": 3, "s2": 0, "sigma2": 0}]))

    def test_not_estimable(self):
        arms = {"n0": "neutral", "a": "pole0", "b": "pole1"}
        neutral, targets = split(arms)
        rows = synth(arms, 30, {"n0": 0.0, "a": 0.2, "b": 0.0})
        poles, _, taus = pan.pole_effects(rows, neutral, targets, [pan.PRIMARY], b=500)
        self.assertEqual([t["family"] for t in taus], ["POOLED"])
        self.assertFalse(pooled(taus)["tau_estimable"])
        for p in poles:
            self.assertFalse(p["tau_estimable"])
            self.assertAlmostEqual(p["se"], np.sqrt(p["v_case"]))


class TestPoleEffect(unittest.TestCase):
    def setUp(self):
        self.arms = {"n0": "neutral", "n1": "neutral", "n2": "neutral",
                     "hx1": "high_x", "hx2": "high_x", "hx3": "high_x",
                     "ly1": "low_y", "ly2": "low_y", "ly3": "low_y",
                     "lz1": "low_z"}
        self.word = {"n0": 0.01, "n1": -0.01, "n2": 0.0, "hx1": 0.02, "hx2": -0.02, "hx3": 0.0,
                     "ly1": 0.05, "ly2": -0.05, "ly3": 0.01, "lz1": 0.0}
        self.rows = synth(self.arms, 60, self.word, pole={"high_x": 0.2, "low_y": 0.0, "low_z": 0.1},
                          noise_sd=0.03, seed=3)
        self.neutral, self.targets = split(self.arms)
        self.poles, self.words, self.taus = pan.pole_effects(self.rows, self.neutral, self.targets, [pan.PRIMARY],
                                                             b=2000)
        self.by = {p["target"]: p for p in self.poles}

    def test_delta_and_sign(self):
        hx = self.by["high_x"]
        pcm = {a: pan.per_case_mean(self.rows, a, pan.PRIMARY) for a in self.arms}
        want = np.mean([np.mean(list(pcm[a].values())) for a in ("hx1", "hx2", "hx3")]) \
            - np.mean([np.mean(list(pcm[a].values())) for a in self.neutral])
        self.assertAlmostEqual(hx["effect"], want)
        self.assertAlmostEqual(hx["effect"], 0.2, delta=0.03)
        self.assertEqual((hx["k_wordings"], hx["k_neutral"], hx["n_cases"]), (3, 3, 60))
        self.assertTrue(hx["ci_lo"] < 0.2 < hx["ci_hi"])
        self.assertTrue(hx["boot2_lo"] < hx["effect"] < hx["boot2_hi"])
        self.assertTrue(hx["sign_consistent"])
        tau2 = pooled(self.taus)["tau2"]
        self.assertGreater(tau2, 0)
        self.assertAlmostEqual(hx["se"] ** 2, hx["v_case"] + tau2 * (1 / 3 + 1 / 3))
        self.assertFalse(self.by["low_y"]["sign_consistent"])
        w = {x["arm"]: x for x in self.words}
        self.assertEqual(set(w), set(self.targets))
        self.assertAlmostEqual(np.mean([w[a]["effect"] for a in ("hx1", "hx2", "hx3")]), hx["effect"])
        self.assertTrue(all(w[a]["same_sign_as_pole"] for a in ("hx1", "hx2", "hx3")))
        self.assertTrue(w["hx1"]["ci_lo"] < w["hx1"]["effect"] < w["hx1"]["ci_hi"])

    def test_holm_over_poles(self):
        self.assertEqual(sorted(self.by), ["high_x", "low_y", "low_z"])
        ps = [p["p"] for p in self.poles]
        self.assertEqual([p["p_holm"] for p in self.poles], pan.holm(ps))
        self.assertLess(self.by["high_x"]["p_holm"], 0.05)
        self.assertGreater(self.by["low_y"]["p_holm"], 0.05)
        # no gate file → nothing counts
        self.assertTrue(all(p["gate"] is None and p["counts"] is False for p in self.poles))
        sec = pan.pole_effects(self.rows, self.neutral, self.targets, [pan.PRIMARY, "FR"], b=200)[0]
        self.assertTrue(all(p["counts"] is None for p in sec if p["metric"] == "FR"))


def noisy_bf_rows(n_cases=40, seed=5):
    arms = {"bf_neutral": "neutral", "bf_neutral_b": "neutral", "bf_low_A": "low_agreeableness",
            "bf_high_N": "high_neuroticism", "bf_low_C": "low_conscientiousness"}
    pole = {"low_agreeableness": -0.3, "high_neuroticism": 0.3, "low_conscientiousness": 0.0}
    return synth(arms, n_cases, {a: 0.0 for a in arms}, pole=pole, noise_sd=0.05, seed=seed)


class TestGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def write_gate(self, recs):
        with open(os.path.join(self.tmp, "pairwise_gate.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["target", "arm", "n", "hits", "hit_rate", "ci_lo", "ci_hi", "pass"])
            w.writeheader()
            for t, a, p in recs:
                w.writerow({"target": t, "arm": a, "n": 10, "hits": 9, "hit_rate": 0.9, "ci_lo": 0.6,
                            "ci_hi": 1.0, "pass": p})

    def test_gate_and_counts(self):
        self.assertIsNone(pan.read_gate(self.tmp))
        self.assertIsNone(pan.read_gate(None))
        self.write_gate([("low_agreeableness", "bf_low_A", False), ("low_agreeableness", "ALL", True),
                         ("high_neuroticism", "bf_high_N", True), ("high_neuroticism", "ALL", False),
                         ("low_conscientiousness", "ALL", True)])
        self.assertEqual(pan.read_gate(self.tmp), {"low_agreeableness": True, "high_neuroticism": False,
                                                   "low_conscientiousness": True})
        res = pan.analyse(noisy_bf_rows(), family="bigfive", run_dir=self.tmp)
        self.assertTrue(res["pole_gate_found"])
        prim = {p["target"]: p for p in res["pole_effects"] if p["metric"] == pan.PRIMARY}
        self.assertEqual({t: p["k_neutral"] for t, p in prim.items()},
                         {"low_agreeableness": 2, "high_neuroticism": 2, "low_conscientiousness": 2})
        self.assertLess(prim["high_neuroticism"]["p_holm"], 0.05)
        self.assertGreater(prim["low_conscientiousness"]["p_holm"], 0.05)
        self.assertEqual({t: (p["gate"], p["counts"]) for t, p in prim.items()},
                         {"low_agreeableness": (True, True), "high_neuroticism": (False, False),
                          "low_conscientiousness": (True, False)})
        self.assertIn("## Pole effects (wording as random effect)", pan.render_markdown(res))

    def test_main_outputs(self):
        results = os.path.join(self.tmp, "results.jsonl")
        rows = [r for r in noisy_bf_rows(20) if r["arm"] != "bf_neutral_b"]
        with open(results, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        out = os.path.join(self.tmp, "an")
        res = pan.main(["--results", results, "--out", out, "--family", "bigfive"])
        for name in ("pole_effects", "pole_wordings", "tau"):
            self.assertTrue(os.path.getsize(os.path.join(out, f"{name}.csv")) > 0, name)
        self.assertFalse(res["pole_gate_found"])
        with open(os.path.join(out, "analysis.md")) as f:
            md = f.read()
        i, j = md.index("## Pole effects (wording as random effect)"), md.index("## Primary: Paired effect")
        self.assertLess(i, j)
        self.assertIn("τ not estimable (need ≥2 wordings in some family)", md)
        self.assertIn("per-arm tables below are descriptive", md)


def head_module():
    try:
        src = subprocess.run(["git", "show", "HEAD:persona_analysis.py"], cwd=REPO, capture_output=True, text=True,
                             check=True).stdout
    except Exception:
        return None
    path = os.path.join(tempfile.mkdtemp(), "persona_analysis_head.py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("persona_analysis_head", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestUnchanged(unittest.TestCase):
    def test_old_keys_unchanged(self):
        rows = bf_rows()
        res = pan.analyse(rows, family="bigfive")
        self.assertTrue(NEW_KEYS <= set(res))
        old = pan.analyse(rows)
        self.assertFalse(NEW_KEYS & set(old))
        head = head_module()
        if head is None or not hasattr(head, "analyse") or hasattr(head, "pole_effects"):
            self.skipTest("HEAD persona_analysis.py unavailable or already has pole_effects")
        ref = head.analyse(rows, family="bigfive")
        self.assertEqual(set(res) - NEW_KEYS, set(ref))
        for k in ref:
            self.assertEqual(res[k], ref[k], k)
        old_head = head.analyse(rows)
        self.assertEqual(old, old_head)
        self.assertEqual(pan.render_markdown(old), head.render_markdown(old_head))


if __name__ == "__main__":
    unittest.main(verbosity=2)
