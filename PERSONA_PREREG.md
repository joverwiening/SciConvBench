# Persona trait study — analysis rules of the full1 run

The rules below were fixed before the full1 run (file last changed 2026-09-28 15:14 EDT; the run started 16:11 EDT) and are copied verbatim. `4c4b51d03` is the local id of the commit that ran; on this branch it is the fact-card commit, whose tree is identical. `runs/full1/cases_foam_full1.txt` is `persona/analysis/full1/cases.txt`. test2, test3 and repilot are earlier pilot runs on other cases. The earlier draft of this file, for the first design (style and policy traits), is in the harness commit of this branch.

---

## full1 — plan and analysis rules, fixed before the run (2026-09-28)

Code: SciConvBench branch ein/persona-harness @ 4c4b51d03 (local). The sbatch script refuses any other or dirty commit.

## Design
- Task: SciConvBench foam, disambiguation mode (facts deleted from the request). 62 ok-card cases, none used in any
  earlier run (runs/full1/cases_foam_full1.txt). 1 run per case-arm.
- GLM-5 on Bedrock in every role (agent, simulated scientist, judges), thinking off.
- Arms (35): 9 Big Five poles × 3 wordings, 6 neutral wordings (bf_neutral, _b.._f), mute (floor), oracle (ceiling).
- Calm (low N) is out: it failed the pairwise gate in repilot (13/24 vs opposite 11/24) and after the rework
  (35/48 vs 19/48). The personas and wordings were tuned on test2/test3/repilot cases only.

## Primary analysis
- Metric: recovered_judge, the share of deleted facts in the agent's final spec, graded by the benchmark's judge.
- Per pole: Δ = mean over its 3 wordings − mean over the 6 neutral wordings, paired by case.
  SE² = V_case + τ̂²(1/3 + 1/6), τ̂² by method of moments; p = max(normal p, two-stage bootstrap p).
- Holm over the 9 poles at α = 0.05.
- A pole counts only if Holm p < 0.05 AND it passes the pairwise gate: vs bf_neutral on the same case, target probe
  hit ≥ 0.75 with Clopper-Pearson lower bound > 0.5, opposite probe hit ≤ 0.25 with upper bound < 0.5, on the pooled
  row and on every wording.

## Secondary, descriptive only
- Fully resolved, delivered, asked_delivered, unasked_delivered, number of questions, cap hits.
- Sensitivity: primary Δ with rows whose final spec is still invalid after the retry (final_valid false) removed.

## Known limits
- One domain, one model family in every role, 1 run per case-arm.
- Reserved (low E) is the only pole whose sheet and reminder name a behaviour (clipped fragments); an effect there may
  be partly the brevity instruction rather than the trait.
- Anxious (high N) on the repilot cases: +0.09, down from +0.24 before the plain-chat rules; full1 is the test.
