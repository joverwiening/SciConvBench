# Persona trait study — preregistration (draft, fix before the main run)

Design and code: PERSONA_STUDY.md. This file fixes what is decided before data exists. Anything
changed after the pilot is logged under "Amendments" with date and reason; nothing is changed after
the main run starts.

## Question

Which single traits of the simulated scientist change how much of the missing information a fixed agent
recovers in SciConvBench disambiguation? The ranking picks 2–3 traits for a PaperBench follow-up.

## Outcome

- Primary: item-recovery rate per case (share of missing entities the FR judge marks recovered),
  averaged over reps.
- Secondary: strict FR, CGRR, SRR (native, IC-gated), delivered rate, unasked-delivered rate, turns,
  cap-hit rate.

## Arms and conditions

paper (T = 0, replication only, separate run), neutral, neutral_b, neutral_c (placebos), 8 traits
(terse, verbose, lay_vocabulary, jargon, uncertain; reluctant, volunteering, skeptical), oracle, mute.
Main run: agent and user temperature 0.7, 2 reps, turn cap 10.

Models (OPEN, Apollo to confirm): agent = the agent planned for PaperBench, so the ranking transfers;
user = Claude Sonnet 4.6 (the paper's user); FR/chat/intent judge = Gemini 2.5 Pro (the paper's
judge); attribution/trait judge = a model family different from both user and agent.

## Sample

All disambiguation cases (570) whose fact cards pass the checks, minus cases over the paper's
20,000-character prompt cap. Jobs that error are rerun up to twice; the remaining failures are
excluded and counted per arm.

Power: σ_d ≈ 0.35 for per-case recovery (paired SD between the paper's two agent prompts in the
released pde/fluids runs, range 0.17–0.54). N = (z + z_β)² σ_d² / δ². For δ = 5 pp at 80 % power:
363 cases unadjusted, 591 with α = 0.05/8. With 570 cases, the minimum detectable effect is ≈ 5 pp
after correction. Two reps at T > 0 lower σ_d if part of it is within-case noise.

## Analysis

1. Per trait: paired Δ = trait − neutral per case, with a 95 % case-bootstrap CI (B = 10,000,
   seed 0) and a two-sided bootstrap p. Holm across the 8 traits, α = 0.05.
2. A trait counts as having an effect only if it passes Holm AND its |Δ| CI lower bound exceeds the
   larger placebo |Δ|.
3. Manipulation check: the trait judge must mark the trait present in ≥ 60 % of the arm's answers and
   ≥ 30 pp above the neutral arm. A trait that fails the check is reported, and its Δ is not read as a
   trait effect.
4. Judge validity: from hand labels (80 per arm), the between-arm difference in FR-judge and
   attribution error must be < 2.5 pp. Otherwise Δ for the affected arms is reported as a bound.
5. Selection: rank traits by |Δ|, report bootstrap P(in top 3), and take the top 3 that satisfy 2–3.
6. Descriptive only: oracle−mute-normalised effect (ratio of means), Δ by gap tercile, delivered vs
   recovered-given-delivered decomposition, Δ by mute-arm item difficulty.

## Stages

0. Build cards for all 7 domains; hand-audit 100 cases. Fix the card prompt if more than 5 % of
   audited cards are wrong.
1. Pilot: 30 cases × all arms × 1 rep. Gives cost per case and first manipulation-check rates, and
   supplies the samples for hand validation. Thresholds in 3–4 may be amended here, never later.
2. Replication: paper arm on the 61 pde cases vs the published pde row.
3. Main run as above.
4. Robustness: top traits + neutral + placebos on a second user model and a second agent, ~300 cases.

## Amendments

(none)
