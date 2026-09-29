# Persona trait study — preregistration (draft, fix before the main run)

Design and code: PERSONA_STUDY.md. This file fixes what is decided before data exists. Anything
changed after the pilot is logged under "Amendments" with date and reason; nothing is changed after
the main run starts.

## Question

Which single traits of the simulated scientist change how much of the missing information a fixed agent
recovers in SciConvBench disambiguation? The ranking picks 2–3 traits for a PaperBench follow-up.

## Outcome

- Primary: item-recovery rate per case = the FR judge's own count a/b (`recovered_judge`), averaged
  over reps. Not the per-item match of the judge's entity list against the dataset items: that match
  misses 4.7% of recovered entities on the released upstream runs (paraphrase).
- Secondary: strict FR, CGRR, SRR (native, IC-gated), per-item recovered, asked, delivered,
  unasked-delivered and unasked-partial rates, turns, cap-hit rate. Holm on secondaries is descriptive.

## Arms and conditions

paper (T = 0, replication only, separate run), neutral, neutral_b, neutral_c (placebos), 10 traits
(style: terse, verbose, lay_vocabulary, jargon, uncertain; policy: reluctant, volunteering, skeptical,
annoyed, busy), oracle, mute. Main run: agent and user temperature 0.7, 2 reps, turn cap 10.

Every LLM persona arm has the same structure: precedence line + neutral text + (placebo or trait
paragraph + style-only examples with placeholders) in the system prompt, and a one-line "stay in
character" reminder appended to every agent message. The placebos add a content-free paragraph and
plain examples, so they control for added prompt length and structure, not only for rewording. busy is
enforced in code: questions 1, 3, 5, … get a fixed "can't look at this now" reply with no LLM call.

Models (OPEN, Apollo to confirm): agent = the agent planned for PaperBench, so the ranking transfers;
user = Claude Sonnet 4.6 (the paper's user); FR/chat/intent judge = Gemini 2.5 Pro (the paper's
judge); attribution/trait judge = a model family different from both user and agent.

## Sample

All disambiguation cases (570) whose fact cards pass the checks, minus cases over the paper's
20,000-character prompt cap (both filters applied in the analysis and counted). Jobs that error are
rerun up to twice (the runner stops at 3 attempts); remaining failures are excluded and reported per
arm with job, done, error and missing counts. Judge parse failures count as missing, never as 0.
Primary estimate on available cases (each trait paired with neutral on the cases both have);
sensitivity estimate on the cases present in every arm with both reps.

Power: σ_d ≈ 0.35 for per-case recovery (paired SD between the paper's two agent prompts in the
released pde/fluids runs at T = 0, range 0.17–0.54; T = 0.7 adds within-case noise, two reps remove
part of it). N = (z + z_β)² σ_d² / δ². For δ = 5 pp at 80 % power: 385 cases unadjusted, 652 with
α = 0.05/10. At 570 cases the Holm-alone MDE is 5.4 pp. The full decision rule (Holm AND beats
placebo) is weaker: simulated power 0.62 at δ = 5 pp, 0.80 at 6 pp, 0.93 at 7 pp (10 traits and 2
placebos sharing neutral, σ_d = 0.35, normal approximation). Planned MDE of the study: 6 pp.

## Analysis

1. Per trait: paired Δ = trait − neutral per case (reps averaged within case first), with a 95 %
   case-bootstrap percentile CI and a two-sided bootstrap p (B = 100,000 for the primary, seed 0).
   Holm across the 10 traits, α = 0.05; the family is fixed at 10 and a missing trait arm is an error.
2. A trait counts as having an effect only if it passes Holm AND the bound of its unadjusted 95 % CI
   nearer to zero, in absolute value, exceeds the larger of the two placebo |Δ| point estimates.
3. Manipulation check: the trait judge must mark the trait present in ≥ 60 % of the arm's answers and
   ≥ 30 pp above the neutral arm. A trait that fails the check is reported, and its Δ is not read as a
   trait effect. busy is not judged: it passes by construction if deferrals happened (≥ 1 per case with
   a question).
4. Judge validity: from hand labels (80 per arm, FR and attribution incl. "asked"), per trait the
   error-rate difference vs neutral with a case-clustered 95 % bootstrap CI. If that CI excludes 0, the
   trait's Δ is reported as the interval [Δ − e_hi, Δ − e_lo] (e = recovery-label error difference)
   and the trait is selected only if that interval excludes 0 and satisfies 2.
5. Selection: rank the traits that satisfy 1–4 by |Δ| (sign ignored on purpose: a harmful trait
   matters as much as a helpful one) and take the top 3, or all of them if fewer than 3 pass. Bootstrap
   P(in top 3) is descriptive.
6. Descriptive only: oracle−mute-normalised effect (ratio of summed Δ to summed gap over cases with
   gap > 0), Δ by gap tercile, delivered vs recovered-given-delivered decomposition, Δ by mute-arm item
   difficulty, questions asked per arm.

## Stages

0. Build cards for all 7 domains; hand-audit 100 cases. Fix the card prompt if more than 5 % of
   audited cards are wrong.
1. Pilot: 30 cases × all arms × 1 rep. Gives cost per case and first manipulation-check rates, and
   supplies the samples for hand validation. Thresholds in 3–4 may be amended here, never later.
2. Replication: paper arm on the 61 pde cases vs the published pde row.
3. Main run as above.
4. Robustness: top traits + neutral + placebos on a second user model and a second agent, ~300 cases.

## Amendments

- 2026-09-27, before the pilot, no persona data exists yet. From a code and design review:
  primary outcome switched to the FR judge's own count (per-item matching misses paraphrased
  entities); traits annoyed and busy added (Holm family 8 → 10) after the interactive-PaperBench
  batch (50 runs) showed agents asking ~3× fewer questions with an annoyed researcher, and both an
  annoyed and a deferring researcher ~0.22 lower on the appendix-setting checks (not significant
  after Holm, p 0.06 and 0.12); examples and a per-turn reminder added to every persona arm;
  placebos restructured to match trait prompt length; power corrected (the old 363/591 used
  σ = 0.34) and stated for the full rule; failure accounting, attempt cap, CI level in rule 2, the
  judge-validity rule (the old max−min spread over ~13 arms at 80 labels could never pass) and the
  selection rule made explicit.
