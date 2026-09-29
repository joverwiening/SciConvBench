# Persona trait study on SciConvBench — harness design

Question: which personality traits of the simulated scientist (the user who holds the complete spec) change how
much of the missing information a fixed agent recovers. One trait per arm, every arm on the same cases, so effects
are compared within case.

The design below is the one used in the full1 run (62 foam disambiguation cases × 35 arms, 2026-09-28). Its
analysis rules, fixed before the run, are in `PERSONA_PREREG.md`; the scripts behind the report are in
`persona/analysis/full1/`. The code also still contains the arms of an earlier design (style and policy traits such
as `terse`, `reluctant`; `persona/personas/`), which that run did not use.

## Arms

| arm | scientist | purpose |
|---|---|---|
| `bf_<high/low>_<A/C/E/N/O>`, `_w2`, `_w3` | one Big Five pole, the other four average; 3 independent wordings | the manipulation |
| `bf_neutral`, `bf_neutral_b` … `_f` | average on all five traits; 6 wordings of the same content | control; the spread over wordings measures rewording noise |
| `oracle` | scripted: every reply = all fact-card texts | ceiling: all information delivered |
| `mute` | scripted: every reply = "Make a reasonable assumption based on best practices." (the paper user's own fallback line) | floor: no information delivered |

A = agreeableness, C = conscientiousness, E = extraversion, N = neuroticism, O = openness. `bf_low_N` (calm) exists
but was dropped from full1: it failed the visibility check in two pilots. The `_noex` files are an ablation without
example exchanges, also not run.

Trait effect = mean over the pole's 3 wordings − mean over the 6 neutral wordings, paired by case; the standard
error includes the between-wording variance (`persona_analysis.py --family bigfive`).

## Simulated scientist (`persona_users.BigFiveUser`)

System prompt, in order: shared identity, persona sheet, shared situation, persona appraisal (one sentence of
feelings about the situation), shared notes intro, the COMPLETE requirement as the scientist's notes, the INCOMPLETE
requirement as "the request you sent", shared rules, examples intro. Shared text is `persona/bigfive/shared.json`,
byte-identical in every arm; persona text is `persona/bigfive/personas/*.json`.
- Differences from the paper's user: the incomplete text is framed as the scientist's own message, and nothing states
  outright that the agent lacks facts (the notes intro only calls the request a shorter version). No "be concise", "do not provide information not asked for" or "provide concrete values",
  and no canned "Make a reasonable assumption" reply. Kept: never state a value that is not in the notes. Added:
  never change a value in the notes, do not propose values of your own, plain chat text.
- The request is shown because without it the disagreeable scientist accused the agent of ignoring facts that were
  never in the request.
- Persona text carries personality, facts about the person's life and feelings about them, never rules on sharing
  or claims about what the person knows (lint in `test_bigfive.py`). Some texts describe how the person writes:
  `bf_low_E` (reserved; sheet and reminder ask for clipped replies), `bf_low_C` (typos, half-finished sentences) and
  the unused `bf_low_N` (easygoing tone, no apologies).
- Each persona's 3 example exchanges (unrelated tasks) are sent as earlier chat turns; they never reach the
  clarification log. History goes as real chat turns (agent = user role, scientist = assistant role). The persona's
  one-line reminder plus a shared truth reminder is appended to the current message only.
- If a reply claims an earlier disclosure and carries a number from the notes that has not been sent yet, it is
  regenerated once with a note (`truth_retries` in the record, draft kept in `user_calls.jsonl`).
- Every scientist call is logged raw to `user_calls.jsonl`; the system prompt to `user_system_prompt.txt`.

Visibility check (`persona_pairwise.py`): a judge reads the trait chat and the `bf_neutral` chat of the same case
(scientist lines only matter) and picks which scientist is more like the pole's description; `--probe opposite` asks
for the opposite pole on the same pairs, so a judge that picks the more colourful chat fails. The single-chat trait
check in `persona_eval.py` (`--trait_check bigfive`) exists but full1 ran with `--trait_check none`.

When the agent reaches the turn cap, `run_dialogue(final_fix=True)` gives it the scientist's last reply and asks for
the final spec, with one retry if the output is shorter than 300 characters or ends in a question (`final_attempts`,
`final_valid` in the record). The upstream default path is unchanged.

Models: GLM-5 on Bedrock (`bedrock_glm5`) for agent, scientist, judges and aux. No released paper runs use GLM-5.
Runner, eval, pairwise judge and card builder take `--budget_usd` and stop scheduling work once the priced spend
reaches it (calls in flight still finish, so each stage can overshoot slightly).

## Fact cards (persona/fact_cards/<domain>/<mode>_<domain>.cards.json)

One card per missing entity (disambiguation) or per planted inconsistency. Built once by
`persona_fact_cards.py` with an LLM, then checked deterministically:
- disambiguation card = verbatim span(s) of the complete requirement that specify the entity. Every span
  must be a substring of the complete requirement (whitespace-normalised) or the card is rejected.
- inconsistency card = one-sentence correction grounded in the correct prompt, plus the quoted span of
  the correct prompt it rests on when one exists (often the fix is a deletion, so none).
- coverage: word-level diff (incomplete → complete). Inserted/replaced tokens of the complete text not
  covered by any span are recorded per case; high uncovered share → `needs_review`.
Cards are used by the oracle arm and by the attribution judge. They are never shown to the simulated scientist.

## Measurement (all post hoc, blind to arm, identical across arms)

Native metrics, unchanged: the repo's judges (FR judge, chat judge, intent judge) and
`make_paper_artifacts._score_disambiguation / _score_inconsistency` → FR, grounded, IC, MC, CR, CP,
CGRR, SRR. Judge model = `SCICONV_PROFILE` (paper: Gemini 2.5 Pro).

Per-item metrics (new), from one attribution call per dialogue turn: given the agent question, the user
answer and all cards of the case, label (a) which cards the question asks about, (b) for each card
whether the answer conveys it: full / partial / none, with an evidence quote that must be a substring
of the answer (otherwise downgraded to none).
- asked_i: some question targets card i
- delivered_i: some answer conveys card i fully
- recovered_i: native FR judge lists entity i as recovered (disamb) / breakdown resolved (incons)
- grounded_i = recovered_i ∧ delivered_i; silent_i = recovered_i ∧ ¬delivered_i
Upstream disambiguation "grounded" is len(questions) ≥ #entities (question count, no matching to
entities); per-item grounding replaces that and is reported next to it.

Decomposition of a trait effect on recovery: Δ delivered rate vs Δ P(recovered | delivered).

Trait check of the earlier design (`--trait_check per_turn`): one call per user answer, multi-label over ALL traits (judge not told the
arm): present 0/1 + evidence quote. Report on-target rate, max off-target rate, neutral base rates,
and on-target rate by turn index (fading).

Judge validity tooling: `persona_validate.py` exports a blind hand-label sample stratified by arm (style can
change how detectable a fact is) and scores the attribution and trait judges against it.

## Protocol invariants

- Dialogue loop is the upstream loop (`test_runner.run_dialogue`, shared by `run_single_case`), so all
  arms and the upstream entry point run the same protocol.
- Agent, user, judge and aux (attribution/trait) models are separate config instances; the `Config`
  class is never mutated by the persona runner.
- Output tree keeps the native per-case files so the upstream judges and aggregators run unchanged:
  `<out>/<arm>/rep<k>/<domain>/<AgentLabel>_<Mode>/<case_id>/`.
- `manifest.json` per run: git commit + dirty flag, argv, model ids and temperatures per role, sha256 of
  agent system prompt, user prompt template, each persona file and the cards file, price table.
- Temperatures default to the paper's profiles (0.0); a flag sets them per role. Repeats are
  independent re-runs of the same (arm, case).

## Files

- `persona_fact_cards.py` build + verify cards
- `persona_users.py` persona rendering, `OracleUser`, `MuteUser`
- `persona_runner.py` arms × reps × cases, parallel, resumable, per-role token/cost log
- `persona_pairwise.py` visibility check: trait chat vs neutral chat on the same case, target and opposite probe
- `persona_eval.py` native judges + attribution + trait check → per-case `persona_eval.json`, run-level `results.jsonl`
- `persona_analysis.py` paired effects with case-bootstrap CIs, gap-normalised effects, decomposition, manipulation-check tables, case screen
- `persona_validate.py` judge validity: blind hand-label export, card audit sheet, scorer
- `persona/analysis/full1/` scripts behind the full1 report (reply content, first-answer pairs, decoding, figures); `run_all.sh`
- `test_persona_harness.py` offline end-to-end test with fake models (no API calls); the other `test_*persona*`, `test_bigfive.py`,
  `test_pairwise.py`, `test_pole_effects.py`, `test_harness_fixes.py` are offline too

## Implementation notes (where the code pins down or departs from the above)

- Cards: `index` is 0-based and follows the dataset item order; `card_id = <domain>/<mode>/<case_id>#<index>`.
  Uncovered share only sets `needs_review` for disambiguation (`--max_uncovered_share`, default 0.5). For
  inconsistency it is computed and stored but ignored, because spans are optional there. Re-invoking the builder
  re-queries every case that is not `ok`; `--verify_only` re-runs the checks on the stored LLM output with no calls.
- Per-item flags: `partial_i` = some answer conveys card i partially and none conveys it fully. An attribution
  entry whose evidence is not a substring of the answer is dropped and counted. A trait marked present with
  unverifiable evidence counts as absent. The attribution prompt counts confirming a value the question proposed
  as conveying it.
- `recovered_i` is the raw FR-judge output; the IC gate applies only to the native case metrics (FR, grounded,
  CGRR, SRR), as in `make_paper_artifacts.build_case_df`. The prompt-length cap and the shallow-inconsistency
  drop are not applied: rows carry `native.over_length_cap`, and case selection is done by `--cases`.
- Upstream disambiguation "asked" (so CR and grounded) is `total_questions_asked_in_chat`, the count of every
  question, not only entity-targeted ones. CR is not clamped, so it can exceed 1.
- The native chat judges read `partial_score` from the FR judge's file, so the order is FR → chat → intent.
- `LLMUserInterface` swallows its own API failure and replies "I am unable to answer that at this time.". The
  persona runner marks such a job `error` (the job reruns on the next invocation). `run_single_case` keeps the
  upstream behaviour.
- `test_runner.run_dialogue` returns `DialogueResult(final_spec, turns, completed_by_agent, cap_hit, error)`.
  The `error` field lets `run_single_case` keep the upstream turn count and success flag when an exception occurs.
- One out dir may hold several domains or modes. The manifest's `cards` entry is keyed by `<domain>/<mode>`, and a
  later invocation adds its key. Role models, max_turns, baseline, the agent prompt, the user template and the
  persona hashes must match, or the runner refuses to continue (`--force` overrides). The git dirty flag ignores
  untracked files.
- `persona_eval.py` writes `eval_manifest.json` (judge/aux roles, prompt hashes, OpenRouter prices for the
  judge/aux models) because the run manifest predates the choice of judge. Judge, attribution and trait outputs
  are cached per case. They are reused when the model (and, for aux, the prompt, cards and persona hashes) are
  unchanged. The judge classes are instantiated as-is and their `.llm` is replaced with a thin wrapper around
  `LLMService(judge role)` that keeps raw responses so parse failures can be counted.
- API token counters come only from the langchain providers (openrouter, openai, google, bedrock, ollama).
  `vertex_gemini`, `bedrock_boto3`, `claude_sdk` and `codex` fall back to the tiktoken estimates, which miss
  reasoning tokens. API usage is counted on every response, including empty ones that trigger a retry, because
  those are billed.
