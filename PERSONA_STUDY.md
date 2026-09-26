# Persona trait study on SciConvBench — harness design

Question: which traits of the simulated scientist (the user who holds the complete spec) change how
well a fixed agent recovers the missing / contradictory information. One trait per arm, every arm on
the same cases, so effects are compared within case.

Primary mode: disambiguation (570 cases, 2119 missing entities). Inconsistency is supported by the
same code path.

## Arms

| arm | user | purpose |
|---|---|---|
| `paper` | upstream `LLMUserInterface`, prompt byte-identical to upstream | replication of the paper's setup |
| `neutral` | paper prompt + neutral persona block | control for every trait arm |
| `<trait>` | paper prompt + neutral block + one trait paragraph | the manipulation |
| `oracle` | deterministic: every reply = all fact-card texts | ceiling: all information delivered |
| `mute` | deterministic: every reply = "Make a reasonable assumption based on best practices." (the paper user's own fallback line) | floor: no information delivered |

Trait effect = trait − neutral, paired by case. Reported raw and as a fraction of the case's
oracle − mute gap. Cases where oracle does not beat mute carry no information about the user and are
reported separately (screen).

Traits (persona/personas/*.json, `kind` in {style, policy}):
- style (should not change WHICH facts reach the agent when asked): terse, verbose, lay_vocabulary,
  jargon, uncertain
- policy (may change which facts reach the agent): reluctant, volunteering, skeptical

The persona block states precedence: it overrides the paper prompt's style lines ("Be concise", "Do
not provide information not asked for") where they conflict, and never overrides the facts of the
complete requirement.

## Fact cards (persona/fact_cards/<domain>/<mode>_<domain>.cards.json)

One card per missing entity (disambiguation) or per planted inconsistency. Built once by
`persona_fact_cards.py` with an LLM, then checked deterministically:
- disambiguation card = verbatim span(s) of the complete requirement that specify the entity. Every span
  must be a substring of the complete requirement (whitespace-normalised) or the card is rejected.
- inconsistency card = one-sentence correction grounded in the correct prompt, plus the quoted span of
  the correct prompt it rests on when one exists (often the fix is a deletion, so none).
- coverage: word-level diff (incomplete → complete). Inserted/replaced tokens of the complete text not
  covered by any span are recorded per case; high uncovered share → `needs_review`.
Cards are used by the oracle arm and by the attribution judge. They are never shown to the LLM user,
so the user protocol stays the paper's.

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

Trait manipulation check: one call per user answer, multi-label over ALL traits (judge not told the
arm): present 0/1 + evidence quote. Report on-target rate, max off-target rate, neutral base rates,
and on-target rate by turn index (fading).

Judge validity: attribution and trait judges are validated against a hand-labelled sample stratified
by arm (style can change how detectable a fact is).

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
- `persona_eval.py` native judges + attribution + trait check → per-case `persona_eval.json`, run-level `results.jsonl`
- `persona_analysis.py` paired effects with case-bootstrap CIs, gap-normalised effects, decomposition, manipulation-check tables, case screen
- `test_persona_harness.py` offline end-to-end test with fake models (no API calls)

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
