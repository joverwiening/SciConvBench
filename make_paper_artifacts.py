"""
make_paper_artifacts.py
=======================

Single-file script that regenerates every figure and table in the paper
"CompSciClarBench: Benchmarking LLMs on Multi-Turn Clarification for Task
Formulation in Computational Science" directly from the per-case JSON
judge outputs. No CSV intermediates are required: this script reads the
raw judge JSONs, builds per-case and per-entity DataFrames in memory,
and renders all figures + LaTeX tables.

Inputs (relative to ROOT, one filesystem layout per benchmark archive):
    {ROOT}/{component}/{task.lower()}_{component}.json
        Dataset definition. List of dicts with keys:
            id, incomplete_user_req | inconsistent_prompt,
            missing_entities | inconsistencies, ...

    {ROOT}/{component}/{MODEL}_{Task}/{case_id}/
        llm_judge_{task.lower()}_score.json       # FRR
        llm_judge_{task.lower()}_chat_score.json  # asked / total / MC
        disambiguation_coverage_score.json        # disambig only
        llm_judge_intent_score.json               # IC

    {ROOT}/{component}_baseline/{MODEL}_{Task}/{case_id}/...
        Unguided runs (used only for the guided-vs-unguided figure).

Outputs (under OUT_DIR):
    figs/fig_dataset_distribution.{png,pdf}            -- Figure 3
    figs/fig_main_domain_outcomes.{png,pdf}            -- Figure 4
    figs/fig_ontology_case_level.{png,pdf}             -- Figure 5
    figs/fig_pareto_analysis.{png,pdf}                 -- Figure 6
    figs/fig_general_numeric_outcomes.{png,pdf}        -- Figure 7  (appendix)
    figs/fig_tool_use_outcomes.{png,pdf}               -- Figure 8  (appendix)
    figs/fig_component_specific_ontology.{png,pdf}     -- Appendix per-issue ontology
    figs/outcomes_micro_avg.{png,pdf}                  -- Appendix per-domain micro-avg
    figs/fig_appendix_guided_vs_unguided.{png,pdf}     -- Appendix guided vs. unguided

    tables/table_main_macro.tex                        -- Main domain × model summary
    tables/table_clamber_vs_sciconv.tex                -- Table 1
    tables/table_appendix_disambiguation.tex           -- Appendix per-domain (Disambiguation)
    tables/table_appendix_inconsistency.tex            -- Appendix per-domain (Inconsistency)
    tables/table_robustness_summary.tex                -- Table 2 spread summary
    tables/table_judge_agreement.tex                   -- Table 2(a) judge ablation
    tables/table_prompt_paraphrase.tex                 -- Table 2(b) prompt ablation
    tables/table_simulator_ablation.tex                -- Table 2(c) simulator ablation

Ablation result-JSON inputs (read by the four tables above):
    {ROOT}/judge_ablation/{openai_gpt5,bedrock_4_6_sonnet,human_annotation_form_annotated}_results.json
    {ROOT}/prompt_ablation/OPTION_{1,2}/paraphrased_option_{1,2}_results.json
    {ROOT}/user_llm_ablation/{GEMINI_PRO,GPT_5_2,SONNET_4_6}/<simulator>_results.json
    Each result JSON has shape {"cases": [{... "llm_scores": {FR/CGR/CP/CR/DR/IC/MC: {value, reason}}}, ...]}.

Usage:
    python make_paper_artifacts.py [--root /path/to/ConvAgent] \
                                   [--out figs] [--tables tables] \
                                   [--only fig_3,fig_4,table_1]

Notes (filters applied throughout — match the paper exactly):
    1. Prompt-length cap: drop cases whose `incomplete_user_req` (or
       `inconsistent_prompt`) exceeds 20,000 characters.
    2. Shallow-inconsistency drop: drop Inconsistency cases on which all
       five guided models score CGRR=1 (after the SRR->CGRR correction
       below). Length cap and shallow drop together define the kept pool.
    3. Intent-capture (IC) gate: when `llm_judge_intent_score.json`
       reports `score == "0/1"` (UNFAITHFUL), the case's `recovered` and
       `grounded` flags are forced to 0 before any aggregation. This is
       what the paper calls intent gating.
    4. SRR->CGRR demotion correction: when the chat judge would mark a
       case silent (asked < total), we re-check whether the missing
       entity (or its numeric value) actually appears in the original
       prompt, in any assistant Q-line, or in any user A-line of
       `conversation_log.txt`. If it does, the case is NOT a genuine
       silent default and we promote SRR=1 -> CGRR=1. The same
       correction is applied per-issue when building the entity table
       used in Figure 5 + the per-issue appendix figure.
    5. Units allowlist: entities matching `^Units:` (e.g. "Units: N-mm-MPa")
       are exempt from the silent check — silently picking a standard
       engineering unit convention is not counted as a silent failure.

All four ablation tables (robustness summary, judge agreement, prompt
paraphrase, simulator ablation) are computed directly from the precomputed
result JSONs listed above; no API calls or LLM-judge re-runs are needed
to regenerate them. If you re-run the underlying ablations and replace
the result JSONs, the tables update automatically on the next invocation.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent  # ConvAgent/
DEFAULT_OUT = HERE / "figs"
DEFAULT_TABLES = HERE / "tables"

MODELS = [
    "CLAUDE_SONNET_4.6",
    "GEMINI_2_5_PRO",
    "GEMINI_2_5_FLASH",
    "GPT_5_2",
    "OLLAMA",
]
MODEL_LABEL = {
    "CLAUDE_SONNET_4.6": "Claude Sonnet 4.6",
    "GEMINI_2_5_PRO":    "Gemini 2.5 Pro",
    "GEMINI_2_5_FLASH":  "Gemini 2.5 Flash",
    "GPT_5_2":           "GPT-5.2",
    "OLLAMA":            "GPT-OSS-120B",
}
MODEL_COLOR = {
    "CLAUDE_SONNET_4.6": "#1f77b4",
    "GEMINI_2_5_PRO":    "#d62728",
    "GEMINI_2_5_FLASH":  "#ff7f0e",
    "GPT_5_2":           "#2ca02c",
    "OLLAMA":            "#e377c2",
}

COMPONENTS = ["fluids", "foam", "solMech",
              "solToolUse", "matSci", "matToolUse", "pde"]
COMPONENT_TO_DOMAIN = {
    "fluids":     "Fluid mechanics",
    "foam":       "Fluid mechanics",
    "solMech":    "Solid mechanics",
    "solToolUse": "Solid mechanics",
    "matSci":     "Materials science",
    "matToolUse": "Materials science",
    "pde":        "PDEs",
}
DOMAIN_ORDER = ["Fluid mechanics",
                "Solid mechanics", "Materials science", "PDEs"]
DOMAIN_COLOUR = {
    "Fluid mechanics":   "#2E6FB8",
    "Solid mechanics":   "#D8701A",
    "Materials science": "#1FA882",
    "PDEs":              "#C86FB0",
}

TASKS = ["Disambiguation", "Inconsistency"]
PROMPT_FIELD = {
    "Disambiguation": "incomplete_user_req",
    "Inconsistency":  "inconsistent_prompt",
}
ITEMS_FIELD = {
    "Disambiguation": "missing_entities",
    "Inconsistency":  "inconsistencies",
}

MAX_PROMPT_CHARS = 20_000  # length cap (paper Section 3 filter)

# Ontology components used in Figure 5 and the appendix per-issue plot.
ONT_ORDER = ["G", "M", "B", "D", "N", "U"]
ONT_NAME = {
    "G": "Governing\nphysics\nand regime",
    "M": "Material and\nphysical props",
    "B": "Boundary / IC",
    "D": "Geometry\nand domain",
    "N": "Numerics\nand solver",
    "U": "Units and\nmagnitude",
}

# ---------------------------------------------------------------------------
# 2. JSON / FILESYSTEM HELPERS
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict | list:
    with open(path, "r") as f:
        return json.load(f)


def _maybe_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return _load_json(path)
    except (json.JSONDecodeError, OSError):
        return None


def dataset_path(root: Path, component: str, task: str) -> Path:
    return root / component / f"{task.lower()}_{component}.json"


def case_dir(root: Path, component: str, model: str, task: str, case_id: str,
             baseline: bool = False) -> Path:
    parent = f"{component}_baseline" if baseline else component
    return root / parent / f"{model}_{task}" / case_id


# ---------------------------------------------------------------------------
# 3a. SILENT-RESOLUTION CORRECTION (SRR -> CGRR demotion, paper Section 3)
# ---------------------------------------------------------------------------
# When the chat-based judge reports that fewer questions were asked than
# there are missing entities (asked < total), the case is marked as a
# silent resolution by default. The paper applies a heuristic correction:
# if the missing entity (or its numeric value) actually appears in the
# original incomplete prompt, in any assistant question line, or in any
# user answer line, then the model did NOT silently default — it had
# the information visible — and we demote SRR=1 -> CGRR=1.
# Mirrors `_has_genuine_silent` / `_has_genuine_silent_inconsistency` in
# `paper/sciconvbench_make_plots_all_domains.py`.

_SRR_STOPWORDS = {"the", "and", "for", "with", "from", "into",
                  "that", "this", "use", "are", "per"}
_VALUE_NUM_RE = re.compile(r"\d+(?:[\.,]\d+)?(?:e[+-]?\d+)?")


def _entity_label_tokens(ent: str) -> list[str]:
    label = ent.split(":", 1)[0] if ":" in ent else ent
    toks = [t.lower()
            for t in re.split(r"[\s_\-/,;()]+", label) if len(t) >= 3]
    return [t for t in toks if t not in _SRR_STOPWORDS]


def _entity_value_numeric_tokens(ent: str) -> list[str]:
    val = ent.split(":", 1)[1] if ":" in ent else ""
    return [m.group(0) for m in _VALUE_NUM_RE.finditer(val) if len(m.group(0)) >= 2]


def _is_units_convention(ent: str) -> bool:
    """Allowlist: 'Units: N-mm-MPa' style defaults are not silent failures."""
    return ent.strip().lower().startswith("units:")


def _conv_q_text(conv_path: Path) -> str:
    if not conv_path.exists():
        return ""
    return "\n".join(l[2:] for l in conv_path.read_text().splitlines()
                     if l.startswith("Q:")).lower()


def _conv_a_text(conv_path: Path) -> str:
    if not conv_path.exists():
        return ""
    out, in_a = [], False
    for line in conv_path.read_text().splitlines():
        if line.startswith("A:"):
            in_a = True
            out.append(line[2:])
        elif line.startswith("Q:") or line.startswith("==="):
            in_a = False
        elif in_a:
            out.append(line)
    return "\n".join(out).lower()


def _has_genuine_silent_disambig(case_path: Path, ds_row: dict) -> bool:
    """True iff at least one missing entity is neither in the prompt, in any
    Q-line, nor in any A-line — i.e., the model truly silently defaulted."""
    missing = ds_row.get("missing_entities", []) or []
    if not missing:
        return False
    inc = ds_row.get("incomplete_user_req", "").lower()
    qtxt = _conv_q_text(case_path / "conversation_log.txt")
    atxt = _conv_a_text(case_path / "conversation_log.txt")
    for ent in missing:
        if _is_units_convention(ent):
            continue
        toks = _entity_label_tokens(ent)
        if not toks:
            continue
        if any(t in inc for t in toks):
            continue
        if any(t in qtxt for t in toks):
            continue
        if any(t in atxt for t in toks):
            continue
        nums = _entity_value_numeric_tokens(ent)
        if nums and any(n in atxt for n in nums):
            continue
        return True
    return False


def _inc_tokens(inc_text: str) -> list[str]:
    out = [t.lower() for t in re.split(r"[\s_\-/,;()=\.]+", inc_text)
           if len(t) >= 4]
    extra_stop = {"inconsistent", "inconsistency", "incorrect", "correctly",
                  "requires", "because", "though", "while", "rather", "often",
                  "typically", "therefore", "assumes", "assumed", "assumption"}
    return [t for t in out if t not in _SRR_STOPWORDS and t not in extra_stop]


def _has_genuine_silent_inconsistency(case_path: Path, ds_row: dict) -> bool:
    incs = ds_row.get("inconsistencies", []) or []
    if not incs:
        return False
    qtxt = _conv_q_text(case_path / "conversation_log.txt")
    atxt = _conv_a_text(case_path / "conversation_log.txt")
    for inc in incs:
        toks = _inc_tokens(inc)
        if not toks:
            continue
        q_match = sum(1 for t in toks if t in qtxt)
        a_match = sum(1 for t in toks if t in atxt)
        if q_match < 2 and a_match < 2:
            return True
    return False


def _entity_grounded_disambig(entity: str, case_path: Path, ds_row: dict,
                              addressed_set: set[str]) -> bool:
    """Per-entity grounding for Disambiguation. Returns True if the entity
    was either addressed (per the coverage JSON) or surfaced in the prompt /
    a Q-line / an A-line (or its numeric value was volunteered). Mirrors
    `_entity_grounded_disambig` in `make_component_specific_ontology_plot.py`."""
    if entity in addressed_set:
        return True
    if _is_units_convention(entity):
        return True
    inc = (ds_row or {}).get("incomplete_user_req", "").lower()
    qtxt = _conv_q_text(case_path / "conversation_log.txt")
    atxt = _conv_a_text(case_path / "conversation_log.txt")
    toks = _entity_label_tokens(entity)
    if toks:
        if any(t in inc for t in toks):
            return True
        if any(t in qtxt for t in toks):
            return True
        if any(t in atxt for t in toks):
            return True
    nums = _entity_value_numeric_tokens(entity)
    if nums and any(n in atxt for n in nums):
        return True
    return False


def _issue_grounded_inconsistency(issue_desc: str, case_path: Path,
                                  warns_ge_total: bool) -> bool:
    """Per-issue grounding for Inconsistency. True if (a) the case-level
    explicit_warnings count covers all issues, OR (b) the issue's
    distinctive tokens are mentioned >=2 times in Q-lines or A-lines.
    Mirrors `_issue_grounded_inconsistency` in the per-issue plotter."""
    if warns_ge_total:
        return True
    qtxt = _conv_q_text(case_path / "conversation_log.txt")
    atxt = _conv_a_text(case_path / "conversation_log.txt")
    toks = _inc_tokens(issue_desc)
    if not toks:
        return False
    q_match = sum(1 for t in toks if t in qtxt)
    a_match = sum(1 for t in toks if t in atxt)
    return q_match >= 2 or a_match >= 2


# ---------------------------------------------------------------------------
# 3b. PER-CASE SCORING (FRR / CGRR / SRR / IC / MC / CR / CP / DR)
# ---------------------------------------------------------------------------

@dataclass
class CaseScore:
    component: str
    model: str
    task: str
    case_id: str
    n_items: int
    FR: int     # final resolution (strict score == "1/1")
    asked: int  # # questions about missing items
    grounded: int  # asked >= n_items (silently-resolved correction applied)
    IC: int     # intent capture (1 = faithful)
    MC: int     # memory consistency
    CR: float   # clarification recall
    CP: float   # clarification precision
    DR: int     # detection rate (Inconsistency only): warnings >= n_items
    prompt_length: int


def _score_disambiguation(case_path: Path, prompt_len: int,
                          component: str, model: str, case_id: str,
                          ds_row: dict | None = None) -> CaseScore | None:
    """Score one Disambiguation case from its judge JSONs.
    Applies the SRR->CGRR demotion if the case is not a *genuine* silent."""
    fr_data = _maybe_json(case_path / "llm_judge_disambiguation_score.json")
    chat_data = _maybe_json(
        case_path / "llm_judge_disambiguation_chat_score.json")
    cov_data = _maybe_json(case_path / "disambiguation_coverage_score.json")
    intent = _maybe_json(case_path / "llm_judge_intent_score.json")
    if fr_data is None and chat_data is None:
        return None

    fr = 1 if (fr_data and fr_data.get("strict_score") == "1/1") else 0

    total = asked = None
    if chat_data:
        total = chat_data.get("total_missing_entities", 0)
        asked = chat_data.get("total_questions_asked_in_chat") or \
            chat_data.get("relevant_questions", {}).get("count")
    if (asked is None or not total) and cov_data:
        total = cov_data.get("ambiguities_total", total or 0)
        asked = cov_data.get("ambiguities_addressed_in_conversation", 0)
    total = int(total or 0)
    asked = int(asked or 0)

    grounded = int(total > 0 and asked >= total)
    # SRR -> CGRR demotion correction (paper Section 3 silent-resolution check).
    if fr == 1 and grounded == 0 and total > 0 and ds_row is not None:
        if not _has_genuine_silent_disambig(case_path, ds_row):
            grounded = 1
    cr = (asked / total) if total > 0 else 0.0
    cp = ((chat_data.get("relevant_questions", {}).get("count", 0)
           / asked) if asked > 0 else 0.0) if chat_data else 0.0
    mc = int((chat_data or {}).get(
        "memory_consistency", {}).get("is_consistent", 0))
    ic = 1 if (intent and intent.get("score") == "1/1") else 0

    return CaseScore(
        component=component, model=model, task="Disambiguation", case_id=case_id,
        n_items=total, FR=fr, asked=asked, grounded=grounded,
        IC=ic, MC=mc, CR=cr, CP=cp, DR=0, prompt_length=prompt_len,
    )


def _score_inconsistency(case_path: Path, prompt_len: int,
                         component: str, model: str, case_id: str,
                         ds_row: dict | None = None) -> CaseScore | None:
    fr_data = _maybe_json(case_path / "llm_judge_inconsistency_score.json")
    chat_data = _maybe_json(
        case_path / "llm_judge_inconsistency_chat_score.json")
    intent = _maybe_json(case_path / "llm_judge_intent_score.json")
    if fr_data is None and chat_data is None:
        return None

    fr = 1 if (fr_data and fr_data.get("strict_score") == "1/1") else 0
    total = (chat_data or {}).get("total_planted_inconsistencies", 0)
    asked = (chat_data or {}).get(
        "questions_about_inconsistencies", {}).get("count", 0)
    warns = (chat_data or {}).get("explicit_warnings", {}).get("count", 0)
    total, asked, warns = int(total or 0), int(asked or 0), int(warns or 0)

    grounded = int(total > 0 and (asked >= total or warns >= total))
    # SRR -> CGRR demotion correction.
    if fr == 1 and grounded == 0 and total > 0 and ds_row is not None:
        if not _has_genuine_silent_inconsistency(case_path, ds_row):
            grounded = 1
    cr = (asked / total) if total > 0 else 0.0
    cp = ((chat_data or {}).get("relevant_questions", {}).get("count", 0)
          / asked) if asked > 0 else 0.0
    mc = int((chat_data or {}).get(
        "memory_consistency", {}).get("is_consistent", 0))
    ic = 1 if (intent and intent.get("score") == "1/1") else 0
    dr = int(total > 0 and warns >= total)

    return CaseScore(
        component=component, model=model, task="Inconsistency", case_id=case_id,
        n_items=total, FR=fr, asked=asked, grounded=grounded,
        IC=ic, MC=mc, CR=cr, CP=cp, DR=dr, prompt_length=prompt_len,
    )


# ---------------------------------------------------------------------------
# 4. FILTERS (length cap + shallow inconsistency drop)
# ---------------------------------------------------------------------------

def kept_case_ids(root: Path, component: str, task: str,
                  shallow_drop: bool = True
                  ) -> tuple[list[str], dict[str, int], dict[str, dict]]:
    """Return (kept_case_ids, prompt_length_by_id, ds_row_by_id) after filters."""
    ds_path = dataset_path(root, component, task)
    if not ds_path.exists():
        return [], {}, {}
    ds = _load_json(ds_path)
    field = PROMPT_FIELD[task]
    prompt_len = {r["id"]: len(r.get(field, "")) for r in ds if "id" in r}
    ds_rows = {r["id"]: r for r in ds if "id" in r}

    # Filter 1: prompt length cap
    kept_len = {cid for cid, n in prompt_len.items() if n <= MAX_PROMPT_CHARS}

    # Filter 2: shallow drop (Inconsistency only) — cases solved by all 5
    # guided models with CGRR == 1 (after SRR->CGRR demotion) are removed.
    shallow = set()
    if shallow_drop and task == "Inconsistency":
        for cid in prompt_len:
            cgrr_all = []
            for m in MODELS:
                cd = case_dir(root, component, m, task, cid)
                sc = _score_inconsistency(cd, prompt_len[cid], component, m, cid,
                                          ds_rows.get(cid))
                if sc is None:
                    cgrr_all = None
                    break
                cgrr_all.append(sc.FR == 1 and sc.grounded == 1)
            if cgrr_all and all(cgrr_all):
                shallow.add(cid)
    keep = [cid for cid in prompt_len if cid in kept_len and cid not in shallow]
    return keep, prompt_len, ds_rows


# ---------------------------------------------------------------------------
# 5. BUILD CASE-LEVEL DATAFRAME
# ---------------------------------------------------------------------------

def build_case_df(root: Path) -> pd.DataFrame:
    """Read every (component, model, task, case) and return one row per case.
    Columns: component, domain, model, task, case_id, n_items, FR, asked,
             grounded, IC, MC, CR, CP, DR, prompt_length, CGRR, SRR.
    Filters (length cap, shallow drop) and IC gate are applied."""
    rows = []
    scorer = {"Disambiguation": _score_disambiguation,
              "Inconsistency":  _score_inconsistency}
    for component in COMPONENTS:
        for task in TASKS:
            kept, prompt_len, ds_rows = kept_case_ids(root, component, task)
            for cid in kept:
                for model in MODELS:
                    cd = case_dir(root, component, model, task, cid)
                    sc = scorer[task](cd, prompt_len[cid], component, model,
                                      cid, ds_rows.get(cid))
                    if sc is None:
                        continue
                    rows.append(sc.__dict__)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Apply IC gate: if intent fail, demote FR/grounded to 0.
    fail = df["IC"] == 0
    df.loc[fail, "FR"] = 0
    df.loc[fail, "grounded"] = 0
    df["CGRR_case"] = ((df["FR"] == 1) & (df["grounded"] == 1)).astype(int)
    df["SRR_case"] = ((df["FR"] == 1) & (df["grounded"] == 0)).astype(int)
    df["domain"] = df["component"].map(COMPONENT_TO_DOMAIN)
    return df


# ---------------------------------------------------------------------------
# 6. ONTOLOGY CLASSIFIER (lexical, used for Figure 5 + appendix per-issue)
# ---------------------------------------------------------------------------
#
# Verbatim copy of the classifier in `make_component_specific_ontology_plot.py`
# (the script behind Figure 5) and `add_ontology_to_dataset.py` (which writes
# the per-case `ontology_components` field into dataset/<component>/<task>.json).
# All three classifiers must stay in sync so Figure 5, the appendix per-issue
# plots, and the dataset JSONs use the same labels.
#
# Order: U -> G -> N -> B -> M -> D -> O (Other). Order matters: M comes
# AFTER B so that "ambient_temperature" hits B's rule first instead of being
# swallowed by M's broad property tokens.
# ---------------------------------------------------------------------------
_ONT_U_PAT = [
    r"\bunit", r"kj/mol", r"\bk_?b\b.*mol", r"\bin k\b", r"\bin °?c\b",
    r"celsius", r"kelvin", r"\bmpa\b", r"\bgpa\b",
    r"sign error", r"exceeds?", r"impossible",
    r"sum[^a-z]*(>\s*1|exceed)", r"sums? to 1[0-9]{2}",
    r"\>\s*100[^0-9]?%", r"mass.*>.*mass",
    r"density[^a-z]*>.*density", r"bulk.*>.*theoretical",
    r"relative density\s*[>1]\.",
    r"contact angle\s*(1[89][0-9])",
    r"fraction.*(>\s*1|exceed)",
    r"deg(rees)? used where", r"radian",
    r"unit mismatch", r"dimensional", r"magnitude",
]

_ONT_G_PAT = [
    r"bernoulli", r"incompressible.*compressible", r"compressibility",
    r"laminar", r"turbulen", r"reynolds", r"\bre\b", r"mach",
    r"shock", r"isentropic", r"supersonic", r"subsonic",
    r"euler.?bernoulli", r"plastic", r"elastic", r"linear.?elastic",
    r"constitutive", r"plane stress", r"plane strain", r"beam theory",
    r"castigliano", r"rule of mixtures", r"schmid", r"hall.?petch",
    r"rayleigh", r"stokes", r"navier", r"cavitation",
    r"\bbcc\b|\bfcc\b|\bhcp\b", r"crystal structure", r"unit.?cell", r"bragg",
    r"arrhenius", r"diffusion", r"eutectic", r"peritectic",
    r"hyper.?eutec", r"hypo.?eutec", r"gibbs phase", r"phase rule",
    r"tauc", r"curie", r"superconduct", r"intrinsic silicon",
    r"darcy.?weisbach", r"colebrook", r"moody",
    r"theory", r"regime", r"governing", r"correlation",
    r"friction[ _-]?factor", r"fanning", r"turbulence[ _-]?model",
    r"vapou?r",
    r"ideal[ _-]?gas", r"choked", r"froude", r"speed of sound",
    r"\bgamma\b", r"equation[ _-]?of[ _-]?state", r"centripetal",
    r"conservation", r"equation[ _-]?to[ _-]?use",
    r"flexure", r"bending", r"torque\b", r"shear stress", r"shear modulus",
    r"neutral axis", r"factor of safety", r"\bfos\b",
    r"yield stress", r"yield strength",
    r"engineering strain", r"true strain",
    r"faraday", r"valence\b",
    r"three phases", r"binary system",
    r"formulation", r"convention", r"definition\b", r"formula\b",
    r"airfoil[ _-]?camber",
    r"combustion",
    r"\bfr1?\b", r"hydraulic jump", r"thin.?walled", r"thick.?walled",
    r"axial force", r"axial spring", r"shear strain", r"thermal stress",
    r"thermal restraint", r"rigid in shear", r"point loads?",
    r"piecewise constant", r"linear axial",
    r"\bgriffith\b", r"\bsigma_?f\b",
    r"\bnebmake|lclimb|climbing.?image",
    r"stress.?state|stress state",
    # augmentations to reduce Other bucket
    r"\bassumption", r"\bproblem\b", r"\bparameter[s']*\b",
    r"\blossless\b", r"\bsmall strain",
    r"lift[ _-]?coefficient", r"drag[ _-]?coefficient",
    r"\bmodel\b|\bmodels\b", r"\binterfacial\b",
    r"\bheating\b|\bcooling\b",
]

_ONT_N_PAT = [
    r"\bcfl\b", r"\bdt\b", r"\bdx\b", r"\bnx\b", r"\bnt\b",
    r"time[ _-]*step", r"time[ _-]*integration",
    r"numerical method", r"solver", r"scheme", r"discreti",
    r"output[ _-]?requirements?", r"final[ _-]?time", r"output\b",
    r"boundaryfield|time[ _-]?controls?|solver[ _-]?settings?",
    r"element type", r"\bc3d\b", r"abaqus", r"\*static\b", r"\*dynamic\b",
    r"keyword.*deck", r"\bmesh\b", r"geometry_[23]d", r"element",
    r"isif|ispin|icharg|incar|vasp", r"charge state", r"\bstep\b",
    r"workflow", r"relax", r"convergence",
    r"freysoldt", r"dielectric constant",
    r"diffract", r"planes? \(", r"\bhkl\b",
    r"defect complex", r"chemical system",
    r"competing.?phase", r"fermi.?level", r"supercell",
    r"stable.?entry", r"parameter values?",
    r"resolution", r"discretization",
    r"image.?folder", r"image folder", r"file layout", r"filename",
    r"intermediate images?", r"folder layout",
    r"\bcount\b.*labels?", r"vacancy result",
    r"molecular.?weight labels", r"weight labels",
    # augmentations to reduce Other bucket
    r"\bmethod\b",
    r"\blocpot\b|\bkpoints?\b|\bmagmom\b",
]

_ONT_B_PAT = [
    r"boundary[ _-]?conditions?", r"\bbc\b",
    r"dirichlet", r"neumann", r"periodic", r"fixedvalue",
    r"no.?slip", r"\bwall\b", r"adiabatic", r"convect", r"traction",
    r"pressure[ _-]?outlet", r"\binlet\b", r"\boutlet\b", r"symmetry",
    r"load[ _-]?definition", r"applied[ _-]?load", r"support",
    r"cload|dload", r"clamped", r"fixed[ _-]?end", r"free[ _-]?end",
    r"initial[ _-]?conditions?", r"\bic\b",
    r"load[: ]", r"support and span",
    r"follower", r"dead[ _-]?load",
    r"displacement.?only", r"u[ _-]?[xyz]\s*=", r"loaded[ _-]?top",
    r"\broller\b", r"\bpin\b", r"simply[ _-]?supported",
    r"ambient", r"external[ _-]?h\b", r"internal[ _-]?h\b",
    r"plate[ _-]?velocity", r"pressure[ _-]?gradient",
    r"steam[ _-]?temperature", r"insulation[ _-]?thickness",
    r"freestream|free.?stream", r"farfield|far.?field",
    r"heat[ _-]?transfer[ _-]?coefficient",
    # augmentations to reduce Other bucket
    r"\bloading\b", r"\bload\b",
    r"flow[_ -]?rate|flowrate",
    r"angular[_ -]?velocit",
    r"\bnet[_ -]?head\b|\bhead\b",
    r"\bcontact[_ -]angle\b", r"\befficienc",
    r"\binterface\b|interface[ _-]?conditions?",
]

_ONT_M_PAT = [
    r"density", r"viscosity", r"kinematic", r"dynamic[ _-]?viscosity",
    r"modulus", r"young", r"shear[ _-]?modulus", r"\be\s*=", r"\bg\s*=",
    r"poisson", r"\bnu\b", r"conductivity", r"specific[ _-]?heat",
    r"thermal[ _-]?diffusivity", r"prandtl", r"surface[ _-]?tension",
    r"\bmaterial\b", r"steel", r"aluminum|aluminium", r"copper",
    r"\baisi\b", r"ref[ _-]?physical[ _-]?properties",
    r"physical[ _-]?properties", r"porous[ _-]?properties",
    r"thermo[ _-]?physical", r"elastic[ _-]?constants",
    r"property", r"composition", r"wt%", r"weight[ _-]?fraction",
    r"alloy", r"concentration",
    r"fiber", r"matrix", r"composite",
    r"porosity", r"superficial", r"fluid[ _-]?propert", r"roughness",
    r"permeability",
    r"insulation[ _-]?conductivity",
    r"oil[ _-]?density", r"oil[ _-]?viscosity", r"particle[ _-]?density",
    r"water[ _-]?temperature",
    r"porous[ _-]?media", r"combustion[ _-]?properties",
    r"terminal[ _-]?velocity", r"\bgravity\b", r"\bvelocity\b\s*$",
]

_ONT_D_PAT = [
    r"geometry", r"dimension", r"length", r"diameter", r"\bradius\b",
    r"thickness", r"\barea\b", r"\bvolume\b", r"\b[23]d\b",
    r"two.?dimensional", r"three.?dimensional", r"span", r"height",
    r"\bwidth\b", r"shape", r"cross.?section", r"domain", r"aspect ratio",
    r"cell size", r"grain size", r"particle size",
    r"problem[ _-]?statement",
    r"structure", r"primitive", r"unit[ _-]?cell",
    r"camber", r"airfoil", r"segment",
    r"centreline|centerline", r"\baxis\b", r"notch", r"hole",
    r"cutout", r"re.?entrant",
    # augmentations to reduce Other bucket
    r"\bdepth\b|\bsubmergence\b",
]


def classify_ontology(text: str) -> str:
    """Classify a missing-entity name or inconsistency description into one
    of G/M/B/D/N/U, or 'O' for genuine residue. Identical logic to
    `make_component_specific_ontology_plot.py` and
    `add_ontology_to_dataset.py`. Input is normalized (lowercase +
    underscore-to-space) so snake_case tokens hit word-boundary patterns."""
    if not text:
        return "O"
    t = str(text).lower().replace("_", " ")
    if any(re.search(p, t) for p in _ONT_U_PAT):
        return "U"
    if any(re.search(p, t) for p in _ONT_G_PAT):
        return "G"
    if any(re.search(p, t) for p in _ONT_N_PAT):
        return "N"
    if any(re.search(p, t) for p in _ONT_B_PAT):
        return "B"
    if any(re.search(p, t) for p in _ONT_M_PAT):
        return "M"
    if any(re.search(p, t) for p in _ONT_D_PAT):
        return "D"
    return "O"


def build_entity_df(root: Path) -> pd.DataFrame:
    """One row per (case, missing-item) pair, with ontology slot, recovery,
    and grounding flags. Used for Figure 5 + appendix per-issue plots."""
    rows = []
    for component in COMPONENTS:
        for task in TASKS:
            ds_path = dataset_path(root, component, task)
            if not ds_path.exists():
                continue
            ds = _load_json(ds_path)
            field = ITEMS_FIELD[task]
            kept, prompt_len, ds_rows = kept_case_ids(root, component, task)
            kept_set = set(kept)
            for case in ds:
                cid = case.get("id")
                if cid not in kept_set:
                    continue
                items = case.get(field, []) or []
                ds_row = ds_rows.get(cid, {})
                for model in MODELS:
                    cd = case_dir(root, component, model, task, cid)
                    intent = _maybe_json(cd / "llm_judge_intent_score.json")
                    ic = 1 if (intent and intent.get("score") == "1/1") else 0

                    if task == "Disambiguation":
                        fr_data = _maybe_json(
                            cd / "llm_judge_disambiguation_score.json")
                        cov = _maybe_json(
                            cd / "disambiguation_coverage_score.json")
                        recovered_set = set(
                            (fr_data or {}).get("recovered_entities", []))
                        addressed_set = set(
                            (cov or {}).get("entities_addressed", []))
                        for item in items:
                            slot = classify_ontology(item)
                            is_recovered = (item in recovered_set) and ic == 1
                            is_grounded = ic == 1 and _entity_grounded_disambig(
                                item, cd, ds_row, addressed_set)
                            rows.append({
                                "component": component,
                                "domain":    COMPONENT_TO_DOMAIN[component],
                                "model":     model, "task": task,
                                "case_id":   cid, "item": item, "slot": slot,
                                "recovered": is_recovered,
                                "grounded":  is_grounded, "IC": ic,
                            })
                    else:
                        chat = _maybe_json(
                            cd / "llm_judge_inconsistency_chat_score.json")
                        fr_data = _maybe_json(
                            cd / "llm_judge_inconsistency_score.json")
                        case_fr = 1 if (fr_data and fr_data.get(
                            "strict_score") == "1/1") else 0
                        warns = int((chat or {}).get(
                            "explicit_warnings", {}).get("count", 0))
                        warns_ok = warns >= len(items) and len(items) > 0
                        for item in items:
                            slot = classify_ontology(item)
                            is_recovered = (case_fr == 1) and ic == 1
                            is_grounded = ic == 1 and _issue_grounded_inconsistency(
                                item, cd, warns_ok)
                            rows.append({
                                "component": component,
                                "domain":    COMPONENT_TO_DOMAIN[component],
                                "model":     model, "task": task,
                                "case_id":   cid, "item": item, "slot": slot,
                                "recovered": is_recovered,
                                "grounded":  is_grounded, "IC": ic,
                            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. STYLE SETUP (matches Figure 4 typography)
# ---------------------------------------------------------------------------

def setup_style() -> None:
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "ieee", "tableau-colorblind10"])
    except Exception:
        plt.style.use("tableau-colorblind10")
    mpl.rcParams.update({
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "figure.titlesize": 14, "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
    })
    for k in ("font.size", "axes.titlesize", "axes.labelsize",
              "xtick.labelsize", "ytick.labelsize", "legend.fontsize",
              "figure.titlesize"):
        mpl.rcParams[k] = mpl.rcParams[k] * 1.5


# ---------------------------------------------------------------------------
# 8. FIGURE 3 — DATASET DISTRIBUTION (donut charts)
# ---------------------------------------------------------------------------

def figure_3_dataset_distribution(case_df: pd.DataFrame, out_dir: Path) -> Path:
    """Two donut charts (Disambiguation, Inconsistency resolution) showing
    per-domain case counts after filtering. Maps to Figure 3 in the paper."""
    counts = {t: defaultdict(int) for t in TASKS}
    for (task, dom), grp in case_df.drop_duplicates(["task", "case_id", "component"]) \
                                   .groupby(["task", "domain"]):
        counts[task][dom] = grp[grp["model"] == MODELS[0]]["case_id"].nunique()
    # Recount more carefully: number of unique case_ids per (task, domain).
    counts = {t: {d: 0 for d in DOMAIN_ORDER} for t in TASKS}
    for t in TASKS:
        sub = case_df[case_df["task"] == t]
        for dom in DOMAIN_ORDER:
            counts[t][dom] = sub[sub["domain"] == dom][["component", "case_id"]] \
                .drop_duplicates().shape[0]

    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.6))
    for ax, t, title in [(axes[0], "Disambiguation", "Disambiguation"),
                         (axes[1], "Inconsistency",  "Inconsistency resolution")]:
        values = [counts[t][d] for d in DOMAIN_ORDER]
        total = sum(values)
        wedges, _ = ax.pie(
            values, colors=[DOMAIN_COLOUR[d] for d in DOMAIN_ORDER],
            startangle=90, counterclock=False,
            wedgeprops=dict(width=0.38, edgecolor="white", linewidth=1.0),
        )
        ax.text(0, 0.05, f"{total}", ha="center", va="center",
                fontsize=12, fontweight="bold")
        ax.text(0, -0.18, "cases", ha="center", va="center", fontsize=10)
        ax.set_title(title, fontsize=12, pad=12)
        for w, d, v in zip(wedges, DOMAIN_ORDER, values):
            if v == 0:
                continue
            ang = (w.theta2 + w.theta1) / 2
            x, y = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            ha = "left" if x >= 0 else "right"
            va = "bottom" if y > 0.7 else ("top" if y < -0.7 else "center")
            pct = 100 * v / total
            ax.annotate(f"{d}\n{v} ({pct:.1f}%)",
                        xy=(0.82*x, 0.82*y), xytext=(1.15*x, 1.08*y),
                        ha=ha, va=va, fontsize=9)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94,
                        bottom=0.04, hspace=0.45)
    out = out_dir / "fig_dataset_distribution"
    fig.savefig(out.with_suffix(".png"), dpi=300,
                bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".pdf"),
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out.with_suffix(".png")


# ---------------------------------------------------------------------------
# 9. AGGREGATIONS USED BY FIGURES 4, 6, 7, 8 + APPENDIX TABLES
# ---------------------------------------------------------------------------

def aggregate_by_domain(case_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (domain, model, task) with mean FR, CGRR, SRR, IC, MC, CR,
    CP, DR. Used by Figures 4, 6, 7, 8."""
    grp = case_df.groupby(["domain", "model", "task"], as_index=False).agg(
        FR=("FR", "mean"), CGRR=("CGRR_case", "mean"), SRR=("SRR_case", "mean"),
        IC=("IC", "mean"), MC=("MC", "mean"),
        CR=("CR", "mean"), CP=("CP", "mean"), DR=("DR", "mean"),
        n=("case_id", "count"),
    )
    for c in ["FR", "CGRR", "SRR", "IC", "MC", "CR", "CP", "DR"]:
        grp[c] = 100 * grp[c]
    return grp


# ---------------------------------------------------------------------------
# 10. SHARED BAR-PLOT PRIMITIVE (CGRR solid + SRR hatched, single thick border)
# ---------------------------------------------------------------------------

def _draw_outcome_bars(ax, sub: pd.DataFrame, models: list[str],
                       x_categories: list[str], category_col: str,
                       bar_w: float = 0.15) -> None:
    x = np.arange(len(x_categories))
    n = len(models)
    offsets = (np.arange(n) - (n - 1) / 2) * bar_w
    for i, model in enumerate(models):
        ms = sub[sub["model"] == model].set_index(category_col)
        cgrr = np.array([ms.loc[c, "CGRR"] if c in ms.index else 0.0
                         for c in x_categories])
        srr = np.array([ms.loc[c, "SRR"] if c in ms.index else 0.0
                        for c in x_categories])
        xi = x + offsets[i]
        ax.bar(xi, cgrr, bar_w, color=MODEL_COLOR[model],
               edgecolor="none", linewidth=0, zorder=3)
        ax.bar(xi, srr, bar_w, bottom=cgrr,
               facecolor=MODEL_COLOR[model], alpha=0.35,
               edgecolor="black", linewidth=0, hatch="///", zorder=3)
        ax.bar(xi, cgrr + srr, bar_w, fill=False,
               edgecolor="black", linewidth=1.4, zorder=4)


def _two_legend_layout(fig, models: list[str]) -> None:
    extras = [
        mpatches.Patch(facecolor="#555", edgecolor="black", linewidth=1.4,
                       label="CGRR (conversation-grounded)"),
        mpatches.Patch(facecolor="#555", alpha=0.35, edgecolor="black",
                       linewidth=1.4, hatch="///",
                       label="SRR (silently resolved)"),
    ]
    model_handles = [
        mpatches.Patch(facecolor=MODEL_COLOR[m], edgecolor="black",
                       linewidth=1.4, label=MODEL_LABEL[m])
        for m in models
    ]
    fig.legend(handles=extras, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.97), frameon=False)
    fig.legend(handles=model_handles, loc="upper center",
               ncol=len(model_handles),
               bbox_to_anchor=(0.5, 0.06), frameon=False)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.91,
                        bottom=0.14, wspace=0.05)


# ---------------------------------------------------------------------------
# 11. FIGURE 4 — MAIN BENCHMARK RESULTS (per domain, both tasks)
# ---------------------------------------------------------------------------

def figure_4_main_outcomes(case_df: pd.DataFrame, out_dir: Path) -> Path:
    """Per-domain CGRR + SRR (FRR = sum) for both tasks. Figure 4 in paper."""
    agg = aggregate_by_domain(case_df)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.4), sharey=True)
    pct = r"\%" if mpl.rcParams.get("text.usetex", False) else "%"
    for ax, task, title in zip(axes, TASKS,
                               ["Disambiguation", "Inconsistency Resolution"]):
        sub = agg[agg["task"] == task]
        _draw_outcome_bars(ax, sub, MODELS, DOMAIN_ORDER, "domain")
        ax.set_xticks(np.arange(len(DOMAIN_ORDER)))
        ax.set_xticklabels([d.replace(" ", "\n", 1) for d in DOMAIN_ORDER],
                           linespacing=0.95)
        ax.set_ylim(0, 105)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_title(title, pad=6)
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    axes[0].set_ylabel(f"FRR ({pct})")
    _two_legend_layout(fig, MODELS)
    out = out_dir / "fig_main_domain_outcomes"
    fig.savefig(out.with_suffix(".png"), dpi=300,
                bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".pdf"),
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out.with_suffix(".png")


# ---------------------------------------------------------------------------
# 12. FIGURE 5 — PER-ONTOLOGY-COMPONENT METRICS (per-issue denominator)
# ---------------------------------------------------------------------------

def aggregate_by_ontology_per_issue(entity_df: pd.DataFrame) -> pd.DataFrame:
    df = entity_df.copy()
    df["is_cgrr"] = df["recovered"] & df["grounded"]
    df["is_srr"] = df["recovered"] & ~df["grounded"]
    grp = df.groupby(["task", "model", "slot"], as_index=False).agg(
        CGRR=("is_cgrr", "mean"), SRR=("is_srr", "mean"),
        n=("item", "count"),
    )
    for c in ["CGRR", "SRR"]:
        grp[c] = 100 * grp[c]
    grp["FRR"] = grp["CGRR"] + grp["SRR"]
    return grp


def figure_5_ontology(entity_df: pd.DataFrame, out_dir: Path) -> Path:
    """Per-ontology-component, per-issue rates. Figure 5 in paper."""
    agg = aggregate_by_ontology_per_issue(entity_df)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.4), sharey=True)
    pct = r"\%" if mpl.rcParams.get("text.usetex", False) else "%"
    for ax, task, title in zip(axes, TASKS,
                               ["Disambiguation (per missing entity)",
                                "Inconsistency resolution (per planted issue)"]):
        sub = agg[agg["task"] == task]
        _draw_outcome_bars(ax, sub, MODELS, ONT_ORDER, "slot")
        ax.set_xticks(np.arange(len(ONT_ORDER)))
        ax.set_xticklabels([ONT_NAME[o] for o in ONT_ORDER], linespacing=0.95,
                           fontsize=9*1.5)
        ax.set_ylim(0, 105)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_title(title, pad=6)
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    if mpl.rcParams.get("text.usetex", False):
        axes[0].set_ylabel(r"$\mathrm{FRR}(c)$ " + f"({pct})")
    else:
        axes[0].set_ylabel(f"FRR(c) ({pct})")
    _two_legend_layout(fig, MODELS)
    out = out_dir / "fig_component_specific_ontology"
    fig.savefig(out.with_suffix(".png"), dpi=300,
                bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".pdf"),
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out.with_suffix(".png")


# ---------------------------------------------------------------------------
# 13. FIGURE 6 — PARETO RADAR (Capability / Robustness / Usability)
# ---------------------------------------------------------------------------

def _pareto_axes_row(row) -> pd.Series:
    cr, cp = row["CR"] / 100, row["CP"] / 100
    srr = row["SRR"] / 100
    mc, ic, dr = row["MC"] / 100, row["IC"] / 100, row["DR"] / 100
    capability = float(np.nanmean([cr, cp]))
    robustness = float(np.nanmean([1 - srr, mc, dr]) if row["task"] == "Inconsistency"
                       else np.nanmean([1 - srr, mc]))
    usability = float(ic)
    return pd.Series({"Capability": capability,
                      "Robustness": robustness,
                      "Usability":  usability})


def figure_6_pareto(case_df: pd.DataFrame, out_dir: Path) -> Path:
    """Capability / Robustness / Usability radar, 2 rows × 4 cols. Figure 6."""
    agg = aggregate_by_domain(case_df)
    agg[["Capability", "Robustness", "Usability"]
        ] = agg.apply(_pareto_axes_row, axis=1)
    CATEGORIES = ["Capability", "Robustness", "Usability"]
    ABBREV = {"Capability": "Cap", "Robustness": "Rob", "Usability": "Use"}
    BASE_ANGLES = [np.radians(90), np.radians(-30), np.radians(-150)]
    angles = BASE_ANGLES + [BASE_ANGLES[0]]

    fig = plt.figure(figsize=(28, 14))
    GL, GB, HSP, WSP = 0.09, 0.13, 0.02, 0.02
    gs = fig.add_gridspec(2, 4, left=GL, right=0.99, top=0.90, bottom=GB,
                          hspace=HSP, wspace=WSP)
    FS_AXIS, FS_TICK, FS_COL, FS_ROW = 38, 31, 43, 42
    pct = r"\%" if mpl.rcParams.get("text.usetex", False) else "%"

    def draw_radar(ax, dom, task):
        sub = agg[(agg.domain == dom) & (agg.task == task)]
        theta = np.linspace(0, 2*np.pi, 300)
        for r in [0.25, 0.5, 0.75, 1.0]:
            ax.plot(theta, [r]*300, color="#b0b0b0",
                    linewidth=0.9 if r in (0.5, 1.0) else 0.45,
                    linestyle="--" if r in (0.5, 1.0) else ":", zorder=1)
        for a in BASE_ANGLES:
            ax.plot([a, a], [0, 1.0], color="#b0b0b0", linewidth=0.6, zorder=1)
        for m in MODELS:
            r = sub[sub.model == m]
            if r.empty:
                continue
            vals = [max(0.0, float(r.iloc[0][c])) if pd.notna(r.iloc[0][c]) else 0.0
                    for c in CATEGORIES]
            vals_c = vals + [vals[0]]
            ax.plot(angles, vals_c, "o-", linewidth=2.4,
                    color=MODEL_COLOR[m], alpha=0.92, markersize=6, zorder=3)
            ax.fill(angles, vals_c, alpha=0.09, color=MODEL_COLOR[m], zorder=2)
        for cat, a in zip(CATEGORIES, BASE_ANGLES):
            ax.text(a, 1.18, ABBREV[cat], ha="center", va="center",
                    fontsize=FS_AXIS, fontweight="bold", color="#111")
        for r, lbl in [(0.5, f"50{pct}"), (1.0, f"100{pct}")]:
            ax.text(BASE_ANGLES[1] - 0.26, r, lbl,
                    ha="right", va="center", fontsize=FS_TICK,
                    color="#222", fontweight="bold")
        ax.set_ylim(0, 1.32)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines["polar"].set_visible(False)
        ax.grid(False)
        ax.set_facecolor("white")

    col_w = (0.99 - GL) / (4 + 3 * WSP)
    col_x = [GL + col_w * (0.5 + i*(1+WSP)) for i in range(4)]
    for cx, dom in zip(col_x, DOMAIN_ORDER):
        fig.text(cx, 0.93, dom, ha="center", va="bottom",
                 fontsize=FS_COL, fontweight="bold")
    sub_h = (0.90 - GB) / (2 + HSP)
    fig.text(GL - 0.005, 0.90 - sub_h/2, "Disambiguation", ha="right",
             va="center", fontsize=FS_ROW, fontweight="bold", rotation=90)
    fig.text(GL - 0.005, GB + sub_h/2, "Inconsistency\nResolution",
             ha="right", va="center", multialignment="center",
             fontsize=FS_ROW, fontweight="bold", rotation=90)
    for ri, t in enumerate(TASKS):
        for ci, dom in enumerate(DOMAIN_ORDER):
            ax = fig.add_subplot(gs[ri, ci], projection="polar")
            draw_radar(ax, dom, t)
    legend_handles = [
        mlines.Line2D([], [], color=MODEL_COLOR[m], linewidth=3.2,
                      marker="o", markersize=10, label=MODEL_LABEL[m])
        for m in MODELS
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(MODELS),
               frameon=False, fontsize=36, bbox_to_anchor=(0.5, 0.03),
               columnspacing=1.6, handlelength=2.0, handletextpad=0.6)
    out = out_dir / "fig_pareto_analysis"
    fig.savefig(out.with_suffix(".png"), dpi=220,
                bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".pdf"),
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out.with_suffix(".png")


# ---------------------------------------------------------------------------
# 14. FIGURES 7, 8 — APPENDIX SPLIT (general-numeric vs. tool-use)
# ---------------------------------------------------------------------------

GENERAL_COMPONENTS = ["fluids", "solMech", "matSci"]
TOOL_COMPONENTS = ["foam", "solToolUse", "matToolUse", "pde"]


def _split_outcomes(case_df: pd.DataFrame, components: list[str],
                    domains: list[str], out_dir: Path,
                    out_name: str, title_prefix: str = "") -> Path:
    sub = case_df[case_df["component"].isin(components)].copy()
    agg = aggregate_by_domain(sub)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.4), sharey=True)
    pct = r"\%" if mpl.rcParams.get("text.usetex", False) else "%"
    for ax, task, title in zip(axes, TASKS,
                               [f"{title_prefix}Disambiguation",
                                f"{title_prefix}Inconsistency Resolution"]):
        sub2 = agg[agg["task"] == task]
        _draw_outcome_bars(ax, sub2, MODELS, domains, "domain")
        ax.set_xticks(np.arange(len(domains)))
        ax.set_xticklabels([d.replace(" ", "\n", 1) for d in domains],
                           linespacing=0.95)
        ax.set_ylim(0, 105)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_title(title, pad=6)
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    axes[0].set_ylabel(f"FRR ({pct})")
    _two_legend_layout(fig, MODELS)
    out = out_dir / out_name
    fig.savefig(out.with_suffix(".png"), dpi=300,
                bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".pdf"),
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out.with_suffix(".png")


def figure_7_general_split(case_df: pd.DataFrame, out_dir: Path) -> Path:
    """General-numeric (fluids, solMech, matSci) split. Figure 7 in appendix."""
    return _split_outcomes(case_df, GENERAL_COMPONENTS,
                           ["Fluid mechanics", "Solid mechanics",
                               "Materials science"],
                           out_dir, "fig_general_numeric_outcomes")


def figure_8_tooluse_split(case_df: pd.DataFrame, out_dir: Path) -> Path:
    """Tool-use (foam, solToolUse, matToolUse, pde) split. Figure 8 in appendix."""
    return _split_outcomes(case_df, TOOL_COMPONENTS,
                           DOMAIN_ORDER, out_dir, "fig_tool_use_outcomes")


# ---------------------------------------------------------------------------
# 15. APPENDIX FIGURE — PER-DOMAIN MICRO-AVG (per-issue denominator)
# ---------------------------------------------------------------------------

def figure_appendix_per_domain(entity_df: pd.DataFrame, out_dir: Path) -> Path:
    """Per-domain breakdown using per-issue denominator. Appendix figure."""
    df = entity_df.copy()
    df["is_cgrr"] = df["recovered"] & df["grounded"]
    df["is_srr"] = df["recovered"] & ~df["grounded"]
    grp = df.groupby(["task", "model", "domain"], as_index=False).agg(
        CGRR=("is_cgrr", "mean"), SRR=("is_srr", "mean"),
        n=("item", "count"),
    )
    for c in ["CGRR", "SRR"]:
        grp[c] = 100 * grp[c]
    grp["FRR"] = grp["CGRR"] + grp["SRR"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6.4), sharey=True)
    pct = r"\%" if mpl.rcParams.get("text.usetex", False) else "%"
    for ax, task, title in zip(axes, TASKS,
                               ["Disambiguation (per missing entity)",
                                "Inconsistency resolution (per planted issue)"]):
        sub = grp[grp["task"] == task]
        _draw_outcome_bars(ax, sub, MODELS, DOMAIN_ORDER, "domain")
        ax.set_xticks(np.arange(len(DOMAIN_ORDER)))
        ax.set_xticklabels(DOMAIN_ORDER, linespacing=0.95)
        ax.set_ylim(0, 105)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_title(title, pad=6)
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    axes[0].set_ylabel(f"FRR ({pct}, per item)")
    _two_legend_layout(fig, MODELS)
    out = out_dir / "outcomes_micro_avg"
    fig.savefig(out.with_suffix(".png"), dpi=300,
                bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".pdf"),
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out.with_suffix(".png")


# ---------------------------------------------------------------------------
# 16. APPENDIX FIGURE — GUIDED VS UNGUIDED (Gemini 2.5 Pro)
# ---------------------------------------------------------------------------

def figure_appendix_guided_vs_unguided(root: Path, case_df: pd.DataFrame,
                                       out_dir: Path) -> Path:
    """Guided vs unguided agent on Gemini 2.5 Pro. Appendix figure."""
    target_model = "GEMINI_2_5_PRO"
    guided = case_df[case_df["model"] == target_model].copy()

    # Score baseline runs on the *same* kept case ids.
    rows = []
    for component in COMPONENTS:
        for task in TASKS:
            ds_path = dataset_path(root, component, task)
            ds_rows_map = ({r["id"]: r for r in _load_json(ds_path) if "id" in r}
                           if ds_path.exists() else {})
            kept_ids = guided[(guided["component"] == component)
                              & (guided["task"] == task)]["case_id"].unique()
            for cid in kept_ids:
                cd = case_dir(root, component, target_model,
                              task, cid, baseline=True)
                if not cd.is_dir():
                    rows.append({"component": component,
                                 "domain": COMPONENT_TO_DOMAIN[component],
                                 "task": task, "case_id": cid,
                                 "FR": 0, "CGRR_case": 0, "SRR_case": 0})
                    continue
                scorer = (_score_disambiguation if task == "Disambiguation"
                          else _score_inconsistency)
                sc = scorer(cd, 0, component, target_model, cid,
                            ds_rows_map.get(cid))
                if sc is None:
                    rows.append({"component": component,
                                 "domain": COMPONENT_TO_DOMAIN[component],
                                 "task": task, "case_id": cid,
                                 "FR": 0, "CGRR_case": 0, "SRR_case": 0})
                    continue
                cgrr = int(sc.FR == 1 and sc.grounded == 1)
                srr = int(sc.FR == 1 and sc.grounded == 0)
                rows.append({"component": component,
                             "domain": COMPONENT_TO_DOMAIN[component],
                             "task": task, "case_id": cid,
                             "FR": sc.FR, "CGRR_case": cgrr, "SRR_case": srr})
    unguided = pd.DataFrame(rows)

    def _agg(df):
        g = df.groupby(["domain", "task"]).agg(
            FR=("FR", "mean"), CGRR=("CGRR_case", "mean"),
            SRR=("SRR_case", "mean"), n=("case_id", "count"),
        ).reset_index()
        for c in ["FR", "CGRR", "SRR"]:
            g[c] = 100 * g[c]
        return g

    g_agg, u_agg = _agg(guided), _agg(unguided)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.4), sharey=True)
    width = 0.35
    x = np.arange(len(DOMAIN_ORDER))
    COL_GUIDED, COL_UNGUIDED = "#0072B2", "#D55E00"
    pct = r"\%" if mpl.rcParams.get("text.usetex", False) else "%"
    for ax, task, title in zip(axes, TASKS,
                               ["Disambiguation", "Inconsistency Resolution"]):
        def vals(df):
            out = []
            for d in DOMAIN_ORDER:
                r = df[(df["domain"] == d) & (df["task"] == task)]
                if r.empty:
                    out.append((0.0, 0.0, 0.0))
                else:
                    rr = r.iloc[0]
                    out.append((rr["CGRR"], rr["SRR"], rr["FR"]))
            return out
        u, g = vals(u_agg), vals(g_agg)
        for offs, dat, col in [(- width/2, u, COL_UNGUIDED),
                               (+ width/2, g, COL_GUIDED)]:
            cgrr = np.array([v[0] for v in dat])
            srr = np.array([v[1] for v in dat])
            ax.bar(x + offs, cgrr, width, color=col,
                   edgecolor="none", zorder=3)
            ax.bar(x + offs, srr, width, bottom=cgrr, facecolor=col,
                   alpha=0.35, edgecolor="black", linewidth=0,
                   hatch="///", zorder=3)
            ax.bar(x + offs, cgrr + srr, width, fill=False,
                   edgecolor="black", linewidth=1.4, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([d.replace(" ", "\n", 1) for d in DOMAIN_ORDER],
                           linespacing=0.95)
        ax.set_ylim(0, 105)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_title(title, pad=6)
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    axes[0].set_ylabel(f"FRR ({pct})")

    pattern_handles = [
        mpatches.Patch(facecolor="#555", edgecolor="black", linewidth=1.4,
                       label="CGRR (conversation-grounded)"),
        mpatches.Patch(facecolor="#555", alpha=0.35, edgecolor="black",
                       linewidth=1.4, hatch="///", label="SRR (silently resolved)"),
    ]
    color_handles = [
        mpatches.Patch(facecolor=COL_GUIDED, edgecolor="black", linewidth=1.4,
                       label="Guided"),
        mpatches.Patch(facecolor=COL_UNGUIDED, edgecolor="black", linewidth=1.4,
                       label="Unguided"),
    ]
    fig.legend(handles=pattern_handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.97), frameon=False)
    fig.legend(handles=color_handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.06), frameon=False)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.91,
                        bottom=0.14, wspace=0.05)
    out = out_dir / "fig_appendix_guided_vs_unguided"
    fig.savefig(out.with_suffix(".png"), dpi=300,
                bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".pdf"),
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out.with_suffix(".png")


# ---------------------------------------------------------------------------
# 17. TABLES
# ---------------------------------------------------------------------------

def _latex_pct(x: float) -> str:
    return f"{x:.1f}" if pd.notna(x) else "--"


def table_main_macro(case_df: pd.DataFrame, out_dir: Path) -> Path:
    """Per-domain × per-model FRR/CGRR/SRR macro table. Appendix tables A.1, A.2."""
    agg = aggregate_by_domain(case_df)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for task, fname in [("Disambiguation", "table_appendix_disambiguation.tex"),
                        ("Inconsistency",  "table_appendix_inconsistency.tex")]:
        sub = agg[agg["task"] == task]
        lines = [
            r"\begin{tabular}{llrrrrrrrr}",
            r"\toprule",
            r"Domain & Model & $n$ & FRR & CGRR & SRR & IC & MC & CR & CP \\",
            r"\midrule",
        ]
        for dom in DOMAIN_ORDER:
            for m in MODELS:
                r = sub[(sub["domain"] == dom) & (sub["model"] == m)]
                if r.empty:
                    continue
                r = r.iloc[0]
                lines.append(
                    f"{dom} & {MODEL_LABEL[m]} & {int(r['n'])} & "
                    f"{_latex_pct(r['FR'])} & {_latex_pct(r['CGRR'])} & "
                    f"{_latex_pct(r['SRR'])} & {_latex_pct(r['IC'])} & "
                    f"{_latex_pct(r['MC'])} & {_latex_pct(r['CR'])} & "
                    f"{_latex_pct(r['CP'])} \\\\"
                )
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        out = out_dir / fname
        out.write_text("\n".join(lines))
        paths.append(out)
    return paths[0]


def table_clamber_vs_sciconv(case_df: pd.DataFrame, out_dir: Path,
                             clamber_n: int = 115, clamber_rate: float = 86.1) -> Path:
    """Headline comparison table. Table 1 in paper. Uses Gemini 2.5 Pro."""
    sub = case_df[(case_df["model"] == "GEMINI_2_5_PRO")
                  & (case_df["task"] == "Disambiguation")]
    rows_per_dom = []
    for dom in DOMAIN_ORDER:
        d = sub[sub["domain"] == dom]
        if d.empty:
            continue
        rows_per_dom.append(
            (dom, d["case_id"].nunique(), 100 * d["FR"].mean()))

    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"\textbf{Benchmark} & \textbf{Domain} & $n$ & \textbf{Resolution Rate} \\",
        r"\midrule",
        f"CLAMBER \\cite{{zhang2024clamber}} & General & {clamber_n} & {clamber_rate}\\,\\% \\\\",
        r"\midrule",
    ]
    for i, (dom, n, rate) in enumerate(rows_per_dom):
        prefix = (r"\multirow{4}{*}{\shortstack[l]{\textsc{CompSciClarBench}\\(ours)}}"
                  if i == 0 else "")
        lines.append(f"{prefix} & {dom} & {n} & {rate:.1f}\\,\\% \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out = out_dir / "table_clamber_vs_sciconv.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return out


# ---------- Ablation result-file readers (judge / prompt / simulator) -----

def _ablation_paths(root: Path) -> dict:
    """Map of ablation-name -> result-JSON path. Each result JSON contains a
    'cases' list where each case has a 'llm_scores' dict with FR/CGR/CP/CR/DR/IC/MC."""
    return {
        "judge_gpt5":         root / "judge_ablation" / "openai_gpt5_results.json",
        "judge_sonnet":       root / "judge_ablation" / "bedrock_4_6_sonnet_results.json",
        "judge_human":        root / "judge_ablation" / "human_annotation_form_annotated.json",
        "prompt_option_1":    root / "prompt_ablation" / "OPTION_1" / "paraphrased_option_1_results.json",
        "prompt_option_2":    root / "prompt_ablation" / "OPTION_2" / "paraphrased_option_2_results.json",
        "sim_gemini":         root / "user_llm_ablation" / "GEMINI_PRO" / "gemini_pro_results.json",
        "sim_gpt5":           root / "user_llm_ablation" / "GPT_5_2" / "openai_gpt5_results.json",
        "sim_sonnet":         root / "user_llm_ablation" / "SONNET_4_6" / "bedrock_4_6_sonnet_results.json",
    }


def _read_ablation_cases(path: Path) -> pd.DataFrame:
    """Read a {cases: [...]} ablation results JSON into a DataFrame with
    columns: case_id, component, domain, task, FR, CGR, SRR, CP, CR, DR, IC, MC."""
    if not path.exists():
        return pd.DataFrame()
    data = _load_json(path)
    rows = []
    for c in data.get("cases", []):
        sc = c.get("llm_scores", {})
        def _val(k): return sc.get(k, {}).get(
            "value", 0) if isinstance(sc.get(k), dict) else sc.get(k, 0)
        fr = int(_val("FR"))
        cgr = int(_val("CGR"))
        rows.append({
            "case_id":   c.get("case_id"),
            "component": c.get("component"),
            "domain":    c.get("domain"),
            "task":      c.get("task"),
            "FR":  fr,
            "CGR": cgr,
            "SRR": int(fr == 1 and cgr == 0),
            "CP":  float(_val("CP")),
            "CR":  float(_val("CR")),
            "DR":  float(_val("DR")),
            "IC":  float(_val("IC")),
            "MC":  float(_val("MC")),
        })
    return pd.DataFrame(rows)


def _agg_simple(df: pd.DataFrame) -> dict:
    """Mean each metric * 100."""
    if df.empty:
        return {k: float("nan") for k in ["FR", "CGR", "SRR", "IC", "MC", "CR", "CP"]}
    return {
        "n":   int(len(df)),
        "FR":  100 * df["FR"].mean(),
        "CGR": 100 * df["CGR"].mean(),
        "SRR": 100 * df["SRR"].mean(),
        "IC":  100 * df["IC"].mean(),
        "MC":  100 * df["MC"].mean(),
        "CR":  100 * df["CR"].mean(),
        "CP":  100 * df["CP"].mean(),
    }


# ---------- Tables that DO read from ablation JSONs -----------------------

def table_judge_agreement(root: Path, out_dir: Path) -> Path:
    """Judge-vs-human exact-match agreement on FRR and CGRR. Reads
    judge_ablation/{human, gpt5, sonnet}_results.json plus the standard
    Gemini judge JSONs in each case_dir referenced by the human file.
    Table 2 (a) in paper."""
    paths = _ablation_paths(root)
    human = _read_ablation_cases(paths["judge_human"])
    gpt5 = _read_ablation_cases(paths["judge_gpt5"])
    sonnet = _read_ablation_cases(paths["judge_sonnet"])
    # Gemini 2.5 Pro judge: load from each case's standard JSON via the
    # path stored in the human results file (each case has a "paths" dict).
    human_raw = _load_json(
        paths["judge_human"]) if paths["judge_human"].exists() else {"cases": []}
    gemini_rows = []
    for c in human_raw.get("cases", []):
        cd = Path(c.get("paths", {}).get("case_dir", ""))
        task = c.get("task", "Disambiguation")
        if not cd.is_dir():
            continue
        scorer = (_score_disambiguation if task == "Disambiguation"
                  else _score_inconsistency)
        # We need the dataset row for the demotion check; reconstruct from
        # planted_ground_truth fields stored in the human file.
        ds_row = {
            "incomplete_user_req": c.get("planted_ground_truth", {})
                                    .get("original_inconsistent_prompt")
            or c.get("planted_ground_truth", {})
            .get("original_incomplete_prompt", ""),
            "missing_entities": c.get("planted_ground_truth", {}).get("planted_missing_entities", []),
            "inconsistencies":  c.get("planted_ground_truth", {}).get("planted_inconsistencies", []),
        }
        sc = scorer(cd, 0, c.get("component", ""), c.get("model", ""),
                    c.get("case_id", ""), ds_row)
        if sc is None:
            continue
        gemini_rows.append({
            "case_id": c.get("case_id"),
            "FR": sc.FR,
            "CGR": int(sc.FR == 1 and sc.grounded == 1),
        })
    gemini = pd.DataFrame(gemini_rows)

    def agree(judge: pd.DataFrame, metric: str) -> tuple[int, float]:
        if judge.empty or human.empty:
            return 0, float("nan")
        m = judge.merge(human[["case_id", metric]], on="case_id",
                        suffixes=("_j", "_h"))
        if m.empty:
            return 0, float("nan")
        return len(m), 100 * (m[f"{metric}_j"] == m[f"{metric}_h"]).mean()

    rows = []
    for label, df in [("Gemini 2.5 Pro", gemini),
                      ("GPT-5.2", gpt5),
                      ("Sonnet 4.6", sonnet)]:
        n_fr, fr_agr = agree(df, "FR")
        n_cg, cg_agr = agree(df, "CGR")
        rows.append((label, n_fr, fr_agr, cg_agr))

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Judge} & $n$ & \textbf{FRR agreement} & \textbf{CGRR agreement} \\",
        r"\midrule",
    ]
    for label, n, fr_agr, cg_agr in rows:
        lines.append(
            f"{label} & {n} & {fr_agr:.1f}\\,\\% & {cg_agr:.1f}\\,\\% \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out = out_dir / "table_judge_agreement.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return out


def table_prompt_paraphrase(root: Path, out_dir: Path) -> Path:
    """Prompt-paraphrase ablation (Table 2 (b)). Reads
    prompt_ablation/OPTION_{1,2}/paraphrased_option_*_results.json plus the
    same 80-case subset under the original prompt (judge_ablation files)."""
    paths = _ablation_paths(root)
    # original judge run on same 80 cases
    original = _read_ablation_cases(paths["judge_gpt5"])
    if original.empty:
        original = _read_ablation_cases(paths["judge_sonnet"])
    opt1 = _read_ablation_cases(paths["prompt_option_1"])
    opt2 = _read_ablation_cases(paths["prompt_option_2"])

    rows = []
    for label, df in [("Original", original),
                      ("Variant A", opt1),
                      ("Variant B", opt2)]:
        if df.empty:
            continue
        a = _agg_simple(df)
        rows.append((label, a["n"], a["FR"], a["CGR"]))

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Variant} & $n$ & \textbf{FRR (\%)} & \textbf{CGRR (\%)} \\",
        r"\midrule",
    ]
    for label, n, fr, cg in rows:
        lines.append(f"{label} & {n} & {fr:.1f} & {cg:.1f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out = out_dir / "table_prompt_paraphrase.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return out


def table_simulator_ablation(root: Path, out_dir: Path) -> Path:
    """Simulator ablation (Table 2 (c) / Appendix). Reads the three result
    JSONs under user_llm_ablation/ and reports per-task FRR/CGRR/SRR/IC/MC/CR/CP."""
    paths = _ablation_paths(root)
    sims = [
        ("Claude Sonnet 4.6 (default)",
         _read_ablation_cases(paths["sim_sonnet"])),
        ("GPT-5.2",
         _read_ablation_cases(paths["sim_gpt5"])),
        ("Gemini 2.5 Pro",
         _read_ablation_cases(paths["sim_gemini"])),
    ]

    lines = [
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r"\textbf{Simulator} & \textbf{Task} & $n$ & FRR & CGRR & SRR "
        r"& IC & MC & CR & CP \\",
        r"\midrule",
    ]
    for label, df in sims:
        if df.empty:
            continue
        for task in ["Disambiguation", "Inconsistency"]:
            sub = df[df["task"] == task]
            a = _agg_simple(sub)
            row = (f"\\multirow{{3}}{{*}}{{\\textsc{{{label}}}}} & {task} "
                   if task == "Disambiguation" else f" & {task} ")
            lines.append(
                f"{row} & {a['n']} & {a['FR']:.1f} & {a['CGR']:.1f} & "
                f"{a['SRR']:.1f} & {a['IC']:.1f} & {a['MC']:.1f} & "
                f"{a['CR']:.1f} & {a['CP']:.1f} \\\\"
            )
        a = _agg_simple(df)
        lines.append(
            f" & Overall & {a['n']} & {a['FR']:.1f} & {a['CGR']:.1f} & "
            f"{a['SRR']:.1f} & {a['IC']:.1f} & {a['MC']:.1f} & "
            f"{a['CR']:.1f} & {a['CP']:.1f} \\\\"
        )
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out = out_dir / "table_simulator_ablation.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return out


def table_robustness_summary(root: Path, out_dir: Path) -> Path:
    """Compact robustness summary (Table 2). Computes the FRR/CGRR spread
    (max - min) across each ablation axis and writes one row per axis."""
    paths = _ablation_paths(root)

    judge_dfs = [_read_ablation_cases(paths[k])
                 for k in ("judge_gpt5", "judge_sonnet")]
    prompt_dfs = [_read_ablation_cases(paths[k])
                  for k in ("prompt_option_1", "prompt_option_2")]
    sim_dfs = [_read_ablation_cases(paths[k])
               for k in ("sim_gemini", "sim_gpt5", "sim_sonnet")]

    def spread(dfs):
        frs, cgrs = [], []
        for df in dfs:
            if df.empty:
                continue
            frs.append(100 * df["FR"].mean())
            cgrs.append(100 * df["CGR"].mean())
        if len(frs) < 2:
            return float("nan"), float("nan")
        return max(frs) - min(frs), max(cgrs) - min(cgrs)

    rows = [
        ("Judge model",       *spread(judge_dfs)),
        ("Prompt paraphrase", *spread(prompt_dfs)),
        ("User simulator",    *spread(sim_dfs)),
    ]

    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Variation & FRR spread (pp) & CGRR spread (pp) \\",
        r"\midrule",
    ]
    for label, fr_sp, cg_sp in rows:
        fr_str = f"{fr_sp:.1f}" if not math.isnan(fr_sp) else "--"
        cg_str = f"{cg_sp:.1f}" if not math.isnan(cg_sp) else "--"
        lines.append(f"{label} & {fr_str} & {cg_str} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out = out_dir / "table_robustness_summary.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return out


# ---------------------------------------------------------------------------
# 18. MAIN
# ---------------------------------------------------------------------------

ARTIFACTS: dict[str, str] = {
    "fig_3":  "Figure 3  -- dataset distribution donuts",
    "fig_4":  "Figure 4  -- main per-domain CGRR/SRR/FRR",
    "fig_5":  "Figure 5  -- per-ontology-component (per-issue) FRR(c)",
    "fig_6":  "Figure 6  -- Pareto radar (Cap/Rob/Use)",
    "fig_7":  "Figure 7  -- appendix general-numeric split",
    "fig_8":  "Figure 8  -- appendix tool-use split",
    "fig_microavg":      "Appendix per-domain micro-avg",
    "fig_guided_vs_un":  "Appendix guided vs unguided",
    "table_main":        "Appendix per-domain tables (Disambig + Incon)",
    "table_clamber":     "Table 1  -- CLAMBER vs CompSciClarBench",
    "table_robustness":  "Table 2  -- robustness summary (judge/prompt/simulator spreads)",
    "table_judge":       "Table 2(a) -- judge-vs-human agreement",
    "table_prompt":      "Table 2(b) -- prompt-paraphrase ablation",
    "table_simulator":   "Table 2(c) -- simulator ablation (per task + overall)",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help="ConvAgent/ benchmark root (contains fluids/, foam/, ...).")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Directory for figures.")
    p.add_argument("--tables", type=Path, default=DEFAULT_TABLES,
                   help="Directory for LaTeX tables.")
    p.add_argument("--only", type=str, default="",
                   help="Comma-separated subset of artifact keys; empty = all. "
                        "Keys: " + ", ".join(ARTIFACTS))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.tables.mkdir(parents=True, exist_ok=True)
    setup_style()

    todo = (set(k.strip() for k in args.only.split(",") if k.strip())
            if args.only else set(ARTIFACTS))
    print("Artifacts to generate:", sorted(todo))

    print("Building per-case dataframe ...")
    case_df = build_case_df(args.root)
    print(f"  case_df: {len(case_df):,} rows over "
          f"{case_df['component'].nunique()} components × "
          f"{case_df['model'].nunique()} models")

    need_entity = bool({"fig_5", "fig_microavg"} & todo)
    entity_df = build_entity_df(args.root) if need_entity else pd.DataFrame()
    if need_entity:
        print(f"  entity_df: {len(entity_df):,} rows")

    if "fig_3" in todo:
        print("→", figure_3_dataset_distribution(case_df, args.out))
    if "fig_4" in todo:
        print("→", figure_4_main_outcomes(case_df, args.out))
    if "fig_5" in todo:
        print("→", figure_5_ontology(entity_df, args.out))
    if "fig_6" in todo:
        print("→", figure_6_pareto(case_df, args.out))
    if "fig_7" in todo:
        print("→", figure_7_general_split(case_df, args.out))
    if "fig_8" in todo:
        print("→", figure_8_tooluse_split(case_df, args.out))
    if "fig_microavg" in todo:
        print("→", figure_appendix_per_domain(entity_df, args.out))
    if "fig_guided_vs_un" in todo:
        print("→", figure_appendix_guided_vs_unguided(
            args.root, case_df, args.out))

    if "table_main" in todo:
        print("→", table_main_macro(case_df, args.tables))
    if "table_clamber" in todo:
        print("→", table_clamber_vs_sciconv(case_df, args.tables))
    if "table_robustness" in todo:
        print("→", table_robustness_summary(args.root, args.tables))
    if "table_judge" in todo:
        print("→", table_judge_agreement(args.root, args.tables))
    if "table_prompt" in todo:
        print("→", table_prompt_paraphrase(args.root, args.tables))
    if "table_simulator" in todo:
        print("→", table_simulator_ablation(args.root, args.tables))

    print("done.")


if __name__ == "__main__":
    main()
