import os
import sys
import json
import time
import difflib
import argparse
from typing import Dict, List, Any, Optional

from config import Config
from utils import LLMService
import persona_users as pu

TEXT_KEYS = {
    "disambiguation": ("incomplete_user_req", "complete_user_req"),
    "inconsistency": ("inconsistent_prompt", "correct_prompt"),
}

SYSTEM_PROMPTS = {
    "disambiguation": """
        You build fact cards for a clarification benchmark.
        You are given an [INCOMPLETE_TEXT], the [COMPLETE_TEXT] it was derived from, and a numbered list of [ITEMS]:
        pieces of information that the complete text specifies and the incomplete text lacks.

        For EACH item:
        - "spans": the passage(s) of the [COMPLETE_TEXT] that specify this item, copied VERBATIM
          (character for character, same numbers, units and punctuation). Give every passage needed to
          state the item completely, and nothing unrelated. At least one span is required.
        - "statement": one sentence stating what the complete text specifies for this item.

        **OUTPUT FORMAT:**
        You must output your final answer as a JSON object enclosed EXACTLY within <FINAL_JSON> tags,
        with one entry per item, using the item's index.
        <FINAL_JSON>
        {
            "items": [
                {"index": 0, "spans": ["verbatim passage"], "statement": "One sentence."}
            ]
        }
        </FINAL_JSON>
        """,
    "inconsistency": """
        You build fact cards for a clarification benchmark.
        You are given an [INCONSISTENT_TEXT], the [CORRECT_TEXT] it was derived from, and a numbered list of [ITEMS]:
        inconsistencies that were planted in the inconsistent text.

        For EACH item:
        - "statement": one sentence giving the correction, i.e. what the [CORRECT_TEXT] says or does instead.
          Required.
        - "spans": the passage(s) of the [CORRECT_TEXT] the correction rests on, copied VERBATIM
          (character for character). If the correction is a deletion and no passage of the correct text
          supports it, give an empty list.

        **OUTPUT FORMAT:**
        You must output your final answer as a JSON object enclosed EXACTLY within <FINAL_JSON> tags,
        with one entry per item, using the item's index.
        <FINAL_JSON>
        {
            "items": [
                {"index": 0, "spans": [], "statement": "One sentence."}
            ]
        }
        </FINAL_JSON>
        """,
}

TAGS = {"disambiguation": ("INCOMPLETE_TEXT", "COMPLETE_TEXT"),
        "inconsistency": ("INCONSISTENT_TEXT", "CORRECT_TEXT")}


def build_user_prompt(mode: str, base_text: str, source_text: str, items: List[str],
                      errors: Optional[List[str]] = None) -> str:
    base_tag, src_tag = TAGS[mode]
    item_lines = "\n".join(f"[{i}] {it}" for i, it in enumerate(items))
    prompt = f"""
        <{base_tag}>
        {base_text}
        </{base_tag}>

        <{src_tag}>
        {source_text}
        </{src_tag}>

        <ITEMS>
        {item_lines}
        </ITEMS>
        """
    if errors:
        err_lines = "\n".join(f"- {e}" for e in errors)
        prompt += f"""
        Your previous answer failed these checks. Fix them:
        {err_lines}
        """
    return prompt


SPAN_IN_DIFF_MIN = 0.5  # below this share of a card's span tokens inside the diff, the span is too broad
SPAN_OVERLAP_MAX = 0.5  # two cards sharing more than this share of the shorter span's tokens overlap


def token_index(base_text: str, source_text: str) -> dict:
    """Source tokens, their character starts in the space-joined source, and the inserted/replaced ones."""
    a, b = base_text.split(), source_text.split()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    changed = [j for tag, i1, i2, j1, j2 in sm.get_opcodes()
               if tag in ("insert", "replace") for j in range(j1, j2)]
    starts, pos = [], 0
    for tok in b:
        starts.append(pos)
        pos += len(tok) + 1
    return {"tokens": b, "joined": " ".join(b), "starts": starts, "changed": changed}


def span_tokens(idx: dict, spans: List[str]) -> set:
    b, joined, starts = idx["tokens"], idx["joined"], idx["starts"]
    covered = set()
    for span in spans:
        q = pu.norm_ws(span)
        if not q:
            continue
        s = joined.find(q)
        while s != -1:
            e = s + len(q)
            for j, st in enumerate(starts):
                if st < e and st + len(b[j]) > s:
                    covered.add(j)
            s = joined.find(q, s + 1)
    return covered


def coverage(base_text: str, source_text: str, spans: List[str]) -> dict:
    """Word-level diff base -> source; share of inserted/replaced source tokens not inside any span."""
    idx = token_index(base_text, source_text)
    changed, covered = idx["changed"], span_tokens(idx, spans)
    uncovered = [idx["tokens"][j] for j in changed if j not in covered]
    return {"changed_tokens": len(changed), "uncovered_tokens": uncovered,
            "uncovered_share": (len(uncovered) / len(changed)) if changed else 0.0}


def review_spans(base_text: str, source_text: str, cards: List[dict]) -> None:
    """Sets needs_review (with review_reason) on ok cards whose span is too broad or overlaps another card's."""
    idx = token_index(base_text, source_text)
    changed = set(idx["changed"])
    toks = [span_tokens(idx, c["spans"]) for c in cards]
    for k, c in enumerate(cards):
        reasons = ["non-verbatim span(s) dropped"] if c["checks"]["dropped_spans"] else []
        if toks[k]:
            share = len(toks[k] & changed) / len(toks[k])
            c["checks"]["span_in_diff_share"] = share
            if share < SPAN_IN_DIFF_MIN:
                reasons.append(f"span too broad: {share:.0%} of its tokens are inserted/replaced")
        for m, other in enumerate(toks):
            if m != k and toks[k] and other and \
                    len(toks[k] & other) > SPAN_OVERLAP_MAX * min(len(toks[k]), len(other)):
                reasons.append(f"span overlaps card {cards[m]['index']}")
        c["review_reason"] = "; ".join(reasons) or None
        if c["status"] == "ok" and reasons:
            c["status"] = "needs_review"


def check_case(domain: str, mode: str, case: dict, llm_items: Optional[List[dict]],
               max_uncovered_share: float) -> dict:
    base_key, src_key = TEXT_KEYS[mode]
    base_text, source_text = case.get(base_key, ""), case.get(src_key, "")
    items = case.get(pu.ITEMS_KEY[mode], []) or []
    errors = []
    if llm_items is None:
        errors.append("no parseable <FINAL_JSON> block with an 'items' list")
        llm_items = []
    by_index = {}
    for entry in llm_items:
        try:
            by_index[int(entry.get("index"))] = entry
        except (TypeError, ValueError, AttributeError):
            continue

    cards, kept_all = [], []
    for i, item in enumerate(items):
        entry = by_index.get(i, {})
        if i not in by_index:
            errors.append(f"item {i}: no entry returned")
        raw_spans = entry.get("spans") or []
        if isinstance(raw_spans, str):
            raw_spans = [raw_spans]
        spans = [s for s in raw_spans if isinstance(s, str) and pu.is_verbatim(s, source_text)]
        dropped = [s for s in raw_spans if not (isinstance(s, str) and pu.is_verbatim(s, source_text))]
        statement = entry.get("statement") or ""
        statement = statement.strip() if isinstance(statement, str) else ""
        for s in dropped:
            errors.append(f"item {i}: span is not a verbatim substring of the {TAGS[mode][1]}: {s!r}")
        if mode == "disambiguation":
            failed = not spans
            if failed:
                errors.append(f"item {i}: needs at least one verbatim span")
            text = " ".join(spans)
        else:
            failed = not statement
            if failed:
                errors.append(f"item {i}: statement is required")
            text = statement
        status = "failed" if failed else ("needs_review" if dropped else "ok")
        kept_all.extend(spans)
        cards.append({
            "card_id": f"{domain}/{mode}/{case['id']}#{i}",
            "case_id": case["id"], "index": i, "item": item,
            "spans": spans, "statement": statement, "text": text,
            "checks": {"dropped_spans": dropped, "n_spans": len(spans),
                       "has_statement": bool(statement)},
            "status": status,
        })

    if mode == "disambiguation":
        review_spans(base_text, source_text, cards)
    cov = coverage(base_text, source_text, kept_all)
    high_uncovered = mode == "disambiguation" and cov["uncovered_share"] > max_uncovered_share
    if high_uncovered:
        errors.append("these tokens inserted in the complete text are not covered by any span: "
                      + " ".join(cov["uncovered_tokens"][:60]))
    if not by_index or any(c["status"] == "failed" for c in cards):
        status = "failed"
    elif high_uncovered or any(c["status"] == "needs_review" for c in cards):
        status = "needs_review"
    else:
        status = "ok"
    return {"status": status, "cards": cards, "uncovered_tokens": cov["uncovered_tokens"],
            "uncovered_share": cov["uncovered_share"], "changed_tokens": cov["changed_tokens"],
            "errors": errors, "llm_items": llm_items}


def build_case(llm: LLMService, domain: str, mode: str, case: dict, max_uncovered_share: float) -> dict:
    base_key, src_key = TEXT_KEYS[mode]
    items = case.get(pu.ITEMS_KEY[mode], []) or []
    errors, rec = None, None
    for attempt in (1, 2):
        user_prompt = build_user_prompt(mode, case.get(base_key, ""), case.get(src_key, ""), items, errors)
        response = llm.invoke(user_prompt, SYSTEM_PROMPTS[mode])
        parsed = pu.parse_final_json(response)
        llm_items = parsed.get("items") if parsed and isinstance(parsed.get("items"), list) else None
        rec = check_case(domain, mode, case, llm_items, max_uncovered_share)
        rec["attempts"] = attempt
        if not rec["errors"]:
            break
        errors = rec["errors"]
    return rec


def write_json_atomic(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build and verify persona-study fact cards")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--mode", required=True, choices=list(TEXT_KEYS))
    parser.add_argument("--model", required=True, choices=list(Config.LLM_PROFILES))
    parser.add_argument("--cases", default=None, help="Comma-separated case ids (default: all)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--verify_only", action="store_true",
                        help="Re-run the deterministic checks on stored LLM output; no LLM calls")
    parser.add_argument("--max_uncovered_share", type=float, default=0.5,
                        help="Disambiguation cases above this uncovered share are needs_review")
    args = parser.parse_args(argv)

    out = args.out or pu.default_cards_path(args.domain, args.mode)
    src = pu.data_path(args.domain, args.mode)
    cases = pu.load_cases(args.domain, args.mode)
    wanted = args.cases.split(",") if args.cases else list(cases)
    meta = {"model": pu.role_config(args.model).as_dict(),
            "prompt_sha256": pu.sha256_text(SYSTEM_PROMPTS[args.mode]),
            "source_file": os.path.relpath(src, pu.REPO_DIR),
            "source_sha256": pu.sha256_file(src),
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}

    data = {"meta": meta, "cases": {}}
    if os.path.exists(out):
        with open(out) as f:
            data = json.load(f)
        old = data["meta"]
        for k in ("prompt_sha256", "source_sha256"):
            if old.get(k) != meta[k]:
                sys.exit(f"{out}: {k} differs from the current one; refusing to mix cards")
        if not args.verify_only and old.get("model") != meta["model"]:
            sys.exit(f"{out}: built with {old.get('model')}, not {meta['model']}; refusing to mix cards")

    if args.verify_only:
        for cid, rec in data["cases"].items():
            new = check_case(args.domain, args.mode, cases[cid], rec.get("llm_items"),
                             args.max_uncovered_share)
            new["attempts"] = rec.get("attempts")
            new["usage"] = rec.get("usage")
            data["cases"][cid] = new
        write_json_atomic(out, data)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        for cid in wanted:
            if data["cases"].get(cid, {}).get("status") == "ok":
                continue
            llm = LLMService(pu.role_config(args.model))
            try:
                rec = build_case(llm, args.domain, args.mode, cases[cid], args.max_uncovered_share)
            except Exception as e:
                rec = {"status": "failed", "cards": [], "errors": [f"LLM call failed: {e}"]}
            rec["usage"] = llm.get_statistics()
            data["cases"][cid] = rec
            write_json_atomic(out, data)
            print(f"{cid}: {rec['status']}")

    counts = {}
    for rec in data["cases"].values():
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
    print(f"{out}: {counts}")
    return data


if __name__ == "__main__":
    main()
