import os
import sys
import csv
import glob
import json
import random
import difflib
import argparse

import numpy as np

import persona_users as pu
import persona_eval as pev

ATTR_FILE, FR_FILE, KEY_FILE = "attribution_sample.csv", "fr_sample.csv", "key.json"
ATTR_COLS = ["row_id", "question", "answer", "card_item", "card_text", "label_a", "asked_a", "label_b", "asked_b"]
FR_COLS = ["row_id", "entity", "complete_requirement", "final_spec", "label_a", "label_b"]
LEVELS = ("full", "partial", "none")
FR_LABELS = ("1", "0")
ASKED_LABELS = ("1", "0")


def read_text(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def done_case_dirs(run_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(run_dir, "*", "rep*", "*", "*", "*", "persona_run.json"))):
        run = pev.read_json(p) or {}
        if run.get("status") == "done":
            out.append((os.path.dirname(p), run))
    return out


def load_run_cards(run_dir):
    manifest = pev.read_json(os.path.join(run_dir, "manifest.json"))
    if manifest is None:
        sys.exit(f"No manifest.json in {run_dir}")
    cards = {}
    for key, entry in (manifest.get("cards") or {}).items():
        if entry:
            with open(entry["path"]) as f:
                cards[key] = {"data": json.load(f), "sha256": pu.sha256_file(entry["path"])}
    return cards


def collect_pools(run_dir):
    """Per arm: attribution turn units and FR item units, with everything needed for the rows."""
    cards_by_key = load_run_cards(run_dir)
    datasets = {}
    attr_pool, fr_pool = {}, {}
    skipped = {"attribution_cards_sha_mismatch": 0, "attribution_turn_mismatch": 0, "fr_no_final_spec": 0,
               "fr_no_judge_file": 0}
    for folder, run in done_case_dirs(run_dir):
        domain, mode, case_id, arm = run["domain"], run["mode"], run["case_id"], run["arm"]
        if (domain, mode) not in datasets:
            datasets[(domain, mode)] = pu.load_cases(domain, mode)
        row = datasets[(domain, mode)][case_id]
        rel = os.path.relpath(folder, run_dir)
        base = {"arm": arm, "rep": run["rep"], "case_dir": rel, "domain": domain, "mode": mode, "case_id": case_id}

        attribution = pev.read_json(os.path.join(folder, "persona_attribution.json"))
        cards_rec = cards_by_key.get(f"{domain}/{mode}")
        if attribution is not None and cards_rec is not None:
            if attribution.get("cards_sha256") != cards_rec["sha256"]:
                skipped["attribution_cards_sha_mismatch"] += 1
            else:
                cards = [c for c in cards_rec["data"]["cases"].get(case_id, {}).get("cards") or [] if c.get("text")]
                turns, _ = pev.dialogue_turns(folder)
                if len(turns) != len(attribution["turns"]):
                    skipped["attribution_turn_mismatch"] += 1
                elif cards:
                    for k, (d, t) in enumerate(zip(turns, attribution["turns"]), 1):
                        attr_pool.setdefault(arm, []).append(dict(base, turn=k, question=d["question"],
                                                                  answer=d["answer"], cards=cards, labels=t))

        final_spec = read_text(os.path.join(folder, "generated_prompt.txt"))
        if not (final_spec or "").strip():
            skipped["fr_no_final_spec"] += 1
            continue
        fr_file = pev.read_json(os.path.join(folder, pev.NATIVE_FILES[mode][0]))
        if fr_file is None:
            skipped["fr_no_judge_file"] += 1
            continue
        items = row.get(pu.ITEMS_KEY[mode], []) or []
        if mode == "disambiguation":
            recovered = [m is not None for m in pev.match_entities(items, fr_file.get("recovered_entities", []) or [])]
        else:
            bd = fr_file.get("breakdown", []) or []
            recovered = [bool(bd[i].get("resolved") is True) if i < len(bd) else False for i in range(len(items))]
        complete = read_text(os.path.join(folder, "complete_prompt.txt")) or ""
        for i, item in enumerate(items):
            fr_pool.setdefault(arm, []).append(dict(base, index=i, entity=item, complete_requirement=complete,
                                                    final_spec=final_spec, model_label=int(recovered[i])))
    return attr_pool, fr_pool, skipped


def stratified(pool, n_per_arm, rng):
    out = {}
    for arm in sorted(pool):
        units = pool[arm]
        out[arm] = rng.sample(units, min(n_per_arm, len(units)))
    return out


def write_csv(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def export(run_dir, n_per_arm, seed, out):
    rng = random.Random(seed)
    attr_pool, fr_pool, skipped = collect_pools(run_dir)
    attr_sample = stratified(attr_pool, n_per_arm, rng)
    fr_sample = stratified(fr_pool, n_per_arm, rng)

    attr_rows = []
    for arm, units in attr_sample.items():
        for u in units:
            conveyed, asked = u["labels"].get("conveyed"), u["labels"].get("asked")
            for c in u["cards"]:
                # None = the judge output for this turn could not be parsed; such rows are not scored
                level = None if conveyed is None else (conveyed.get(c["card_id"], {}).get("level") or "none")
                model_asked = None if asked is None else str(int(c["card_id"] in asked))
                meta = {k: u[k] for k in ("arm", "rep", "case_dir", "domain", "mode", "case_id", "turn")}
                attr_rows.append(({"question": u["question"], "answer": u["answer"], "card_item": c["item"],
                                   "card_text": c["text"]},
                                  dict(meta, card_id=c["card_id"], card_index=c["index"], model_label=level,
                                       model_asked=model_asked,
                                       attribution_parse_failure=bool(u["labels"].get("parse_failure")),
                                       ask_parse_failure=bool(u["labels"].get("ask_parse_failure")))))
    fr_rows = []
    for arm, units in fr_sample.items():
        for u in units:
            meta = {k: u[k] for k in ("arm", "rep", "case_dir", "domain", "mode", "case_id", "index", "model_label")}
            fr_rows.append(({"entity": u["entity"], "complete_requirement": u["complete_requirement"],
                             "final_spec": u["final_spec"]}, meta))

    key = {}
    rng.shuffle(attr_rows)
    rng.shuffle(fr_rows)
    for prefix, rows in (("A", attr_rows), ("F", fr_rows)):
        for k, (cells, meta) in enumerate(rows, 1):
            rid = f"{prefix}{k:05d}"
            cells["row_id"] = rid
            key[rid] = dict(meta, task="attribution" if prefix == "A" else "fr")

    os.makedirs(out, exist_ok=True)
    write_csv(os.path.join(out, ATTR_FILE), ATTR_COLS, [c for c, _ in attr_rows])
    write_csv(os.path.join(out, FR_FILE), FR_COLS, [c for c, _ in fr_rows])
    pev.write_json(os.path.join(out, KEY_FILE), key)
    counts = {"attribution_turns": {a: len(u) for a, u in attr_sample.items()},
              "attribution_rows": len(attr_rows),
              "fr_items": {a: len(u) for a, u in fr_sample.items()},
              "skipped": skipped}
    print(json.dumps(counts, indent=2))
    return counts


# ---- fact-card audit sheet ----

def word_diff(base_text, source_text):
    a, b = base_text.split(), source_text.split()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    parts = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            parts.append(" ".join(a[i1:i2]))
        if tag in ("delete", "replace"):
            parts.append("[-" + " ".join(a[i1:i2]) + "-]")
        if tag in ("insert", "replace"):
            parts.append("{+" + " ".join(b[j1:j2]) + "+}")
    return " ".join(parts)


def export_cards(domain, mode, n, seed, cards_path, out):
    import persona_fact_cards as pfc
    cards_path = cards_path or pu.default_cards_path(domain, mode)
    with open(cards_path) as f:
        data = json.load(f)
    cases = pu.load_cases(domain, mode)
    ids = sorted(data["cases"])
    sampled = random.Random(seed).sample(ids, min(n, len(ids)))
    base_key, src_key = pfc.TEXT_KEYS[mode]
    lines = [f"# Fact-card audit: {domain} / {mode}", "",
             f"cards: `{cards_path}`, sha256 `{pu.sha256_file(cards_path)}`, n={len(sampled)}, seed={seed}", "",
             "Diff marks: `[-deleted-]` `{+inserted+}` (" + base_key + " -> " + src_key + ").", ""]
    for cid in sampled:
        rec, case = data["cases"][cid], cases[cid]
        lines += [f"## {cid}", "",
                  f"case status: {rec.get('status')}, uncovered_share: {rec.get('uncovered_share')}", "",
                  "Items:", ""]
        lines += [f"{i}. {it}" for i, it in enumerate(case.get(pu.ITEMS_KEY[mode], []) or [])]
        lines += ["", "Diff:", "", "```text", word_diff(case.get(base_key, ""), case.get(src_key, "")), "```", ""]
        for c in rec.get("cards") or []:
            lines += [f"### card {c['index']}: {c['item']}", "",
                      f"- status: {c.get('status')}",
                      f"- uncovered_share (case): {rec.get('uncovered_share')}",
                      f"- statement: {c.get('statement')}",
                      "- spans:"]
            lines += [f"  - `{s}`" for s in c.get("spans") or []] or ["  - (none)"]
            lines += ["", "verdict: correct / wrong / incomplete", "", "note:", ""]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{out}: {len(sampled)} cases")
    return sampled


# ---- scoring ----

def cohen_kappa(a, b, labels):
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    if pe == 1:
        return None  # a single class used by both raters: kappa undefined
    return (po - pe) / (1 - pe)


def arm_vs_neutral(errors, B=10000, seed=0, ref="neutral"):
    """errors: [(arm, case, 0/1)]. Per arm: error rate minus the ref arm's, with a 95% bootstrap CI that
    resamples cases (all rows of a drawn case, in every arm), not rows."""
    arms = sorted({a for a, _, _ in errors})
    if ref not in arms or len(arms) < 2:
        return None
    cases = sorted({c for _, c, _ in errors})
    ai, ci = {a: k for k, a in enumerate(arms)}, {c: k for k, c in enumerate(cases)}
    s, n = np.zeros((len(arms), len(cases))), np.zeros((len(arms), len(cases)))
    for a, c, e in errors:
        s[ai[a], ci[c]] += e
        n[ai[a], ci[c]] += 1
    rng = np.random.default_rng(seed)
    w = np.zeros((B, len(cases)))
    np.add.at(w, (np.arange(B)[:, None], rng.integers(0, len(cases), size=(B, len(cases)))), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = (w @ s.T) / (w @ n.T)  # B x arms; nan where a draw has no rows of an arm
    rates = s.sum(axis=1) / n.sum(axis=1)
    r = ai[ref]
    out = {}
    for a in arms:
        if a == ref:
            continue
        d = boot[:, ai[a]] - boot[:, r]
        d = d[~np.isnan(d)]
        lo, hi = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) if len(d) else (None, None)
        out[a] = {"diff": float(rates[ai[a]] - rates[r]), "error_rate": float(rates[ai[a]]),
                  "ref_error_rate": float(rates[r]), "n": int(n[ai[a]].sum()), "n_ref": int(n[r].sum()),
                  "n_cases": len(cases), "ci_lo": lo, "ci_hi": hi,
                  "ci_excludes_0": lo is not None and (lo > 0 or hi < 0), "B": B, "B_valid": len(d)}
    return out


def labelled_rows(path, key, allowed, task, cols=("label_a", "label_b"), model_key="model_label"):
    rows = read_csv(path)
    out, bad = [], []
    for r in rows:
        rid = r["row_id"]
        if rid not in key or key[rid]["task"] != task:
            sys.exit(f"{path}: row {rid} not in the key")
        la, lb = (r.get(cols[0]) or "").strip().lower(), (r.get(cols[1]) or "").strip().lower()
        for v in (la, lb):
            if v and v not in allowed:
                bad.append(f"{rid}={v!r}")
        k = key[rid]
        model = k.get(model_key)
        out.append({"row_id": rid, "a": la, "b": lb, "ref": la or lb, "arm": k["arm"],
                    "case": (k["domain"], k["mode"], k["case_id"]), "model": None if model is None else str(model)})
    if bad:
        sys.exit(f"{path}: invalid labels (allowed {allowed}): {', '.join(bad[:20])}")
    return out, len(rows)


def prec_rec(rows, labels):
    out = {}
    for c in labels:
        tp = sum(r["model"] == c and r["ref"] == c for r in rows)
        pm = sum(r["model"] == c for r in rows)
        pr = sum(r["ref"] == c for r in rows)
        out[c] = {"precision": tp / pm if pm else None, "recall": tp / pr if pr else None,
                  "n_model": pm, "n_ref": pr}
    return out


def score_task(rows, n_total, labels, binary_fn, B, seed):
    lab = [r for r in rows if r["ref"] and r["model"] is not None]
    by_arm = {}
    for r in lab:
        by_arm.setdefault(r["arm"], []).append(r)
    per_arm = {}
    for arm, rs in sorted(by_arm.items()):
        d = {"n": len(rs), "accuracy": sum(r["model"] == r["ref"] for r in rs) / len(rs)}
        if binary_fn is not None:
            d["binary_accuracy"] = sum(binary_fn(r["model"]) == binary_fn(r["ref"]) for r in rs) / len(rs)
            d["per_class"] = prec_rec(rs, labels)
        per_arm[arm] = d
    overall = {"n": len(lab), "accuracy": (sum(r["model"] == r["ref"] for r in lab) / len(lab)) if lab else None}
    if binary_fn is not None and lab:
        overall["binary_accuracy"] = sum(binary_fn(r["model"]) == binary_fn(r["ref"]) for r in lab) / len(lab)
        overall["per_class"] = prec_rec(lab, labels)
    both = [r for r in rows if r["a"] and r["b"]]
    kappa = {"n": len(both), "kappa": cohen_kappa([r["a"] for r in both], [r["b"] for r in both], labels)}
    to_bin = binary_fn or (lambda x: x)
    errors = [(r["arm"], r["case"], int(to_bin(r["model"]) != to_bin(r["ref"]))) for r in lab]
    return {"n_rows": n_total, "n_labelled": len(lab), "per_arm": per_arm, "overall": overall,
            "kappa_a_b": kappa, "arm_vs_neutral": arm_vs_neutral(errors, B, seed)}


def fmt(x):
    return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def render_markdown(res):
    lines = ["# Judge validation", ""]
    for task, title in (("attribution", "Attribution judge"), ("attribution_asked", "Ask judge"), ("fr", "FR judge")):
        t = res.get(task)
        if t is None:
            continue
        binary = task == "attribution"
        cols = "asked_a, else asked_b" if task == "attribution_asked" else "label_a, else label_b"
        lines += [f"## {title}", "", f"labelled {t['n_labelled']} / {t['n_rows']} rows (rows whose judge output "
                                     f"was unparsed are excluded). Reference = {cols}.", ""]
        head = "| arm | n | accuracy |" + (" binary acc (full vs not) |" if binary else "")
        lines += [head, "|" + "---|" * (head.count("|") - 1)]
        for arm, d in list(t["per_arm"].items()) + [("ALL", t["overall"])]:
            lines.append(f"| {arm} | {d['n']} | {fmt(d['accuracy'])} |"
                         + (f" {fmt(d.get('binary_accuracy'))} |" if binary else ""))
        if binary:
            lines += ["", "Per class (precision / recall, model vs reference):", "",
                      "| arm | " + " | ".join(LEVELS) + " |", "|---|" + "---|" * len(LEVELS)]
            for arm, d in list(t["per_arm"].items()) + [("ALL", t["overall"])]:
                pc = d.get("per_class") or {}
                lines.append(f"| {arm} | " + " | ".join(
                    f"{fmt(pc.get(c, {}).get('precision'))} / {fmt(pc.get(c, {}).get('recall'))}" for c in LEVELS) + " |")
        k = t["kappa_a_b"]
        lines += ["", f"Cohen's kappa label_a vs label_b ({'3-class' if binary else 'binary'}): "
                      f"{fmt(k['kappa'])} (n={k['n']})", ""]
        m = t["arm_vs_neutral"]
        if m is None:
            lines += ["Arm vs neutral error-rate difference: no labelled `neutral` arm or no other arm.", ""]
        else:
            lines += [f"Model error-rate difference arm - neutral{' (full vs not)' if binary else ''}, "
                      f"95% bootstrap CI resampling cases:", "",
                      "| arm | error rate | neutral | diff | 95% CI | CI excludes 0 |", "|---|---|---|---|---|---|"]
            for arm, d in m.items():
                lines.append(f"| {arm} | {fmt(d['error_rate'])} | {fmt(d['ref_error_rate'])} | {fmt(d['diff'])} | "
                             f"[{fmt(d['ci_lo'])}, {fmt(d['ci_hi'])}] | {d['ci_excludes_0']} |")
            lines.append("")
    return "\n".join(lines) + "\n"


def score(export_dir, key_path=None, B=10000, seed=0):
    key = pev.read_json(key_path or os.path.join(export_dir, KEY_FILE))
    if key is None:
        sys.exit("key.json not found or unreadable")
    res = {}
    path = os.path.join(export_dir, ATTR_FILE)
    if os.path.exists(path):
        rows, n = labelled_rows(path, key, LEVELS, "attribution")
        res["attribution"] = score_task(rows, n, LEVELS, lambda x: x == "full", B, seed)
        rows, n = labelled_rows(path, key, ASKED_LABELS, "attribution", ("asked_a", "asked_b"), "model_asked")
        res["attribution_asked"] = score_task(rows, n, ASKED_LABELS, None, B, seed)
    path = os.path.join(export_dir, FR_FILE)
    if os.path.exists(path):
        rows, n = labelled_rows(path, key, FR_LABELS, "fr")
        res["fr"] = score_task(rows, n, FR_LABELS, None, B, seed)
    pev.write_json(os.path.join(export_dir, "validation_scores.json"), res)
    with open(os.path.join(export_dir, "validation_report.md"), "w") as f:
        f.write(render_markdown(res))
    return res


def main(argv=None):
    parser = argparse.ArgumentParser(description="Blind hand-validation of the persona-study judges and fact cards")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export", help="Blind attribution / FR samples stratified by arm")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--n_per_arm", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p = sub.add_parser("export_cards", help="Markdown audit sheet for the fact cards")
    p.add_argument("--domain", required=True)
    p.add_argument("--mode", required=True, choices=list(pu.ITEMS_KEY))
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cards", default=None)
    p.add_argument("--out", required=True)
    p = sub.add_parser("score", help="Score filled samples against the key")
    p.add_argument("--dir", required=True)
    p.add_argument("--key", default=None)
    args = parser.parse_args(argv)

    if args.cmd == "export":
        return export(args.run_dir, args.n_per_arm, args.seed, args.out)
    if args.cmd == "export_cards":
        return export_cards(args.domain, args.mode, args.n, args.seed, args.cards, args.out)
    res = score(args.dir, args.key)
    print(render_markdown(res))
    return res


if __name__ == "__main__":
    main()
