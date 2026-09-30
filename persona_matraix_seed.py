"""Seed a MatrAIx run with the bf_neutral family case dirs and results rows of an earlier run (file copy only)."""
import os
import sys
import json
import time
import argparse
import subprocess

import persona_users as pu
import persona_runner as pr

NEUTRAL_ARMS = ("bf_neutral", "bf_neutral_b", "bf_neutral_c", "bf_neutral_d", "bf_neutral_e", "bf_neutral_f")


def check_hashes(src_manifest):
    """The copied dialogues are only a valid baseline if the neutral personas and user template are unchanged."""
    now = pr.persona_hashes(pu.load_personas(), pu.load_bigfive(), pu.load_matraix())
    old = src_manifest["sha256"]["personas"]
    bad = [k for k in ("bigfive/shared",) + NEUTRAL_ARMS if old.get(k) != now.get(k)]
    if src_manifest["sha256"]["user_prompt_template"] != pr.user_prompt_template_sha():
        bad.append("user_prompt_template")
    return bad


def seed_baseline(src_run, dst_run, cases, domain="foam", mode="disambiguation", agent_profile="bedrock_glm5",
                  rep=1):
    manifest_path = os.path.join(src_run, "manifest.json")
    with open(manifest_path) as f:
        src_manifest = json.load(f)
    bad = check_hashes(src_manifest)
    if bad:
        sys.exit(f"{manifest_path}: hashes differ from the current code for {bad}; not seeding")
    copied, missing = [], []
    for arm in NEUTRAL_ARMS:
        for cid in cases:
            src = pr.case_dir(src_run, arm, rep, domain, agent_profile, mode, cid)
            dst = pr.case_dir(dst_run, arm, rep, domain, agent_profile, mode, cid)
            if not os.path.isdir(src):
                missing.append(f"{arm}/{cid}")
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            subprocess.run(["rsync", "-a", "--no-g", src + "/", dst + "/"], check=True)
            copied.append(f"{arm}/{cid}")

    want = {tuple(x.split("/")) for x in copied}
    src_rows = []
    with open(os.path.join(src_run, "results.jsonl")) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if (r["arm"], r["case_id"]) in want and r["rep"] == rep and r["domain"] == domain and r["mode"] == mode:
                    src_rows.append(r)
    keys = {(r["arm"], r["rep"], r["domain"], r["mode"], r["case_id"]) for r in src_rows}
    dst_path = os.path.join(dst_run, "results.jsonl")
    rows = []
    if os.path.exists(dst_path):
        with open(dst_path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
    rows = [r for r in rows if (r["arm"], r["rep"], r["domain"], r["mode"], r["case_id"]) not in keys] + src_rows
    rows.sort(key=lambda r: (r["arm"], r["rep"], r["domain"], r["mode"], r["case_id"]))
    os.makedirs(dst_run, exist_ok=True)
    with open(dst_path + ".tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(dst_path + ".tmp", dst_path)

    prov = {"src_run": os.path.abspath(src_run), "src_manifest_sha256": pu.sha256_file(manifest_path),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "arms": list(NEUTRAL_ARMS), "rep": rep,
            "domain": domain, "mode": mode, "agent_profile": agent_profile, "cases": list(cases),
            "copied": copied, "missing": missing, "results_rows_copied": len(src_rows)}
    with open(os.path.join(dst_run, "baseline_provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    return prov


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sb = sub.add_parser("seed-baseline")
    sb.add_argument("--src_run", required=True)
    sb.add_argument("--dst_run", required=True)
    sb.add_argument("--cases", required=True, help="Comma list or @file")
    a = ap.parse_args(argv)
    prov = seed_baseline(a.src_run, a.dst_run, pr.parse_cases(a.cases))
    print(f"copied {len(prov['copied'])}, missing {len(prov['missing'])}, "
          f"results rows {prov['results_rows_copied']}")
    for m in prov["missing"]:
        print(f"  missing {m}")
    return prov


if __name__ == "__main__":
    main()
