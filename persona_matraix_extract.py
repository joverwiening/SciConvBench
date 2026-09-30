"""Extract MatrAIx full_persona_v2 persona blocks (verbatim) from PersonaScienceBench and draw the mx1 selection."""
import argparse, hashlib, json, os, random, re

QUOTAS = {"real_human_survey": 2, "synthetic": 2, "wiki": 2, "stackoverflow": 2, "prism": 1, "amazon": 1}
FENCE = re.compile(r"```text\n(.*?)\n```", re.S)


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psb", required=True)
    ap.add_argument("--psb_profiles_commit", required=True)
    ap.add_argument("--psb_cohort_commit", required=True)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "persona", "matraix"))
    a = ap.parse_args()
    cohort = json.load(open(os.path.join(a.psb, "configs", "full_persona_v2", "cohort.json")))
    os.makedirs(os.path.join(a.out, "personas"), exist_ok=True)
    meta = {"psb_profiles_commit": a.psb_profiles_commit, "psb_cohort_commit": a.psb_cohort_commit,
            "profiles_dir": "reports/full_persona_v2/profiles", "personas": {}}
    for p in cohort["profiles"]:
        cid = p["candidate_id"]
        md = open(os.path.join(a.psb, "reports", "full_persona_v2", "profiles", f"{cid}.md"), encoding="utf-8").read()
        blocks = FENCE.findall(md)
        assert len(blocks) == 1, (cid, len(blocks))
        text = blocks[0]
        with open(os.path.join(a.out, "personas", f"{cid}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        meta["personas"][cid] = {"source": p["source"], "chars": len(text), "sha256": sha(text),
                                 "cohort_prompt_characters": p["prompt_characters"],
                                 "cohort_prompt_sha256": p["prompt_sha256"],
                                 "sha_match": p["prompt_sha256"] in (sha(text), sha(text + "\n"), sha(text.strip()))}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    rng = random.Random(a.seed)
    by_src = {}
    for cid, m in sorted(meta["personas"].items()):
        by_src.setdefault(m["source"], []).append(cid)
    picked = []
    for src, k in QUOTAS.items():
        picked += sorted(rng.sample(by_src[src], k))
    json.dump({"seed": a.seed, "quotas": QUOTAS, "selected": picked}, open(os.path.join(a.out, "selection.json"), "w"),
              indent=2)
    for cid in picked:
        m = meta["personas"][cid]
        print(cid, m["source"], m["chars"], "cohort_chars", m["cohort_prompt_characters"], "sha_match", m["sha_match"])


if __name__ == "__main__":
    main()
