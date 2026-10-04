"""tone3 first-order checks with GLM-5 (Bedrock), on scientist replies as the agent saw them (clarifications.json).
(a) pairwise: which of two chats' scientists is more disagreeable (per case, A/B order randomised with a fixed seed).
(b) remark labels per reply: remark on the question/assistant, or wanting the conversation to end."""
import os, sys, json, glob, math, random, argparse, threading, collections
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
RUNS = os.environ.get('PILOT_RUNS', 'runs')
PRICE_IN, PRICE_OUT = 1.2e-6, 3.2e-6
SEED = 0
CHUNK = 10

os.environ.setdefault('AWS_DEFAULT_REGION', 'us-west-2')
os.environ['SCICONV_PROFILE'] = 'bedrock_glm5'
sys.path.insert(0, REPO)
from config import Config  # noqa: E402
from utils import LLMService  # noqa: E402

PAIR_SYSTEM = 'You rate the tone of scientists in conversations with an AI assistant. Answer with JSON only.'
PAIR_PROMPT = """Below are the replies a scientist gave to an AI assistant's clarifying questions, from two separate conversations (scientist A and scientist B). Each reply is shown exactly as the assistant saw it.

Scientist A's replies:
{a}

Scientist B's replies:
{b}

Which scientist is more disagreeable (cold, blunt, hostile)?
Answer with JSON only: {{"more_disagreeable": "A"}} or {{"more_disagreeable": "B"}}"""

REMARK_SYSTEM = 'You label replies a scientist gave to an AI assistant. Answer with JSON only.'
REMARK_PROMPT = """Below are replies a scientist gave to an AI assistant's clarifying questions in one conversation, shown exactly as the assistant saw them.

{replies}

For each reply, decide: does it contain a remark that comments on the question or the assistant (that the question is unnecessary, obvious or basic, that the assistant should know it, that it was already given), or that signals wanting the conversation to end or the assistant to stop asking?
If yes, quote the remark verbatim; if no, use an empty quote.
Answer with JSON only, one entry per reply, in order:
{{"labels": [{{"reply": 1, "remark": true, "quote": "..."}}, ...]}}"""


def fmt(answers):
    return '\n'.join(f'{i}. {a}' for i, a in enumerate(answers, 1))


def chat_dir(run, arm, case):
    return f'{run}/{arm}/rep1/foam/bedrock_glm5_Disambiguation/{case}'


def answers(run, arm, case):
    try:
        return [t['answer'] for t in json.load(open(chat_dir(run, arm, case) + '/clarifications.json'))]
    except Exception:
        return None


def parse_json(text):
    i, j = text.find('{'), text.rfind('}')
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(text[i:j + 1])
    except Exception:
        return None


class Spend:
    def __init__(self, cap):
        self.cap, self.usd, self.calls, self.lock = cap, 0.0, 0, threading.Lock()

    def add(self, tin, tout):
        with self.lock:
            self.usd += tin * PRICE_IN + tout * PRICE_OUT
            self.calls += 1

    def over(self):
        with self.lock:
            return self.usd >= self.cap


def call(prompt, system, spend):
    """one judge call (one retry on unparseable output); returns (parsed, raw, tokens_in, tokens_out)."""
    llm = LLMService(Config)
    raw, parsed = None, None
    try:
        for _ in range(2):
            if spend.over():
                break
            tin0, tout0 = llm.api_input_tokens, llm.api_output_tokens
            try:
                raw = llm.invoke(prompt, system)
            finally:
                spend.add(llm.api_input_tokens - tin0, llm.api_output_tokens - tout0)
            parsed = parse_json(raw)
            if parsed is not None:
                break
    except Exception as e:
        raw = f'error: {type(e).__name__}: {e}'
    return parsed, raw, llm.api_input_tokens, llm.api_output_tokens


def do_pair(item, run, spend):
    case, x, y = item['case'], item['x'], item['y']
    ax, ay = answers(run, x, case), answers(run, y, case)
    first_is_a = random.Random(f'{SEED}:{case}:{x}:{y}').random() < .5
    a, b = (ax, ay) if first_is_a else (ay, ax)
    parsed, raw, tin, tout = call(PAIR_PROMPT.format(a=fmt(a), b=fmt(b)), PAIR_SYSTEM, spend)
    pick = (parsed or {}).get('more_disagreeable')
    pick = pick.strip().upper() if isinstance(pick, str) else None
    ok = pick in ('A', 'B')
    return dict(item, A=x if first_is_a else y, B=y if first_is_a else x, pick=pick if ok else None,
                first_picked=(pick == ('A' if first_is_a else 'B')) if ok else None,
                status='ok' if ok else 'parse_fail', raw=raw, tokens_in=tin, tokens_out=tout)


def do_remark(item, run, spend):
    ans = answers(run, item['arm'], item['case'])[item['start']:item['start'] + CHUNK]
    parsed, raw, tin, tout = call(REMARK_PROMPT.format(replies=fmt(ans)), REMARK_SYSTEM, spend)
    labels = (parsed or {}).get('labels')
    ok = isinstance(labels, list) and len(labels) == len(ans) and all(isinstance(l, dict) and isinstance(l.get('remark'), bool) for l in labels)
    return dict(item, n_replies=len(ans), labels=[{'remark': l['remark'], 'quote': l.get('quote', '')} for l in labels] if ok else None,
                status='ok' if ok else 'parse_fail', raw=raw, tokens_in=tin, tokens_out=tout)


def wilson(k, n, z=1.96):
    if not n:
        return float('nan'), float('nan')
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', default=RUNS + '/tone3/foam/run')
    ap.add_argument('--budget_usd', type=float, required=True)
    ap.add_argument('--cases', default=None, help='comma list, e.g. case_031,case_047')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tone3_judge.jsonl'))
    ap.add_argument('--pairs', default='rs_lowAc:rs_para,rs_lowA:rs_para,rs_lowAc:rs_lowA',
                    help='comma list of first:second arm pairs for (a); empty string skips (a)')
    ap.add_argument('--arms', default='rs_lowA,rs_lowAc,rs_para', help='arms for (b) remark labels; empty string skips (b)')
    args = ap.parse_args()
    run = os.path.abspath(args.run_dir)
    pairs = [tuple(p.split(':')) for p in args.pairs.split(',') if p]
    arms = [a for a in args.arms.split(',') if a]

    def cases_of(arm):
        return {os.path.basename(d) for d in glob.glob(chat_dir(run, arm, 'case_*')) if os.path.exists(d + '/clarifications.json')}
    want = set(args.cases.split(',')) if args.cases else None

    items = []
    for x, y in pairs:
        for c in sorted(cases_of(x) & cases_of(y)):
            if want is None or c in want:
                items.append(dict(kind='pair', run=run, case=c, x=x, y=y))
    for arm in arms:
        for c in sorted(cases_of(arm)):
            if want is None or c in want:
                for s in range(0, len(answers(run, arm, c)), CHUNK):
                    items.append(dict(kind='remark', run=run, case=c, arm=arm, start=s))

    key = lambda r: (r['kind'], r['run'], r['case'], r.get('x'), r.get('y'), r.get('arm'), r.get('start'))
    done = {}
    if os.path.exists(args.out):
        for l in open(args.out):
            r = json.loads(l)
            if r.get('status') == 'ok':
                done[key(r)] = r
    todo = [it for it in items if key(it) not in done]
    print(f'run {run}: {len(items)} items ({len(items) - len(todo)} cached, {len(todo)} to judge)')

    spend, lock, n_budget = Spend(args.budget_usd), threading.Lock(), 0

    def work(it):
        if spend.over():
            return None
        r = (do_pair if it['kind'] == 'pair' else do_remark)(it, run, spend)
        with lock:
            with open(args.out, 'a') as f:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        return r

    with ThreadPoolExecutor(args.workers) as ex:
        for fut in as_completed([ex.submit(work, it) for it in todo]):
            r = fut.result()
            if r is None:
                n_budget += 1
            elif r['status'] == 'ok':
                done[key(r)] = r
    n_fail = sum(key(it) not in done for it in todo) - n_budget
    print(f'this call: {spend.calls} model calls, spend ${spend.usd:.4f} (cap ${args.budget_usd}), '
          f'{n_fail} unparsed/failed, {n_budget} skipped (budget)')

    recs = [done[key(it)] for it in items if key(it) in done]
    if pairs:
        print('\n## (a) pairwise: share of cases where the first-named arm is judged more disagreeable (Wilson 95%)')
        for x, y in pairs:
            rs = [r for r in recs if r['kind'] == 'pair' and r['x'] == x and r['y'] == y]
            k, n = sum(r['first_picked'] for r in rs), len(rs)
            lo, hi = wilson(k, n)
            print(f'{x} vs {y}: {k}/{n} = {k / n if n else float("nan"):.3f} [{lo:.3f}, {hi:.3f}]  '
                  f'(first arm shown as A in {sum(r["A"] == x for r in rs)}/{n})')
    if arms:
        print('\n## (b) remark labels (chats counted only when all their replies are labelled)')
        for arm in arms:
            per = collections.defaultdict(list)
            for it in items:
                if it['kind'] == 'remark' and it['arm'] == arm:
                    per[it['case']].append(done.get(key(it)))
            full = {c: [l for r in v for l in r['labels']] for c, v in per.items() if all(v)}
            labs = [l['remark'] for v in full.values() for l in v]
            nc = sum(any(l['remark'] for l in v) for v in full.values())
            print(f'{arm}: replies {sum(labs)}/{len(labs)} = {sum(labs) / len(labs) if labs else float("nan"):.3f}  '
                  f'chats>=1 {nc}/{len(full)} = {nc / len(full) if full else float("nan"):.3f}  (chats total {len(per)})')


if __name__ == '__main__':
    main()
