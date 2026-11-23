"""Training-free layer looping on the champion (D78).

  python scripts/loop_sweep.py setup               write runs/loop/configs/*/loop.json and queue the direct screen
  python scripts/loop_sweep.py screen              paired direct-answer screen vs the champion
  python scripts/loop_sweep.py hold <name>...      queue thinking-mode passes (8,192) on the 48-question holdout bank
  python scripts/loop_sweep.py holdreport          paired vs the champion's banked holdout48 answers

Stage 1 (screen): direct-answer protocol (letter after the prompt, PyTorch engine) on 200 practice questions that are
not in any bank. Cheap, and catches configs that break the model. Stage 2 (hold): thinking mode at 8,192 tokens on the
48-question holdout bank (champion answers already banked, same seeds). Stage 3: the usual 256-question selection with
the D65 rule, via a GA generation."""
import json, math, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wsp import ga, data
from wsp.config import ROOT

LDIR = os.path.join(ROOT, "runs", "loop")
CHAMP = os.path.join(ROOT, "runs", "champions", "gen008_carry_stack_shortans_L23_finalist_L22", "weights")
HOLD_BANK = os.path.join(ROOT, "runs", "shared", "holdout48_bank_gen008champ_cap8192_det1.jsonl")
HOLD_IDS = os.path.join(ROOT, "runs", "shared", "holdout48_ids.json")

# span (inclusive), repeats, beta. LFM3 has 24 layers; GLM-5.4 loops its middle 28 of 78 (36%) once.
CONFIGS = {
    "L08-15_b1.0": ([8, 15], 1, 1.0), "L08-15_b0.5": ([8, 15], 1, 0.5),     # middle third (closest to GLM-5.4)
    "L06-17_b1.0": ([6, 17], 1, 1.0), "L06-17_b0.5": ([6, 17], 1, 0.5),     # middle half
    "L10-13_b1.0": ([10, 13], 1, 1.0), "L10-13_b0.5": ([10, 13], 1, 0.5),   # narrow middle
    "L04-19_b1.0": ([4, 19], 1, 1.0), "L04-19_b0.5": ([4, 19], 1, 0.5),     # wide
    "L04-11_b1.0": ([4, 11], 1, 1.0), "L04-11_b0.5": ([4, 11], 1, 0.5),     # lower middle
    "L12-19_b1.0": ([12, 19], 1, 1.0), "L12-19_b0.5": ([12, 19], 1, 0.5),   # upper middle
    "L08-15_x2_b0.5": ([8, 15], 2, 0.5),                                     # two extra passes
}


def cdir(name):
    return os.path.join(LDIR, "configs", name)


def screen_ids():
    p = os.path.join(LDIR, "screen_ids.json")
    if os.path.exists(p):
        return json.load(open(p))
    used = set()
    for f in os.listdir(os.path.join(ROOT, "runs", "shared")):
        if f.endswith("_ids.json"):
            used |= set(json.load(open(os.path.join(ROOT, "runs", "shared", f))))
    for f in os.listdir(os.path.join(ROOT, "runs", "shared")):
        if f.endswith(".jsonl"):
            used |= {json.loads(l)["id"] for l in open(os.path.join(ROOT, "runs", "shared", f))}
    pool = sorted(set(data.splits()["practice"]) - used)
    ids = sorted(random.Random("loop-screen").sample(pool, 200))
    os.makedirs(LDIR, exist_ok=True); json.dump(ids, open(p, "w"))
    return ids


def setup(priority=3):
    ids = screen_ids()
    ga.enqueue_eval("loop_screen_champion", [CHAMP], ids, os.path.join(LDIR, "screen", "champion.jsonl"), priority)
    for name, (span, rep, beta) in CONFIGS.items():
        os.makedirs(cdir(name), exist_ok=True)
        json.dump({"span": span, "repeats": rep, "beta": beta}, open(os.path.join(cdir(name), "loop.json"), "w"))
        ga.enqueue_eval(f"loop_screen_{name}", [CHAMP, cdir(name)], ids, os.path.join(LDIR, "screen", f"{name}.jsonl"), priority)
    print(f"queued {len(CONFIGS) + 1} screen jobs on {len(ids)} practice questions")


def _paired(a, b, key):
    ids = sorted(set(a) & set(b)); n = len(ids)
    d = [b[i][key] - a[i][key] for i in ids]; m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)
    return n, m, se


def screen():
    load = lambda p: {r["id"]: r for r in map(json.loads, open(p))}
    p0 = os.path.join(LDIR, "screen", "champion.jsonl")
    if not os.path.exists(p0):
        sys.exit("champion screen not done")
    ch = load(p0); out = []
    print(f"champion: acc {100 * sum(r['correct'] for r in ch.values()) / len(ch):.1f}  mean logp(gold) "
          f"{sum(r['logp_correct'] for r in ch.values()) / len(ch):.3f}  (n {len(ch)})")
    for name in CONFIGS:
        p = os.path.join(LDIR, "screen", f"{name}.jsonl")
        if not os.path.exists(p):
            print(f"{name:16s} not done"); continue
        c = load(p)
        n, da, sa = _paired(ch, c, "correct"); _, dl, sl = _paired(ch, c, "logp_correct")
        mass = sum(r["letter_mass"] for r in c.values()) / len(c)
        out.append((name, 100 * da, 100 * sa, dl, sl, mass))
        print(f"{name:16s} acc {100 * sum(r['correct'] for r in c.values()) / len(c):5.1f}  diff {100 * da:+5.1f} (SE {100 * sa:.1f})"
              f"  dlogp {dl:+.3f} (SE {sl:.3f})  letter mass {mass:.3f}")
    json.dump(out, open(os.path.join(LDIR, "screen_summary.json"), "w"), indent=1)


def hold(names, priority=3):
    ids = json.load(open(HOLD_IDS))
    for name in names:
        assert name in CONFIGS, name
        ga.enqueue_eval(f"loop_hold48_{name}", [CHAMP, cdir(name)], ids,
                        os.path.join(LDIR, "hold48", f"{name}.jsonl"), priority, protocol="think", cap=8192)
        print("queued", name)


def holdreport():
    bank = {(r["id"], r.get("sample", 0)): r for r in map(json.loads, open(HOLD_BANK))}
    bank = {k[0]: v for k, v in bank.items() if k[1] == 0}
    d = os.path.join(LDIR, "hold48")
    print(f"champion (bank): acc {100 * sum(r['correct'] for r in bank.values()) / len(bank):.1f}  "
          f"cut {sum(r['cut'] for r in bank.values())}  mean tokens {sum(r['tokens'] for r in bank.values()) / len(bank):.0f}")
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        c = {r["id"]: r for r in map(json.loads, open(os.path.join(d, f)))}
        n, m, se = _paired(bank, c, "correct")
        both = [i for i in c if i in bank and not c[i]["cut"] and not bank[i]["cut"]]
        print(f"{f[:-6]:16s} acc {100 * sum(r['correct'] for r in c.values()) / len(c):5.1f}  diff {100 * m:+5.1f} (SE {100 * se:.1f}, z {m / se if se else 0:+.2f})"
              f"  +{sum(c[i]['correct'] > bank[i]['correct'] for i in c)}/-{sum(c[i]['correct'] < bank[i]['correct'] for i in c)}"
              f"  cut {sum(r['cut'] for r in c.values())}  mean tokens {sum(r['tokens'] for r in c.values()) / len(c):.0f}  both-finished n {len(both)}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "setup": setup()
    elif cmd == "screen": screen()
    elif cmd == "hold": hold(sys.argv[2:])
    elif cmd == "holdreport": holdreport()
