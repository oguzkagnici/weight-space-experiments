"""MMLU-Pro splits and the fixed evaluation prompt."""
import json, os, random, collections
from .config import DATA, LETTERS

SPLIT_FILE = os.path.join(DATA, "mmlupro_splits.json")
QFILE = os.path.join(DATA, "mmlupro_test.jsonl")
# fraction per category: practice (editors see it, with answers), selection pool (hidden, used by the GA to choose
# winners), heldout (never used until checkpoint/final evaluations)
FRACTIONS = {"practice": 0.25, "selection": 0.17}

INSTRUCTION = "Answer with the letter of the correct option."
ASSISTANT_PREFIX = "The answer is ("


def build_splits():
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro")["test"]
    os.makedirs(DATA, exist_ok=True)
    by = collections.defaultdict(list)
    with open(QFILE, "w") as f:
        for ex in ds:
            q = {"id": f"mmlupro_{ex['question_id']}", "category": ex["category"], "src": ex["src"],
                 "question": ex["question"], "options": ex["options"], "answer": ex["answer"],
                 "answer_index": ex["answer_index"]}
            f.write(json.dumps(q) + "\n")
            by[q["category"]].append(q["id"])
    splits = {"practice": [], "selection": [], "heldout": []}
    for cat in sorted(by):
        ids = sorted(by[cat]); random.Random(f"wsp-2026-{cat}").shuffle(ids)
        a = round(len(ids) * FRACTIONS["practice"]); b = a + round(len(ids) * FRACTIONS["selection"])
        splits["practice"] += ids[:a]; splits["selection"] += ids[a:b]; splits["heldout"] += ids[b:]
    json.dump(splits, open(SPLIT_FILE, "w"))
    return {k: len(v) for k, v in splits.items()}


GPQA_FILE = os.path.join(DATA, "gpqa_diamond.jsonl")


def build_gpqa():
    """GPQA Diamond (gated: needs HF_TOKEN). Options shuffled once with a fixed seed; same prompt format.
    Only used for checkpoint evaluations; never shown to editors."""
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
    with open(GPQA_FILE, "w") as f:
        for i, ex in enumerate(ds):
            opts = [ex["Correct Answer"], ex["Incorrect Answer 1"], ex["Incorrect Answer 2"], ex["Incorrect Answer 3"]]
            opts = [o.strip() for o in opts]
            order = list(range(4)); random.Random(f"gpqa-{i}").shuffle(order)
            q = {"id": f"gpqa_{i:03d}", "category": "gpqa:" + (ex.get("High-level domain") or "?"), "src": "gpqa_diamond",
                 "question": ex["Question"].strip(), "options": [opts[j] for j in order],
                 "answer": LETTERS[order.index(0)], "answer_index": order.index(0)}
            f.write(json.dumps(q) + "\n")
    return len(ds)


import threading
_Q = None
_QSIG = None
_QLOCK = threading.Lock()
def questions():
    """All questions by id (thread-safe; reloads if the data files change)."""
    global _Q, _QSIG
    sig = tuple(os.path.getmtime(f) if os.path.exists(f) else 0 for f in (QFILE, GPQA_FILE))
    if _Q is not None and sig == _QSIG:
        return _Q
    with _QLOCK:
        if _Q is None or sig != _QSIG:
            new = {}
            for fn in (QFILE, GPQA_FILE):
                if os.path.exists(fn):
                    for line in open(fn):
                        q = json.loads(line); new[q["id"]] = q
            _Q, _QSIG = new, sig            # publish the complete dict atomically
    return _Q


def gpqa_ids():
    return sorted(k for k in questions() if k.startswith("gpqa_"))


def splits():
    return json.load(open(SPLIT_FILE))


def user_message(q):
    opts = "\n".join(f"({LETTERS[j]}) {o}" for j, o in enumerate(q["options"]))
    return f"{q['question']}\n\n{opts}\n\n{INSTRUCTION}"


def render(q, tok):
    """The exact evaluation string: chat template, thinking disabled, assistant turn opened with 'The answer is ('.
    The model's next-token distribution over the option letters is its answer."""
    prompt = tok.apply_chat_template([{"role": "user", "content": user_message(q)}], add_generation_prompt=True,
                                     tokenize=False, enable_thinking=False)
    return prompt + ASSISTANT_PREFIX
