"""Editor API (what the editing LLM gets; the analogue of the paper's pre-defined kernel functions).

    from wsp.editor import *
    NAMES                      all tensor names
    load(name)                 float32 tensor: your current edit if any, else the generation's original (= parent)
    load_original(name)        the parent's tensor (the current champion of the GA, which is the starting point)
    load_base(name)            the unedited LFM3-1.7B tensor (generation 0)
    save_edit(name, tensor)    stage an edit (refused for out-of-scope tensors / whole-tensor scaling)
    drop_edit(name)            remove an edit
    edited()                   names you currently edit
    submit_candidate(label, description)   freeze /edited as a candidate for the GA selection
    tok                        the tokenizer
    questions()                {id: question dict} for the PRACTICE split only
    render(qid)                the exact evaluation string for a practice question (ends with 'The answer is (')
    LETTER_IDS                 token ids of the answer letters A..J as they follow the rendered prompt

The editor's folder is $WSP_EDITOR_DIR (meta.json, edited/, work/, code/, notes.md, candidates/)."""
import json, os, shutil, hashlib, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file

from .config import WEIGHTS_FILE, FORBIDDEN, MODEL_DIR, LETTERS
from . import data as _data

torch.set_grad_enabled(False)
EDIR = os.environ.get("WSP_EDITOR_DIR")
assert EDIR and os.path.isdir(EDIR), "set WSP_EDITOR_DIR to your editor folder"
META = json.load(open(os.path.join(EDIR, "meta.json")))
EDITED = os.path.join(EDIR, "edited")
os.makedirs(EDITED, exist_ok=True)
PARENT = META.get("parent_weights")   # dir of the champion's edited tensors (None in generation 1 = base)

_f = safe_open(WEIGHTS_FILE, "pt")
NAMES = sorted(_f.keys())
SHAPES = {n: tuple(_f.get_slice(n).get_shape()) for n in NAMES}


def _p(d, name):
    return os.path.join(d, name + ".safetensors")


def load_base(name):
    return _f.get_tensor(name).float()


def load_original(name):
    if PARENT and os.path.exists(_p(PARENT, name)):
        return load_file(_p(PARENT, name))["tensor"].float()
    return load_base(name)


def load(name):
    if os.path.exists(_p(EDITED, name)):
        return load_file(_p(EDITED, name))["tensor"].float()
    return load_original(name)


def edited():
    return sorted(fn[:-12] for fn in os.listdir(EDITED) if fn.endswith(".safetensors"))


def save_edit(name, tensor):
    if name in FORBIDDEN or name not in SHAPES:
        raise ValueError(f"refused: {name} is out of scope (embedding / output layer / final norm) or unknown")
    t = torch.as_tensor(tensor).detach().float().contiguous()
    if tuple(t.shape) != SHAPES[name]:
        raise ValueError(f"refused: shape {tuple(t.shape)} != {SHAPES[name]}")
    if not torch.isfinite(t).all():
        raise ValueError("refused: non-finite values")
    o = load_original(name)
    # what is evaluated is the bfloat16 cast
    tb, ob = t.bfloat16(), o.bfloat16()
    changed = int((tb != ob).sum())
    if changed == 0:
        drop_edit(name)
        return f"no entry differs from the original after bfloat16 rounding: edit for {name} dropped"
    c = float((t * o).sum() / (o * o).sum().clamp_min(1e-30))
    if abs(c - 1) > 1e-6 and float((t - c * o).norm() / o.norm().clamp_min(1e-30)) < 1e-4:
        raise ValueError(f"refused: the edit only multiplies the whole tensor by a constant ({c:.4f})")
    save_file({"tensor": t}, _p(EDITED, name))
    rel = float((t - o).norm() / o.norm().clamp_min(1e-30))
    return f"saved {name}: {changed} of {t.numel()} entries differ after bf16 rounding, relative change {rel:.4%}"


def drop_edit(name):
    if os.path.exists(_p(EDITED, name)):
        os.remove(_p(EDITED, name))


def _sha(t):
    return hashlib.sha256(t.bfloat16().view(torch.int16).numpy().tobytes()).hexdigest()[:16]


def submit_candidate(label, description):
    """Freeze the current /edited as a candidate. description: what was changed and the evidence (shown to the GA)."""
    cdir = os.path.join(EDIR, "candidates")
    os.makedirs(cdir, exist_ok=True)
    existing = [d for d in os.listdir(cdir) if os.path.isdir(os.path.join(cdir, d))]
    if label in existing:
        raise ValueError(f"candidate {label} exists")
    if len(existing) >= META["max_candidates"]:
        raise ValueError(f"at most {META['max_candidates']} candidates per editor per generation")
    names = edited()
    if not names:
        raise ValueError("no edits staged")
    dst = os.path.join(cdir, label)
    os.makedirs(os.path.join(dst, "edited"))
    man = {"label": label, "editor": META["editor"], "gen": META["gen"], "time": time.time(), "tensors": {}}
    for n in names:
        shutil.copy2(_p(EDITED, n), _p(os.path.join(dst, "edited"), n))
        t, o = load(n), load_original(n)
        man["tensors"][n] = {"changed_bf16": int((t.bfloat16() != o.bfloat16()).sum()), "numel": t.numel(),
                             "rel_change": float((t - o).norm() / o.norm()), "sha_bf16": _sha(t)}
    open(os.path.join(dst, "description.md"), "w").write(description)
    json.dump(man, open(os.path.join(dst, "manifest.json"), "w"), indent=1)
    tot = sum(v["changed_bf16"] for v in man["tensors"].values())
    return f"candidate {label} submitted: {len(names)} tensors, {tot} weights changed"


# ------------------------------------------------------------------ data
tok = None
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
except Exception as e:  # pragma: no cover
    print("tokenizer not loaded:", e)

_PRACTICE = set(_data.splits()["practice"])


def questions():
    Q = _data.questions()
    return {k: Q[k] for k in Q if k in _PRACTICE}


def render(qid):
    if qid not in _PRACTICE:
        raise KeyError("only practice questions are available")
    return _data.render(_data.questions()[qid], tok)


def render_think(qid):
    """The exact thinking-mode prompt (phase 2) of a practice question; ends with "<think>"."""
    if qid not in _PRACTICE:
        raise KeyError("only practice questions are available")
    from .gen import render_prompt
    return render_prompt(_data.questions()[qid], tok)


def trace_text(row):
    """Prompt + exactly what the model generated, for a row of a generate pass (gen_<k>_<model>.jsonl)."""
    return render_think(row["id"]) + row["raw"]


LETTER_IDS = [tok.encode(_data.ASSISTANT_PREFIX + l, add_special_tokens=False)[-1] for l in LETTERS] if tok else None

__all__ = ["NAMES", "SHAPES", "load", "load_original", "load_base", "save_edit", "drop_edit", "edited",
           "submit_candidate", "tok", "questions", "render", "render_think", "trace_text", "LETTER_IDS", "EDIR", "META"]
