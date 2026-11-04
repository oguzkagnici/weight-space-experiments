"""Model engine: holds LFM3-1.7B (bf16, CPU) in memory, swaps candidate tensors in, scores MMLU-Pro questions with
the fixed direct-answer protocol, and records internals (hidden states, MLP activations, module inputs, attention).

Only the forward-pass server uses this module; editors never run the model themselves."""
import hashlib, json, os, time
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import MODEL_DIR, WEIGHTS_FILE, LETTERS, CACHE, N_LAYERS, MAX_ATTN_TEXTS
from . import data

torch.set_num_threads(int(os.environ.get("WSP_THREADS", "4")))


def weights_key(dirs):
    """Identity of a weight state = the ordered list of edit dirs with each file's size and mtime."""
    h = hashlib.sha256()
    for d in dirs:
        for fn in sorted(os.listdir(d)) if d and os.path.isdir(d) else []:
            if fn.endswith(".safetensors"):
                st = os.stat(os.path.join(d, fn))
                h.update(f"{os.path.realpath(d)}/{fn}:{st.st_size}:{st.st_mtime_ns};".encode())
    return h.hexdigest()[:20]


def read_edit_dir(d):
    out = {}
    if d and os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".safetensors"):
                t = load_file(os.path.join(d, fn))["tensor"]
                out[fn[: -len(".safetensors")]] = t
    return out


class Engine:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa").eval()
        self.params = dict(self.model.named_parameters())
        self.stash = {}            # name -> original bf16 tensor, for tensors currently replaced
        self.current = None
        # the rendered prompt ends with the token " (" ; the answer is the next token, the bare letter
        ctx = self.tok.encode(data.ASSISTANT_PREFIX, add_special_tokens=False)
        self.letter_ids = []
        for l in LETTERS:
            full = self.tok.encode(data.ASSISTANT_PREFIX + l, add_special_tokens=False)
            assert full[:-1] == ctx and len(full) == len(ctx) + 1, (l, full, ctx)
            self.letter_ids.append(full[-1])
        assert len(set(self.letter_ids)) == 10
        self.layers = self.model.model.layers

    # ------------------------------------------------------------------ weights
    def set_weights(self, dirs):
        key = weights_key(dirs)
        if key == self.current:
            return key
        new = {}
        for d in dirs:
            new.update(read_edit_dir(d))
        with torch.no_grad():
            for name in list(self.stash):
                if name not in new:
                    self.params[name].data.copy_(self.stash.pop(name))
            for name, t in new.items():
                p = self.params[name]
                assert tuple(t.shape) == tuple(p.shape), (name, t.shape, p.shape)
                if name not in self.stash:
                    self.stash[name] = p.data.clone()
                p.data.copy_(t.to(torch.bfloat16))
        self.current = key
        return key

    # ------------------------------------------------------------------ hooks
    def _hooks(self, spec, store):
        """spec: {"hidden_states": [l...] (0=embeddings, l=after layer l), "mlp": [layer idx 0..23] (input of w2),
        "inputs": [module names] (input of that module), "outputs": [module names]}"""
        hs = []
        mods = dict(self.model.named_modules())
        def cap(key, which):
            def f(mod, inp, out):
                x = inp[0] if which == "in" else (out[0] if isinstance(out, tuple) else out)
                store[key] = x.detach()[0].float()
            return f
        for l in spec.get("hidden_states", []) or []:
            if l == 0:
                hs.append(self.model.model.embed_tokens.register_forward_hook(cap("h0", "out")))
            else:
                hs.append(self.layers[l - 1].register_forward_hook(cap(f"h{l}", "out")))
        for l in spec.get("mlp", []) or []:
            hs.append(self.layers[l].feed_forward.w2.register_forward_hook(cap(f"mlp{l}", "in")))
        for name in spec.get("inputs", []) or []:
            hs.append(mods[name].register_forward_hook(cap(f"in:{name}", "in")))
        for name in spec.get("outputs", []) or []:
            hs.append(mods[name].register_forward_hook(cap(f"out:{name}", "out")))
        return hs

    # ------------------------------------------------------------------ scoring
    def score(self, qids, acts=None, cache_key=None):
        """Score questions. Returns list of per-question dicts; optionally collects activations at the answer
        position (last token) -> dict key -> np.ndarray [n, d]."""
        Q = data.questions()
        cache = {}
        cpath = os.path.join(CACHE, f"score_{cache_key}.jsonl") if cache_key else None
        if cpath and os.path.exists(cpath) and not acts:
            for line in open(cpath):
                r = json.loads(line); cache[r["id"]] = r
        rows, act_out = [], {}
        new_rows = []
        for qid in qids:
            if qid in cache:
                rows.append(cache[qid]); continue
            q = Q[qid]
            ids = self.tok(data.render(q, self.tok), return_tensors="pt", add_special_tokens=False).input_ids
            store = {}
            hooks = self._hooks(acts, store) if acts else []
            try:
                with torch.inference_mode():
                    logits = self.model(ids).logits[0, -1].float()
            finally:
                for h in hooks: h.remove()
            k = len(q["options"])
            ll = logits[self.letter_ids[:k]]
            lp = torch.log_softmax(ll, -1)
            pred = int(torch.argmax(ll)); a = q["answer_index"]
            other = torch.cat([ll[:a], ll[a + 1:]])
            # mass the full vocabulary puts on the option letters (format adherence)
            full_lp = torch.log_softmax(logits, -1)
            r = {"id": qid, "category": q["category"], "answer": a, "pred": pred, "correct": int(pred == a),
                 "logp_correct": round(float(lp[a]), 5), "margin": round(float(ll[a] - other.max()), 4),
                 "letter_mass": round(float(full_lp[self.letter_ids[:k]].exp().sum()), 5),
                 "probs": [round(float(x), 5) for x in lp.exp()], "n_tokens": int(ids.shape[1])}
            rows.append(r); new_rows.append(r)
            for key, v in store.items():
                act_out.setdefault(key, []).append(v[-1].half().numpy())
        if cpath and new_rows:
            with open(cpath, "a") as f:
                for r in new_rows: f.write(json.dumps(r) + "\n")
        return rows, {k: np.stack(v) for k, v in act_out.items()}

    # ------------------------------------------------------------------ teacher-forced read
    def read(self, texts, ret, out_dir):
        """Teacher-forced pass over raw texts. Saves .npy files to out_dir. ret keys: token_logprobs, topk,
        hidden_states/mlp/inputs/outputs (as in _hooks), positions ('last' | 'all' | 'mean'), attn_layers."""
        os.makedirs(out_dir, exist_ok=True)
        positions = ret.get("positions", "last")
        topk = int(ret.get("topk", 0) or 0)
        attn_layers = ret.get("attn_layers") or []
        meta = {"texts": len(texts), "positions": positions, "n_tokens": [], "files": {}}
        tok_lp, topk_ids, topk_lp, acts = [], [], [], {}
        attn_out = {}
        for ti, text in enumerate(texts):
            ids = self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids
            meta["n_tokens"].append(int(ids.shape[1]))
            store = {}
            hooks = self._hooks(ret, store)
            want_attn = bool(attn_layers) and ti < MAX_ATTN_TEXTS
            if want_attn:
                self.model.set_attn_implementation("eager")
            try:
                with torch.inference_mode():
                    out = self.model(ids, output_attentions=want_attn)
            finally:
                for h in hooks: h.remove()
                if want_attn:
                    self.model.set_attn_implementation("sdpa")
            logits = out.logits[0].float()
            lsm = torch.log_softmax(logits, -1)
            if ret.get("token_logprobs"):
                tgt = ids[0, 1:]
                tok_lp.append(lsm[:-1].gather(1, tgt[:, None])[:, 0].numpy().astype(np.float32))
            if topk:
                v, i = lsm.topk(topk, -1)
                if positions == "all":
                    topk_ids.append(i.numpy().astype(np.int32)); topk_lp.append(v.numpy().astype(np.float32))
                else:
                    topk_ids.append(i[-1].numpy().astype(np.int32)); topk_lp.append(v[-1].numpy().astype(np.float32))
            for key, v in store.items():
                if positions == "all":
                    acts.setdefault(key, []).append(v.half().numpy())
                elif positions == "mean":
                    acts.setdefault(key, []).append(v.mean(0).half().numpy())
                else:
                    acts.setdefault(key, []).append(v[-1].half().numpy())
            if want_attn and out.attentions is not None:
                for l in attn_layers:
                    a = out.attentions[l]
                    if a is not None:
                        attn_out[f"attn_L{l}_text{ti}"] = a[0].float().numpy().astype(np.float16)
        def save(name, arr):
            np.save(os.path.join(out_dir, name + ".npy"), arr); meta["files"][name] = list(arr.shape)
        lens = np.array(meta["n_tokens"])
        offsets = np.concatenate([[0], np.cumsum(lens)])
        if tok_lp:
            save("token_logprobs", np.concatenate(tok_lp)); meta["token_logprobs_note"] = (
                "concatenated over texts; text i has n_tokens[i]-1 entries (logprob of token t+1 given prefix)")
        if topk:
            save("topk_ids", np.concatenate(topk_ids) if positions == "all" else np.stack(topk_ids))
            save("topk_logprobs", np.concatenate(topk_lp) if positions == "all" else np.stack(topk_lp))
        for key, v in acts.items():
            fname = key.replace(":", "_").replace("/", "_")
            save(fname, np.concatenate(v) if positions == "all" else np.stack(v))
        for key, v in attn_out.items():
            save(key, v)
        if positions == "all":
            save("offsets", offsets)
        json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
        return meta
