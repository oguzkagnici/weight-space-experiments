"""Layer looping without training (D78), after the "Loop Transformer" of Z.ai's GLM-5.4: a span of layers runs again
right after itself, with the same weights. GLM-5.4 learns per-token damp gates and adapts the model by continued
training; here nothing is trained, so the only knob is a hand-set constant damping beta per extra pass.

Spec, `loop.json` in an edit dir (the last one among the weight dirs wins):
    {"span": [a, b], "repeats": 1, "beta": 1.0}      # or "beta": [b1, b2, ...], one per extra pass
Layers 0..b run, then a..b again `repeats` times, then b+1..23. In an extra pass every layer's residual writes are
multiplied by beta: LFM3 uses sandwich norms (x + post_norm(op(norm(x)))), so scaling the post-norm weights
(post_operator_norm / post_ffn_norm; GGUF: post_attention_norm / post_ffw_norm) scales each write exactly. Weight edits
in the other files of the weight dirs apply to every pass (the copies share the tensors)."""
import json, os, re

N_LAYERS = 24
POST_NORMS_HF = ("post_operator_norm", "post_ffn_norm")
POST_NORMS_GGUF = ("post_attention_norm.weight", "post_ffw_norm.weight")


def read_loop(dirs):
    spec = None
    for d in dirs or []:
        p = os.path.join(d, "loop.json") if d else None
        if p and os.path.exists(p):
            spec = json.load(open(p))
    return spec


def plan(spec, n=N_LAYERS):
    """-> (order, scale): order[i] = original layer run at position i; scale[i] = beta for extra-pass copies, else None."""
    a, b = spec["span"]; R = int(spec.get("repeats", 1))
    assert 0 <= a <= b < n and R >= 1, spec
    betas = spec.get("beta", 1.0)
    betas = [float(betas)] * R if not isinstance(betas, list) else [float(x) for x in betas]
    assert len(betas) == R, spec
    order = list(range(b + 1)); scale = [None] * (b + 1)
    for r in range(R):
        order += list(range(a, b + 1)); scale += [betas[r]] * (b - a + 1)
    order += list(range(b + 1, n)); scale += [None] * (n - b - 1)
    return order, scale


def write_looped_gguf(src, dst, spec):
    """Copy a GGUF (any quantisation) with the loop applied: repeated blocks are raw copies of the source tensors; the
    per-layer metadata lists are extended; extra-pass post-norm weights (F32) are multiplied by beta."""
    import numpy as np, gguf
    r = gguf.GGUFReader(src)
    arch = r.fields["general.architecture"].contents()
    n = int(r.fields[f"{arch}.block_count"].contents())
    order, scale = plan(spec, n)
    w = gguf.GGUFWriter(dst, arch)
    for k, f in r.fields.items():
        if k.startswith("GGUF.") or k == "general.architecture":
            continue
        v = f.contents()
        if k == f"{arch}.block_count":
            v = len(order)
        elif k in (f"{arch}.attention.head_count_kv", f"{arch}.attention.sliding_window_pattern"):
            v = [v[o] for o in order]
        if f.types[0] == gguf.GGUFValueType.ARRAY:
            w.add_key_value(k, v, f.types[0], sub_type=f.types[-1])
        else:
            w.add_key_value(k, v, f.types[0])
    by_layer = {}
    for t in r.tensors:
        m = re.match(r"blk\.(\d+)\.(.+)$", t.name)
        if m:
            by_layer.setdefault(int(m.group(1)), []).append((m.group(2), t))

    def add(name, t, s=None):
        data = t.data                     # memmap view: no copy, so a looped build needs no extra RAM per block
        if s is not None:
            assert t.tensor_type == gguf.GGMLQuantizationType.F32, (name, t.tensor_type)
            data = np.asarray(data) * np.float32(s)
        plain = t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
        w.add_tensor(name, data, raw_shape=data.shape if plain else None, raw_dtype=t.tensor_type)

    for t in r.tensors:
        if not t.name.startswith("blk."):
            add(t.name, t)
    for i, (o, s) in enumerate(zip(order, scale)):
        for suffix, t in by_layer[o]:
            add(f"blk.{i}.{suffix}", t, s if suffix in POST_NORMS_GGUF else None)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return order, scale


def apply_torch(model, spec, orig_layers, orig_types):
    """Rebuild model.model.layers for the spec (None restores the original stack). Extra-pass copies share every
    submodule with the original layer except their own scaled post-norms. Callers must run with use_cache=False."""
    import copy, torch
    cfg = model.config
    if not spec:
        model.model.layers = orig_layers
        cfg.num_hidden_layers = len(orig_layers); cfg.layer_types = list(orig_types)
        return orig_layers
    order, scale = plan(spec, len(orig_layers))
    mods = []
    for o, s in zip(order, scale):
        layer = orig_layers[o]
        if s is None:
            mods.append(layer); continue
        c = copy.copy(layer)
        c._modules = dict(layer._modules)           # own module table, shared submodules
        for nm in POST_NORMS_HF:
            nrm = copy.deepcopy(layer._modules[nm])
            with torch.no_grad():
                nrm.weight.mul_(s)
            c._modules[nm] = nrm
        mods.append(c)
    model.model.layers = torch.nn.ModuleList(mods)
    cfg.num_hidden_layers = len(mods); cfg.layer_types = [orig_types[o] for o in order]
    return model.model.layers
