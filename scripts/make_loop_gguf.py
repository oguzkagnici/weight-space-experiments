"""Write a looped GGUF: layers [a, b] run a second time right after layer b (copies of the same tensors),
with the second pass's residual writes scaled by beta (post-norm weights x beta; LFM3 uses sandwich norms)."""
import sys, re, numpy as np, gguf
src, dst, a, b, beta = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
r = gguf.GGUFReader(src)
arch = "lfm3"
n = int(r.fields[f"{arch}.block_count"].contents())
order = list(range(0, b + 1)) + list(range(a, b + 1)) + list(range(b + 1, n))   # new index -> old layer
second = set(range(b + 1, b + 1 + (b - a + 1)))                                   # new indices of the pass-2 copies
w = gguf.GGUFWriter(dst, arch)
for k, f in r.fields.items():
    if k.startswith("GGUF.") or k == "general.architecture":
        continue
    v = f.contents()
    if k == f"{arch}.block_count":
        v = len(order)
    elif k in (f"{arch}.attention.head_count_kv", f"{arch}.attention.sliding_window_pattern"):
        v = [v[o] for o in order]
    vt = f.types[0]
    if vt == gguf.GGUFValueType.ARRAY:
        w.add_key_value(k, v, vt, sub_type=f.types[-1])
    else:
        w.add_key_value(k, v, vt)
tens = {t.name: t for t in r.tensors}
def add(name, t, scale=None):
    data = np.array(t.data)
    if scale is not None:
        assert t.tensor_type == gguf.GGMLQuantizationType.F32, (name, t.tensor_type)
        data = data * np.float32(scale)
    w.add_tensor(name, data, raw_shape=data.shape if t.tensor_type in (0, 1) else None, raw_dtype=t.tensor_type)
for t in r.tensors:
    if not t.name.startswith("blk."):
        add(t.name, t)
for new, old in enumerate(order):
    for t in r.tensors:
        m = re.match(rf"blk\.{old}\.(.+)$", t.name)
        if not m:
            continue
        sc = beta if (new in second and m.group(1) in ("post_attention_norm.weight", "post_ffw_norm.weight")) else None
        add(f"blk.{new}.{m.group(1)}", t, sc)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("layers", len(order), "order", order)
