"""Map LFM3 HF tensor names to the official GGUF and patch edited tensors into a copy of the BF16 GGUF.

Avoids running llama.cpp's converter: the official LiquidAI GGUF is copied and only the edited tensors' bytes are
overwritten in place (same shape and dtype). verify() checks the mapping value-for-value against the HF weights."""
import os, re, shutil
import numpy as np
import torch

GGUF_BF16 = "/home/user/models/LFM3-1.7B-GGUF/LFM3-1.7B-BF16.gguf"
MAP = {"operator_norm.weight": "attn_norm.weight", "post_operator_norm.weight": "post_attention_norm.weight",
       "ffn_norm.weight": "ffn_norm.weight", "post_ffn_norm.weight": "post_ffw_norm.weight",
       "feed_forward.w1.weight": "ffn_gate.weight", "feed_forward.w3.weight": "ffn_up.weight",
       "feed_forward.w2.weight": "ffn_down.weight", "conv.conv.weight": "shortconv.conv.weight",
       "conv.in_proj.weight": "shortconv.in_proj.weight", "conv.out_proj.weight": "shortconv.out_proj.weight",
       "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight",
       "self_attn.v_proj.weight": "attn_v.weight", "self_attn.o_proj.weight": "attn_output.weight",
       "self_attn.gate_proj.weight": "attn_gate.weight", "self_attn.q_norm.weight": "attn_q_norm.weight",
       "self_attn.k_norm.weight": "attn_k_norm.weight"}
TOP = {"model.embed_tokens.weight": "token_embd.weight", "model.embedding_norm.weight": "output_norm.weight"}


def gguf_name(hf):
    if hf in TOP:
        return TOP[hf]
    m = re.match(r"model\.layers\.(\d+)\.(.+)$", hf)
    return f"blk.{m.group(1)}.{MAP[m.group(2)]}"


def to_gguf_array(t, gtype):
    """HF tensor -> numpy array with the GGUF tensor's memory layout (numpy shape = reversed GGUF ne)."""
    t = t.detach().float()
    if t.dim() == 3:                       # depthwise conv (2048, 1, 7) -> (2048, 7)
        t = t[:, 0, :]
    if gtype == "BF16":
        return t.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    return t.numpy().astype(np.float32)


def verify(hf_file="/home/user/models/LFM3-1.7B/model.safetensors"):
    import gguf
    from safetensors import safe_open
    r = gguf.GGUFReader(GGUF_BF16)
    byname = {t.name: t for t in r.tensors}
    f = safe_open(hf_file, "pt"); bad = []
    for hf in f.keys():
        g = byname[gguf_name(hf)]
        want = to_gguf_array(f.get_tensor(hf), g.tensor_type.name)
        raw = np.asarray(g.data)
        have = raw.view(np.uint16) if g.tensor_type.name == "BF16" else raw   # BF16 is exposed as raw bytes
        if have.size != want.size or not np.array_equal(have.reshape(want.shape), want):
            bad.append(hf)
    return len(list(f.keys())), bad


def patch(edits, out_path):
    """edits: {hf_name: tensor}. Copies the official BF16 GGUF to out_path and overwrites the edited tensors."""
    import gguf
    shutil.copyfile(GGUF_BF16, out_path)
    r = gguf.GGUFReader(out_path, "r+")
    byname = {t.name: t for t in r.tensors}
    for hf, t in edits.items():
        g = byname[gguf_name(hf)]
        arr = to_gguf_array(t, g.tensor_type.name)
        dst = g.data                                   # BF16: uint8 byte view; F32: float32 view
        src = arr.view(np.uint8) if g.tensor_type.name == "BF16" else arr
        assert dst.size == src.size, (hf, dst.shape, src.shape)
        dst[...] = src.reshape(dst.shape)
    r.data.flush() if hasattr(r.data, "flush") else None
    del r
    return out_path
