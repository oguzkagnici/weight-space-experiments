"""Export / restore a champion's edited tensors as a sparse diff against the base model (D16, D64).

The full edited tensors (runs/champions/<name>/weights/*.safetensors, ~63 MB each) are gitignored. Edits touch few
weights, so the diff (flat index + edited bf16 value per changed weight) is a few hundred KB and is committed instead.

  python scripts/champion_delta.py export <name>     # weights/ -> delta.npz (+ check)
  python scripts/champion_delta.py restore <name>    # delta.npz + base model -> weights/ (same mtimes -> same weights key)
"""
import json, os, sys
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = os.environ.get("WSP_ROOT", "/home/user/weight-space-experiments")
BASE = "/home/user/models/LFM3-1.7B/model.safetensors"


def _base(name):
    with safe_open(BASE, "pt") as f:
        return f.get_tensor(name)


def export(champ):
    d = os.path.join(ROOT, "runs", "champions", champ)
    wdir = os.path.join(d, "weights")
    arrays, meta = {}, {}
    total = 0
    for fn in sorted(os.listdir(wdir)):
        if not fn.endswith(".safetensors"):
            continue
        tname = fn[: -len(".safetensors")]
        raw = load_file(os.path.join(wdir, fn))["tensor"]          # float32 (engine.py key "tensor")
        # Both consumers round to bf16 (engine.py: t.to(bfloat16); gguf_patch.to_gguf_array -> BF16 GGUF), so the
        # bf16-rounded tensor IS the model that was evaluated. The diff stores exactly that.
        new = raw.to(torch.bfloat16)
        base = _base(tname)
        assert new.shape == base.shape and base.dtype == torch.bfloat16, (tname, new.shape, base.dtype)
        nb, bb = new.view(torch.int16).flatten(), base.view(torch.int16).flatten()
        idx = torch.nonzero(nb != bb).flatten()
        k = tname.replace(".", "_")
        arrays[f"{k}__idx"] = idx.numpy().astype(np.int64)
        arrays[f"{k}__val"] = nb[idx].numpy().astype(np.int16)        # raw bf16 bits
        meta[tname] = {"file": fn, "shape": list(new.shape), "n_changed": int(idx.numel()), "dtype": str(raw.dtype),
                       "mtime_ns": os.stat(os.path.join(wdir, fn)).st_mtime_ns}
        total += idx.numel()
    out = os.path.join(d, "delta.npz")
    np.savez_compressed(out, **arrays, meta=np.array(json.dumps(meta)))
    print(f"{champ}: {total} changed weights in {len(meta)} tensors -> {out} ({os.path.getsize(out)/1e3:.0f} kB)")
    return total


def restore(champ, out_dir=None):
    d = os.path.join(ROOT, "runs", "champions", champ)
    z = np.load(os.path.join(d, "delta.npz"))
    meta = json.loads(str(z["meta"]))
    wdir = out_dir or os.path.join(d, "weights")
    os.makedirs(wdir, exist_ok=True)
    for tname, m in meta.items():
        k = tname.replace(".", "_")
        t = _base(tname).clone()
        flat = t.view(torch.int16).flatten()
        flat[torch.from_numpy(z[f"{k}__idx"])] = torch.from_numpy(z[f"{k}__val"])
        t = flat.view(torch.bfloat16).reshape(m["shape"])
        p = os.path.join(wdir, m["file"])
        save_file({"tensor": t.to(getattr(torch, m.get("dtype", "torch.float32").split(".")[-1]))}, p)
        os.utime(p, ns=(m["mtime_ns"], m["mtime_ns"]))   # weights_key hashes path+size+mtime: keep it identical
    print(f"{champ}: restored {len(meta)} tensors into {wdir}")


if __name__ == "__main__":
    cmd, champ = sys.argv[1], sys.argv[2]
    if cmd == "export":
        export(champ)
        # round-trip check into a scratch dir
        import tempfile, shutil
        tmp = tempfile.mkdtemp()
        restore(champ, tmp)
        wdir = os.path.join(ROOT, "runs", "champions", champ, "weights")
        for fn in os.listdir(tmp):
            a = load_file(os.path.join(tmp, fn)); b = load_file(os.path.join(wdir, fn))
            assert torch.equal(a["tensor"].to(torch.bfloat16), b["tensor"].to(torch.bfloat16)), fn
            assert os.path.getsize(os.path.join(tmp, fn)) == os.path.getsize(os.path.join(wdir, fn)), fn
        shutil.rmtree(tmp)
        print("round trip: identical bf16 weights (what the engine and GGUF build use) and file sizes")
    elif cmd == "restore":
        restore(champ)
    else:
        sys.exit(__doc__)
