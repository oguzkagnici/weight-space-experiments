import time, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(4)
path = sys.argv[1]
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
t=time.time()
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, trust_remote_code=True).eval()
print(type(model).__name__, f"load {time.time()-t:.1f}s", sum(p.numel() for p in model.parameters())/1e9, "B params", flush=True)
text = "The mitochondria is the powerhouse of the cell. " * 40
ids = tok(text, return_tensors="pt").input_ids[:, :320]
with torch.inference_mode():
    model(ids)  # warmup
    for bs in (1, 8):
        x = ids.repeat(bs, 1)
        t = time.time(); model(x); dt = time.time()-t
        print(f"prefill bs={bs} len=320: {dt:.2f}s -> {bs*320/dt:.0f} tok/s", flush=True)
    msgs=[{"role":"user","content":"What is 17*23? Think briefly."}]
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    for bs in (1, 8):
        inp = {k: v.repeat(bs,1) for k,v in p.items()}
        t=time.time(); out = model.generate(**inp, max_new_tokens=64, do_sample=False, min_new_tokens=64); dt=time.time()-t
        print(f"decode bs={bs}: {bs*64/dt:.1f} tok/s total", flush=True)
