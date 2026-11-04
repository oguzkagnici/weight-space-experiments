import sys, time, torch, random, collections
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(4)
path = sys.argv[1]
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, trust_remote_code=True).eval()
ds = load_dataset("TIGER-Lab/MMLU-Pro")["test"]
by = collections.defaultdict(list)
for i, c in enumerate(ds["category"]): by[c].append(i)
rng = random.Random(0); idx = [j for c in sorted(by) for j in rng.sample(by[c], 10)]
L = "ABCDEFGHIJ"
letter_ids = [tok.encode(f"({l}", add_special_tokens=False)[-1] for l in L]
assert len(set(letter_ids)) == 10, letter_ids
correct = 0; t = time.time(); ntok = 0; pos = collections.Counter()
for k, i in enumerate(idx):
    ex = ds[i]; opts = ex["options"]
    q = ex["question"] + "\n\n" + "\n".join(f"({L[j]}) {o}" for j, o in enumerate(opts)) + "\n\nAnswer with the letter of the correct option."
    prompt = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    ids = tok(prompt + "The answer is (", return_tensors="pt", add_special_tokens=False).input_ids
    ntok += ids.shape[1]
    with torch.inference_mode():
        logits = model(ids).logits[0, -1].float()
    pred = int(torch.argmax(logits[letter_ids[:len(opts)]])); pos[L[pred]] += 1
    correct += pred == ex["answer_index"]
print(f"{path.split('/')[-1]}: acc {correct}/{len(idx)} = {100*correct/len(idx):.1f}%  ({time.time()-t:.0f}s, {ntok/len(idx):.0f} tok/q)  pred-letter dist {dict(sorted(pos.items()))}")
print("prompt tail:", repr(prompt[-120:]))
