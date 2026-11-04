# Hypothesis: the L22 retrieval head 12 itself has a letter-position bias (via where it attends), which E1 (MLP neurons only) does not address.
import json, numpy as np
exec(open('code/011_screen.py').read().split('# heads layer 22')[0].split("for thr in")[0])
x = np.load(A + 'in_model.layers.22.self_attn.o_proj.npy').astype(np.float32)
Wo = load_original('model.layers.22.self_attn.o_proj.weight').numpy(); pw = load_original('model.layers.22.post_operator_norm.weight').numpy()
y = x @ Wo.T; rq = np.sqrt((y ** 2).mean(1) + 1e-5)
for h in [12, 15]:
    yh = x[:, h*128:(h+1)*128] @ Wo[:, h*128:(h+1)*128].T; d = (pw * yh / (rq * rms24)[:, None]) @ Uc
    dm = np.where(mask, d, -1e9)
    print('head', h, 'pick counts', np.bincount(dm.argmax(1), minlength=10), 'gold', np.bincount(ans, minlength=10))
    dc = d - d.mean(1, keepdims=True)
    print('   mean centered push (10-opt)', np.round(dc[m10].mean(0), 2))
    print('   acc by gold letter', [round(float((dm.argmax(1) == ans)[ans == k].mean()), 2) for k in range(10)])
