# Hypothesis: the residual at the newline after option s carries the next-token prediction "(s+1" (tied unembedding
# direction of the "(X" tokens), which L22.H12's value reads as letter s+1 -> the off-by-one in the transfer matrix.
import json, numpy as np, torch
from wsp.editor import *
X21 = np.load('work/p2_nlX_h21.npy'); X14 = np.load('work/p2_nlX_h14.npy')
N = X21.shape[0]; nopt = np.load('work/p1_nopt.npy')[:N]
m10 = nopt == 10
PAR = [tok.encode('\n(' + L, add_special_tokens=False)[-1] for L in 'ABCDEFGHIJ']
print('"(X" token ids', PAR)
E = load_base('model.embed_tokens.weight').numpy(); wn = load_base('model.embedding_norm.weight').numpy()
np.set_printoptions(precision=2, suppress=True, linewidth=160)
for nm, X in [('h14', X14), ('h21', X21)]:
    x = X[m10].astype(np.float32)
    x = x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-5) * wn
    for lab, rows in [('"(X" tokens', PAR), ('bare letters', LETTER_IDS)]:
        lg = x @ E[rows].T  # n x slot x 10
        lg = lg - lg.mean(-1, keepdims=True)
        M = lg.mean(0)
        print(f'{nm} lens at newline of slot s (rows) over {lab} (cols), centered:')
        print(M)
# cosine between "(X" embedding and bare letter embedding
Ep = E[PAR]; El = E[LETTER_IDS]
Ep_c = Ep - Ep.mean(0); El_c = El - El.mean(0)
C = (Ep_c / np.linalg.norm(Ep_c, axis=1, keepdims=True)) @ (El_c / np.linalg.norm(El_c, axis=1, keepdims=True)).T
print('cos(centered E["(X"], centered E[letter])'); print(C)
