# Hypothesis: the early-letter prior at forced-commit positions is written by a few late MLP neurons (near-constant prior writers); DLA of each neuron to the letter ramp from recorded activations.
import json, numpy as np, sys, glob
from wsp.editor import *
d=sys.argv[1]; meta=json.load(open(d+'/meta.json'))
files=meta['files']
E=load('model.embed_tokens.weight')[LETTER_IDS].numpy(); g=load('model.embedding_norm.weight').numpy()
Ec=E-E.mean(0)
h24=np.load(d+'/'+[f for k,f in files.items() if k in('hidden_states_24','hidden_states.24') or f.startswith('hidden_states_24')][0]) if False else None
def f(key):
    for k,v in files.items():
        if k==key: return np.load(d+'/'+v) if isinstance(v,str) else None
    raise KeyError(key)
print(list(files.keys())[:50])
