# Hypothesis: specific heads carry the inserted cue quote to the conclusion position (attention from the forced-answer "(" to the quote), and their attention to the quote / to the stem relates to the size of the cue effect.
# usage: 010_attn_regions.py <pass> <dir>
import json, sys, numpy as np, importlib, collections
sys.path.insert(0,'code'); b=importlib.import_module('005_build_ins')
from wsp.editor import *
name,D=sys.argv[1],sys.argv[2]; Q=questions()
meta=json.load(open('work/%s_meta.json'%name)); texts=json.load(open('work/%s_texts.json'%name)); off=np.load(D+'/offsets_tok.npy')
LAY=[2,4,6,8,10,12,14,16,18,20,22]
REG=['stem','cue_src','options','instr','trace','ins_quote','ins_other','probe','special']
def regions(text,mt):
    e=tok(text,add_special_tokens=False,return_offsets_mapping=True); om=e['offset_mapping']; n=len(om)
    q=Q[mt['id']]['question']; qs=text.find(q); assert qs>=0
    th=text.find('<think>'); osn=qs+len(q); instr=text.find('Think step by step')
    lab=np.zeros(n,int)
    cue=b.ANN[mt['id']][0] if mt['id'] in b.ANN else None; cs=text.find(cue) if cue else -1
    ins_s=text.rfind('Wait, let me re-read') if mt.get('arm') in('CUE','NON','INF') else -1
    if mt.get('arm')=='NON' and ins_s<0: ins_s=text.rfind('Wait, let me think about this again')
    pr=text.rfind(b.PROBE) if mt['kind']=='ins' else len(text)+1
    qq=text.find('"',ins_s)+1 if ins_s>=0 else -1; qe=text.find('".',qq) if ins_s>=0 else -1
    for i,(a,c) in enumerate(om):
        if a>=pr: r='probe'
        elif ins_s>=0 and a>=ins_s: r='ins_quote' if (qq<=a<qe and 're-read' in text[ins_s:ins_s+30]) else 'ins_other'
        elif a>=th+7: r='trace'
        elif a>=instr and instr>0: r='instr'
        elif a>=osn: r='options'
        elif a>=qs: r='cue_src' if (cs>=0 and cs<=a<cs+len(cue)) else 'stem'
        else: r='special'
        lab[i]=REG.index(r)
    return lab
M=np.zeros((len(texts),len(LAY),16,len(REG)),np.float32)
A=[np.load(D+'/attn_last_L%d.npy'%l,mmap_mode='r') for l in LAY]
for i,(t,mt) in enumerate(zip(texts,meta)):
    if mt['kind']=='key': continue
    lab=regions(t,mt); n=off[i+1]-off[i]; assert n==len(lab),(n,len(lab))
    for li,a in enumerate(A):
        x=np.asarray(a[off[i]:off[i+1]],dtype=np.float32)
        for r in range(len(REG)): M[i,li,:,r]=x[lab==r].sum(0)
np.save('work/%s_attn_regions.npy'%name,M); json.dump(dict(REG=REG,LAY=LAY),open('work/attn_regions_meta.json','w'))
print('saved',M.shape)
