# Hypothesis: (library) token-region labels of a probe text: stem, cue source, options, instruction, trace, inserted quote, other insertion text, probe, special.
import importlib, sys, numpy as np
sys.path.insert(0,'code'); b=importlib.import_module('005_build_ins')
from wsp.editor import *
Q=questions()
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
