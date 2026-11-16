# Hypothesis: on cue questions with right samples, right samples mention the decisive cue's key words more often / earlier than wrong samples.
import json, sys, re, importlib, collections
sys.path.insert(0,'code'); m=importlib.import_module('001_overview')
rows=m.load_rows()
KEY={'mmlupro_6110':r'informed|open[- ]label|blind','mmlupro_2654':r'falling|dream(?:ed|t)|content of the dream|frighten|scar','mmlupro_6562':r'smok|COPD|emphysema|chronic obstructive','mmlupro_6800':r'transillumination|sinusitis|maxillary',
     'mmlupro_1196':r'30[- ]day|another 30|extend','mmlupro_1275':r'permission of the mayor|discretion|prior restraint|standards','mmlupro_1643':r'made that day|after (?:the )?(?:trial|judgment)|at the time of (?:the )?trial|existed','mmlupro_4811':r'1450|1750','mmlupro_6614':r'BMI|obes|weight','mmlupro_1705':r'fidget|nervous','mmlupro_1409':r'WEAKEST|weakest','mmlupro_1445':r'all commercial advertising|more extensive|ban'}
for q,pat in KEY.items():
    for r in sorted([r for r in rows if r['id']==q],key=lambda r:r['sample']):
        raw=r['raw'].split('</think>')[0]; n=len(raw.split())
        st='cut  ' if r['cut'] else ('RIGHT' if r['correct'] else 'wrong')
        ms=[mm.start() for mm in re.finditer(pat,raw)]
        late=[x for x in ms if x>0.25*len(raw)]
        print(q,r['sample'],st,'words %5d | cue-word mentions %3d (%.1f per 1,000 words) | after the first quarter %3d (%.1f per 1,000)'%(n,len(ms),1000*len(ms)/n,len(late),1000*len(late)/(0.75*n)))
