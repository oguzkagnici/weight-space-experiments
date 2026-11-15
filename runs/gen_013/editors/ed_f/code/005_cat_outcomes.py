# Hypothesis: finished-wrong answers in quantitative categories are few and contain arithmetic slips; cut traces hold most slips.
import json, collections, sys
sys.path.insert(0,'code')
import importlib; m=importlib.import_module('003_arith_check2')
rows=m.load_rows(); eqs=m.scan(rows)
per=collections.defaultdict(lambda:[0,0])
for e in eqs:
    per[(e['id'],e['sample'])][0]+=1; per[(e['id'],e['sample'])][1]+=(not e['ok'] and e['rel']>0.003)
c=collections.defaultdict(lambda:[0,0,0])
for r in rows: c[r['category']][0 if r['correct'] else (2 if r['cut'] else 1)]+=1
print('category right/wrongfin/cut');[print('  %-18s'%k,v) for k,v in sorted(c.items())]
print('finished-wrong traces with >=1 equation:')
for r in rows:
    k=(r['id'],r['sample'])
    if not r['correct'] and not r['cut'] and per[k][0]>=1: print('  ',k,r['category'],'gold',r['gold'],'pred',r['pred'],'eqs',per[k][0],'bad(rel>.3%)',per[k][1],'tokens',r['tokens'])
# big slips (rel>0.3%), distinct, by outcome
big=[e for e in eqs if not e['ok'] and e['rel']>0.003]
print('equations',len(eqs),'slips with rel>0.3%:',len(big),'in right',sum(e['correct'] for e in big),'wrongfin',sum((not e['correct']) and (not e['cut']) for e in big),'cut',sum(e['cut'] for e in big))
tot=collections.Counter('right' if e['correct'] else ('cut' if e['cut'] else 'wrongfin') for e in eqs); print(tot)
