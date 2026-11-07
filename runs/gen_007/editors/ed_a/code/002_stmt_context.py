# Hypothesis: mid-trace answer statements in cut traces are followed by re-verification openers that n5414 fails to suppress; inspect context.
import json, re
B={json.loads(l)['id']:json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')}
pat=re.compile(r'answer is \(?([A-J])\)?')
ids=['mmlupro_8262','mmlupro_10964','mmlupro_1196','mmlupro_2055','mmlupro_1275','mmlupro_5510','mmlupro_6941','mmlupro_8932','mmlupro_6143','mmlupro_7373','mmlupro_10770','mmlupro_2687','mmlupro_5325','mmlupro_7261']
for i in ids:
    th=B[i]['think']
    for m in list(pat.finditer(th))[:2]:
        a=max(0,m.start()-200); b=m.end()+250
        s=th[a:b].encode('ascii','replace').decode().replace('\n','//')
        print(f"== {i} gold {B[i]['gold']} pos {m.start()/len(th):.2f}: {s}")
