# Hypothesis: none (inspection of API, tensor names, question format, tokenization of options).
from wsp.editor import *
import collections, wsp.editor as E
print([n for n in dir(E) if not n.startswith('_')])
print(len(NAMES)); print([ (n,SHAPES[n]) for n in NAMES if 'layers.22.' in n])
qs=questions(); ids=sorted(qs); print(len(ids), ids[:3])
c=collections.Counter(len(q['options']) for q in qs.values()); print(c)
print(collections.Counter(q['category'] for q in qs.values()))
s=render(ids[0]); print(repr(s[-600:]))
enc=tok.encode(s) if hasattr(tok,'encode') else None
print(type(tok))
e=tok.encode(s); ids_=e.ids if hasattr(e,'ids') else e
print(len(ids_)); print(LETTER_IDS)
toks=[tok.decode([i]) for i in ids_[-80:]]; print(toks)
