import json, sys
import numpy as np, jax.numpy as jnp
from bindcraft.protein import Protein, StructurePrediction
from bindcraft.loss import build_losses
d = np.load(sys.argv[1])
def prot(prefix):
    return Protein(**{f: jnp.asarray(d[f'{prefix}.{f}']) for f in ('sequence','atoms','atom_mask','flags','residue_index')})
states = {'complex': {'binder': prot('in.binder'), 'target': prot('in.target')}}
met = {k.split('met.',1)[1]: jnp.asarray(d[k]) for k in d.files if k.startswith('met.')}
preds = {'complex': StructurePrediction({'binder': prot('pred.binder'), 'target': prot('pred.target')}, met)}
settings = json.load(open('settings/core/default.json'))
losses = build_losses(settings, seed=0)
out = {}
for name, entry in sorted(losses.items()):
    try:
        v = entry.function(states, preds)
        out[name] = repr(float(np.asarray(v)))
    except Exception as e:
        out[name] = f'ERR {type(e).__name__}: {e}'
json.dump(out, open(sys.argv[2],'w'), indent=1, sort_keys=True)
for k,v in sorted(out.items()): print(f'  {k:26s} {v}')
