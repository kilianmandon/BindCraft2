"""Steady-state wall time of one sequence_gradients call: isolates kernel speed from
compilation, Python overhead and campaign bookkeeping."""
import json, sys, time
import numpy as np, jax, jax.numpy as jnp
from bindcraft.protein import Protein, ResidueFlags
from bindcraft.af2 import AlphaFoldDesignModel
from bindcraft.loss import build_losses
from bindcraft.model_weights import model_weights

params, _ = model_weights(download=False)
target = Protein.from_structure('settings/target/structures/hPDL1.pdb')['A']
rng = np.random.default_rng(0)
a0, m0, i0 = Protein.unresolved_atom_arrays(55)
binder = Protein(sequence=jnp.asarray(0.01*rng.normal(size=(55,20)), dtype=jnp.float16),
                 atoms=a0, atom_mask=m0, residue_index=i0,
                 flags=jnp.full((55,), int(ResidueFlags.DESIGN), dtype=jnp.uint8))
settings = json.load(open('settings/core/default.json'))
losses = build_losses(settings, seed=0)
model = AlphaFoldDesignModel(presets='model_1_multimer_v3', data_dir=params, num_recycle=1,
                             dropout=True, key=jax.random.PRNGKey(0))
states = {'complex': {'binder': binder, 'target': target}}
def once():
    preds, grads = model.sequence_gradients(states, losses, model='model_1_multimer_v3',
                                            softmax_weight=0.3, one_hot_weight=0.0,
                                            temperature=1.0, logit_scale=2.0)[:2]
    jax.block_until_ready([g for g in grads.values()])
t0 = time.perf_counter(); once(); warm = time.perf_counter() - t0      # includes compile
times = []
for _ in range(6):
    t = time.perf_counter(); once(); times.append(time.perf_counter() - t)
out = {'first_call_with_compile': round(warm, 3), 'steady_times': [round(x,4) for x in times],
       'steady_median': round(float(np.median(times)), 4), 'steady_min': round(min(times), 4)}
json.dump(out, open(sys.argv[1],'w'), indent=1)
print(f"  first call (incl compile): {warm:7.2f}s")
print(f"  steady median            : {np.median(times):7.4f}s   min {min(times):.4f}s")
print(f"  all: {[round(x,3) for x in times]}")
